"""Public /v1/tools routes: the catalog, call records, tool hints and approval suggestions
(design/TOOL_MEMORY.md).

The service never executes a tool. ``record`` is what an adapter calls after it ran one;
``hints`` answers which tool, which plan, which next step and which arguments, from what
previous runs did and what the catalog and the graph know. Every answer names only tools the
caller may call and is built from records the caller's visibility keys cover.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_service.api.caching import conditional_model
from memory_service.api.deps import (
    AdministeredTenantDep,
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
    request_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.headers import LINK_HEADER
from memory_service.api.idempotent import run_idempotent
from memory_service.api.pagination import CursorQuery, decode_cursor, link_next, next_link, page
from memory_service.api.params import SkillDraftIdPath, SuggestionIdPath, limit_query
from memory_service.api.schemas.context import ToolHintsResponse
from memory_service.api.validation import ToolJson, ToolOutput
from memory_service.domain.enums import Visibility
from memory_service.domain.learning import (
    APPROVAL_MIN_SUPPORT,
    APPROVAL_SUGGESTIONS_MAX,
    Suggestion,
)
from memory_service.domain.tools import (
    TOOL_SOURCE_DESCRIPTION,
    SideEffects,
    SkillDecision,
    ToolAnnotations,
    ToolDescriptor,
    ToolSource,
    ToolStats,
    ToolStatus,
)
from memory_service.modules.context.views import hints_view
from memory_service.modules.tools import approvals
from memory_service.modules.tools.hints import HINTS_K_MAX
from memory_service.modules.tools.service import CATALOG_MAX
from memory_service.modules.tools.skills import NAME_MAX, SkillDraft
from trellis.memory.approval import MAX_EXPRESSION_CHARS, parse

router = APIRouter()
#: 404: WORKSPACE visibility naming a workspace that is not a team (modules/tenancy/gate.py)
_ERRORS = error_responses(401, 403, 404, 422, 503)

_SCOPE: dict[str, Any] = {
    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "agent_run_id": "run_01J8ZK",
}


class SubCallIn(BaseModel):
    """One ``server.tool(...)`` call parsed out of a code-mode script, in script order."""

    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(..., ge=0, description="Its position in the script, from 0.")
    tool: str = Field(description="The tool the script called.")
    args: ToolJson = Field(
        default_factory=dict, description="The arguments it was called with (bounded JSON)."
    )
    bindings: dict[str, str] = Field(
        default_factory=dict,
        description="argument path -> expression it was bound from (e.g. 'id': 'step0.result.id')",
    )


class RecordRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scope": _SCOPE,
                    "tool": "pricing-lookup_price",
                    "args": {"sku": "SKU-22"},
                    "output": {"price": 1200, "currency": "EUR"},
                    "status": "ok",
                    "latency_ms": 42.0,
                    "task": "update quote Q-1183 with EMEA price for SKU-22",
                    "step": 0,
                }
            ]
        },
    )

    scope: ScopeBody = Field(
        default_factory=ScopeBody,
        description="The lineage the call acts in (thread, session, turn, work, agent, "
        "run). Tenant, workspace and user come from the trusted headers; a "
        "value here must agree with them. The agent run (agent_run_id) keys the"
        " step.",
    )
    tool: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The tool that ran (its catalog name; an unknown one is added).",
    )
    args: ToolJson = Field(
        default_factory=dict,
        description="The arguments it ran with (at most 64 KiB of JSON); redacted paths are"
        " never stored.",
    )
    output: ToolOutput = Field(
        default=None,
        description="What it returned (at most 256 KiB of JSON); large outputs go to the "
        "blob store and a summary is kept.",
    )
    output_summary: str | None = Field(
        default=None, description="A summary of the output to keep instead of deriving one."
    )
    status: ToolStatus = Field(
        default="ok",
        description="How the call ended: ok, error (the tool raised; name it in error_class), "
        "timeout, rejected (the agent or a policy refused to run it), or cancelled (the run "
        "stopped before the call finished).",
    )
    error_class: str | None = Field(
        default=None, description="The exception or error type, when status is error."
    )
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="How long the call took, in milliseconds."
    )
    cost: float | None = Field(
        default=None, ge=0.0, description="What the call cost, in the caller's unit (e.g. USD)."
    )
    task: str = Field(
        default="",
        max_length=4000,
        description="What the run was asked to do, in words: the pattern procedures are keyed on.",
    )
    step: int | None = Field(
        default=None,
        ge=0,
        description="The call's position in the run (from 0); omitted: the run's next.",
    )
    sub_calls: list[SubCallIn] = Field(
        default_factory=list,
        max_length=64,
        description="For a code-mode script: the tool calls it made, in order (at most 64).",
    )
    visibility: Visibility = Field(
        default=Visibility.PRIVATE,
        description=(
            "Who may see the record, narrowest first: PRIVATE (this agent alone, across runs "
            "- the default, and what a procedure is learned from), RUN (this run and the run "
            "that spawned it), THREAD, AGENT_GROUP, USER, WORKSPACE, TENANT."
        ),
    )


class RecordResponse(BaseModel):
    invocation_id: str = Field(description="The recorded call (tiv_...).")
    step: int = Field(description="The call's position in the run.")
    args_hash: str = Field(
        description="Digest of the full arguments: an identical call has the same one."
    )
    recorded: bool = Field(description="False when an identical call was already recorded.")


class CatalogEntry(BaseModel):
    """What a tool is and does."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The tool's name (1-200 characters), unique in the scope; the upsert key.",
    )
    description: str = Field(
        default="", max_length=4000, description="What the tool does (at most 4000 characters)."
    )
    input_schema: ToolJson = Field(
        default_factory=lambda: {"type": "object"},
        description="The JSON Schema of the tool's arguments (at most 64 KiB).",
    )
    required: list[str] | None = Field(
        default=None,
        max_length=100,
        description="required argument names; omitted, the input schema's `required`",
    )
    argument_entity_types: dict[str, str] = Field(
        default_factory=dict,
        description="argument name -> the entity type its value names (supplier: ORG), what "
        "hints fill from the knowledge graph and tool recording links in it",
    )
    side_effects: SideEffects | None = Field(
        default=None,
        description="read (changes nothing), write (changes what can be changed back), "
        "irreversible (cannot be undone); omitted when unknown",
    )
    source: ToolSource = Field(default="manual", description=TOOL_SOURCE_DESCRIPTION)
    server: str | None = Field(
        default=None, max_length=200, description="The MCP server that provides it, for source mcp."
    )
    examples: list[ToolJson] = Field(
        default_factory=list, max_length=20, description="Example argument objects (at most 20)."
    )
    redact: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="dotted argument paths whose values never reach storage",
    )
    annotations: ToolAnnotations = Field(
        default_factory=ToolAnnotations,
        description="the MCP annotations (readOnlyHint, destructiveHint, idempotentHint, "
        "openWorldHint); the risk tier follows them when side_effects is not set",
    )
    approve_when: str | None = Field(
        default=None,
        max_length=MAX_EXPRESSION_CHARS,
        description="ask a person before a call when this expression over the arguments is "
        'true, e.g. `amount >= 1000 or shape == "amount:num:1e3"` '
        "(trellis.memory.approval); an empty string removes it",
    )

    @field_validator("approve_when")
    @classmethod
    def _parses(cls, value: str | None) -> str | None:
        if value:
            parse(value)
        return value or None

    def to_domain(self) -> tuple[ToolDescriptor, frozenset[str]]:
        """The entry and the fields this request sets: an existing entry keeps the others."""
        required = self.required
        if required is None:
            required = [str(r) for r in self.input_schema.get("required", []) or []]
        entry = ToolDescriptor(
            tenant_id="",
            **self.model_dump(exclude={"required", "annotations"}),
            annotations=self.annotations,
            required=required,
        )
        fields = self.model_fields_set | {"description", "input_schema", "required"}
        return entry, frozenset(fields - {"name"})


class CatalogRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tools": [
                        {
                            "name": "erp-create_po",
                            "description": "Create a purchase order",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "supplier": {"type": "string"},
                                    "amount": {"type": "number"},
                                },
                                "required": ["supplier", "amount"],
                            },
                            "argument_entity_types": {"supplier": "ORG"},
                            "side_effects": "write",
                            "source": "mcp",
                            "server": "erp",
                        }
                    ]
                }
            ]
        },
    )

    scope: ScopeBody = Field(
        default_factory=ScopeBody,
        description="The lineage the call acts in (thread, session, turn, work, agent, "
        "run). Tenant, workspace and user come from the trusted headers; a "
        "value here must agree with them. A workspace in the scope makes the "
        "entries the workspace's own.",
    )
    tools: list[CatalogEntry] = Field(
        ...,
        min_length=1,
        max_length=CATALOG_MAX,
        description="The entries to upsert by name (1-500).",
    )


class ToolStatsBody(BaseModel):
    calls: int = Field(default=0, description="Recorded calls.")
    successes: int = Field(default=0, description="Calls that ended ok.")
    failures: int = Field(
        default=0, description="Calls that ended otherwise (error, timeout, rejected, cancelled)."
    )
    success_rate: float | None = Field(
        default=None, description="0..1, successes / calls; null before the first call."
    )
    avg_latency_ms: float | None = Field(
        default=None,
        description="Mean latency of the calls, in milliseconds; null when none reported one.",
    )
    approvals: int = Field(default=0, description="Approval decisions recorded for its calls.")
    rejections: int = Field(default=0, description="Rejection decisions recorded for its calls.")
    edits: int = Field(
        default=0, description="Decisions that changed the arguments before approving."
    )
    last_used_at: datetime | None = Field(
        default=None, description="When it was last called (ISO 8601, UTC); null: never."
    )

    @classmethod
    def of(cls, stats: ToolStats) -> ToolStatsBody:
        return cls(
            calls=stats.calls,
            successes=stats.successes,
            failures=stats.failures,
            success_rate=stats.success_rate,
            avg_latency_ms=stats.avg_latency_ms,
            approvals=stats.approvals,
            rejections=stats.rejections,
            edits=stats.edits,
            last_used_at=stats.last_used_at,
        )


class CatalogTool(BaseModel):
    tool_id: str = Field(description="The entry's id (tol_...).")
    name: str = Field(description="The tool's name.")
    version: int = Field(description="Bumped whenever the description or the schema changes.")
    description: str = Field(description="What the tool does.")
    input_schema: dict[str, Any] = Field(description="The JSON Schema of its arguments.")
    required: list[str] = Field(description="The required argument names.")
    argument_entity_types: dict[str, str] = Field(
        description="Argument name -> the entity type its value names (e.g. supplier: ORG)."
    )
    side_effects: SideEffects | None = Field(
        description="What a call changes, as set: read, write or irreversible; null when "
        "unknown (risk then follows the annotations)."
    )
    source: ToolSource = Field(description=TOOL_SOURCE_DESCRIPTION)
    server: str | None = Field(description="The MCP server that provides it, for source mcp.")
    examples: list[dict[str, Any]] = Field(description="Example argument objects.")
    annotations: dict[str, bool] = Field(
        description="the MCP annotations that were given (readOnlyHint, destructiveHint, "
        "idempotentHint, openWorldHint)"
    )
    approve_when: str | None = Field(
        description="The expression over the arguments that asks a person before a call "
        "(trellis.memory.approval); null: none."
    )
    risk: SideEffects = Field(
        description="read (run), write (run and notify) or irreversible (ask): side_effects "
        "when set, else the annotations"
    )
    schema_hash: str = Field(description="Digest of the input schema: equal hashes, equal schemas.")
    workspace_id: str | None = Field(
        description="The workspace whose own entry this is; null: the tenant's."
    )
    stats: ToolStatsBody = Field(
        default_factory=ToolStatsBody, description="What the calls recorded for it add up to."
    )

    @classmethod
    def of(cls, entry: ToolDescriptor, stats: ToolStats | None = None) -> CatalogTool:
        return cls(
            **entry.model_dump(
                include={
                    "tool_id",
                    "name",
                    "version",
                    "description",
                    "required",
                    "argument_entity_types",
                    "side_effects",
                    "source",
                    "server",
                    "examples",
                    "approve_when",
                    "schema_hash",
                    "workspace_id",
                }
            ),
            annotations=entry.annotations.model_dump(by_alias=True, exclude_none=True),
            risk=entry.risk,
            input_schema=entry.input_schema or {"type": "object"},
            stats=ToolStatsBody.of(stats or ToolStats(tool_name=entry.name)),
        )


class CatalogResponse(BaseModel):
    tools: list[CatalogTool] = Field(description="The entries, by name.")


class CatalogPageResponse(CatalogResponse):
    next_cursor: str | None = Field(
        default=None,
        description="Pass as `cursor` for the next page (also in `Link`); null on the last.",
    )


class HintsRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scope": _SCOPE,
                    "task": "order 500 sheets of A4 from Acme",
                    "available": ["erp-get_stock", "erp-create_po"],
                    "k": 8,
                }
            ]
        },
    )

    scope: ScopeBody = Field(
        default_factory=ScopeBody,
        description="The lineage the call acts in (thread, session, turn, work, agent, "
        "run). Tenant, workspace and user come from the trusted headers; a "
        "value here must agree with them.",
    )
    task: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="What the agent has to do, in words (1-4000 characters).",
    )
    available: list[str] | None = Field(
        default=None,
        max_length=500,
        description="the tools the caller can call; omitted, every catalog tool may be named",
    )
    k: int = Field(default=8, ge=1, le=HINTS_K_MAX, description="The most tools to return.")


class ApprovalSuggestionBody(BaseModel):
    id: str = Field(description="accept it with POST /v1/tools/approval-suggestions/{id}/accept")
    tool: str = Field(description="The tool the rule is for.")
    arg_shape: str = Field(
        description="argument names with their value kinds (numbers by magnitude)"
    )
    suggestion: Suggestion = Field(
        description="auto_approve: nearly every decision approved calls of this shape; "
        "always_ask: half or more were rejected or edited. Never applied by the service."
    )
    approvals: int = Field(description="How many calls of this shape were approved.")
    rejections: int = Field(description="How many were rejected.")
    edits: int = Field(description="How many had their arguments changed before approval.")
    support: int = Field(
        description="approvals + rejections + edits: the decisions the rule rests on."
    )
    approve_rate: float = Field(description="0..1, approvals / support.")
    agent_id: str | None = Field(
        description="The agent whose decisions these are; null: the tenant's."
    )
    accepted: bool = Field(description="the rule is already part of the tool's approve_when")


class ApprovalSuggestionsResponse(BaseModel):
    suggestions: list[ApprovalSuggestionBody] = Field(
        description="The rules offered, most supported first."
    )
    next_cursor: str | None = Field(
        default=None,
        description="Pass as `cursor` for the next page (also in `Link`); null on the last.",
    )


@router.post(
    "/tools/invocations",
    name="record_invocation",
    response_model=RecordResponse,
    status_code=202,
    tags=["tools"],
    summary="Record one tool call (idempotent on run + step + tool + arguments)",
    responses=_ERRORS,
)
async def record_invocation(
    request: Request, body: RecordRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response:
    """Recorded in the request, so there is no job to poll and no ``Location``: 202 says the
    call is counted, and what it teaches (procedures, statistics) is learned later."""
    ctx = build_context(request, container, body.scope)
    service = container.services["tool_memory"]

    async def handler(uow):  # type: ignore[no-untyped-def]
        invocation, created = await service.record(
            uow,
            ctx,
            **body.model_dump(exclude={"scope", "sub_calls"}),
            sub_calls=[c.model_dump() for c in body.sub_calls],
        )
        out = RecordResponse(
            invocation_id=invocation.invocation_id,
            step=invocation.step,
            args_hash=invocation.args_hash,
            recorded=created,
        )
        return 202, out.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=body.model_dump(mode="json"),
        handler=handler,
    )


#: The catalog changes when an entry or a statistic does, so a client revalidates each time
#: (``no-cache``) and a 304 spares it the body; ``private``: the answer is per tenant.
CATALOG_CACHE_CONTROL = "private, no-cache"


@router.get(
    "/tools",
    response_model=CatalogPageResponse,
    tags=["tools"],
    summary="The tool catalog visible in this scope, by name, with each tool's statistics "
    "(cursor paged; ETag / If-None-Match answer 304 when unchanged)",
    description="The response carries `ETag` (a digest of the page) and `Cache-Control: "
    "private, no-cache`; send the tag back in `If-None-Match` and an unchanged page is a "
    "`304` without a body - how a harness refreshes the approval tiers (`risk`, "
    "`approve_when`) cheaply. A page holds `limit` tools by name; the default is the whole "
    "catalog of a tenant at its 500-entry bound.",
    responses={**_ERRORS, 304: {"description": "Not modified: If-None-Match names the ETag"}},
)
async def list_tools(
    request: Request,
    ctx: HeaderContextDep,
    container: ContainerDep,
    names: Annotated[
        list[str] | None,
        Query(
            max_length=CATALOG_MAX,
            description="Only these tools (repeat the parameter); a name the catalog does not "
            "know is absent from the answer. Omit for every tool.",
        ),
    ] = None,
    cursor: CursorQuery = None,
    limit: Annotated[int, limit_query(CATALOG_MAX, "tools")] = CATALOG_MAX,
) -> Response:
    position = decode_cursor(cursor, fields=("name",))
    async with container.services["uow_factory"]() as uow:
        rows = await container.services["tool_memory"].catalog(
            uow, ctx, names, after=position["name"] if position else "", limit=limit + 1
        )
    items, next_cursor = page(rows, limit=limit, position=lambda row: {"name": row[0].name})
    body = CatalogPageResponse(
        tools=[CatalogTool.of(entry, stats) for entry, stats in items], next_cursor=next_cursor
    )
    headers = {LINK_HEADER: next_link(request, next_cursor)} if next_cursor else None
    return conditional_model(request, body, cache_control=CATALOG_CACHE_CONTROL, headers=headers)


@router.put(
    "/tools/catalog",
    response_model=CatalogResponse,
    tags=["tools"],
    summary="Upsert catalog entries by name (idempotent; unchanged entries are left alone)",
    responses=_ERRORS,
)
async def put_catalog(
    request: Request, body: CatalogRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response:
    ctx = build_context(request, container, body.scope)

    async def handler(uow):  # type: ignore[no-untyped-def]
        stored = await container.services["tool_memory"].put_catalog(
            uow, ctx, [entry.to_domain() for entry in body.tools]
        )
        out = CatalogResponse(tools=[CatalogTool.of(entry) for entry in stored])
        return 200, out.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=body.model_dump(mode="json"),
        handler=handler,
    )


@router.post(
    "/tools/hints",
    response_model=ToolHintsResponse,
    response_model_exclude_none=True,
    tags=["tools"],
    summary="Which tools fit a task, best first: confidence, track record, arguments, plan",
    responses=_ERRORS,
)
async def tool_hints(
    request: Request, body: HintsRequest, container: ContainerDep, _: ServicePrincipalDep
) -> dict[str, Any]:
    ctx = build_context(request, container, body.scope)
    visibility = await container.services["authz"].visibility(ctx)
    async with container.services["uow_factory"]() as uow:
        profile = await container.services["profile"].blocks(uow, ctx)
    hints = await container.services["tool_hints"].hints(
        ctx,
        body.task,
        available=body.available,
        k=body.k,
        scope_keys=list(visibility.keys),
        profile=profile,
    )
    return hints_view(hints)


@router.get(
    "/tools/approval-suggestions",
    response_model=ApprovalSuggestionsResponse,
    tags=["tools"],
    summary="Approval rules this agent's approve / reject / edit decisions support "
    "(suggestions only: never applied)",
    responses=_ERRORS,
)
async def approval_suggestions(
    request: Request,
    response: Response,
    ctx: HeaderContextDep,
    container: ContainerDep,
    tool: Annotated[
        str | None,
        Query(max_length=200, description="Only the suggestions for this tool (its name)."),
    ] = None,
    cursor: CursorQuery = None,
    limit: Annotated[
        int, limit_query(APPROVAL_SUGGESTIONS_MAX, "patterns")
    ] = APPROVAL_SUGGESTIONS_MAX,
) -> ApprovalSuggestionsResponse:
    """Most supported first. A page reads ``limit`` decision patterns and offers those that
    support a rule the tool does not have yet, so it may hold fewer suggestions than
    ``limit`` and still have a next page."""
    position = decode_cursor(cursor, fields={"support": int, "tool": str, "shape": str})
    after = (position["support"], position["tool"], position["shape"]) if position else None
    async with container.services["uow_factory"]() as uow:
        rows = await uow.tools.approval_patterns(
            ctx.tenant_id,
            ctx.agent_id or "",
            tool_name=tool,
            min_support=APPROVAL_MIN_SUPPORT,
            limit=limit + 1,
            after=after,
        )
        counts, next_cursor = page(
            rows,
            limit=limit,
            position=lambda c: {"support": c.support, "tool": c.tool, "shape": c.arg_shape},
        )
        names = sorted({c.tool for c in counts})
        entries = {
            e.name: e
            for e in await uow.tools.catalog(
                ctx.tenant_id, workspace_id=ctx.workspace_id, names=names, limit=len(names) + 1
            )
        }
    link_next(request, response, next_cursor)
    return ApprovalSuggestionsResponse(
        next_cursor=next_cursor,
        suggestions=[
            ApprovalSuggestionBody(
                id=approvals.suggestion_id(c),
                accepted=approvals.accepted(entries.get(c.tool), c.arg_shape),
                tool=c.tool,
                arg_shape=c.arg_shape,
                suggestion=suggestion,
                approvals=c.approvals,
                rejections=c.rejections,
                edits=c.edits,
                support=c.support,
                approve_rate=round(c.approve_rate, 4),
                agent_id=c.agent_id or None,
            )
            for c in counts
            if (suggestion := approvals.offered(c, entries.get(c.tool))) is not None
        ],
    )


@router.post(
    "/tools/approval-suggestions/{suggestion_id}/accept",
    response_model=CatalogTool,
    tags=["tools"],
    summary="Accept an approval suggestion: its rule is written into the tool's approve_when",
    responses=error_responses(401, 403, 404, 409, 422, 503),
)
async def accept_approval_suggestion(
    request: Request,
    suggestion_id: SuggestionIdPath,
    ctx: HeaderContextDep,
    container: ContainerDep,
) -> Response:
    async def handler(uow):  # type: ignore[no-untyped-def]
        entry = await approvals.accept(uow, ctx, suggestion_id)
        return 200, CatalogTool.of(entry).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload={"action": "accept", "suggestion_id": suggestion_id},
        handler=handler,
    )


# ------------------------------------------------------------------ learned skills


class SkillDraftsResponse(BaseModel):
    drafts: list[SkillDraft] = Field(description="The drafts, best supported first.")


class PublishSkillBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        max_length=NAME_MAX,
        description="Publish under this name instead of the draft's (lowercase letters, "
        "digits and hyphens).",
    )
    description: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description="Publish with this description instead of the draft's.",
    )


_SKILL_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
PublishSkill = Annotated[
    PublishSkillBody | None,
    Body(
        openapi_examples={
            "under another name": {"value": {"name": "refund-order"}},
            "as drafted": {"value": {}},
        }
    ),
]


def _reviewer(request: Request) -> str:
    return f"key:{request.state.service_principal.service_id}"


@router.get(
    "/tools/skill-drafts",
    response_model=SkillDraftsResponse,
    tags=["tools"],
    summary="Learned procedures as draft Agent Skills, for an administrator to publish",
    description="The tenant's administrator credential. A draft is an active procedure no "
    "one has published or dismissed for its current steps (new), or a published one whose "
    "steps changed since (changed).",
    responses=error_responses(401, 403, 422, 503),
)
async def skill_drafts(
    container: ContainerDep, tenant_id: AdministeredTenantDep
) -> SkillDraftsResponse:
    return SkillDraftsResponse(drafts=await container.services["skill_drafts"].list(tenant_id))


@router.post(
    "/tools/skill-drafts/{draft_id}/publish",
    response_model=SkillDecision,
    tags=["tools"],
    summary="Publish a skill draft where agents load skills from (SKILLS_DIR or the gateway)",
    description="The tenant's administrator credential. Its next version: 1.0.0, then the "
    "next minor. 409 when there is no draft for the procedure's current steps, or a skill of "
    "that name exists that this tenant did not publish; 503 when the deployment has no "
    "skills store.",
    responses=_SKILL_ERRORS,
)
async def publish_skill_draft(
    request: Request,
    draft_id: SkillDraftIdPath,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    body: PublishSkill = None,
) -> Response:
    """With ``Idempotency-Key``, a retried publication that succeeded gets the same decision
    again rather than the 409 a decided draft would earn."""
    name, description = (body.name, body.description) if body else (None, None)

    async def handler(uow):  # type: ignore[no-untyped-def]
        decision = await container.services["skill_drafts"].publish(
            tenant_id, draft_id, by=_reviewer(request), name=name, description=description
        )
        return 200, decision.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        request_context(request, tenant_id),
        key=request.state.idempotency_key,
        payload={"action": "publish", "draft": draft_id, "name": name, "description": description},
        handler=handler,
    )


@router.post(
    "/tools/skill-drafts/{draft_id}/dismiss",
    response_model=SkillDecision,
    tags=["tools"],
    summary="Dismiss a skill draft: not offered again until the procedure's steps change",
    description="The tenant's administrator credential. A published skill stays published. "
    "409 when there is no draft for the procedure's current steps.",
    responses=_SKILL_ERRORS,
)
async def dismiss_skill_draft(
    request: Request,
    draft_id: SkillDraftIdPath,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> Response:
    async def handler(uow):  # type: ignore[no-untyped-def]
        decision = await container.services["skill_drafts"].dismiss(
            tenant_id, draft_id, by=_reviewer(request)
        )
        return 200, decision.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        request_context(request, tenant_id),
        key=request.state.idempotency_key,
        payload={"action": "dismiss", "draft": draft_id},
        handler=handler,
    )
