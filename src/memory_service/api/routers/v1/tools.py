"""Public /v1/tools routes: the catalog, call records, run outcomes, tool hints and approval
suggestions (TOOL_MEMORY.md).

The service never executes a tool. ``record`` is what an adapter calls after it ran one;
``hints`` answers which tool, which plan, which next step and which arguments, from what
previous runs did and what the catalog and the graph know. Every answer names only tools the
caller may call and is built from records the caller's visibility keys cover.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.validation import ToolJson, ToolOutput
from memory_service.domain.enums import Visibility
from memory_service.domain.learning import (
    APPROVAL_MIN_SUPPORT,
    APPROVAL_SUGGESTIONS_MAX,
    Suggestion,
)
from memory_service.domain.tools import (
    SOURCE_MAX_CHARS,
    SideEffects,
    ToolDescriptor,
    ToolHints,
    ToolStats,
    ToolStatus,
)
from memory_service.modules.tools.hints import HINTS_K_MAX
from memory_service.modules.tools.service import CATALOG_MAX

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

    ordinal: int = Field(..., ge=0)
    tool: str
    args: ToolJson = Field(default_factory=dict)
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

    scope: ScopeBody = Field(default_factory=ScopeBody)
    tool: str = Field(..., min_length=1, max_length=200)
    args: ToolJson = Field(default_factory=dict)
    output: ToolOutput = None
    output_summary: str | None = None
    status: ToolStatus = Field(
        default="ok",
        description="How the call ended: ok, error (the tool raised; name it in error_class), "
        "timeout, rejected (the agent or a policy refused to run it), or cancelled (the run "
        "stopped before the call finished).",
    )
    error_class: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost: float | None = Field(default=None, ge=0.0)
    task: str = Field(
        default="",
        max_length=4000,
        description="What the run was asked to do, in words: the pattern procedures are keyed on.",
    )
    step: int | None = Field(default=None, ge=0)
    sub_calls: list[SubCallIn] = Field(default_factory=list, max_length=64)
    visibility: Visibility = Field(
        default=Visibility.PRIVATE,
        description=(
            "Who may see the record, narrowest first: PRIVATE (this agent alone, across runs "
            "- the default, and what a procedure is learned from), RUN (this run and the run "
            "that spawned it), THREAD, AGENT_GROUP, USER, WORKSPACE, TENANT."
        ),
    )


class RecordResponse(BaseModel):
    invocation_id: str
    step: int
    args_hash: str
    recorded: bool = Field(description="False when an identical call was already recorded.")


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"scope": _SCOPE, "success": True, "note": "user accepted the quote"}]
        },
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    success: bool
    note: str | None = Field(default=None, max_length=4000)


class OutcomeResponse(BaseModel):
    run_id: str
    success: bool
    source: str = Field(description="explicit (said by the run or a reviewer) or feedback")


class CatalogEntry(BaseModel):
    """What a tool is and does."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    input_schema: ToolJson = Field(default_factory=lambda: {"type": "object"})
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
    source: str = Field(default="manual", max_length=SOURCE_MAX_CHARS)
    server: str | None = Field(default=None, max_length=200)
    examples: list[ToolJson] = Field(default_factory=list, max_length=20)
    redact: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="dotted argument paths whose values never reach storage",
    )

    def to_domain(self) -> ToolDescriptor:
        required = self.required
        if required is None:
            required = [str(r) for r in self.input_schema.get("required", []) or []]
        return ToolDescriptor(
            tenant_id="",
            **self.model_dump(exclude={"required"}),
            required=required,
        )


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

    scope: ScopeBody = Field(default_factory=ScopeBody)
    tools: list[CatalogEntry] = Field(..., min_length=1, max_length=CATALOG_MAX)


class ToolStatsBody(BaseModel):
    calls: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float | None = None
    avg_latency_ms: float | None = None
    approvals: int = 0
    rejections: int = 0
    edits: int = 0
    last_used_at: datetime | None = None

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
    tool_id: str
    name: str
    version: int
    description: str
    input_schema: dict[str, Any]
    required: list[str]
    argument_entity_types: dict[str, str]
    side_effects: SideEffects | None
    source: str
    server: str | None
    examples: list[dict[str, Any]]
    schema_hash: str
    workspace_id: str | None
    stats: ToolStatsBody = Field(default_factory=ToolStatsBody)

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
                    "schema_hash",
                    "workspace_id",
                }
            ),
            input_schema=entry.input_schema or {"type": "object"},
            stats=ToolStatsBody.of(stats or ToolStats(tool_name=entry.name)),
        )


class CatalogResponse(BaseModel):
    tools: list[CatalogTool]


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

    scope: ScopeBody = Field(default_factory=ScopeBody)
    task: str = Field(..., min_length=1, max_length=4000)
    available: list[str] | None = Field(
        default=None,
        max_length=500,
        description="the tools the caller can call; omitted, every catalog tool may be named",
    )
    k: int = Field(default=8, ge=1, le=HINTS_K_MAX)


class ApprovalSuggestionBody(BaseModel):
    tool: str
    arg_shape: str = Field(
        description="argument names with their value kinds (numbers by magnitude)"
    )
    suggestion: Suggestion = Field(
        description="auto_approve: nearly every decision approved calls of this shape; "
        "always_ask: half or more were rejected or edited. Never applied by the service."
    )
    approvals: int
    rejections: int
    edits: int
    support: int
    approve_rate: float
    agent_id: str | None


class ApprovalSuggestionsResponse(BaseModel):
    suggestions: list[ApprovalSuggestionBody]


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
) -> RecordResponse:
    ctx = build_context(request, container, body.scope)
    service = container.services["tool_memory"]
    async with container.services["uow_factory"]() as uow:
        invocation, created = await service.record(
            uow,
            ctx,
            **body.model_dump(exclude={"scope", "sub_calls"}),
            sub_calls=[c.model_dump() for c in body.sub_calls],
        )
        await uow.commit()
    return RecordResponse(
        invocation_id=invocation.invocation_id,
        step=invocation.step,
        args_hash=invocation.args_hash,
        recorded=created,
    )


@router.post(
    "/runs/{run_id}/outcome",
    response_model=OutcomeResponse,
    tags=["tools"],
    summary="Label an agent run successful or not (only successful runs validate a procedure)",
    responses=_ERRORS,
)
async def set_run_outcome(
    run_id: str,
    request: Request,
    body: OutcomeRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
) -> OutcomeResponse:
    ctx = build_context(request, container, body.scope)
    service = container.services["tool_memory"]
    async with container.services["uow_factory"]() as uow:
        outcome = await service.set_outcome(
            uow, ctx, run_id=run_id, success=body.success, note=body.note
        )
        await uow.commit()
    return OutcomeResponse(run_id=outcome.run_id, success=outcome.success, source=outcome.source)


@router.get(
    "/tools",
    response_model=CatalogResponse,
    tags=["tools"],
    summary="The tool catalog visible in this scope, with each tool's statistics",
    responses=_ERRORS,
)
async def list_tools(
    ctx: HeaderContextDep,
    container: ContainerDep,
    names: Annotated[list[str] | None, Query(max_length=CATALOG_MAX)] = None,
) -> CatalogResponse:
    async with container.services["uow_factory"]() as uow:
        rows = await container.services["tool_memory"].catalog(uow, ctx, names)
    return CatalogResponse(tools=[CatalogTool.of(entry, stats) for entry, stats in rows])


@router.put(
    "/tools/catalog",
    response_model=CatalogResponse,
    tags=["tools"],
    summary="Upsert catalog entries by name (idempotent; unchanged entries are left alone)",
    responses=_ERRORS,
)
async def put_catalog(
    request: Request, body: CatalogRequest, container: ContainerDep, _: ServicePrincipalDep
) -> CatalogResponse:
    ctx = build_context(request, container, body.scope)
    async with container.services["uow_factory"]() as uow:
        stored = await container.services["tool_memory"].put_catalog(
            uow, ctx, [entry.to_domain() for entry in body.tools]
        )
        await uow.commit()
    return CatalogResponse(tools=[CatalogTool.of(entry) for entry in stored])


@router.post(
    "/tools/hints",
    response_model=ToolHints,
    tags=["tools"],
    summary="Which tool fits a task, the learned plan, the next step and its arguments",
    responses=_ERRORS,
)
async def tool_hints(
    request: Request, body: HintsRequest, container: ContainerDep, _: ServicePrincipalDep
) -> ToolHints:
    ctx = build_context(request, container, body.scope)
    visibility = await container.services["authz"].visibility(ctx)
    async with container.services["uow_factory"]() as uow:
        profile = await container.services["profile"].blocks(uow, ctx)
    return await container.services["tool_hints"].hints(
        ctx,
        body.task,
        available=body.available,
        k=body.k,
        scope_keys=list(visibility.keys),
        profile=profile,
    )


@router.get(
    "/tools/approval-suggestions",
    response_model=ApprovalSuggestionsResponse,
    tags=["tools"],
    summary="Approval rules this agent's approve / reject / edit decisions support "
    "(suggestions only: never applied)",
    responses=_ERRORS,
)
async def approval_suggestions(
    ctx: HeaderContextDep,
    container: ContainerDep,
    tool: Annotated[str | None, Query(max_length=200)] = None,
) -> ApprovalSuggestionsResponse:
    async with container.services["uow_factory"]() as uow:
        counts = await uow.tools.approval_patterns(
            ctx.tenant_id,
            ctx.agent_id or "",
            tool_name=tool,
            min_support=APPROVAL_MIN_SUPPORT,
            limit=APPROVAL_SUGGESTIONS_MAX,
        )
    return ApprovalSuggestionsResponse(
        suggestions=[
            ApprovalSuggestionBody(
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
            if (suggestion := c.suggestion()) is not None
        ]
    )
