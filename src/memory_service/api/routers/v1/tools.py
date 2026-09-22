"""Public /v1/tools routes: the tool registry, invocation records, the output cache and the
advice an agent asks for mid-task (TOOL_MEMORY.md §30.0, §30.2, §30.4, §30.6).

The service never executes a tool. ``record`` is what an adapter calls after it ran one,
``lookup`` is what it calls before, and ``suggest`` / ``next`` / ``plan`` answer from what
previous runs did. Every reply names only tools the caller declared as available and is
built from records the caller's visibility keys cover.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.api.validation import ToolJson, ToolOutput
from memory_service.domain.enums import Visibility
from memory_service.domain.tools import ToolSource, ToolStatus

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)

_TOOL_EXAMPLE: dict[str, Any] = {
    "name": "pricing.lookup_price",
    "description": "Current list price for a SKU in a region",
    "input_schema": {
        "type": "object",
        "properties": {"sku": {"type": "string"}, "region": {"type": "string"}},
    },
    "tags": ["pricing"],
    "source": "mcp",
    "policy": {
        "deterministic": True,
        "cacheable": True,
        "cache_ttl_seconds": 900,
        "cache_scope": "thread",
        "side_effects": "read",
    },
}


class DeclaredTool(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    description: str = ""
    schema_: ToolJson | None = Field(default=None, alias="schema")
    output_schema: ToolJson | None = None
    tags: list[str] = Field(default_factory=list)
    source: ToolSource = Field(
        default="manual",
        description=(
            "Where the tool definition comes from: bifrost-mcp (an MCP server behind the "
            "Bifrost gateway), mcp (a directly connected MCP server), langgraph, adk or crewai "
            "(a framework tool node), manual (declared by the caller; the default)."
        ),
    )
    server: str | None = None


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
                    "scope": {
                        "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                        "agent_run_id": "run_01J8ZK",
                    },
                    "tool": "pricing.lookup_price",
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
    tool: str
    args: ToolJson = Field(default_factory=dict)
    output: ToolOutput = None
    output_summary: str | None = None
    status: ToolStatus = Field(
        default="ok",
        description="How the call ended: ok, error (the tool raised; name it in error_class), "
        "timeout, or rejected (the agent or a policy refused to run it).",
    )
    error_class: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost: float | None = Field(default=None, ge=0.0)
    task: str = ""
    step: int | None = Field(default=None, ge=0)
    sub_calls: list[SubCallIn] = Field(default_factory=list, max_length=64)
    visibility: Visibility = Field(
        default=Visibility.RUN,
        description=(
            "Who may see the record, narrowest first: PRIVATE, RUN (this agent run and the "
            "runs it spawns; the default), THREAD, WORK, AGENT_GROUP, GROUP, USER, WORKSPACE, "
            "TENANT, GLOBAL."
        ),
    )


class RecordResponse(BaseModel):
    invocation_id: str
    step: int
    args_hash: str
    recorded: bool = Field(description="False when an identical call was already recorded.")


class PlanRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scope": {
                        "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                        "agent_run_id": "run_01J8ZK",
                    },
                    "task": "update quote Q-1183 with EMEA price for SKU-22",
                    "available_tools": [
                        {"name": "pricing.lookup_price"},
                        {"name": "crm.update_quote"},
                    ],
                }
            ]
        },
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    task: str = Field(..., max_length=4000)
    available_tools: list[DeclaredTool] = Field(default_factory=list)


class PlanResponse(BaseModel):
    task_pattern: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    valid: bool
    reason: str | None = None
    problems: list[str] = Field(default_factory=list)
    support: int = 0
    success_rate: float = 0.0
    script: str | None = None
    rendered: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    invocation_ids: list[str] = Field(default_factory=list)


class ProceduresResponse(BaseModel):
    procedures: list[dict[str, Any]]


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scope": {
                        "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                        "agent_run_id": "run_01J8ZK",
                    },
                    "success": True,
                    "note": "user accepted the quote",
                }
            ]
        },
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    success: bool
    note: str | None = None


class OutcomeResponse(BaseModel):
    run_id: str
    success: bool
    source: str


def _declared(items: list[DeclaredTool]) -> list[dict[str, Any]]:
    return [i.model_dump(by_alias=True, exclude_none=False) for i in items]


async def _scope_keys(container: Any, ctx: Any) -> list[str]:
    visibility = await container.services["authz"].visibility(ctx)
    return list(visibility.keys)


@router.post(
    "/tools/record",
    response_model=RecordResponse,
    status_code=202,
    tags=["tools"],
    summary="Record one tool call (idempotent on run + step + tool + arguments)",
    responses=_ERRORS,
)
async def record_tool(
    request: Request, body: RecordRequest, container: ContainerDep, _: ServicePrincipalDep
) -> RecordResponse:
    ctx = build_context(request, container, body.scope)
    service = container.services["tool_memory"]
    async with container.services["uow_factory"]() as uow:
        before = (
            await uow.tools.invocations_for_run(ctx.tenant_id, ctx.agent_run_id)
            if ctx.agent_run_id
            else []
        )
        invocation = await service.record(
            uow,
            ctx,
            tool=body.tool,
            args=body.args,
            output=body.output,
            output_summary=body.output_summary,
            status=body.status,
            error_class=body.error_class,
            latency_ms=body.latency_ms,
            cost=body.cost,
            task=body.task,
            step=body.step,
            sub_calls=[c.model_dump() for c in body.sub_calls],
            visibility=body.visibility,
        )
        await uow.commit()
    known = {i.invocation_id for i in before}
    return RecordResponse(
        invocation_id=invocation.invocation_id,
        step=invocation.step,
        args_hash=invocation.args_hash,
        recorded=invocation.invocation_id not in known,
    )


@router.post(
    "/tools/plan",
    response_model=PlanResponse,
    tags=["tools"],
    summary="The best-known validated chain for a task, as an ordered plan with bindings",
    responses=_ERRORS,
)
async def plan_tools(
    request: Request, body: PlanRequest, container: ContainerDep, _: ServicePrincipalDep
) -> PlanResponse:
    ctx = build_context(request, container, body.scope)
    service = container.services["tool_memory"]
    keys = await _scope_keys(container, ctx)
    async with container.services["uow_factory"]() as uow:
        payload = await service.plan(
            uow,
            ctx,
            task=body.task,
            available_tools=_declared(body.available_tools),
            scope_keys=keys,
        )
        await uow.commit()
    return PlanResponse(**{k: v for k, v in payload.items() if k in PlanResponse.model_fields})


@router.get(
    "/tools/procedures",
    response_model=ProceduresResponse,
    tags=["tools"],
    summary="Procedures mined for a task pattern",
    responses=_ERRORS,
)
async def list_procedures(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    task: str = "",
    agent_id: str = Query(
        default="",
        description=(
            "The agent whose procedures to list. Required to see agent-scoped invocations: "
            "they are recorded against principal 'agent:<id>', and lineage cannot travel in "
            "a header the way tenant/workspace/user do."
        ),
    ),
    workspace_id: str = Query(default="", description="Narrow to one workspace."),
) -> ProceduresResponse:
    # A GET has no body, and lineage is body-only everywhere else — so it comes in as query
    # parameters here. Passing an empty ScopeBody() discarded the caller's agent entirely:
    # the context fell back to the API-key service principal, which the authorization model
    # does not define, and every call to this route failed with a 500 from OpenFGA.
    ctx = build_context(
        request,
        container,
        ScopeBody(agent_id=agent_id or None, workspace_id=workspace_id or None),
    )
    service = container.services["tool_memory"]
    keys = await _scope_keys(container, ctx)
    async with container.services["uow_factory"]() as uow:
        procedures = await service.procedures(uow, ctx, task=task, scope_keys=keys)
    return ProceduresResponse(procedures=[p.to_payload() for p in procedures])


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
