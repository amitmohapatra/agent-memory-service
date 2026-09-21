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
from memory_service.domain.enums import Visibility

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


class PolicyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic: bool = False
    side_effects: str = "unknown"
    cacheable: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    cache_scope: str = "run"
    cost_hint: float | None = Field(default=None, ge=0.0)
    redact: list[str] = Field(default_factory=list)


class RegisterToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_TOOL_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"
    server: str | None = None
    policy: PolicyBody | None = Field(
        default=None,
        description="Widening a policy requires tenant admin; otherwise the stored policy wins.",
    )


class DeclaredTool(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    description: str = ""
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"
    server: str | None = None


class LookupRequest(BaseModel):
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
                    "args": {"sku": "SKU-22", "region": "EMEA"},
                }
            ]
        },
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


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
    args: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    output_summary: str | None = None
    status: str = "ok"
    error_class: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost: float | None = Field(default=None, ge=0.0)
    task: str = ""
    step: int | None = Field(default=None, ge=0)
    sub_calls: list[dict[str, Any]] = Field(default_factory=list)
    visibility: str = Field(default="RUN", description="RUN | AGENT_GROUP | USER | THREAD | ...")


class RecordResponse(BaseModel):
    invocation_id: str
    step: int
    args_hash: str
    recorded: bool = Field(description="False when an identical call was already recorded.")


class SuggestRequest(BaseModel):
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
    context: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    args_hash: str | None = None
    status: str = "ok"
    output_summary: str | None = None
    output_fields: dict[str, Any] = Field(default_factory=dict)


class NextRequest(BaseModel):
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
                    "trajectory_so_far": [
                        {
                            "tool": "pricing.lookup_price",
                            "status": "ok",
                            "output_fields": {"quote_id": "Q-1183"},
                        }
                    ],
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
    trajectory_so_far: list[TrajectoryStep] = Field(default_factory=list)
    available_tools: list[DeclaredTool] = Field(default_factory=list)
    limit: int = Field(default=3, ge=1, le=10)


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
            sub_calls=body.sub_calls,
            visibility=Visibility(body.visibility),
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
