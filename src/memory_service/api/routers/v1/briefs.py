"""Standing briefs: async refresh and a model-free stored-content read."""

from typing import Literal

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
from memory_service.api.idempotent import default_idempotency_key, run_idempotent
from memory_service.domain.briefs import BriefInfo, BriefOutput, BriefSpec, StoredBrief, brief_scope

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)


class BriefRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: ScopeBody = Field(default_factory=ScopeBody)
    spec: BriefSpec


class BriefResponse(BaseModel):
    brief_id: str
    spec: BriefSpec
    status: Literal["pending", "ready", "stale"] = Field(
        description=(
            "pending means no refresh has completed; ready is a current stored result; "
            "stale hides output until evidence or authorization changes are refreshed."
        )
    )
    output: BriefOutput | None = None


def response(brief: StoredBrief, status: str = "pending") -> dict:
    return {
        "brief_id": brief.brief_id,
        "spec": brief.spec.model_dump(mode="json"),
        "status": status,
        "output": brief.output.model_dump(mode="json") if brief.output else None,
    }


@router.post(
    "/briefs",
    response_model=BriefResponse,
    status_code=202,
    tags=["briefs"],
    responses=_ERRORS,
    summary="Create a standing question or knowledge page and queue its refresh",
)
async def create_brief(
    request: Request, body: BriefRequest, container: ContainerDep, _: ServicePrincipalDep
):
    ctx = build_context(request, container, body.scope)

    async def write(uow):
        brief = await container.services["briefs"].create(uow, ctx, body.spec)
        return 202, response(brief), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.headers.get("Idempotency-Key")
        or default_idempotency_key(ctx, "brief", ctx.request_id),
        payload={"scope": brief_scope(ctx), "spec": body.spec.model_dump(mode="json")},
        handler=write,
    )


@router.put(
    "/briefs/{brief_id}",
    response_model=BriefResponse,
    status_code=202,
    tags=["briefs"],
    responses=_ERRORS,
    summary="Replace a brief definition and queue refresh",
)
async def update_brief(
    brief_id: str,
    request: Request,
    body: BriefRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
):
    ctx = build_context(request, container, body.scope)

    async def write(uow):
        brief = await container.services["briefs"].update(uow, ctx, brief_id, body.spec)
        return 202, response(brief), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.headers.get("Idempotency-Key")
        or default_idempotency_key(ctx, "brief-update", ctx.request_id),
        payload={
            "scope": brief_scope(ctx),
            "brief_id": brief_id,
            "spec": body.spec.model_dump(mode="json"),
        },
        handler=write,
    )


@router.get(
    "/briefs",
    response_model=list[BriefInfo],
    tags=["briefs"],
    responses=_ERRORS,
    summary="List brief definitions in the exact execution scope (no generated content)",
)
async def list_briefs(
    container: ContainerDep,
    ctx: HeaderContextDep,
    after: str = Query(default="", max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
):
    async with container.services["uow_factory"]() as uow:
        rows = await uow.briefs.list_owned(
            ctx.tenant_id, brief_scope(ctx), after=after, limit=limit
        )
    return rows


@router.get(
    "/briefs/{brief_id}",
    response_model=BriefResponse,
    tags=["briefs"],
    responses=_ERRORS,
    summary="Read fresh stored evidence or synthesis without model calls",
)
async def read_brief(brief_id: str, container: ContainerDep, ctx: HeaderContextDep):
    brief, status = await container.services["briefs"].read(ctx, brief_id)
    return response(brief, status)


@router.delete(
    "/briefs/{brief_id}",
    tags=["briefs"],
    responses=_ERRORS,
    summary="Delete a brief definition and its stored output",
)
async def delete_brief(brief_id: str, container: ContainerDep, ctx: HeaderContextDep):
    async with container.services["uow_factory"]() as uow:
        await container.services["briefs"].owned(uow, ctx, brief_id)
        await uow.briefs.delete(ctx.tenant_id, brief_id)
        await uow.commit()
    return {"deleted": True}
