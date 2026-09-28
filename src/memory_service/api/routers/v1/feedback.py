"""Feedback: submit a judgement, read it back, list what was said about one target."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.pagination import CursorQuery, decode_cursor, link_next, page
from memory_service.api.schemas.feedback import (
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from memory_service.domain.feedback import FeedbackTargetKind
from memory_service.modules.feedback.service import FeedbackService

router = APIRouter(tags=["feedback"])
_ERRORS = error_responses(401, 403, 404, 422, 503)
_CURSOR_FIELDS = {"created_at": datetime, "feedback_id": str}


def _service(container) -> FeedbackService:  # type: ignore[no-untyped-def]
    return container.services["feedback"]


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    responses={
        **_ERRORS,
        200: {
            "model": FeedbackResponse,
            "description": "The record an earlier submission stored under this feedback_id",
        },
    },
    summary="Record a judgement on a run, answer, memory, tool call, brief or procedure",
    description=(
        "The body is the `trellis.contracts.Feedback` record. Identity fields come from the "
        "trusted headers; a body value that disagrees is refused. A memory target must be "
        "readable by the caller; reject, correct and edit also need the owner (or a tenant "
        "admin), the rule that governs forgetting. The record is stored and projected "
        "asynchronously: on a "
        "memory, confirm/approve reinforce it, reject retracts it, correct/edit write a "
        "corrected memory that supersedes it. A retry with the same `feedback_id` returns "
        "the stored record with status 200 (with `Idempotency-Key`, the original 201 is "
        "replayed)."
    ),
)
async def submit_feedback(
    request: Request, body: FeedbackRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response:
    ctx = build_context(
        request,
        container,
        ScopeBody(
            tenant_id=body.tenant_id,
            workspace_id=body.workspace_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            agent_run_id=body.agent_run_id,
        ),
    )
    feedback = body.to_domain(ctx)

    async def write(uow):  # type: ignore[no-untyped-def]
        stored, created = await _service(container).submit(uow, ctx, feedback)
        return 201 if created else 200, FeedbackResponse.of(stored).model_dump(mode="json"), None

    # ``feedback_id`` already makes the write idempotent (a retry gets the stored record and
    # 200), so no key is derived when the client sends none.
    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=body.model_dump(mode="json", exclude_none=True),
        handler=write,
    )


@router.get(
    "/feedback",
    response_model=FeedbackListResponse,
    responses=_ERRORS,
    summary="List the feedback on one target, newest first (cursor paged)",
)
async def list_feedback(
    request: Request,
    response: Response,
    container: ContainerDep,
    ctx: HeaderContextDep,
    target_kind: Annotated[FeedbackTargetKind, Query()],
    target_id: Annotated[str, Query(min_length=1, max_length=200)],
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> FeedbackListResponse:
    position = decode_cursor(cursor, fields=_CURSOR_FIELDS)
    before = (position["created_at"], position["feedback_id"]) if position else None
    async with container.services["uow_factory"]() as uow:
        rows = await _service(container).list_for(
            uow, ctx, target_kind=target_kind, target_id=target_id, before=before, limit=limit + 1
        )
    items, next_cursor = page(
        rows,
        limit=limit,
        position=lambda f: {"created_at": f.created_at.isoformat(), "feedback_id": f.feedback_id},
    )
    link_next(request, response, next_cursor)
    return FeedbackListResponse(
        feedback=[FeedbackResponse.of(f) for f in items], next_cursor=next_cursor
    )


@router.get(
    "/feedback/{feedback_id}",
    response_model=FeedbackResponse,
    responses=_ERRORS,
    summary="Read one feedback record, with its projection once it has run",
)
async def get_feedback(
    feedback_id: str, container: ContainerDep, ctx: HeaderContextDep
) -> FeedbackResponse:
    async with container.services["uow_factory"]() as uow:
        return FeedbackResponse.of(await _service(container).get(uow, ctx, feedback_id))
