"""Feedback: submit a judgement, read it back, list what was said about one target, and
review the verdicts that wait for a tenant admin (ADR 0028)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Query, Request, Response

from memory_service.api.deps import (
    AdministeredTenantDep,
    ContainerDep,
    HeaderContextDep,
    LineageDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
    ensure_role,
    existing_administered_tenant,
    is_tenant_administrator,
    request_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.headers import DEPRECATION_HEADER, LINK_HEADER
from memory_service.api.idempotent import resource_at, run_idempotent
from memory_service.api.pagination import CursorQuery, decode_cursor, link_next, next_link, page
from memory_service.api.params import FeedbackIdPath, limit_query
from memory_service.api.schemas.feedback import (
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackReviewRequest,
)
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.feedback import FeedbackTargetKind
from memory_service.domain.tenancy import KeyRole
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
    summary="Record a judgement on a run, memory, tool call or procedure",
    description=(
        "The body is the `trellis.contracts.Feedback` record. Identity fields come from the "
        "trusted headers; a body value that disagrees is refused. A memory target must be "
        "readable by the caller; reject, correct and edit also need the owner (or a tenant "
        "admin), the rule that governs forgetting. The record is stored and projected "
        "asynchronously: on a "
        "memory, confirm/approve reinforce it, reject retracts it, correct/edit write a "
        "corrected memory that supersedes it. A verdict that would change what was learned "
        "on a person's or an agent's word (a confirm of a memory, a verdict on a run or a "
        "procedure) is stored with `review.state=pending` and changes nothing until a tenant "
        "admin approves it; an owner's reject/correct/edit of a memory, a run reporting its "
        "own status, a tool-call decision and what a tenant admin says are applied as they "
        "arrive. A retry with the same `feedback_id` returns "
        "the stored record with status 200 (with `Idempotency-Key`, the original 201 is "
        "replayed)."
    ),
)
async def submit_feedback(
    request: Request, body: FeedbackRequest, container: ContainerDep, principal: ServicePrincipalDep
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
        # the tenant's administrator credential is the reviewer's own: no queue for it
        stored, created = await _service(container).submit(
            uow, ctx, feedback, trusted=is_tenant_administrator(principal)
        )
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
        location=resource_at("/v1/feedback/{}", "feedback_id"),
    )


_PENDING_DESCRIPTION = (
    "The tenant's administrator credential (the platform key names the tenant in "
    "X-Trellis-Tenant). Each verdict carries `author_record`: how its author's verdicts fared "
    "in review (pending, approved, dismissed)."
)


@router.get(
    "/feedback",
    response_model=FeedbackListResponse,
    responses=_ERRORS,
    summary="List the feedback on one target, or the review queue, newest first (cursor paged)",
    description=(
        "Two lists, one route. With `target_kind` and `target_id`: the feedback on that "
        "target the caller may see. With `review=pending`: the review queue, verdicts that "
        "change nothing until a tenant admin approves them. " + _PENDING_DESCRIPTION
    ),
)
async def list_feedback(
    request: Request,
    response: Response,
    container: ContainerDep,
    principal: ServicePrincipalDep,
    lineage: LineageDep,
    target_kind: Annotated[
        FeedbackTargetKind | None,
        Query(
            description="What the feedback judges: run, memory, tool_call or procedure. "
            "Required with target_id unless review=pending."
        ),
    ] = None,
    target_id: Annotated[
        str | None,
        Query(
            min_length=1,
            max_length=200,
            description="The judged object's id (a run id, mem_..., an invocation id or a "
            "procedure id). Required with target_kind unless review=pending.",
        ),
    ] = None,
    review: Annotated[
        Literal["pending"] | None,
        Query(
            description="pending: list the review queue instead of one target's feedback "
            "(the tenant's administrator credential)."
        ),
    ] = None,
    cursor: CursorQuery = None,
    limit: Annotated[int, limit_query(500, "feedback records")] = 100,
) -> FeedbackListResponse:
    if review is not None:
        if target_kind is not None or target_id is not None:
            raise ValidationFailed("review=pending lists the queue; send no target with it")
        ensure_role(principal, KeyRole.ADMIN, KeyRole.PLATFORM)
        tenant_id = await existing_administered_tenant(request, principal, container)
        return await _pending(request, response, container, tenant_id, cursor, limit)
    if target_kind is None or target_id is None:
        raise ValidationFailed("send target_kind and target_id, or review=pending")
    ctx = build_context(request, container, lineage)
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
    "/feedback/pending",
    response_model=FeedbackListResponse,
    responses=_ERRORS,
    deprecated=True,
    summary="The review queue (deprecated: GET /v1/feedback?review=pending)",
    description="The same list as `GET /v1/feedback?review=pending`, kept as an alias; "
    "answered with `Deprecation: true` and a `Link` to the successor. " + _PENDING_DESCRIPTION,
)
async def pending_feedback(
    request: Request,
    response: Response,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, limit_query(500, "feedback records")] = 100,
) -> FeedbackListResponse:
    response.headers[DEPRECATION_HEADER] = "true"
    successor = request.url.replace(path=request.url.path.removesuffix("/pending"))
    successor = successor.include_query_params(review="pending")
    response.headers[LINK_HEADER] = f'<{successor}>; rel="successor-version"'
    listed = await _pending(request, response, container, tenant_id, cursor, limit)
    if listed.next_cursor is not None:
        # the page link stays on this route; the successor rides beside it
        response.headers[LINK_HEADER] += ", " + next_link(request, listed.next_cursor)
    return listed


async def _pending(
    request: Request,
    response: Response,
    container: Any,
    tenant_id: str,
    cursor: str | None,
    limit: int,
) -> FeedbackListResponse:
    position = decode_cursor(cursor, fields=_CURSOR_FIELDS)
    before = (position["created_at"], position["feedback_id"]) if position else None
    async with container.services["uow_factory"]() as uow:
        queued = await _service(container).pending(uow, tenant_id, before=before, limit=limit + 1)
    items, next_cursor = page(
        queued,
        limit=limit,
        position=lambda q: {
            "created_at": q[0].created_at.isoformat(),
            "feedback_id": q[0].feedback_id,
        },
    )
    if LINK_HEADER not in response.headers:
        link_next(request, response, next_cursor)
    return FeedbackListResponse(
        feedback=[FeedbackResponse.of(f, record) for f, record in items], next_cursor=next_cursor
    )


@router.get(
    "/feedback/{feedback_id}",
    response_model=FeedbackResponse,
    responses=_ERRORS,
    summary="Read one feedback record, with its projection once it has run",
)
async def get_feedback(
    feedback_id: FeedbackIdPath, container: ContainerDep, ctx: HeaderContextDep
) -> FeedbackResponse:
    async with container.services["uow_factory"]() as uow:
        return FeedbackResponse.of(await _service(container).get(uow, ctx, feedback_id))


async def _review(
    request: Request,
    container: Any,
    tenant_id: str,
    feedback_id: str,
    body: FeedbackReviewRequest | None,
    *,
    approve: bool,
) -> Response:
    """With ``Idempotency-Key``, a retried review that succeeded gets the reviewed record
    again rather than the 409 a verdict no longer pending would earn."""
    principal = request.state.service_principal

    async def handler(uow):  # type: ignore[no-untyped-def]
        reviewed = await _service(container).review(
            uow,
            tenant_id,
            feedback_id,
            approve=approve,
            reviewed_by=f"key:{principal.service_id}",
            note=body.note if body else None,
        )
        return 200, FeedbackResponse.of(reviewed).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        request_context(request, tenant_id),
        key=request.state.idempotency_key,
        payload={
            "feedback_id": feedback_id,
            "approve": approve,
            "note": body.note if body else None,
        },
        handler=handler,
    )


_REVIEW_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
#: the note is optional, and so is the body
ReviewBody = Annotated[
    FeedbackReviewRequest | None,
    Body(
        openapi_examples={
            "with a note": {"value": {"note": "Checked against the signed contract."}},
            "without": {"value": {}},
        }
    ),
]


@router.post(
    "/feedback/{feedback_id}/approve",
    response_model=FeedbackResponse,
    responses=_REVIEW_ERRORS,
    summary="Approve a pending verdict: it is applied as if it had just arrived",
    description="The tenant's administrator credential. 409 when it is not pending.",
)
async def approve_feedback(
    request: Request,
    feedback_id: FeedbackIdPath,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    body: ReviewBody = None,
) -> Response:
    return await _review(request, container, tenant_id, feedback_id, body, approve=True)


@router.post(
    "/feedback/{feedback_id}/dismiss",
    response_model=FeedbackResponse,
    responses=_REVIEW_ERRORS,
    summary="Dismiss a pending verdict: kept for statistics, never applied",
    description="The tenant's administrator credential. 409 when it is not pending.",
)
async def dismiss_feedback(
    request: Request,
    feedback_id: FeedbackIdPath,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    body: ReviewBody = None,
) -> Response:
    return await _review(request, container, tenant_id, feedback_id, body, approve=False)
