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
    credential_vouches_for_agent,
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
    FeedbackBulkReviewRequest,
    FeedbackBulkReviewResponse,
    FeedbackGroupListResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackReviewRequest,
    FeedbackTargetGroupResponse,
    FeedbackUnchanged,
)
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.feedback import FeedbackTargetKind, ReviewState
from memory_service.domain.tenancy import KeyRole
from memory_service.modules.feedback.service import FeedbackService

router = APIRouter(tags=["feedback"])
_ERRORS = error_responses(401, 403, 404, 422, 503)
_CURSOR_FIELDS = {"created_at": datetime, "feedback_id": str}
_GROUP_CURSOR_FIELDS = {"latest_at": datetime, "target_kind": str, "target_id": str}


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
        "readable by the caller; a correct or edit that rewrites it also needs the owner (or "
        "the user an agent acts for, or a tenant admin in person), the rule that governs "
        "forgetting. "
        "The record is stored and projected "
        "asynchronously: on a "
        "memory, confirm/approve reinforce it, reject retracts it, correct/edit write a "
        "corrected memory that supersedes it. An end user's vote (a confirm or reject of a "
        "memory, a verdict on a run or a procedure) is stored with `review.state=pending` "
        "and changes nothing until a tenant admin approves it; the same voter's newer vote "
        "on the same target replaces their earlier pending one (dismissed, with "
        "`review.replaced_by`), "
        "so repeated clicks count once. An owner's correct/edit of a memory carrying new "
        "text, a run reporting its own status, a tool-call decision, the service's own "
        "/v1/verify verdict and what a tenant admin says are applied as they arrive. A "
        "retry with the same `feedback_id` returns "
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
            uow,
            ctx,
            feedback,
            trusted=is_tenant_administrator(principal),
            run_vouched=credential_vouches_for_agent(request, ctx.agent_id),
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
    response_model=FeedbackListResponse | FeedbackGroupListResponse,
    responses=_ERRORS,
    summary="List the feedback on one target, or the review queue, newest first (cursor paged)",
    description=(
        "Two lists, one route. With `target_kind` and `target_id`: the feedback on that "
        "target the caller may see. With `review=pending`: the review queue, verdicts that "
        "change nothing until a tenant admin approves them (`review=approved`, `dismissed` "
        "lists those, for reporting). With `group=target` as well: one row "
        "per target (`groups`) with its thumbs up and down, distinct voters and the latest "
        "vote's time, the most recently voted target first. " + _PENDING_DESCRIPTION
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
        ReviewState | None,
        Query(
            description="pending: list the review queue instead of one target's feedback "
            "(the tenant's administrator credential); approved or dismissed: the "
            "verdicts in that state, for reporting."
        ),
    ] = None,
    group: Annotated[
        Literal["target"] | None,
        Query(
            description="target (with review): one row per target with its counts instead "
            "of one row per verdict."
        ),
    ] = None,
    cursor: CursorQuery = None,
    limit: Annotated[int, limit_query(500, "feedback records")] = 100,
) -> FeedbackListResponse | FeedbackGroupListResponse:
    if group is not None and review is None:
        raise ValidationFailed("group=target groups a review list; send review with it")
    if review is not None:
        if target_kind is not None or target_id is not None:
            raise ValidationFailed("review lists the queue; send no target with it")
        ensure_role(principal, KeyRole.ADMIN, KeyRole.PLATFORM)
        tenant_id = await existing_administered_tenant(request, principal, container)
        if group is not None:
            return await _grouped(request, response, container, tenant_id, review, cursor, limit)
        return await _pending(request, response, container, tenant_id, cursor, limit, review)
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
    state: ReviewState = ReviewState.PENDING,
) -> FeedbackListResponse:
    position = decode_cursor(cursor, fields=_CURSOR_FIELDS)
    before = (position["created_at"], position["feedback_id"]) if position else None
    async with container.services["uow_factory"]() as uow:
        queued = await _service(container).pending(
            uow, tenant_id, state=state, before=before, limit=limit + 1
        )
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


async def _grouped(
    request: Request,
    response: Response,
    container: Any,
    tenant_id: str,
    state: ReviewState,
    cursor: str | None,
    limit: int,
) -> FeedbackGroupListResponse:
    position = decode_cursor(cursor, fields=_GROUP_CURSOR_FIELDS)
    before = (
        (position["latest_at"], position["target_kind"], position["target_id"])
        if position
        else None
    )
    async with container.services["uow_factory"]() as uow:
        groups = await _service(container).pending_by_target(
            uow, tenant_id, state=state, before=before, limit=limit + 1
        )
    items, next_cursor = page(
        groups,
        limit=limit,
        position=lambda g: {
            "latest_at": g.latest_at.isoformat(),
            "target_kind": g.target_kind.value,
            "target_id": g.target_id,
        },
    )
    link_next(request, response, next_cursor)
    return FeedbackGroupListResponse(
        groups=[FeedbackTargetGroupResponse.of(g) for g in items], next_cursor=next_cursor
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


async def _review_many(
    request: Request,
    container: Any,
    tenant_id: str,
    body: FeedbackBulkReviewRequest,
    *,
    approve: bool,
) -> Response:
    """Idempotent by construction (a verdict already decided is left as it is); with
    ``Idempotency-Key`` a retry also gets the original answer back."""
    principal = request.state.service_principal

    async def handler(uow):  # type: ignore[no-untyped-def]
        target = (
            (body.target_kind, body.target_id)
            if body.target_kind is not None and body.target_id is not None
            else None
        )
        decided, unchanged, more = await _service(container).review_many(
            uow,
            tenant_id,
            feedback_ids=body.feedback_ids,
            target=target,
            approve=approve,
            reviewed_by=f"key:{principal.service_id}",
            note=body.note,
        )
        answer = FeedbackBulkReviewResponse(
            state=ReviewState.APPROVED if approve else ReviewState.DISMISSED,
            decided=[f.feedback_id for f in decided],
            unchanged=[
                FeedbackUnchanged(
                    feedback_id=f.feedback_id, state=f.review.state if f.review else None
                )
                for f in unchanged
            ],
            more=more,
        )
        return 200, answer.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        request_context(request, tenant_id),
        key=request.state.idempotency_key,
        payload={"approve": approve, **body.model_dump(mode="json", exclude_none=True)},
        handler=handler,
    )


_BULK_ERRORS = error_responses(401, 403, 404, 422, 503)
_BULK_DESCRIPTION = (
    "The tenant's administrator credential. Send `feedback_ids` (1 to 500), or `target_kind` "
    "and `target_id` for every verdict pending on that target. Idempotent: a verdict that is "
    "no longer pending (approved or dismissed) is listed in `unchanged` and left as "
    "it is, so a repeated call applies nothing twice. An id the tenant does not have refuses "
    "the whole call (404) and decides nothing."
)


@router.post(
    "/feedback/approve",
    response_model=FeedbackBulkReviewResponse,
    responses=_BULK_ERRORS,
    summary="Approve pending verdicts in bulk: each is applied as if it had just arrived",
    description=_BULK_DESCRIPTION,
)
async def approve_feedback_many(
    request: Request,
    body: FeedbackBulkReviewRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> Response:
    return await _review_many(request, container, tenant_id, body, approve=True)


@router.post(
    "/feedback/dismiss",
    response_model=FeedbackBulkReviewResponse,
    responses=_BULK_ERRORS,
    summary="Dismiss pending verdicts in bulk: kept for statistics, never applied",
    description=_BULK_DESCRIPTION,
)
async def dismiss_feedback_many(
    request: Request,
    body: FeedbackBulkReviewRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> Response:
    return await _review_many(request, container, tenant_id, body, approve=False)
