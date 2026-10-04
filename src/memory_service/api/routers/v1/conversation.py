"""Public /v1 routes: threads, messages, jobs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ServicePrincipalDep,
    ThreadContextDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import (
    NO_CONTENT,
    default_idempotency_key,
    derived_or_body,
    first_job,
    run_idempotent,
)
from memory_service.api.pagination import CursorQuery, decode_cursor, encode_cursor, link_next
from memory_service.api.params import ThreadIdPath, limit_query
from memory_service.api.schemas.conversation import (
    CreateMessagesRequest,
    JobResponse,
    MessageAckResponse,
    MessageListResponse,
    MessageResponse,
    MessagesAckResponse,
    PatchThreadRequest,
    ThreadResponse,
    ThreadSummaryBody,
)
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Attachment, Message, Thread
from memory_service.domain.enums import ArchiveStatus, JobStatus
from memory_service.domain.errors import NotFound, ValidationFailed
from memory_service.domain.profile import ThreadSummary
from memory_service.modules.conversation.service import ConversationService
from memory_service.ports.tasks import Queue

router = APIRouter()

_WRITE_ERRORS = error_responses(401, 403, 409, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)


def _thread_response(t: Thread, summary: ThreadSummary | None = None) -> ThreadResponse:
    return ThreadResponse(
        thread_id=t.thread_id,
        tenant_id=t.tenant_id,
        workspace_id=t.workspace_id,
        owner_user_id=t.owner_user_id,
        title=t.title,
        revision=t.revision,
        created_at=t.created_at,
        updated_at=t.updated_at,
        custom_metadata=t.custom_metadata,
        summary=ThreadSummaryBody.model_validate(summary.model_dump()) if summary else None,
    )


def _message_response(m: Message) -> MessageResponse:
    return MessageResponse(
        message_id=m.message_id,
        thread_id=m.thread_id,
        session_id=m.session_id,
        turn_id=m.turn_id,
        role=m.role,
        kind=m.kind,
        sequence=m.sequence,
        content=m.content,
        author_principal=m.author_principal,
        agent_run_id=m.agent_run_id,
        occurred_at=m.occurred_at,
        archive_status=m.archive_status,
        attachments=[a.model_dump(exclude={"attachment_id", "message_id"}) for a in m.attachments],  # type: ignore[misc]
        custom_metadata=m.custom_metadata,
    )


def _service(container) -> ConversationService:  # type: ignore[no-untyped-def]
    return container.services["conversation"]


async def _hydrate(archive, message: Message) -> Message:  # type: ignore[no-untyped-def]
    """Fill in content that was purged from the hot store after archival."""
    if archive is None or message.archive_status is not ArchiveStatus.PURGED:
        return message
    return message.model_copy(update={"content": await archive.load_message_content(message)})


# --------------------------------------------------------------------------- threads


@router.patch(
    "/threads/{thread_id}",
    response_model=ThreadResponse,
    tags=["threads"],
    summary="Set a thread's title and metadata (the thread is created when it does not exist)",
    responses=_WRITE_ERRORS,
)
async def patch_thread(
    thread_id: ThreadIdPath,
    request: Request,
    body: PatchThreadRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
) -> Response:
    ctx = build_context(request, container, body.scope.model_copy(update={"thread_id": thread_id}))

    async def handler(uow):  # type: ignore[no-untyped-def]
        thread = await _service(container).patch_thread(
            uow, ctx, thread_id, title=body.title, custom_metadata=body.custom_metadata
        )
        return 200, _thread_response(thread).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload={"thread_id": thread_id, **body.model_dump(mode="json")},
        handler=handler,
    )


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadResponse,
    tags=["threads"],
    summary="Get a thread, with its durable summary once it has one",
    responses=_READ_ERRORS,
)
async def get_thread(
    thread_id: ThreadIdPath, ctx: ThreadContextDep, container: ContainerDep
) -> ThreadResponse:
    async with container.services["uow_factory"]() as uow:
        thread = await _service(container).get_thread(uow, ctx, thread_id)
        summary = await uow.summaries.latest(ctx.tenant_id, thread_id)
    return _thread_response(thread, summary)


@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    tags=["threads"],
    summary="Soft-delete a thread",
    responses=_READ_ERRORS,
)
async def delete_thread(
    request: Request, thread_id: ThreadIdPath, ctx: ThreadContextDep, container: ContainerDep
) -> Response:
    """With ``Idempotency-Key``, a retry of a delete that succeeded is its 204 again, not
    the 404 the deleted thread would now earn."""

    async def handler(uow):  # type: ignore[no-untyped-def]
        await _service(container).delete_thread(uow, ctx, thread_id)
        return NO_CONTENT, {}, None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload={"action": "delete", "thread_id": thread_id},
        handler=handler,
    )


@router.get(
    "/threads/{thread_id}/messages",
    response_model=MessageListResponse,
    tags=["messages"],
    summary="List messages (newest page last; page backwards with before_sequence)",
    responses=_READ_ERRORS,
)
async def list_messages(
    request: Request,
    response: Response,
    thread_id: ThreadIdPath,
    ctx: ThreadContextDep,
    container: ContainerDep,
    limit: Annotated[int, limit_query(500, "messages")] = 50,
    before_sequence: Annotated[
        int | None, Query(ge=1, description="Return messages with sequence < this value")
    ] = None,
    cursor: CursorQuery = None,
    include_internal: Annotated[
        bool, Query(description="Include INTERNAL agent/tool messages (lineage owners only)")
    ] = False,
) -> MessageListResponse:
    position = decode_cursor(cursor, fields={"before_sequence": int})
    if position is not None:
        before_sequence = position["before_sequence"] or None
    archive = container.services.get("archive_service")
    async with container.services["uow_factory"]() as uow:
        rows = await _service(container).list_messages(
            uow,
            ctx,
            thread_id,
            limit=limit + 1,
            before_sequence=before_sequence,
            include_internal=include_internal,
        )
    # one row more than the page proves an older page exists; the page is the newest ``limit``
    has_more = len(rows) > limit
    messages = [await _hydrate(archive, m) for m in (rows[-limit:] if has_more else rows)]
    next_before = messages[0].sequence if has_more and messages else None
    next_cursor = encode_cursor({"before_sequence": next_before}) if next_before else None
    link_next(request, response, next_cursor)
    return MessageListResponse(
        thread_id=thread_id,
        messages=[_message_response(m) for m in messages],
        next_before_sequence=next_before,
        next_cursor=next_cursor,
    )


# --------------------------------------------------------------------------- messages


def _with_thread(ctx: MemoryExecutionContext) -> MemoryExecutionContext:
    """The thread messages go to: the scope's, else the agent run's own (a run is a
    conversation of its own when nobody named one)."""
    if ctx.thread_id:
        return ctx
    if ctx.agent_run_id:
        return ctx.model_copy(update={"thread_id": ctx.agent_run_id})
    raise ValidationFailed("messages need a thread_id, or an agent_run_id to default it to")


@router.post(
    "/messages",
    response_model=MessagesAckResponse,
    status_code=202,
    tags=["messages"],
    summary="Append messages (durably acknowledged, processed asynchronously)",
    description=(
        "Persists the messages, their observations and the processing jobs in one transaction "
        "and returns 202 only after COMMIT. role=EVENT tells the service something that "
        "happened, to learn from. Retries with the same Idempotency-Key (or the same lineage "
        "+ contents when the header is absent) return the original acknowledgements."
    ),
    responses={
        **_WRITE_ERRORS,
        202: {"model": MessagesAckResponse, "description": "Durably acknowledged"},
    },
)
async def create_messages(
    request: Request, body: CreateMessagesRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response:
    ctx = _with_thread(build_context(request, container, body.scope))
    identity: tuple[str, ...] = (
        "messages",
        *(
            f"{m.role.value}\x1f{m.kind.value}\x1f{m.content}\x1f"
            + (m.occurred_at.isoformat() if m.occurred_at and ctx.turn_id is None else "")
            for m in body.messages
        ),
    )
    key = request.state.idempotency_key or default_idempotency_key(ctx, *identity)
    payload = derived_or_body(request, body, identity)
    service = _service(container)

    async def handler(uow):  # type: ignore[no-untyped-def]
        results = []
        for m in body.messages:
            results.append(
                await service.append_message(
                    uow,
                    ctx,
                    role=m.role,
                    kind=m.kind,
                    content=m.content,
                    attachments=[
                        Attachment(message_id="pending", **a.model_dump()) for a in m.attachments
                    ],
                    custom_metadata=m.custom_metadata,
                    occurred_at=m.occurred_at,
                    source_system=m.source_system,
                    source_message_id=m.source_message_id,
                    parent_message_id=m.parent_message_id,
                )
            )
        acks = MessagesAckResponse(
            messages=[MessageAckResponse(**r.ack.__dict__) for r in results]
        ).model_dump(mode="json")

        async def after_commit() -> None:
            for result in results:
                await service.after_commit(result)

        return 202, acks, after_commit

    return await run_idempotent(
        request,
        container,
        ctx,
        key=key,
        payload=payload,
        handler=handler,
        location=_first_message_job,
    )


def _first_message_job(body: dict[str, Any]) -> str | None:
    """``Location`` of an accepted batch: the first job any of its messages queued."""
    for ack in body.get("messages") or []:
        if where := first_job(ack):
            return where
    return None


@router.get(
    "/messages/{message_id}",
    response_model=MessageResponse,
    tags=["messages"],
    summary="Get a message",
    responses=_READ_ERRORS,
)
async def get_message(
    message_id: str, ctx: HeaderContextDep, container: ContainerDep
) -> MessageResponse:
    async with container.services["uow_factory"]() as uow:
        message = await _service(container).get_message(uow, ctx, message_id)
    return _message_response(await _hydrate(container.services.get("archive_service"), message))


# --------------------------------------------------------------------------- jobs


def _queue(name: str | None) -> Queue | None:
    """The queue a job runs on; ``None`` when the queue no longer reports one."""
    return Queue(name) if name in {q.value for q in Queue} else None


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    tags=["jobs"],
    summary="Background job status",
    responses=_READ_ERRORS,
)
async def get_job(job_id: str, ctx: HeaderContextDep, container: ContainerDep) -> JobResponse:
    """Accepts outbox references (``obx_<n>``) and task-queue ids."""
    if job_id.startswith("obx_"):
        from memory_service.adapters.db.orm import OutboxRow

        async with container.database.session() as session:
            row = await session.get(OutboxRow, int(job_id[4:]))
        if row is None or (row.tenant_id and row.tenant_id != ctx.tenant_id):
            raise NotFound("Job not found")
        if row.job_id is None:
            return JobResponse(
                job_id=job_id,
                task_name=row.task_name,
                queue=_queue(row.queue),
                status=JobStatus.PENDING,
                attempts=row.attempts,
                last_error=row.last_error,
            )
        info = await container.tasks.get(row.job_id) if container.tasks is not None else None
        if info is None:
            return JobResponse(
                job_id=job_id,
                task_name=row.task_name,
                queue=_queue(row.queue),
                status=JobStatus.PENDING,
                attempts=row.attempts,
            )
        return JobResponse(
            job_id=job_id,
            task_name=row.task_name,
            queue=_queue(row.queue),
            status=info.status,
            attempts=info.attempts,
            last_error=info.last_error,
        )
    # A task-queue id carries no tenant of its own. Procrastinate ids are sequential
    # integers and the TaskQueue port never modelled a tenant, so this branch used to hand
    # back any tenant's task_name, queue, status, attempts and schedule to whoever named an
    # id - and sequential ids do not have to be guessed. Every other read in the service is
    # scoped by visibility keys; this one had nothing to scope by.
    #
    # The outbox is what knows who dispatched a job, so a job this tenant's outbox never
    # dispatched is not found. The null-tenant case is allowed through exactly as the
    # ``obx_`` branch above allows it: those rows are the service's own periodic work and
    # belong to no tenant.
    async with container.services["uow_factory"]() as uow:
        dispatched, owner = await uow.outbox.dispatcher_of(job_id)
    if not dispatched or (owner and owner != ctx.tenant_id):
        raise NotFound("Job not found")
    info = await container.tasks.get(job_id) if container.tasks is not None else None
    if info is None:
        raise NotFound("Job not found")
    return JobResponse(
        job_id=info.job_id,
        task_name=info.task_name,
        queue=_queue(info.queue),
        status=info.status,
        attempts=info.attempts,
        last_error=info.last_error,
    )
