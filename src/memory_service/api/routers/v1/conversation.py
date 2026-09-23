"""Public /v1 routes: threads, messages, jobs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import default_idempotency_key, run_idempotent
from memory_service.api.schemas.conversation import (
    CreateMessageRequest,
    CreateThreadRequest,
    JobResponse,
    MessageAckResponse,
    MessageListResponse,
    MessageResponse,
    ThreadResponse,
)
from memory_service.domain.conversation import Attachment, Message, Thread
from memory_service.domain.enums import ArchiveStatus, JobStatus
from memory_service.domain.errors import NotFound
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.conversation.service import ConversationService

router = APIRouter()

_WRITE_ERRORS = error_responses(401, 403, 409, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)


def _thread_response(t: Thread) -> dict[str, Any]:
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
    ).model_dump(mode="json")


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


@router.post(
    "/threads",
    response_model=ThreadResponse,
    status_code=201,
    tags=["threads"],
    summary="Create (or idempotently fetch) a thread",
    responses={
        **_WRITE_ERRORS,
        200: {"model": ThreadResponse, "description": "Replayed (Idempotency-Key seen before)"},
    },
)
async def create_thread(
    request: Request, body: CreateThreadRequest, container: ContainerDep, _: ServicePrincipalDep
) -> JSONResponse:
    ctx = build_context(request, container, body.scope)
    thread_id = body.thread_id or ctx.thread_id
    key = request.state.idempotency_key or default_idempotency_key(
        ctx, "thread", thread_id or "", body.title or ""
    )
    payload = body.model_dump(mode="json")

    async def handler(uow):  # type: ignore[no-untyped-def]
        thread = await _service(container).create_thread(
            uow, ctx, thread_id=thread_id, title=body.title, custom_metadata=body.custom_metadata
        )
        return 201, _thread_response(thread), None

    return await run_idempotent(request, container, ctx, key=key, payload=payload, handler=handler)


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadResponse,
    tags=["threads"],
    summary="Get a thread",
    responses=_READ_ERRORS,
)
async def get_thread(
    thread_id: str, ctx: HeaderContextDep, container: ContainerDep
) -> ThreadResponse:
    async with container.services["uow_factory"]() as uow:
        thread = await _service(container).get_thread(uow, ctx, thread_id)
    return ThreadResponse.model_validate(_thread_response(thread))


@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    tags=["threads"],
    summary="Soft-delete a thread",
    responses=_READ_ERRORS,
)
async def delete_thread(thread_id: str, ctx: HeaderContextDep, container: ContainerDep) -> None:
    async with container.services["uow_factory"]() as uow:
        await _service(container).delete_thread(uow, ctx, thread_id)
        await uow.commit()


@router.get(
    "/threads/{thread_id}/messages",
    response_model=MessageListResponse,
    tags=["messages"],
    summary="List messages (newest page last; page backwards with before_sequence)",
    responses=_READ_ERRORS,
)
async def list_messages(
    thread_id: str,
    ctx: HeaderContextDep,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=500, examples=[50])] = 50,
    before_sequence: Annotated[
        int | None, Query(ge=1, description="Return messages with sequence < this value")
    ] = None,
    include_internal: Annotated[
        bool, Query(description="Include INTERNAL agent/tool messages (lineage owners only)")
    ] = False,
) -> MessageListResponse:
    archive = container.services.get("archive_service")
    async with container.services["uow_factory"]() as uow:
        messages = await _service(container).list_messages(
            uow,
            ctx,
            thread_id,
            limit=limit,
            before_sequence=before_sequence,
            include_internal=include_internal,
        )
    messages = [await _hydrate(archive, m) for m in messages]
    next_before = (
        messages[0].sequence if len(messages) == limit and messages[0].sequence > 1 else None
    )
    return MessageListResponse(
        thread_id=thread_id,
        messages=[_message_response(m) for m in messages],
        next_before_sequence=next_before,
    )


# --------------------------------------------------------------------------- messages


@router.post(
    "/messages",
    response_model=MessageAckResponse,
    status_code=202,
    tags=["messages"],
    summary="Append a message (durably acknowledged, processed asynchronously)",
    description=(
        "Persists the message, its observation and the processing jobs in one transaction and "
        "returns 202 only after COMMIT. Retries with the same Idempotency-Key (or the same "
        "lineage + content when the header is absent) return the original acknowledgement."
    ),
    responses={
        **_WRITE_ERRORS,
        202: {"model": MessageAckResponse, "description": "Durably acknowledged"},
    },
)
async def create_message(
    request: Request, body: CreateMessageRequest, container: ContainerDep, _: ServicePrincipalDep
) -> JSONResponse:
    ctx = build_context(request, container, body.scope)
    key = request.state.idempotency_key or default_idempotency_key(
        ctx, "message", body.role.value, body.kind.value, body.content
    )
    payload = body.model_dump(mode="json")
    service = _service(container)

    async def handler(uow):  # type: ignore[no-untyped-def]
        result = await service.append_message(
            uow,
            ctx,
            role=body.role,
            kind=body.kind,
            content=body.content,
            attachments=[
                Attachment(message_id="pending", **a.model_dump()) for a in body.attachments
            ],
            custom_metadata=body.custom_metadata,
            occurred_at=body.occurred_at,
            source_system=body.source_system,
            source_message_id=body.source_message_id,
            hints=ProcessingHints(**body.hints.model_dump()),
            parent_message_id=body.parent_message_id,
        )
        ack = MessageAckResponse(**result.ack.__dict__).model_dump(mode="json")

        async def after_commit() -> None:
            await service.after_commit(result)

        return 202, ack, after_commit

    return await run_idempotent(request, container, ctx, key=key, payload=payload, handler=handler)


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
                queue=row.queue,
                status=JobStatus.PENDING,
                attempts=row.attempts,
                last_error=row.last_error,
            )
        info = await container.tasks.get(row.job_id) if container.tasks is not None else None
        if info is None:
            return JobResponse(
                job_id=job_id,
                task_name=row.task_name,
                queue=row.queue,
                status=JobStatus.PENDING,
                attempts=row.attempts,
            )
        return JobResponse(
            job_id=job_id,
            task_name=row.task_name,
            queue=row.queue,
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
        queue=info.queue,
        status=info.status,
        attempts=info.attempts,
        last_error=info.last_error,
    )
