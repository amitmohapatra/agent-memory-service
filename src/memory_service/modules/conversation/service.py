"""ConversationService: threads, sessions, turns, messages and internal lineage.

Dynamic IDs: clients (ChatGPT-style UI, SDK, LangGraph adapter) mint thread/session/turn ids
and the service creates the rows on first sight, inside the same transaction as the message.

Hot path (synchronous, one transaction):
    authorize -> upsert thread/session/turn -> message row (+version, attachments)
    -> observation row -> outbox jobs (memory + archive) -> revisions -> COMMIT
Everything expensive (extraction, embedding, graph, archive, summaries) is asynchronous.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import AgentRun, Attachment, Message, Session, Thread, Turn
from memory_service.domain.enums import MessageKind, MessageRole, ObservationKind
from memory_service.domain.errors import NotFound, ScopeDenied, ValidationFailed
from memory_service.domain.ids import content_hash, new_id
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.text import sanitise
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.working_memory.hot_thread import HotThreadCache
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)

TASK_PROCESS_OBSERVATION = "memory.process_observation"
TASK_ARCHIVE_STAGE = "archive.stage_message"


@dataclass(frozen=True)
class MessageAck:
    message_id: str
    thread_id: str
    session_id: str
    turn_id: str
    sequence: int
    job_ids: list[str] = field(default_factory=list)
    deduplicated: bool = False
    observation_id: str | None = None


@dataclass(frozen=True)
class AppendResult:
    ack: MessageAck
    message: Message | None
    revision: int = 0  # thread revision after this append (validates the hot cache)


class ConversationService:
    def __init__(
        self,
        authz: AuthorizationService,
        hot_cache: HotThreadCache,
        *,
        archive_enabled: bool = True,
    ) -> None:
        self.authz = authz
        self.hot = hot_cache
        self.archive_enabled = archive_enabled

    # -- threads ----------------------------------------------------------------
    async def create_thread(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        thread_id: str | None = None,
        title: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        source_system: str | None = None,
        source_thread_id: str | None = None,
    ) -> Thread:
        thread_id = thread_id or ctx.thread_id or new_id("thread")
        existing = await uow.threads.get(ctx.tenant_id, thread_id)
        if existing is not None:
            await self.authz.require(ctx, "can_read", "thread", thread_id)
            return existing
        thread = Thread(
            thread_id=thread_id,
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            owner_user_id=ctx.user_id,
            title=title,
            source_system=source_system,
            source_thread_id=source_thread_id,
            custom_metadata=custom_metadata or {},
        )
        await uow.threads.add(thread)
        await self.authz.grant_thread(
            ctx, thread_id, workspace_id=ctx.workspace_id, revisions=uow.revisions
        )
        await uow.revisions.bump(ctx.tenant_id, RevisionKind.THREAD, thread_id)
        if ctx.user_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.USER, ctx.user_id)
        log.info("thread.created", **ctx.log_fields())
        return thread

    async def get_thread(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, thread_id: str
    ) -> Thread:
        thread = await uow.threads.get(ctx.tenant_id, thread_id)
        if thread is None:
            raise NotFound("Thread not found")
        await self.authz.require(ctx, "can_read", "thread", thread_id)
        return thread

    async def delete_thread(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, thread_id: str
    ) -> None:
        if await uow.threads.get(ctx.tenant_id, thread_id) is None:
            raise NotFound("Thread not found")
        await self.authz.require(ctx, "can_write", "thread", thread_id)
        await uow.threads.soft_delete(ctx.tenant_id, thread_id)
        await uow.revisions.bump(ctx.tenant_id, RevisionKind.THREAD, thread_id)
        await self.hot.invalidate(ctx.tenant_id, thread_id)

    # -- messages ---------------------------------------------------------------
    async def append_message(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        role: MessageRole,
        content: str,
        kind: MessageKind = MessageKind.VISIBLE,
        attachments: Sequence[Attachment] = (),
        custom_metadata: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
        source_system: str | None = None,
        source_message_id: str | None = None,
        hints: ProcessingHints | None = None,
        parent_message_id: str | None = None,
    ) -> AppendResult:
        if not ctx.thread_id or not ctx.session_id or not ctx.turn_id:
            raise ValidationFailed("thread_id, session_id and turn_id are required for messages")
        # Same reason as MemoryService.submit_observation: a message is agent-authored text
        # going into a PostgreSQL `text` column, and a NUL byte in it fails the INSERT.
        content = sanitise(content)
        if kind is MessageKind.VISIBLE and role not in (
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.SYSTEM,
        ):
            raise ValidationFailed("visible messages must have role USER, ASSISTANT or SYSTEM")
        if kind is MessageKind.INTERNAL and ctx.agent_id is None and role in (MessageRole.AGENT,):
            raise ValidationFailed("internal AGENT messages require agent_id in the scope")

        with (
            span("conversation.append", tenant_id=ctx.tenant_id),
            stage_seconds.labels("conversation.append").time(),
        ):
            # concurrent first messages of a new thread (and its session/turn) must not
            # race on creation: serialize writers per thread for this transaction
            await uow.serialize(f"thread:{ctx.tenant_id}/{ctx.thread_id}")
            thread = await uow.threads.get(ctx.tenant_id, ctx.thread_id)
            if thread is None:
                await self.create_thread(uow, ctx, thread_id=ctx.thread_id)
            else:
                await self.authz.require(ctx, "can_write", "thread", ctx.thread_id)

            # imports: the same source message must not be stored twice
            if source_system and source_message_id:
                dup = await uow.messages.find_by_source(
                    ctx.tenant_id, source_system, source_message_id
                )
                if dup is not None:
                    ack = MessageAck(
                        dup.message_id,
                        dup.thread_id,
                        dup.session_id,
                        dup.turn_id,
                        dup.sequence,
                        deduplicated=True,
                    )
                    return AppendResult(ack, None)

            await self._ensure_session_and_turn(uow, ctx)
            if ctx.agent_run_id:
                await self._ensure_agent_run(uow, ctx)

            sequence = await uow.messages.next_sequence(ctx.tenant_id, ctx.thread_id)
            now = datetime.now(UTC)
            message = Message(
                thread_id=ctx.thread_id,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
                tenant_id=ctx.tenant_id,
                role=role,
                kind=kind,
                sequence=sequence,
                content=content,
                content_hash=content_hash(content),
                author_principal=ctx.principal_id,
                agent_run_id=ctx.agent_run_id,
                parent_message_id=parent_message_id,
                attachments=list(attachments),
                source_system=source_system,
                source_message_id=source_message_id,
                occurred_at=occurred_at or now,
                created_at=now,
                system_metadata={"trace_id": ctx.trace_id, "request_id": ctx.request_id},
                custom_metadata=custom_metadata or {},
            )
            await uow.messages.add(message)
            await uow.messages.add_version(message)

            observation = Observation(
                tenant_id=ctx.tenant_id,
                kind=ObservationKind.MESSAGE,
                content=content,
                content_hash=message.content_hash,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
                work_id=ctx.work_id,
                task_id=ctx.task_id,
                agent_id=ctx.agent_id,
                agent_group_id=ctx.agent_group_id,
                agent_run_id=ctx.agent_run_id,
                parent_agent_run_id=ctx.parent_agent_run_id,
                principal_id=ctx.principal_id,
                trace_id=ctx.trace_id,
                message_id=message.message_id,
                source_system=source_system,
                source_id=source_message_id,
                hints=hints or ProcessingHints(),
                custom_metadata={"role": role.value, "kind": kind.value},
                occurred_at=message.occurred_at,
            )
            await uow.observations.add(observation)

            job_ids: list[str] = []
            outbox_id = await uow.enqueue(
                JobSpec(
                    task_name=TASK_PROCESS_OBSERVATION,
                    queue=Queue.CHAT_FAST,
                    payload={
                        "tenant_id": ctx.tenant_id,
                        "observation_id": observation.observation_id,
                        "trace_id": ctx.trace_id,
                    },
                    idempotency_key=f"obs:{observation.observation_id}",
                    tenant_id=ctx.tenant_id,
                )
            )
            if outbox_id is not None:
                job_ids.append(f"obx_{outbox_id}")
            if self.archive_enabled:
                outbox_id = await uow.enqueue(
                    JobSpec(
                        task_name=TASK_ARCHIVE_STAGE,
                        queue=Queue.ARCHIVE,
                        payload={"tenant_id": ctx.tenant_id, "thread_id": ctx.thread_id},
                        idempotency_key=f"archive:thread:{ctx.tenant_id}:{ctx.thread_id}",
                        lock=f"archive:{ctx.tenant_id}:{ctx.thread_id}",
                        schedule_in_seconds=60,
                        tenant_id=ctx.tenant_id,
                    )
                )
                if outbox_id is not None:
                    job_ids.append(f"obx_{outbox_id}")

            revision = await uow.threads.touch(ctx.tenant_id, ctx.thread_id)
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.THREAD, ctx.thread_id)
            ack = MessageAck(
                message.message_id,
                ctx.thread_id,
                ctx.session_id,
                ctx.turn_id,
                sequence,
                job_ids=job_ids,
                observation_id=observation.observation_id,
            )
            return AppendResult(ack, message, revision)

    async def after_commit(self, result: AppendResult) -> None:
        """Post-commit side effects that must never affect the acknowledgement."""
        if result.message is not None:
            await self.hot.append(result.message, revision=result.revision)

    async def list_messages(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        thread_id: str,
        *,
        limit: int = 50,
        before_sequence: int | None = None,
        include_internal: bool = False,
    ) -> list[Message]:
        thread = await uow.threads.get(ctx.tenant_id, thread_id)
        if thread is None:
            raise NotFound("Thread not found")
        await self.authz.require(ctx, "can_read", "thread", thread_id)
        newest = not include_internal and before_sequence is None
        if newest:
            # the cache answers only when it is provably current for this thread revision
            # (appends during a cache outage advance the revision without reaching the cache)
            cached = await self.hot.recent(
                ctx.tenant_id, thread_id, limit=limit, revision=thread.revision
            )
            if cached is not None:
                return cached
        messages = await uow.messages.list_thread(
            ctx.tenant_id,
            thread_id,
            limit=limit,
            before_sequence=before_sequence,
            include_internal=include_internal,
        )
        if newest:
            await self.hot.refill(ctx.tenant_id, thread_id, messages, revision=thread.revision)
        return messages

    async def get_message(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, message_id: str
    ) -> Message:
        message = await uow.messages.get(ctx.tenant_id, message_id)
        if message is None:
            raise NotFound("Message not found")
        await self.authz.require(ctx, "can_read", "thread", message.thread_id)
        if (
            message.kind is MessageKind.INTERNAL
            and not ctx.is_agent
            and message.author_principal != ctx.principal_id
        ):
            # internal execution is not exposed as chat; only lineage owners/agents see it
            raise ScopeDenied("Internal messages are not visible to this principal")
        return message

    # -- helpers ----------------------------------------------------------------
    async def _ensure_session_and_turn(self, uow: UnitOfWork, ctx: MemoryExecutionContext) -> None:
        assert ctx.thread_id and ctx.session_id and ctx.turn_id
        session = await uow.sessions.get(ctx.tenant_id, ctx.session_id)
        if session is None:
            await uow.sessions.add(
                Session(
                    session_id=ctx.session_id,
                    thread_id=ctx.thread_id,
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    client=str(ctx.custom_metadata.get("client", "sdk")),
                )
            )
        elif session.thread_id != ctx.thread_id:
            raise ValidationFailed("session_id belongs to a different thread")
        turn = await uow.turns.get(ctx.tenant_id, ctx.turn_id)
        if turn is None:
            seq = await uow.turns.next_sequence(ctx.tenant_id, ctx.thread_id)
            await uow.turns.add(
                Turn(
                    turn_id=ctx.turn_id,
                    session_id=ctx.session_id,
                    thread_id=ctx.thread_id,
                    tenant_id=ctx.tenant_id,
                    sequence=seq,
                )
            )
        elif turn.session_id != ctx.session_id:
            raise ValidationFailed("turn_id belongs to a different session")

    async def _ensure_agent_run(self, uow: UnitOfWork, ctx: MemoryExecutionContext) -> None:
        assert ctx.agent_run_id and ctx.agent_id
        run = await uow.agent_runs.get(ctx.tenant_id, ctx.agent_run_id)
        if run is None:
            await uow.agent_runs.add(
                AgentRun(
                    agent_run_id=ctx.agent_run_id,
                    tenant_id=ctx.tenant_id,
                    thread_id=ctx.thread_id,
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    work_id=ctx.work_id,
                    task_id=ctx.task_id,
                    agent_id=ctx.agent_id,
                    agent_group_id=ctx.agent_group_id,
                    parent_agent_run_id=ctx.parent_agent_run_id,
                    trace_id=ctx.trace_id,
                )
            )
        if ctx.turn_id:
            await uow.turns.link_run(ctx.turn_id, ctx.agent_run_id)
