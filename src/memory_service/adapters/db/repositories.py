"""PostgreSQL repositories (SQLAlchemy 2 async). Translate ORM rows <-> domain models."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import (
    AgentRunRow,
    ArchiveSegmentRow,
    IdempotencyRow,
    MessageAttachmentRow,
    MessageRow,
    MessageVersionRow,
    ObservationRow,
    OutboxRow,
    RevisionRow,
    SessionRow,
    ThreadRow,
    TurnRow,
    TurnRunLinkRow,
)
from memory_service.domain.conversation import (
    AgentRun,
    Attachment,
    Message,
    Session,
    Thread,
    Turn,
)
from memory_service.domain.enums import ArchiveStatus, MessageKind, MessageRole, ObservationKind
from memory_service.domain.ids import new_id
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.domain.revisions import RevisionKind
from memory_service.ports.repositories import ArchiveSegment, IdempotencyRecord, OutboxEntry
from memory_service.ports.tasks import JobSpec, Queue


def _rowcount(result: Any) -> int:
    value = getattr(result, "rowcount", None)
    return int(value) if value is not None and value >= 0 else 0


def _row_to_thread(r: ThreadRow) -> Thread:
    return Thread(
        thread_id=r.thread_id,
        tenant_id=r.tenant_id,
        workspace_id=r.workspace_id,
        owner_user_id=r.owner_user_id,
        title=r.title,
        source_system=r.source_system,
        source_thread_id=r.source_thread_id,
        system_metadata=r.system_metadata or {},
        custom_metadata=r.custom_metadata or {},
        revision=r.revision,
        created_at=r.created_at,
        updated_at=r.updated_at,
        archived_at=r.archived_at,
        deleted_at=r.deleted_at,
    )


async def _next_sequence(session: Any, row: Any, tenant_id: str, thread_id: str) -> int:
    """The next per-thread sequence number, serialised on the thread row.

    Turns and messages both number themselves within a thread and both need the same lock:
    without ``FOR UPDATE`` on the thread, two concurrent writers read the same maximum and
    collide on the unique constraint. The two implementations were identical apart from the
    table, which is exactly the kind of copy that drifts — one of them had the comment
    explaining the lock and the other did not.
    """
    await session.execute(
        select(ThreadRow.thread_id)
        .where(ThreadRow.thread_id == thread_id, ThreadRow.tenant_id == tenant_id)
        .with_for_update()
    )
    current = await session.scalar(
        select(func.coalesce(func.max(row.sequence), 0)).where(
            row.thread_id == thread_id, row.tenant_id == tenant_id
        )
    )
    return int(current or 0) + 1


class SqlThreadRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, thread: Thread) -> None:
        self.s.add(
            ThreadRow(
                thread_id=thread.thread_id,
                tenant_id=thread.tenant_id,
                workspace_id=thread.workspace_id,
                owner_user_id=thread.owner_user_id,
                title=thread.title,
                source_system=thread.source_system,
                source_thread_id=thread.source_thread_id,
                system_metadata=thread.system_metadata,
                custom_metadata=thread.custom_metadata,
                revision=thread.revision,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, thread_id: str) -> Thread | None:
        row = await self.s.get(ThreadRow, thread_id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return _row_to_thread(row)

    async def touch(self, tenant_id: str, thread_id: str, *, title: str | None = None) -> int:
        values: dict[str, Any] = {"revision": ThreadRow.revision + 1, "updated_at": func.now()}
        if title is not None:
            values["title"] = title
        result = await self.s.execute(
            update(ThreadRow)
            .where(ThreadRow.thread_id == thread_id, ThreadRow.tenant_id == tenant_id)
            .values(**values)
            .returning(ThreadRow.revision)
        )
        value = result.scalar_one_or_none()
        return int(value) if value is not None else 0

    async def list_for_user(
        self, tenant_id: str, user_id: str, *, limit: int = 50, before: datetime | None = None
    ) -> list[Thread]:
        stmt = (
            select(ThreadRow)
            .where(
                ThreadRow.tenant_id == tenant_id,
                ThreadRow.owner_user_id == user_id,
                ThreadRow.deleted_at.is_(None),
            )
            .order_by(ThreadRow.updated_at.desc())
            .limit(limit)
        )
        if before is not None:
            stmt = stmt.where(ThreadRow.updated_at < before)
        rows = (await self.s.execute(stmt)).scalars().all()
        return [_row_to_thread(r) for r in rows]

    async def list_active(self, *, since: datetime, limit: int = 500) -> list[Thread]:
        rows = (
            await self.s.execute(
                select(ThreadRow)
                .where(ThreadRow.updated_at >= since, ThreadRow.deleted_at.is_(None))
                .order_by(ThreadRow.updated_at.desc())
                .limit(limit)
            )
        ).scalars()
        return [_row_to_thread(r) for r in rows.all()]

    async def soft_delete(self, tenant_id: str, thread_id: str) -> bool:
        result = await self.s.execute(
            update(ThreadRow)
            .where(
                ThreadRow.thread_id == thread_id,
                ThreadRow.tenant_id == tenant_id,
                ThreadRow.deleted_at.is_(None),
            )
            .values(deleted_at=func.now(), revision=ThreadRow.revision + 1)
        )
        return _rowcount(result) > 0


class SqlSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, session: Session) -> None:
        self.s.add(
            SessionRow(
                session_id=session.session_id,
                thread_id=session.thread_id,
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                client=session.client,
                started_at=session.started_at,
                ended_at=session.ended_at,
                custom_metadata=session.custom_metadata,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, session_id: str) -> Session | None:
        r = await self.s.get(SessionRow, session_id)
        if r is None or r.tenant_id != tenant_id:
            return None
        return Session(
            session_id=r.session_id,
            thread_id=r.thread_id,
            tenant_id=r.tenant_id,
            user_id=r.user_id,
            client=r.client,
            started_at=r.started_at,
            ended_at=r.ended_at,
            custom_metadata=r.custom_metadata or {},
        )

    async def end(self, tenant_id: str, session_id: str) -> None:
        await self.s.execute(
            update(SessionRow)
            .where(SessionRow.session_id == session_id, SessionRow.tenant_id == tenant_id)
            .values(ended_at=func.now())
        )


class SqlTurnRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, turn: Turn) -> None:
        self.s.add(
            TurnRow(
                turn_id=turn.turn_id,
                session_id=turn.session_id,
                thread_id=turn.thread_id,
                tenant_id=turn.tenant_id,
                sequence=turn.sequence,
                started_at=turn.started_at,
                completed_at=turn.completed_at,
                custom_metadata=turn.custom_metadata,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, turn_id: str) -> Turn | None:
        r = await self.s.get(TurnRow, turn_id)
        if r is None or r.tenant_id != tenant_id:
            return None
        return Turn(
            turn_id=r.turn_id,
            session_id=r.session_id,
            thread_id=r.thread_id,
            tenant_id=r.tenant_id,
            sequence=r.sequence,
            started_at=r.started_at,
            completed_at=r.completed_at,
            custom_metadata=r.custom_metadata or {},
        )

    async def next_sequence(self, tenant_id: str, thread_id: str) -> int:
        return await _next_sequence(self.s, TurnRow, tenant_id, thread_id)

    async def complete(self, tenant_id: str, turn_id: str) -> None:
        await self.s.execute(
            update(TurnRow)
            .where(TurnRow.turn_id == turn_id, TurnRow.tenant_id == tenant_id)
            .values(completed_at=func.now())
        )

    async def link_run(self, turn_id: str, agent_run_id: str) -> None:
        await self.s.execute(
            pg_insert(TurnRunLinkRow)
            .values(turn_id=turn_id, agent_run_id=agent_run_id)
            .on_conflict_do_nothing()
        )


def _row_to_message(r: MessageRow, attachments: Sequence[MessageAttachmentRow] = ()) -> Message:
    return Message(
        message_id=r.message_id,
        thread_id=r.thread_id,
        session_id=r.session_id,
        turn_id=r.turn_id,
        tenant_id=r.tenant_id,
        role=MessageRole(r.role),
        kind=MessageKind(r.kind),
        sequence=r.sequence,
        content=r.content or "",
        content_hash=r.content_hash,
        version=r.version,
        author_principal=r.author_principal,
        agent_run_id=r.agent_run_id,
        parent_message_id=r.parent_message_id,
        attachments=[
            Attachment(
                attachment_id=a.attachment_id,
                message_id=a.message_id,
                document_id=a.document_id,
                filename=a.filename,
                media_type=a.media_type,
                size_bytes=a.size_bytes,
                checksum=a.checksum,
            )
            for a in attachments
        ],
        source_system=r.source_system,
        source_message_id=r.source_message_id,
        occurred_at=r.occurred_at,
        created_at=r.created_at,
        archive_status=ArchiveStatus(r.archive_status),
        system_metadata=r.system_metadata or {},
        custom_metadata=r.custom_metadata or {},
        deleted_at=r.deleted_at,
    )


class SqlMessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, message: Message) -> None:
        self.s.add(
            MessageRow(
                message_id=message.message_id,
                thread_id=message.thread_id,
                session_id=message.session_id,
                turn_id=message.turn_id,
                tenant_id=message.tenant_id,
                role=message.role.value,
                kind=message.kind.value,
                sequence=message.sequence,
                content=message.content,
                content_hash=message.content_hash,
                content_bytes=len(message.content.encode("utf-8")),
                version=message.version,
                author_principal=message.author_principal,
                agent_run_id=message.agent_run_id,
                parent_message_id=message.parent_message_id,
                source_system=message.source_system,
                source_message_id=message.source_message_id,
                occurred_at=message.occurred_at,
                created_at=message.created_at,
                archive_status=message.archive_status.value,
                system_metadata=message.system_metadata,
                custom_metadata=message.custom_metadata,
            )
        )
        for att in message.attachments:
            self.s.add(
                MessageAttachmentRow(
                    attachment_id=att.attachment_id,
                    message_id=message.message_id,
                    document_id=att.document_id,
                    filename=att.filename,
                    media_type=att.media_type,
                    size_bytes=att.size_bytes,
                    checksum=att.checksum,
                )
            )
        await self.s.flush()

    async def add_version(self, message: Message) -> None:
        self.s.add(
            MessageVersionRow(
                message_version_id=new_id("message_version"),
                message_id=message.message_id,
                version=message.version,
                content=message.content,
                content_hash=message.content_hash,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, message_id: str) -> Message | None:
        r = await self.s.get(MessageRow, message_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
            return None
        atts = (
            (
                await self.s.execute(
                    select(MessageAttachmentRow).where(
                        MessageAttachmentRow.message_id == message_id
                    )
                )
            )
            .scalars()
            .all()
        )
        return _row_to_message(r, atts)

    async def next_sequence(self, tenant_id: str, thread_id: str) -> int:
        return await _next_sequence(self.s, MessageRow, tenant_id, thread_id)

    async def list_thread(
        self,
        tenant_id: str,
        thread_id: str,
        *,
        limit: int = 50,
        before_sequence: int | None = None,
        include_internal: bool = False,
    ) -> list[Message]:
        stmt = (
            select(MessageRow)
            .where(
                MessageRow.tenant_id == tenant_id,
                MessageRow.thread_id == thread_id,
                MessageRow.deleted_at.is_(None),
            )
            .order_by(MessageRow.sequence.desc())
            .limit(limit)
        )
        if not include_internal:
            stmt = stmt.where(MessageRow.kind == MessageKind.VISIBLE.value)
        if before_sequence is not None:
            stmt = stmt.where(MessageRow.sequence < before_sequence)
        rows = list((await self.s.execute(stmt)).scalars().all())
        rows.reverse()
        if not rows:
            return []
        ids = [r.message_id for r in rows]
        atts = (
            (
                await self.s.execute(
                    select(MessageAttachmentRow).where(MessageAttachmentRow.message_id.in_(ids))
                )
            )
            .scalars()
            .all()
        )
        by_msg: dict[str, list[MessageAttachmentRow]] = {}
        for a in atts:
            by_msg.setdefault(a.message_id, []).append(a)
        return [_row_to_message(r, by_msg.get(r.message_id, ())) for r in rows]

    async def find_by_source(
        self, tenant_id: str, source_system: str, source_message_id: str
    ) -> Message | None:
        r = await self.s.scalar(
            select(MessageRow).where(
                MessageRow.tenant_id == tenant_id,
                MessageRow.source_system == source_system,
                MessageRow.source_message_id == source_message_id,
            )
        )
        return _row_to_message(r) if r is not None else None

    async def list_staged(
        self,
        *,
        older_than: datetime | None = None,
        limit: int = 1000,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[Message]:
        stmt = (
            select(MessageRow)
            .where(MessageRow.archive_status == ArchiveStatus.STAGED.value)
            .order_by(MessageRow.tenant_id, MessageRow.thread_id, MessageRow.sequence)
            .limit(limit)
        )
        if older_than is not None:
            stmt = stmt.where(MessageRow.created_at <= older_than)
        if tenant_id is not None:
            stmt = stmt.where(MessageRow.tenant_id == tenant_id)
        if thread_id is not None:
            stmt = stmt.where(MessageRow.thread_id == thread_id)
        rows = (await self.s.execute(stmt)).scalars().all()
        return [_row_to_message(r) for r in rows]

    async def mark_archived(
        self, message_ids: Sequence[str], *, segment_id: str, archived_at: datetime
    ) -> int:
        if not message_ids:
            return 0
        result = await self.s.execute(
            update(MessageRow)
            .where(MessageRow.message_id.in_(list(message_ids)))
            .values(
                archive_status=ArchiveStatus.ARCHIVED.value,
                archived_at=archived_at,
                archive_segment_id=segment_id,
            )
        )
        return _rowcount(result)

    async def purge_payloads(
        self, *, archived_before: datetime, min_bytes: int, limit: int = 1000
    ) -> int:
        ids = (
            (
                await self.s.execute(
                    select(MessageRow.message_id)
                    .where(
                        MessageRow.archive_status == ArchiveStatus.ARCHIVED.value,
                        MessageRow.archived_at <= archived_before,
                        MessageRow.content_bytes >= min_bytes,
                        MessageRow.content.is_not(None),
                    )
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        if not ids:
            return 0
        result = await self.s.execute(
            update(MessageRow)
            .where(MessageRow.message_id.in_(list(ids)))
            .values(content=None, archive_status=ArchiveStatus.PURGED.value)
        )
        return _rowcount(result)


class SqlAgentRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, run: AgentRun) -> None:
        self.s.add(
            AgentRunRow(
                agent_run_id=run.agent_run_id,
                tenant_id=run.tenant_id,
                thread_id=run.thread_id,
                session_id=run.session_id,
                turn_id=run.turn_id,
                work_id=run.work_id,
                task_id=run.task_id,
                agent_id=run.agent_id,
                agent_group_id=run.agent_group_id,
                parent_agent_run_id=run.parent_agent_run_id,
                status=run.status,
                started_at=run.started_at,
                completed_at=run.completed_at,
                trace_id=run.trace_id,
                custom_metadata=run.custom_metadata,
            )
        )
        await self.s.flush()

    @staticmethod
    def _to_domain(r: AgentRunRow) -> AgentRun:
        return AgentRun(
            agent_run_id=r.agent_run_id,
            tenant_id=r.tenant_id,
            thread_id=r.thread_id,
            session_id=r.session_id,
            turn_id=r.turn_id,
            work_id=r.work_id,
            task_id=r.task_id,
            agent_id=r.agent_id,
            agent_group_id=r.agent_group_id,
            parent_agent_run_id=r.parent_agent_run_id,
            status=r.status,
            started_at=r.started_at,
            completed_at=r.completed_at,
            trace_id=r.trace_id,
            custom_metadata=r.custom_metadata or {},
        )

    async def get(self, tenant_id: str, agent_run_id: str) -> AgentRun | None:
        r = await self.s.get(AgentRunRow, agent_run_id)
        if r is None or r.tenant_id != tenant_id:
            return None
        return self._to_domain(r)

    async def complete(self, tenant_id: str, agent_run_id: str, *, status: str) -> None:
        await self.s.execute(
            update(AgentRunRow)
            .where(AgentRunRow.agent_run_id == agent_run_id, AgentRunRow.tenant_id == tenant_id)
            .values(status=status, completed_at=func.now())
        )

    async def list_for_turn(self, tenant_id: str, turn_id: str) -> list[AgentRun]:
        rows = (
            (
                await self.s.execute(
                    select(AgentRunRow)
                    .where(AgentRunRow.tenant_id == tenant_id, AgentRunRow.turn_id == turn_id)
                    .order_by(AgentRunRow.started_at)
                )
            )
            .scalars()
            .all()
        )
        return [self._to_domain(r) for r in rows]


class SqlObservationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, o: Observation) -> None:
        self.s.add(
            ObservationRow(
                observation_id=o.observation_id,
                tenant_id=o.tenant_id,
                kind=o.kind.value,
                content=o.content,
                content_hash=o.content_hash,
                workspace_id=o.workspace_id,
                user_id=o.user_id,
                thread_id=o.thread_id,
                session_id=o.session_id,
                turn_id=o.turn_id,
                work_id=o.work_id,
                task_id=o.task_id,
                agent_id=o.agent_id,
                agent_group_id=o.agent_group_id,
                agent_run_id=o.agent_run_id,
                parent_agent_run_id=o.parent_agent_run_id,
                principal_id=o.principal_id,
                trace_id=o.trace_id,
                message_id=o.message_id,
                document_id=o.document_id,
                tool_run_id=o.tool_run_id,
                source_system=o.source_system,
                source_id=o.source_id,
                hints=o.hints.model_dump(mode="json", exclude_none=True),
                custom_metadata=o.custom_metadata,
                occurred_at=o.occurred_at,
                created_at=o.created_at,
                processed_at=o.processed_at,
            )
        )
        await self.s.flush()

    @staticmethod
    def _to_domain(r: ObservationRow) -> Observation:
        return Observation(
            observation_id=r.observation_id,
            tenant_id=r.tenant_id,
            kind=ObservationKind(r.kind),
            content=r.content or "",
            content_hash=r.content_hash,
            workspace_id=r.workspace_id,
            user_id=r.user_id,
            thread_id=r.thread_id,
            session_id=r.session_id,
            turn_id=r.turn_id,
            work_id=r.work_id,
            task_id=r.task_id,
            agent_id=r.agent_id,
            agent_group_id=r.agent_group_id,
            agent_run_id=r.agent_run_id,
            parent_agent_run_id=r.parent_agent_run_id,
            principal_id=r.principal_id,
            trace_id=r.trace_id,
            message_id=r.message_id,
            document_id=r.document_id,
            tool_run_id=r.tool_run_id,
            source_system=r.source_system,
            source_id=r.source_id,
            hints=ProcessingHints.model_validate(r.hints or {}),
            custom_metadata=r.custom_metadata or {},
            occurred_at=r.occurred_at,
            created_at=r.created_at,
            processed_at=r.processed_at,
        )

    async def get(self, tenant_id: str, observation_id: str) -> Observation | None:
        r = await self.s.get(ObservationRow, observation_id)
        if r is None or r.tenant_id != tenant_id:
            return None
        return self._to_domain(r)

    async def mark_processed(self, tenant_id: str, observation_id: str, *, status: str) -> None:
        await self.s.execute(
            update(ObservationRow)
            .where(
                ObservationRow.observation_id == observation_id,
                ObservationRow.tenant_id == tenant_id,
            )
            .values(processed_at=func.now(), status=status)
        )

    async def list_unprocessed(
        self, *, older_than: datetime, limit: int = 500
    ) -> list[Observation]:
        rows = (
            (
                await self.s.execute(
                    select(ObservationRow)
                    .where(
                        ObservationRow.processed_at.is_(None),
                        ObservationRow.created_at <= older_than,
                    )
                    .order_by(ObservationRow.created_at)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return [self._to_domain(r) for r in rows]


class SqlRevisionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def bump(self, tenant_id: str, kind: RevisionKind, object_id: str = "") -> int:
        stmt = (
            pg_insert(RevisionRow)
            .values(tenant_id=tenant_id, kind=kind.value, object_id=object_id, value=1)
            .on_conflict_do_update(
                index_elements=[RevisionRow.tenant_id, RevisionRow.kind, RevisionRow.object_id],
                set_={"value": RevisionRow.value + 1, "updated_at": func.now()},
            )
            .returning(RevisionRow.value)
        )
        return int((await self.s.execute(stmt)).scalar_one())

    async def get_many(
        self, tenant_id: str, keys: Sequence[tuple[RevisionKind, str]]
    ) -> dict[str, int]:
        if not keys:
            return {}
        rows = (
            await self.s.execute(
                select(RevisionRow.kind, RevisionRow.object_id, RevisionRow.value).where(
                    RevisionRow.tenant_id == tenant_id,
                    RevisionRow.kind.in_({k.value for k, _ in keys}),
                    RevisionRow.object_id.in_({o for _, o in keys}),
                )
            )
        ).all()
        wanted = {(k.value, o) for k, o in keys}
        out = {f"{k}:{o}": int(v) for k, o, v in rows if (k, o) in wanted}
        for k, o in keys:
            out.setdefault(f"{k.value}:{o}", 0)
        return out


class SqlIdempotencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def get(self, tenant_id: str, key: str) -> IdempotencyRecord | None:
        r = await self.s.get(IdempotencyRow, (tenant_id, key))
        if r is None:
            return None
        return IdempotencyRecord(
            tenant_id=r.tenant_id,
            key=r.key,
            request_hash=r.request_hash,
            response_status=r.response_status,
            response_body=r.response_body,
            expires_at=r.expires_at,
        )

    async def reserve(self, record: IdempotencyRecord) -> bool:
        # RETURNING makes the outcome explicit: psycopg3 reports rowcount=-1 for
        # INSERT ... ON CONFLICT DO NOTHING, so rowcount cannot be trusted here.
        result = await self.s.execute(
            pg_insert(IdempotencyRow)
            .values(
                tenant_id=record.tenant_id,
                key=record.key,
                request_hash=record.request_hash,
                response_status=record.response_status,
                response_body=record.response_body,
                expires_at=record.expires_at,
            )
            .on_conflict_do_nothing(index_elements=[IdempotencyRow.tenant_id, IdempotencyRow.key])
            .returning(IdempotencyRow.key)
        )
        return result.scalar_one_or_none() is not None

    async def complete(
        self, tenant_id: str, key: str, *, status: int, body: dict[str, Any]
    ) -> None:
        await self.s.execute(
            update(IdempotencyRow)
            .where(IdempotencyRow.tenant_id == tenant_id, IdempotencyRow.key == key)
            .values(response_status=status, response_body=body)
        )

    async def purge_expired(self, *, now: datetime, limit: int = 5000) -> int:
        ids = (
            await self.s.execute(
                select(IdempotencyRow.tenant_id, IdempotencyRow.key)
                .where(IdempotencyRow.expires_at < now)
                .limit(limit)
            )
        ).all()
        count = 0
        for tenant_id, key in ids:
            row = await self.s.get(IdempotencyRow, (tenant_id, key))
            if row is not None:
                await self.s.delete(row)
                count += 1
        return count


class SqlOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, spec: JobSpec) -> int | None:
        stmt = pg_insert(OutboxRow).values(
            tenant_id=spec.tenant_id,
            task_name=spec.task_name,
            queue=spec.queue.value,
            payload=spec.payload,
            idempotency_key=spec.idempotency_key,
            lock=spec.lock,
            priority=spec.priority,
            schedule_in_seconds=spec.schedule_in_seconds,
        )
        if spec.idempotency_key is not None:
            stmt = stmt.on_conflict_do_nothing(
                index_elements=[OutboxRow.idempotency_key],
                index_where=OutboxRow.idempotency_key.is_not(None),
            )
        result = await self.s.execute(stmt.returning(OutboxRow.outbox_id))
        value = result.scalar_one_or_none()
        return int(value) if value is not None else None

    async def pending(self, *, limit: int = 200, older_than_seconds: int = 0) -> list[OutboxEntry]:
        stmt = (
            select(OutboxRow)
            .where(OutboxRow.dispatched_at.is_(None), OutboxRow.dead.is_(False))
            .order_by(OutboxRow.outbox_id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        if older_than_seconds > 0:
            cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
            stmt = stmt.where(OutboxRow.created_at <= cutoff)
        rows = (await self.s.execute(stmt)).scalars().all()
        return [
            OutboxEntry(
                outbox_id=r.outbox_id,
                spec=JobSpec(
                    task_name=r.task_name,
                    queue=Queue(r.queue),
                    payload=r.payload or {},
                    idempotency_key=r.idempotency_key,
                    lock=r.lock,
                    priority=r.priority,
                    schedule_in_seconds=r.schedule_in_seconds,
                    tenant_id=r.tenant_id,
                ),
                attempts=r.attempts,
                dispatched_at=r.dispatched_at,
                job_id=r.job_id,
            )
            for r in rows
        ]

    async def mark_dispatched(self, outbox_id: int, *, job_id: str) -> None:
        await self.s.execute(
            update(OutboxRow)
            .where(OutboxRow.outbox_id == outbox_id)
            .values(dispatched_at=func.now(), job_id=job_id, attempts=OutboxRow.attempts + 1)
        )

    async def mark_failed(self, outbox_id: int, *, error: str, dead: bool) -> None:
        await self.s.execute(
            update(OutboxRow)
            .where(OutboxRow.outbox_id == outbox_id)
            .values(attempts=OutboxRow.attempts + 1, last_error=error[:2000], dead=dead)
        )


class SqlArchiveRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    @staticmethod
    def _to_domain(r: ArchiveSegmentRow) -> ArchiveSegment:
        return ArchiveSegment(
            segment_id=r.segment_id,
            tenant_id=r.tenant_id,
            kind=r.kind,
            thread_id=r.thread_id,
            document_id=r.document_id,
            bucket=r.bucket,
            key=r.key,
            generation=r.generation,
            size_bytes=r.size_bytes,
            raw_bytes=r.raw_bytes,
            checksum_sha256=r.checksum_sha256,
            message_count=r.message_count,
            first_sequence=r.first_sequence,
            last_sequence=r.last_sequence,
            first_at=r.first_at,
            last_at=r.last_at,
            status=r.status,
            manifest=r.manifest or {},
            verified_at=r.verified_at,
            last_error=r.last_error,
        )

    async def add(self, segment: ArchiveSegment) -> None:
        self.s.add(
            ArchiveSegmentRow(
                segment_id=segment.segment_id,
                tenant_id=segment.tenant_id,
                kind=segment.kind,
                thread_id=segment.thread_id,
                document_id=segment.document_id,
                bucket=segment.bucket,
                key=segment.key,
                generation=segment.generation,
                size_bytes=segment.size_bytes,
                raw_bytes=segment.raw_bytes,
                checksum_sha256=segment.checksum_sha256,
                message_count=segment.message_count,
                first_sequence=segment.first_sequence,
                last_sequence=segment.last_sequence,
                first_at=segment.first_at,
                last_at=segment.last_at,
                status=segment.status,
                manifest=segment.manifest,
                verified_at=segment.verified_at,
            )
        )
        await self.s.flush()

    async def get(self, segment_id: str) -> ArchiveSegment | None:
        r = await self.s.get(ArchiveSegmentRow, segment_id)
        return self._to_domain(r) if r is not None else None

    async def mark_verified(
        self, segment_id: str, *, generation: str | None, verified_at: datetime
    ) -> None:
        await self.s.execute(
            update(ArchiveSegmentRow)
            .where(ArchiveSegmentRow.segment_id == segment_id)
            .values(
                status="VERIFIED", generation=generation, verified_at=verified_at, last_error=None
            )
        )

    async def mark_failed(self, segment_id: str, *, error: str) -> None:
        await self.s.execute(
            update(ArchiveSegmentRow)
            .where(ArchiveSegmentRow.segment_id == segment_id)
            .values(status="FAILED", last_error=error[:2000])
        )

    async def list_by_status(
        self, status: str, *, older_than: datetime | None = None, limit: int = 200
    ) -> list[ArchiveSegment]:
        stmt = (
            select(ArchiveSegmentRow)
            .where(ArchiveSegmentRow.status == status)
            .order_by(ArchiveSegmentRow.created_at)
            .limit(limit)
        )
        if older_than is not None:
            stmt = stmt.where(ArchiveSegmentRow.created_at <= older_than)
        rows = (await self.s.execute(stmt)).scalars().all()
        return [self._to_domain(r) for r in rows]

    async def list_for_thread(self, tenant_id: str, thread_id: str) -> list[ArchiveSegment]:
        rows = (
            (
                await self.s.execute(
                    select(ArchiveSegmentRow)
                    .where(
                        ArchiveSegmentRow.tenant_id == tenant_id,
                        ArchiveSegmentRow.thread_id == thread_id,
                    )
                    .order_by(ArchiveSegmentRow.first_sequence)
                )
            )
            .scalars()
            .all()
        )
        return [self._to_domain(r) for r in rows]

    async def list_verified(self, *, limit: int = 200, offset: int = 0) -> list[ArchiveSegment]:
        rows = (
            (
                await self.s.execute(
                    select(ArchiveSegmentRow)
                    .where(ArchiveSegmentRow.status == "VERIFIED")
                    .order_by(ArchiveSegmentRow.created_at)
                    .offset(offset)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return [self._to_domain(r) for r in rows]
