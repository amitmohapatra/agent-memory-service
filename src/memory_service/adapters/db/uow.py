"""SQLAlchemy Unit of Work with transactional outbox dispatch."""

from __future__ import annotations

from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_service.adapters.db.repositories import (
    SqlAgentRunRepository,
    SqlArchiveRepository,
    SqlIdempotencyRepository,
    SqlMessageRepository,
    SqlObservationRepository,
    SqlOutboxRepository,
    SqlRevisionRepository,
    SqlSessionRepository,
    SqlThreadRepository,
    SqlTurnRepository,
)
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.tasks import JobSpec

log = get_logger(__name__)


class OutboxRelay:
    """Dispatches committed outbox rows to the task queue and marks them dispatched.

    ``dispatch_ids`` is the fast path used right after commit. ``sweep`` is the periodic
    repair that re-dispatches rows whose fast path never ran (crash between COMMIT and defer).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession], task_queue) -> None:  # type: ignore[no-untyped-def]
        self.session_factory = session_factory
        self.task_queue = task_queue

    async def dispatch_ids(self, outbox_ids: list[int]) -> list[str]:
        if not outbox_ids or self.task_queue is None:
            return []
        job_ids: list[str] = []
        async with self.session_factory() as session, session.begin():
            repo = SqlOutboxRepository(session)
            entries = [
                e
                for e in await repo.pending(limit=len(outbox_ids) + 50)
                if e.outbox_id in set(outbox_ids)
            ]
            for entry in entries:
                job_id = await self._dispatch(repo, entry)
                if job_id:
                    job_ids.append(job_id)
        return job_ids

    async def sweep(self, *, older_than_seconds: int = 30, limit: int = 200) -> int:
        if self.task_queue is None:
            return 0
        count = 0
        async with self.session_factory() as session, session.begin():
            repo = SqlOutboxRepository(session)
            for entry in await repo.pending(limit=limit, older_than_seconds=older_than_seconds):
                if await self._dispatch(repo, entry):
                    count += 1
        return count

    async def _dispatch(self, repo: SqlOutboxRepository, entry) -> str | None:  # type: ignore[no-untyped-def]
        try:
            job_id = await self.task_queue.enqueue(entry.spec)
        except Exception as exc:
            dead = entry.attempts + 1 >= 20
            await repo.mark_failed(entry.outbox_id, error=f"{type(exc).__name__}: {exc}", dead=dead)
            log.warning(
                "outbox.dispatch_failed", outbox_id=entry.outbox_id, error=str(exc), dead=dead
            )
            return None
        await repo.mark_dispatched(entry.outbox_id, job_id=job_id)
        return job_id


class SqlUnitOfWork:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], relay: OutboxRelay | None
    ) -> None:
        self._session_factory = session_factory
        self._relay = relay
        self._session: AsyncSession | None = None
        self._pending_outbox: list[int] = []
        self._committed = False
        self._dispatched: list[str] = []

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        await self._session.__aenter__()
        await self._session.begin()
        s = self._session
        self.threads = SqlThreadRepository(s)
        self.sessions = SqlSessionRepository(s)
        self.turns = SqlTurnRepository(s)
        self.messages = SqlMessageRepository(s)
        self.agent_runs = SqlAgentRunRepository(s)
        self.observations = SqlObservationRepository(s)
        self.revisions = SqlRevisionRepository(s)
        self.idempotency = SqlIdempotencyRepository(s)
        self.outbox = SqlOutboxRepository(s)
        self.archive = SqlArchiveRepository(s)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._session is not None
        try:
            if exc_type is not None or not self._committed:
                await self.rollback()
        finally:
            await self._session.__aexit__(exc_type, exc, tb)
            self._session = None

    @property
    def session(self) -> AsyncSession:
        assert self._session is not None, "UnitOfWork used outside 'async with'"
        return self._session

    async def enqueue(self, spec: JobSpec) -> int | None:
        outbox_id = await self.outbox.add(spec)
        if outbox_id is not None:
            self._pending_outbox.append(outbox_id)
        return outbox_id

    async def commit(self) -> None:
        assert self._session is not None
        with span("db.commit"), stage_seconds.labels("db.commit").time():
            await self._session.commit()
        self._committed = True
        pending, self._pending_outbox = self._pending_outbox, []
        if pending and self._relay is not None:
            try:
                self._dispatched = await self._relay.dispatch_ids(pending)
            except Exception as exc:
                log.warning("outbox.post_commit_dispatch_failed", error=str(exc))
                self._dispatched = []

    async def rollback(self) -> None:
        if self._session is not None and self._session.in_transaction():
            await self._session.rollback()
        self._pending_outbox = []

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def dispatched_job_ids(self) -> list[str]:
        return list(self._dispatched)


class SqlUnitOfWorkFactory:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], relay: OutboxRelay | None
    ) -> None:
        self._session_factory = session_factory
        self._relay = relay

    def __call__(self) -> SqlUnitOfWork:
        return SqlUnitOfWork(self._session_factory, self._relay)
