"""Unit of Work port.

One transaction spans: source rows + observation + revisions + outbox job rows. The point of
acknowledgement is ``commit()``. After a successful commit, queued outbox rows are dispatched
to the task queue (best effort; the relay sweep guarantees eventual dispatch).
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from memory_service.ports.repositories import (
    AgentRunRepository,
    ArchiveRepository,
    DocumentRepository,
    IdempotencyRepository,
    MemoryRepository,
    MessageRepository,
    ObservationRepository,
    OutboxRepository,
    RevisionRepository,
    SessionRepository,
    ThreadRepository,
    TurnRepository,
)
from memory_service.ports.tasks import JobSpec


@runtime_checkable
class UnitOfWork(Protocol):
    threads: ThreadRepository
    sessions: SessionRepository
    turns: TurnRepository
    messages: MessageRepository
    agent_runs: AgentRunRepository
    observations: ObservationRepository
    revisions: RevisionRepository
    idempotency: IdempotencyRepository
    outbox: OutboxRepository
    archive: ArchiveRepository
    documents: DocumentRepository
    memories: MemoryRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    async def enqueue(self, spec: JobSpec) -> int | None:
        """Add a job to the transactional outbox. Dispatched after commit."""
        ...

    async def serialize(self, *keys: str) -> None:
        """Take a transaction-scoped exclusive lock on a logical key (e.g. a thread that may
        not exist yet) so concurrent writers to the same key run one after another. Released
        at commit/rollback; a no-op for stores without locks."""
        ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    @property
    def committed(self) -> bool: ...

    @property
    def dispatched_job_ids(self) -> list[str]:
        """Task-queue job ids dispatched after the last commit (may be empty if relay lagged)."""
        ...


@runtime_checkable
class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...
