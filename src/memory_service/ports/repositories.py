"""Repository ports for canonical state. Implemented by the PostgreSQL adapter.

Repositories speak domain models only. Identity/scope queries always hit indexed columns
(tenant_id + id / sequence); no repository ever returns rows outside the given tenant.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from memory_service.domain.conversation import AgentRun, Message, Session, Thread, Turn
from memory_service.domain.observation import Observation
from memory_service.domain.revisions import RevisionKind
from memory_service.ports.tasks import JobSpec


class IdempotencyRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    key: str
    request_hash: str
    response_status: int | None = None
    response_body: dict[str, Any] | None = None
    expires_at: datetime


class OutboxEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    outbox_id: int
    spec: JobSpec
    attempts: int = 0
    dispatched_at: datetime | None = None
    job_id: str | None = None


@runtime_checkable
class ThreadRepository(Protocol):
    async def add(self, thread: Thread) -> None: ...
    async def get(self, tenant_id: str, thread_id: str) -> Thread | None: ...
    async def touch(self, tenant_id: str, thread_id: str, *, title: str | None = None) -> int:
        """Bump the thread revision/updated_at; returns new revision."""
        ...

    async def list_for_user(
        self, tenant_id: str, user_id: str, *, limit: int = 50, before: datetime | None = None
    ) -> list[Thread]: ...
    async def soft_delete(self, tenant_id: str, thread_id: str) -> bool: ...


@runtime_checkable
class SessionRepository(Protocol):
    async def add(self, session: Session) -> None: ...
    async def get(self, tenant_id: str, session_id: str) -> Session | None: ...
    async def end(self, tenant_id: str, session_id: str) -> None: ...


@runtime_checkable
class TurnRepository(Protocol):
    async def add(self, turn: Turn) -> None: ...
    async def get(self, tenant_id: str, turn_id: str) -> Turn | None: ...
    async def next_sequence(self, tenant_id: str, thread_id: str) -> int:
        """Next 1-based turn sequence for the thread (row-locked to avoid gaps/dupes)."""
        ...

    async def complete(self, tenant_id: str, turn_id: str) -> None: ...
    async def link_run(self, turn_id: str, agent_run_id: str) -> None: ...


@runtime_checkable
class MessageRepository(Protocol):
    async def add(self, message: Message) -> None: ...
    async def get(self, tenant_id: str, message_id: str) -> Message | None: ...
    async def next_sequence(self, tenant_id: str, thread_id: str) -> int: ...
    async def list_thread(
        self,
        tenant_id: str,
        thread_id: str,
        *,
        limit: int = 50,
        before_sequence: int | None = None,
        include_internal: bool = False,
    ) -> list[Message]: ...
    async def find_by_source(
        self, tenant_id: str, source_system: str, source_message_id: str
    ) -> Message | None: ...
    async def list_staged(
        self, *, older_than: datetime | None = None, limit: int = 1000, tenant_id: str | None = None
    ) -> list[Message]: ...
    async def mark_archived(
        self, message_ids: Sequence[str], *, segment_id: str, archived_at: datetime
    ) -> int: ...
    async def purge_payloads(
        self, *, archived_before: datetime, min_bytes: int, limit: int = 1000
    ) -> int: ...
    async def add_version(self, message: Message) -> None: ...


@runtime_checkable
class AgentRunRepository(Protocol):
    async def add(self, run: AgentRun) -> None: ...
    async def get(self, tenant_id: str, agent_run_id: str) -> AgentRun | None: ...
    async def complete(self, tenant_id: str, agent_run_id: str, *, status: str) -> None: ...
    async def list_for_turn(self, tenant_id: str, turn_id: str) -> list[AgentRun]: ...


@runtime_checkable
class ObservationRepository(Protocol):
    async def add(self, observation: Observation) -> None: ...
    async def get(self, tenant_id: str, observation_id: str) -> Observation | None: ...
    async def mark_processed(self, tenant_id: str, observation_id: str, *, status: str) -> None: ...
    async def list_unprocessed(
        self, *, older_than: datetime, limit: int = 500
    ) -> list[Observation]: ...


@runtime_checkable
class RevisionRepository(Protocol):
    async def bump(self, tenant_id: str, kind: RevisionKind, object_id: str = "") -> int: ...
    async def get_many(
        self, tenant_id: str, keys: Sequence[tuple[RevisionKind, str]]
    ) -> dict[str, int]: ...


@runtime_checkable
class IdempotencyRepository(Protocol):
    async def get(self, tenant_id: str, key: str) -> IdempotencyRecord | None: ...
    async def reserve(self, record: IdempotencyRecord) -> bool:
        """Insert if absent. Returns False when the key already exists."""
        ...

    async def complete(
        self, tenant_id: str, key: str, *, status: int, body: dict[str, Any]
    ) -> None: ...
    async def purge_expired(self, *, now: datetime, limit: int = 5000) -> int: ...


@runtime_checkable
class OutboxRepository(Protocol):
    async def add(self, spec: JobSpec) -> int | None:
        """Insert an outbox row. Returns None when ``idempotency_key`` already exists."""
        ...

    async def pending(
        self, *, limit: int = 200, older_than_seconds: int = 0
    ) -> list[OutboxEntry]: ...
    async def mark_dispatched(self, outbox_id: int, *, job_id: str) -> None: ...
    async def mark_failed(self, outbox_id: int, *, error: str, dead: bool) -> None: ...
