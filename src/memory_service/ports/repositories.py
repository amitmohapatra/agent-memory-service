"""Repository ports for canonical state. Implemented by the PostgreSQL adapter.

Repositories speak domain models only. Identity/scope queries always hit indexed columns
(tenant_id + id / sequence); no repository ever returns rows outside the given tenant.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.conversation import AgentRun, Message, Session, Thread, Turn
from memory_service.domain.documents import (
    Chunk,
    ContextEdge,
    Document,
    DocumentNode,
    DocumentVersion,
)
from memory_service.domain.enums import ContextGraphEdge
from memory_service.domain.memory import CanonicalMemory
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
        self,
        *,
        older_than: datetime | None = None,
        limit: int = 1000,
        tenant_id: str | None = None,
        thread_id: str | None = None,
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


class ArchiveSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment_id: str
    tenant_id: str
    kind: str = "chat"
    thread_id: str | None = None
    document_id: str | None = None
    bucket: str
    key: str
    generation: str | None = None
    size_bytes: int
    raw_bytes: int = 0
    checksum_sha256: str
    message_count: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    first_at: datetime | None = None
    last_at: datetime | None = None
    status: str = "UPLOADING"
    manifest: dict[str, Any] = Field(default_factory=dict)
    verified_at: datetime | None = None
    last_error: str | None = None


@runtime_checkable
class ArchiveRepository(Protocol):
    async def add(self, segment: ArchiveSegment) -> None: ...
    async def get(self, segment_id: str) -> ArchiveSegment | None: ...
    async def mark_verified(
        self, segment_id: str, *, generation: str | None, verified_at: datetime
    ) -> None: ...
    async def mark_failed(self, segment_id: str, *, error: str) -> None: ...
    async def list_by_status(
        self, status: str, *, older_than: datetime | None = None, limit: int = 200
    ) -> list[ArchiveSegment]: ...
    async def list_for_thread(self, tenant_id: str, thread_id: str) -> list[ArchiveSegment]: ...
    async def list_verified(self, *, limit: int = 200, offset: int = 0) -> list[ArchiveSegment]: ...


@runtime_checkable
class DocumentRepository(Protocol):
    async def add(
        self, document: Document, *, visibility_keys: Sequence[str], message_id: str | None = None
    ) -> None: ...
    async def get(self, tenant_id: str, document_id: str) -> Document | None: ...
    async def find_by_checksum(self, tenant_id: str, checksum: str) -> Document | None: ...
    async def set_status(
        self,
        tenant_id: str,
        document_id: str,
        *,
        status: str,
        current_version_id: str | None = None,
        error: str | None = None,
    ) -> None: ...
    async def stage_bytes(
        self, tenant_id: str, document_id: str, data: bytes, *, checksum: str
    ) -> None: ...
    async def staged_bytes(self, tenant_id: str, document_id: str) -> bytes | None: ...
    async def purge_staged_bytes(self, tenant_id: str, document_id: str) -> None: ...
    async def mark_archived(
        self, tenant_id: str, document_id: str, *, segment_id: str, archived_at: datetime
    ) -> None: ...
    async def list_staged_archive(
        self, *, older_than: datetime | None = None, limit: int = 200
    ) -> list[Document]: ...
    async def add_version(self, version: DocumentVersion) -> None: ...
    async def get_version(
        self, tenant_id: str, document_version_id: str
    ) -> DocumentVersion | None: ...
    async def add_nodes(self, nodes: Sequence[DocumentNode]) -> None: ...
    async def add_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    async def add_edges(self, edges: Sequence[ContextEdge]) -> None: ...
    async def replace_version_content(self, tenant_id: str, document_id: str) -> None:
        """Delete nodes/chunks/edges of previous versions (re-parse)."""
        ...

    async def list_nodes(
        self, tenant_id: str, document_id: str, *, version_id: str | None = None
    ) -> list[DocumentNode]: ...
    async def get_nodes(self, tenant_id: str, node_ids: Sequence[str]) -> list[DocumentNode]: ...
    async def list_chunks(
        self, tenant_id: str, document_id: str, *, unindexed_only: bool = False, limit: int = 5000
    ) -> list[Chunk]: ...
    async def get_chunks(self, tenant_id: str, chunk_ids: Sequence[str]) -> list[Chunk]: ...
    async def chunks_for_nodes(self, tenant_id: str, node_ids: Sequence[str]) -> list[Chunk]:
        """Chunks under the given nodes, ordered by node then ordinal."""
        ...

    async def set_node_summaries(self, tenant_id: str, summaries: dict[str, str]) -> None:
        """Store hierarchical summaries in ``system_metadata['summary']`` of each node."""
        ...

    async def node_summaries(self, tenant_id: str, node_ids: Sequence[str]) -> dict[str, str]: ...
    async def mark_chunks_indexed(
        self, chunk_ids: Sequence[str], *, fingerprint: str, indexed_at: datetime
    ) -> int: ...
    async def edges_from(
        self,
        tenant_id: str,
        source_ids: Sequence[str],
        *,
        kinds: Sequence[ContextGraphEdge] | None = None,
    ) -> list[ContextEdge]: ...
    async def edges_to(
        self,
        tenant_id: str,
        target_ids: Sequence[str],
        *,
        kinds: Sequence[ContextGraphEdge] | None = None,
    ) -> list[ContextEdge]: ...
    async def visibility_keys(self, tenant_id: str, document_id: str) -> list[str]: ...


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


@runtime_checkable
class MemoryRepository(Protocol):
    """Canonical memories (M7). Rows are never physically deleted by application code:
    ``forget`` soft-deletes and later purge is an operator task."""

    async def add(self, memory: CanonicalMemory, *, visibility_keys: Sequence[str]) -> None: ...
    async def get(self, tenant_id: str, memory_id: str) -> CanonicalMemory | None: ...
    async def get_many(
        self, tenant_id: str, memory_ids: Sequence[str]
    ) -> list[CanonicalMemory]: ...
    async def visibility_keys(self, tenant_id: str, memory_id: str) -> list[str]: ...
    async def update(self, memory: CanonicalMemory) -> None:
        """Persist content/temporal/evidence/counter changes; bumps ``revision``."""
        ...

    async def candidates(
        self,
        tenant_id: str,
        *,
        scope_key: str,
        normalized_hash: str | None = None,
        subject: str | None = None,
        limit: int = 20,
    ) -> list[CanonicalMemory]:
        """Existing CURRENT memories that could be duplicates of a new candidate: same scope
        and (same hash OR same subject OR most recent)."""
        ...

    async def list_scope(
        self,
        tenant_id: str,
        *,
        scope_keys: Sequence[str],
        memory_types: Sequence[str] | None = None,
        current_only: bool = True,
        limit: int = 200,
    ) -> list[CanonicalMemory]: ...
    async def forget(self, tenant_id: str, memory_id: str) -> bool: ...
    async def list_unindexed(
        self, tenant_id: str, *, limit: int = 500
    ) -> list[CanonicalMemory]: ...
    async def mark_indexed(
        self, memory_ids: Sequence[str], *, fingerprint: str, indexed_at: datetime
    ) -> None: ...
    async def expire_due(self, *, now: datetime, limit: int = 500) -> list[tuple[str, str]]:
        """Mark memories past ``expires_at`` EXPIRED; returns (tenant_id, memory_id) pairs."""
        ...
