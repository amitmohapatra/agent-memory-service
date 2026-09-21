"""TaskQueue port. Procrastinate (PostgreSQL-backed) by default; Kafka etc. later.

The critical property: ``enqueue`` participates in the caller's PostgreSQL transaction
when the backend supports it, so 'source row + job' commit atomically (transactional
outbox). Queues are bulkheads: a 500-page PDF must never starve chat memory.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import JobStatus


class Queue(StrEnum):
    CHAT_FAST = "chat-fast"
    MEMORY_EXTRACT = "memory-extract"
    EMBEDDING = "embedding"
    DOCUMENT_PARSE = "document-parse"
    GRAPH = "graph"
    SUMMARY = "summary"
    ARCHIVE = "archive"
    EVALUATION = "evaluation"
    IMPORT = "import"
    RECONCILE = "reconcile"


# interactive > indexing > summaries > eval/import
QUEUE_PRIORITY: dict[Queue, int] = {
    Queue.CHAT_FAST: 100,
    Queue.MEMORY_EXTRACT: 90,
    Queue.EMBEDDING: 70,
    Queue.DOCUMENT_PARSE: 60,
    Queue.GRAPH: 60,
    Queue.ARCHIVE: 50,
    Queue.RECONCILE: 40,
    Queue.SUMMARY: 30,
    Queue.EVALUATION: 10,
    Queue.IMPORT: 5,
}


class JobSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_name: str
    queue: Queue
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(
        default=None, description="deduplicates queued jobs (e.g. one archive job per segment)"
    )
    lock: str | None = Field(default=None, description="serialize jobs sharing this lock")
    priority: int | None = None
    schedule_in_seconds: int | None = None
    tenant_id: str | None = None


class JobInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    task_name: str
    queue: str
    status: JobStatus
    attempts: int = 0
    last_error: str | None = None
    scheduled_at: str | None = None


TaskHandler = Callable[..., Awaitable[Any]]


@runtime_checkable
class TaskQueue(Protocol):
    async def enqueue(self, spec: JobSpec, *, connection: Any | None = None) -> str:
        """Enqueue and return the job id. ``connection`` binds the job to a DB transaction."""
        ...

    async def get(self, job_id: str) -> JobInfo | None: ...

    def register(self, name: str, queue: Queue, handler: TaskHandler, *, retries: int = 5) -> None:
        """Register a task handler. Must be called before workers start."""
        ...

    async def run_worker(
        self, queues: list[Queue] | None = None, *, concurrency: int = 4
    ) -> None: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
