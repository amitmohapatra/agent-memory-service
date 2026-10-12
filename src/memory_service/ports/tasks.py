"""TaskQueue port. Procrastinate (PostgreSQL-backed) by default; Kafka etc. later.

The critical property: ``enqueue`` participates in the caller's PostgreSQL transaction
when the backend supports it, so 'source row + job' commit atomically (transactional
outbox). Queues are bulkheads: a 500-page PDF must never starve chat memory.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
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


#: The queues whose jobs make a write findable (memories, chunks, tools): the worker runs
#: them in a pool of their own (``TaskQueue.run_pools``). In one pool, a slow model's jobs -
#: observation extraction, graph enrichment, summaries - held every slot and indexing waited
#: behind them for minutes, so a write was not found when it was asked for.
INDEXING_QUEUES: tuple[Queue, ...] = (Queue.EMBEDDING,)
#: Graph enrichment runs in a pool of its own too. Its backlog grows with every write - an
#: import queues thousands of jobs - and outranks parsing, archiving, reconciliation and
#: summaries (``QUEUE_PRIORITY``), so in the shared pool a backlog held every slot until it
#: drained: a document's parse, a run's feedback and the skill learned from it waited behind
#: it, and on a slow model each enrichment held its slot for minutes.
GRAPH_QUEUES: tuple[Queue, ...] = (Queue.GRAPH,)


#: The interactive queue: besides its place in the shared pool (at the top of
#: ``QUEUE_PRIORITY``), it has a slot of its own that nothing else can take. On a worker
#: capped at 2 CPUs the shared pool has one slot, and a 500-page parse held it for minutes
#: while every remember waited.
CHAT_QUEUES: tuple[Queue, ...] = (Queue.CHAT_FAST,)
#: How many pools (Procrastinate workers, each with a LISTEN connection) the worker runs
WORKER_POOL_COUNT: int = 4


def worker_pools(
    concurrency: int, indexing: int, *, llm: bool = False
) -> list[tuple[list[Queue], int]]:
    """The worker's pools, sized so the slots whose jobs compute stay about one per CPU
    (``concurrency``): every job of one process shares its CPUs, its GIL and its event loop,
    and each model serves one call at a time.

    * chat: ``concurrency // 2`` slots of its own, which a long parse cannot take, and the
      first claim on the shared pool's (``QUEUE_PRIORITY``). A remember's job is what a
      write waits for; on one pool of ``concurrency`` slots chat had all of them first.
    * indexing: up to ``indexing`` slots and at most a quarter, so a write turns findable
      without a slow model's jobs in front of it.
    * graph: enrichment is a job of its own. Without a model (``llm``) it is CPU work on
      the event loop, and gets one slot; with one it waits on the model, and gets
      ``concurrency // 2`` slots (at least 2) on top of the CPUs.
    * shared: every other queue, on what is left, and at least one slot.

    Every pool has at least one slot, or its queues would never run: below 4 CPUs that is 4
    slots. The bulkhead wins there over the CPU count; the extra slots wait their turn on a
    model rather than multiply its threads."""
    chat = max(1, concurrency // 2)
    embedding = min(indexing, max(1, concurrency // 4))
    graph, graph_computes = (max(2, concurrency // 2), 0) if llm else (1, 1)
    shared = max(1, concurrency - chat - embedding - graph_computes)
    own = (*INDEXING_QUEUES, *GRAPH_QUEUES)
    rest = [q for q in Queue if q not in own]
    return [
        (list(INDEXING_QUEUES), embedding),
        (list(GRAPH_QUEUES), graph),
        (list(CHAT_QUEUES), chat),
        (rest, shared),
    ]


#: Called with a job's payload and the error it last failed with, once it will not be retried.
GiveUpHandler = Callable[[dict[str, Any], str], Awaitable[None]]


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

    def register(
        self,
        name: str,
        queue: Queue,
        handler: TaskHandler,
        *,
        retries: int = 5,
        on_give_up: GiveUpHandler | None = None,
    ) -> None:
        """Register a task handler. Must be called before workers start. ``on_give_up`` is
        awaited with the payload and the error once the job has failed for the last time
        (its retries spent, or its last attempt past the job timeout): what the job was to
        finish must not stay half-done with nothing saying so."""
        ...

    async def run_worker(
        self,
        queues: list[Queue] | None = None,
        *,
        concurrency: int = 4,
        install_signal_handlers: bool = False,
        shutdown_grace_seconds: float | None = None,
    ) -> None:
        """Run jobs; with ``install_signal_handlers``, SIGTERM/SIGINT stop it gracefully,
        waiting up to ``shutdown_grace_seconds`` for running jobs before releasing them."""
        ...

    async def run_pools(
        self,
        pools: Sequence[tuple[list[Queue], int]],
        *,
        install_signal_handlers: bool = False,
        shutdown_grace_seconds: float | None = None,
        db_outage_seconds: float | None = None,
    ) -> None:
        """Run one worker per pool (its queues, its slots), so one pool's long jobs never
        hold another's; SIGTERM/SIGINT stop them all as ``run_worker`` stops. The queue's
        database out of reach is ridden out for ``db_outage_seconds``; a worker that ends
        without being asked to raises, so the process exits non-zero."""
        ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
