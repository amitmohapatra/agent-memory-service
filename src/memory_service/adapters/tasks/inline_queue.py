"""In-process TaskQueue implementations for tests and single-process dev.

``InlineTaskQueue`` runs handlers immediately on enqueue (after the UoW commit, since
dispatch happens via the outbox relay). ``RecordingTaskQueue`` only records jobs so tests
can assert what would have been queued and drain them explicitly.
"""

from __future__ import annotations

from typing import Any

from memory_service.domain.enums import JobStatus
from memory_service.domain.ids import new_id
from memory_service.observability.logging import get_logger
from memory_service.ports.models import ProviderInfo
from memory_service.ports.tasks import JobInfo, JobSpec, Queue, TaskHandler

log = get_logger(__name__)


class RecordingTaskQueue:
    info = ProviderInfo(
        name="memory-queue", license="Apache-2.0", origin="internal", locality="local"
    )

    def __init__(self, *, fail_enqueue: bool = False) -> None:
        self.handlers: dict[str, TaskHandler] = {}
        self.queues: dict[str, Queue] = {}
        self.jobs: dict[str, JobInfo] = {}
        self.payloads: dict[str, JobSpec] = {}
        self.fail_enqueue = fail_enqueue
        self._queueing_locks: set[str] = set()
        self.periodic: dict[str, str] = {}

    def register(self, name: str, queue: Queue, handler: TaskHandler, *, retries: int = 5) -> None:
        self.handlers[name] = handler
        self.queues[name] = queue

    def register_periodic(
        self, name: str, queue: Queue, handler: TaskHandler, *, cron: str
    ) -> None:
        self.handlers[name] = handler
        self.queues[name] = queue
        self.periodic[name] = cron

    async def run_periodic(self, name: str) -> None:
        """Tests/dev: run a periodic task now."""
        await self.handlers[name]({"timestamp": 0})

    async def enqueue(self, spec: JobSpec, *, connection: Any | None = None) -> str:
        if self.fail_enqueue:
            raise ConnectionError("simulated queue outage")
        if spec.task_name not in self.handlers:
            raise KeyError(f"task {spec.task_name!r} is not registered")
        if spec.idempotency_key and spec.idempotency_key in self._queueing_locks:
            return f"dedup:{spec.idempotency_key}"
        job_id = new_id("job")
        if spec.idempotency_key:
            self._queueing_locks.add(spec.idempotency_key)
        self.jobs[job_id] = JobInfo(
            job_id=job_id,
            task_name=spec.task_name,
            queue=spec.queue.value,
            status=JobStatus.PENDING,
        )
        self.payloads[job_id] = spec
        return job_id

    async def get(self, job_id: str) -> JobInfo | None:
        return self.jobs.get(job_id)

    async def drain(self, *, max_rounds: int = 10) -> int:
        """Run all pending jobs (and jobs they enqueue) until none remain."""
        ran = 0
        for _ in range(max_rounds):
            pending = [jid for jid, j in self.jobs.items() if j.status is JobStatus.PENDING]
            if not pending:
                break
            for job_id in pending:
                await self._run(job_id)
                ran += 1
        return ran

    async def _run(self, job_id: str) -> None:
        info = self.jobs[job_id]
        spec = self.payloads[job_id]
        self.jobs[job_id] = info.model_copy(
            update={"status": JobStatus.RUNNING, "attempts": info.attempts + 1}
        )
        if spec.idempotency_key:
            self._queueing_locks.discard(spec.idempotency_key)
        try:
            await self.handlers[spec.task_name](spec.payload)
        except Exception as exc:
            self.jobs[job_id] = self.jobs[job_id].model_copy(
                update={"status": JobStatus.FAILED, "last_error": f"{type(exc).__name__}: {exc}"}
            )
            log.warning("inline_job.failed", task=spec.task_name, error=str(exc))
            return
        self.jobs[job_id] = self.jobs[job_id].model_copy(update={"status": JobStatus.SUCCEEDED})

    async def run_worker(self, queues: list[Queue] | None = None, *, concurrency: int = 4) -> None:
        await self.drain()

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class InlineTaskQueue(RecordingTaskQueue):
    """Runs each job as soon as it is enqueued (i.e. right after the outbox relay dispatches)."""

    info = ProviderInfo(
        name="inline-queue", license="Apache-2.0", origin="internal", locality="local"
    )

    async def enqueue(self, spec: JobSpec, *, connection: Any | None = None) -> str:
        job_id = await super().enqueue(spec, connection=connection)
        if not job_id.startswith("dedup:"):
            await self._run(job_id)
        return job_id
