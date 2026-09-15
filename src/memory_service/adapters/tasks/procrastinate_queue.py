"""Procrastinate-backed TaskQueue (PostgreSQL). MIT-licensed; async; retries, locks, queues."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import procrastinate
from procrastinate import exceptions as pexc

from memory_service.domain.enums import JobStatus
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import jobs_total
from memory_service.observability.tracing import span
from memory_service.ports.models import ProviderInfo
from memory_service.ports.tasks import QUEUE_PRIORITY, JobInfo, JobSpec, Queue, TaskHandler

log = get_logger(__name__)

INFO = ProviderInfo(
    name="procrastinate",
    version=procrastinate.__version__,
    license="MIT",
    origin="procrastinate-org/procrastinate",
    locality="local",
)

_STATUS_MAP = {
    "todo": JobStatus.PENDING,
    "doing": JobStatus.RUNNING,
    "succeeded": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "aborting": JobStatus.CANCELLED,
    "aborted": JobStatus.CANCELLED,
}


def _retry_strategy(retries: int) -> procrastinate.RetryStrategy | bool:
    """``retries`` additional attempts after the first; 0 disables retries entirely."""
    if retries <= 0:
        return False
    return procrastinate.RetryStrategy(max_attempts=retries, wait=1, exponential_wait=2)


class ProcrastinateTaskQueue:
    """Wraps a ``procrastinate.App``. Handlers are registered before the worker starts.

    Each handler receives ``(payload: dict)`` and may raise to trigger a retry.
    """

    info = INFO

    def __init__(
        self, dsn: str, *, default_retries: int = 5, job_timeout_seconds: int = 600
    ) -> None:
        self.app = procrastinate.App(
            connector=procrastinate.PsycopgConnector(
                conninfo=dsn,
                json_dumps=lambda v: json.dumps(v, default=str),
                json_loads=json.loads,
            )
        )
        self.default_retries = default_retries
        self.job_timeout_seconds = job_timeout_seconds
        self._handlers: dict[str, TaskHandler] = {}
        self._opened = False

    # -- lifecycle -----------------------------------------------------------
    async def open(self) -> None:
        if not self._opened:
            await self.app.open_async()
            self._opened = True

    async def close(self) -> None:
        if self._opened:
            await self.app.close_async()
            self._opened = False

    async def ensure_schema(self) -> None:
        """Apply Procrastinate's schema once (idempotent)."""
        await self.open()
        async with self.app.connector.pool.connection() as conn:  # type: ignore[attr-defined]
            row = await (
                await conn.execute("SELECT to_regclass('public.procrastinate_jobs') IS NOT NULL")
            ).fetchone()
        if row and row[0]:
            return
        await self.app.schema_manager.apply_schema_async()

    async def ping(self) -> bool:
        try:
            await self.open()
            return await self.app.check_connection_async()
        except Exception:
            return False

    # -- registration ---------------------------------------------------------
    def register(
        self, name: str, queue: Queue, handler: TaskHandler, *, retries: int | None = None
    ) -> None:
        if name in self._handlers:
            return
        self._handlers[name] = handler
        timeout = self.job_timeout_seconds

        async def _run(context: procrastinate.JobContext, **payload: Any) -> Any:
            attempt = context.job.attempts if context.job else 0
            with span("job.run", task=name, queue=queue.value, attempt=attempt):
                try:
                    result = await asyncio.wait_for(handler(payload), timeout=timeout)
                except Exception:
                    jobs_total.labels(name, "failed").inc()
                    raise
                jobs_total.labels(name, "succeeded").inc()
                return result

        self.app.task(
            name=name,
            queue=queue.value,
            priority=QUEUE_PRIORITY[queue],
            retry=_retry_strategy(retries if retries is not None else self.default_retries),
            pass_context=True,
        )(_run)

    def register_periodic(
        self, name: str, queue: Queue, handler: TaskHandler, *, cron: str
    ) -> None:
        if name in self._handlers:
            return
        self._handlers[name] = handler

        async def _run(context: procrastinate.JobContext, timestamp: int) -> Any:
            with span("job.periodic", task=name):
                return await handler({"timestamp": timestamp})

        self.app.periodic(cron=cron)(
            self.app.task(
                name=name, queue=queue.value, priority=QUEUE_PRIORITY[queue], pass_context=True
            )(_run)
        )

    def registered(self) -> list[str]:
        return sorted(self._handlers)

    # -- enqueue / status -----------------------------------------------------
    async def enqueue(self, spec: JobSpec, *, connection: Any | None = None) -> str:
        await self.open()
        task = self.app.tasks.get(spec.task_name)
        if task is None:
            raise KeyError(f"task {spec.task_name!r} is not registered")
        options: dict[str, Any] = {"queue": spec.queue.value}
        if spec.lock:
            options["lock"] = spec.lock
        if spec.idempotency_key:
            options["queueing_lock"] = spec.idempotency_key
        if spec.priority is not None:
            options["priority"] = spec.priority
        if spec.schedule_in_seconds:
            options["schedule_in"] = {"seconds": spec.schedule_in_seconds}
        try:
            job_id = await task.configure(**options).defer_async(**spec.payload)
        except pexc.AlreadyEnqueued:
            # A job with this queueing lock is already waiting: deduplicated by design.
            return f"dedup:{spec.idempotency_key}"
        return str(job_id)

    async def get(self, job_id: str) -> JobInfo | None:
        if job_id.startswith("dedup:"):
            return JobInfo(job_id=job_id, task_name="", queue="", status=JobStatus.PENDING)
        await self.open()
        try:
            jobs = await self.app.job_manager.list_jobs_async(id=int(job_id))
        except (ValueError, TypeError):
            return None
        for job in jobs:
            return JobInfo(
                job_id=str(job.id),
                task_name=job.task_name,
                queue=job.queue,
                status=_STATUS_MAP.get(str(job.status), JobStatus.PENDING),
                attempts=job.attempts,
                scheduled_at=job.scheduled_at.isoformat() if job.scheduled_at else None,
            )
        return None

    async def cancel(self, job_id: str) -> bool:
        await self.open()
        try:
            return bool(await self.app.job_manager.cancel_job_by_id_async(int(job_id)))
        except (ValueError, TypeError, pexc.ProcrastinateException):
            return False

    async def run_worker(
        self, queues: list[Queue] | None = None, *, concurrency: int = 4, wait: bool = True
    ) -> None:
        await self.open()
        options: dict[str, Any] = {}
        if queues:
            options["queues"] = [q.value for q in queues]
        await self.app.run_worker_async(
            **options,
            concurrency=concurrency,
            wait=wait,
            install_signal_handlers=False,
            fetch_job_polling_interval=2.0,
        )

    async def recover_stalled(self, *, seconds_since_heartbeat: float = 30) -> int:
        """Re-queue jobs left in ``doing`` by a worker that died (kill -9, OOM, node loss).

        Procrastinate workers heartbeat; a job whose worker stopped heartbeating is stalled.
        The handler's idempotency (observation ``processed_at``, outbox keys, index upserts)
        makes the replay safe. Returns the number of jobs re-queued. Called from the
        periodic reconcile with ``tasks.stalled_after_seconds`` (never below a heartbeat
        interval there, or a live worker would prune itself)."""
        await self.open()
        manager = self.app.job_manager
        await manager.prune_stalled_workers(seconds_since_heartbeat=seconds_since_heartbeat)
        stalled = list(
            await manager.get_stalled_jobs(seconds_since_heartbeat=seconds_since_heartbeat)
        )
        for job in stalled:
            await manager.retry_job(job)
        if stalled:
            log.warning("jobs.stalled_recovered", count=len(stalled))
        return len(stalled)

    async def run_until_idle(
        self, queues: list[Queue] | None = None, *, concurrency: int = 4
    ) -> None:
        """Process everything currently queued, then return (tests / batch jobs)."""
        await self.run_worker(queues, concurrency=concurrency, wait=False)


def handler(fn: Callable[[dict[str, Any]], Awaitable[Any]]) -> TaskHandler:
    """Identity decorator documenting the handler signature."""
    return fn
