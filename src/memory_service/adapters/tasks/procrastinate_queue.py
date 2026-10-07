"""Procrastinate-backed TaskQueue (PostgreSQL). MIT-licensed; async; retries, locks, queues."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import procrastinate
from procrastinate import exceptions as pexc

from memory_service.config.constants import DATABASE
from memory_service.domain.enums import JobStatus
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import jobs_total, worker_running_jobs
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

#: Procrastinate records that a job failed but not why, so ``get`` had nothing to put in
#: ``last_error`` and a FAILED job's status never said what went wrong. The worker writes the
#: last failure here. The ``procrastinate_`` prefix keeps it out of Alembic's autogenerate
#: (migrations/env.py), like the rest of the queue's schema, and the cascade drops it with
#: its job.
_ERRORS_TABLE = """
CREATE TABLE IF NOT EXISTS procrastinate_job_errors (
    job_id bigint PRIMARY KEY REFERENCES procrastinate_jobs(id) ON DELETE CASCADE,
    error text NOT NULL,
    at timestamp with time zone NOT NULL DEFAULT NOW()
)
"""
_RECORD_ERROR = """
INSERT INTO procrastinate_job_errors (job_id, error) VALUES (%(job_id)s, %(error)s)
ON CONFLICT (job_id) DO UPDATE SET error = EXCLUDED.error, at = NOW()
"""


def _retry_strategy(retries: int) -> procrastinate.RetryStrategy | bool:
    """``retries`` additional attempts after the first (Procrastinate's ``job.attempts`` is
    the number of *previous* runs, so ``max_attempts=N`` retries while fewer than N have
    failed: N + 1 runs in all); 0 disables retries entirely."""
    if retries <= 0:
        return False
    return procrastinate.RetryStrategy(max_attempts=retries, wait=1, exponential_wait=2)


class ProcrastinateTaskQueue:
    """Wraps a ``procrastinate.App``. Handlers are registered before the worker starts.

    Each handler receives ``(payload: dict)`` and may raise to trigger a retry.
    """

    info = INFO

    def __init__(
        self,
        dsn: str,
        *,
        default_retries: int = 5,
        job_timeout_seconds: int = 600,
        max_size: int = 4,
    ) -> None:
        """``dsn`` must reach PostgreSQL with a session of its own (``direct_url``): the
        worker LISTENs for new jobs on a connection it keeps, which a transaction-mode
        pooler would hand to someone else between statements.

        The pool is sized from the pod's connection budget and bounded in time like the
        request pool: Procrastinate's default had no statement timeout and no connect
        timeout, so a stalled server held a job fetch, and the worker's slot, forever."""
        self.app = procrastinate.App(
            connector=procrastinate.PsycopgConnector(
                conninfo=dsn,
                json_dumps=lambda v: json.dumps(v, default=str),
                json_loads=json.loads,
                min_size=1,
                max_size=max(2, max_size),
                timeout=DATABASE.pool_timeout_seconds,
                kwargs={
                    "options": f"-c statement_timeout={DATABASE.statement_timeout_ms}",
                    "connect_timeout": DATABASE.connect_timeout_seconds,
                },
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
        if not (row and row[0]):
            await self.app.schema_manager.apply_schema_async()
        # outside the check above: a database whose queue schema predates the table gets it
        async with self.app.connector.pool.connection() as conn:  # type: ignore[attr-defined]
            await conn.execute(_ERRORS_TABLE)

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
                worker_running_jobs.inc()
                try:
                    result = await asyncio.wait_for(handler(payload), timeout=timeout)
                except Exception as exc:
                    jobs_total.labels(name, "failed").inc()
                    if context.job and context.job.id is not None:
                        await self._record_error(context.job.id, exc)
                    raise
                finally:
                    worker_running_jobs.dec()
                jobs_total.labels(name, "succeeded").inc()
                return result

        self.app.task(
            name=name,
            queue=queue.value,
            priority=QUEUE_PRIORITY[queue],
            retry=_retry_strategy(retries if retries is not None else self.default_retries),
            pass_context=True,
        )(_run)

    async def _record_error(self, job_id: int, exc: BaseException) -> None:
        """Best effort: failing to record why a job failed must not change how it fails."""
        try:
            await self.app.connector.execute_query_async(
                _RECORD_ERROR, job_id=job_id, error=f"{type(exc).__name__}: {exc}"[:2000]
            )
        except Exception as record_exc:
            log.warning("jobs.error_not_recorded", job_id=job_id, error=str(record_exc))

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
            errors = await self.app.connector.execute_query_all_async(
                "SELECT error FROM procrastinate_job_errors WHERE job_id = %(job_id)s",
                job_id=job.id,
            )
            return JobInfo(
                job_id=str(job.id),
                task_name=job.task_name,
                queue=job.queue,
                status=_STATUS_MAP.get(str(job.status), JobStatus.PENDING),
                attempts=job.attempts,
                last_error=errors[0]["error"] if errors else None,
                scheduled_at=job.scheduled_at.isoformat() if job.scheduled_at else None,
            )
        return None

    async def run_worker(
        self,
        queues: list[Queue] | None = None,
        *,
        concurrency: int = 4,
        wait: bool = True,
        install_signal_handlers: bool = False,
        shutdown_grace_seconds: float | None = None,
    ) -> None:
        """Run jobs until stopped (or, ``wait=False``, until the queues are empty).

        ``install_signal_handlers`` is for the worker process alone (``memory_service.worker``):
        SIGTERM/SIGINT then stop the fetch loop and wait up to ``shutdown_grace_seconds`` for
        running jobs, after which Procrastinate aborts them with ``AbortReason.SHUTDOWN`` and
        puts them back to be retried. Tests and drains run it inside a loop they own, where a
        handler on SIGINT would take Ctrl-C away from the test runner.
        """
        await self.open()
        options: dict[str, Any] = {}
        if queues:
            options["queues"] = [q.value for q in queues]
        if shutdown_grace_seconds is not None:
            options["shutdown_graceful_timeout"] = shutdown_grace_seconds
        await self.app.run_worker_async(
            **options,
            concurrency=concurrency,
            wait=wait,
            install_signal_handlers=install_signal_handlers,
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
        """Process everything currently queued, then return (tests / batch jobs).

        Without the periodic deferrer: a drain has nothing to schedule, and procrastinate's
        deferrer swallows the cancellation that ends the worker while it defers, then sleeps
        until the next cron tick - so the worker's shutdown waited out that sleep and an
        intermittent drain hung for minutes."""
        from procrastinate.periodic import PeriodicRegistry

        registry = self.app.periodic_registry
        self.app.periodic_registry = PeriodicRegistry()
        try:
            await self.run_worker(queues, concurrency=concurrency, wait=False)
        finally:
            self.app.periodic_registry = registry


def handler(fn: Callable[[dict[str, Any]], Awaitable[Any]]) -> TaskHandler:
    """Identity decorator documenting the handler signature."""
    return fn
