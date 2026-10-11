"""Procrastinate-backed TaskQueue (PostgreSQL). MIT-licensed; async; retries, locks, queues."""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Final

import procrastinate
import psycopg_pool
from procrastinate import exceptions as pexc
from procrastinate.connector import Notify

from memory_service.config.constants import DATABASE, TASKS
from memory_service.domain.enums import JobStatus
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import (
    jobs_total,
    worker_db_retries_total,
    worker_db_unavailable_seconds,
    worker_running_jobs,
)
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


class WorkerStoppedError(RuntimeError):
    """A pool's worker ended although nobody asked it to stop: Procrastinate stops a worker
    whose heartbeat, periodic deferrer or listener failed, and returns as if stopped. The
    process must then exit non-zero so its supervisor restarts it."""


#: SQLSTATEs that say the database is out of reach or out of room, not that the statement
#: is wrong: the connection failed or was refused (08), the server is shutting down or
#: starting (57P01-57P03), it ran out of disk, memory or connections (53).
_UNAVAILABLE_CLASSES: Final = ("08", "53")
_UNAVAILABLE_STATES: Final = frozenset({"57P01", "57P02", "57P03"})


def _unavailable(exc: BaseException) -> bool:
    """The database could not do the statement: no connection was free in the queue pool in
    time (it never ran), or the server refused it for want of disk, memory or connections, or
    lost the connection (it was rolled back). Running it again is safe: Procrastinate's
    statements claim, finish and retry jobs only from the status they expect, and a job
    claimed by a statement whose answer was lost is re-queued as stalled."""
    if not isinstance(exc, pexc.ConnectorException):
        return False
    cause = exc.__cause__
    if isinstance(cause, psycopg_pool.PoolTimeout):
        return True
    state = getattr(cause, "sqlstate", None) or ""
    return state.startswith(_UNAVAILABLE_CLASSES) or state in _UNAVAILABLE_STATES


class ResilientConnector(procrastinate.PsycopgConnector):
    """Procrastinate's connector riding out a database it cannot reach for a while.

    With ``outage_seconds`` set (the worker's pools, :meth:`ProcrastinateTaskQueue.run_pools`),
    a query the database could not do (:func:`_unavailable`: no free connection in the pool,
    a full disk, a lost connection), and a LISTEN connection that was lost or could not be
    opened, are tried again after a growing pause (``backoff``), each retry logged
    (``worker.db_retry``) and counted (``memory_worker_db_retries_total``), and how long the
    database has been out of reach is a gauge (``memory_worker_db_unavailable_seconds``),
    until that reaches ``outage_seconds``; then the error is raised as before. Procrastinate
    stops a worker whose heartbeat or periodic deferrer raises, so one pool timeout under
    memory pressure, or one write to a full disk, took the whole worker process down.

    Unset (the API's queue), every error is raised at once: the API enqueues through the
    outbox, whose sweep dispatches what a failed enqueue missed, and a request must not wait
    out an outage."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.outage_seconds: float | None = None
        self.backoff: tuple[float, float] = TASKS.worker_db_backoff_seconds

    async def _retry(
        self, operation: str, attempt: int, pause: float, exc: BaseException, down: float
    ) -> None:
        worker_db_retries_total.labels(operation).inc()
        worker_db_unavailable_seconds.set(down)
        log.warning(
            "worker.db_retry",
            operation=operation,
            attempt=attempt,
            pause_seconds=pause,
            error=repr(exc.__cause__ or exc),
        )
        await asyncio.sleep(pause)

    async def _ride_out(self, call: Callable[[], Awaitable[Any]]) -> Any:
        if self.outage_seconds is None:
            return await call()
        loop = asyncio.get_running_loop()
        began, (pause, cap), attempt = loop.time(), self.backoff, 0
        while True:
            try:
                result = await call()
            except pexc.ConnectorException as exc:
                down = loop.time() - began
                if not _unavailable(exc) or down + pause > self.outage_seconds:
                    if attempt:
                        log.error("worker.db_unavailable", operation="query", seconds=round(down))
                    raise
                attempt += 1
                await self._retry("query", attempt, pause, exc, down)
                pause = min(cap, pause * 2)
            else:
                if attempt:
                    worker_db_unavailable_seconds.set(0)
                return result

    async def execute_query_async(self, query: Any, **arguments: Any) -> None:
        run = super().execute_query_async
        return await self._ride_out(lambda: run(query, **arguments))

    async def execute_query_one_async(self, query: Any, **arguments: Any) -> dict[str, Any]:
        run = super().execute_query_one_async
        return await self._ride_out(lambda: run(query, **arguments))

    async def execute_query_all_async(self, query: Any, **arguments: Any) -> list[dict[str, Any]]:
        run = super().execute_query_all_async
        return await self._ride_out(lambda: run(query, **arguments))

    async def listen_notify(self, on_notification: Notify, channels: Iterable[str]) -> None:
        """Listen again after the connection was lost or could not be opened. Jobs are
        fetched every few seconds as well, so a gap in listening delays, never loses, one."""
        channels = list(channels)
        listen = super().listen_notify
        if self.outage_seconds is None:
            return await listen(on_notification=on_notification, channels=channels)
        loop = asyncio.get_running_loop()
        first, cap = self.backoff
        pause, down_since, attempt = first, None, 0
        while True:
            began = loop.time()
            try:
                return await listen(on_notification=on_notification, channels=channels)
            except pexc.ConnectorException as exc:
                now = loop.time()
                if down_since is None or now - began > cap:  # it was listening: a new outage
                    pause, down_since, attempt = first, began, 0
                if now - down_since + pause > self.outage_seconds:
                    log.error(
                        "worker.db_unavailable", operation="listen", seconds=round(now - down_since)
                    )
                    raise
                attempt += 1
                await self._retry("listen", attempt, pause, exc, now - down_since)
                pause = min(cap, pause * 2)


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
            connector=ResilientConnector(
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

    async def run_pools(
        self,
        pools: Sequence[tuple[list[Queue], int]],
        *,
        install_signal_handlers: bool = False,
        shutdown_grace_seconds: float | None = None,
        db_outage_seconds: float | None = None,
    ) -> None:
        """One Procrastinate worker per pool, each on its queues with its own slots, until
        stopped. Procrastinate installs one worker's signal handlers at a time, so the pools
        take theirs from here: SIGTERM/SIGINT cancel every worker, and each still waits up to
        ``shutdown_grace_seconds`` for its running jobs before releasing them for a retry.

        The queue's database out of reach is ridden out for ``db_outage_seconds``
        (:class:`ResilientConnector`). A worker that ends on its own - an error, or
        Procrastinate stopping it after its heartbeat, deferrer or listener failed - stops the
        others and raises (:class:`WorkerStoppedError` when it raised nothing), so the process
        exits non-zero."""
        connector = self.app.connector
        if isinstance(connector, ResilientConnector):
            connector.outage_seconds = db_outage_seconds
        await self.open()
        runs = [
            asyncio.create_task(
                self.run_worker(
                    queues, concurrency=slots, shutdown_grace_seconds=shutdown_grace_seconds
                ),
                name=f"worker:{','.join(q.value for q in queues)}",
            )
            for queues, slots in pools
        ]
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        signals = (signal.SIGINT, signal.SIGTERM) if install_signal_handlers else ()
        for sig in signals:
            loop.add_signal_handler(sig, stop.set)
        stopped = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait([*runs, stopped], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for sig in signals:
                loop.remove_signal_handler(sig)
            for task in (*runs, stopped):
                task.cancel()
            ended = await asyncio.gather(*runs, return_exceptions=True)
        for result in ended:
            if isinstance(result, Exception):
                raise result
        if not stop.is_set():
            names = ", ".join(t.get_name() for t in runs if t in done)
            raise WorkerStoppedError(f"{names} stopped although no stop was asked for")

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
