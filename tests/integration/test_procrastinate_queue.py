"""Real Procrastinate: enqueue, run, retry, queueing-lock dedup, status."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
from memory_service.domain.enums import JobStatus
from memory_service.ports.tasks import JobSpec, Queue, worker_pools
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


@pytest.fixture
async def queue(queue_database: str):
    # The queueing lock dedups against jobs that are still pending, and these tests use fixed
    # lock keys — so a job left behind by an earlier run would make the *first* enqueue of the
    # next one look like a duplicate. The database is this suite's own; start it empty.
    import psycopg

    with psycopg.connect(queue_database, autocommit=True) as conn:
        conn.execute("TRUNCATE procrastinate_jobs, procrastinate_events RESTART IDENTITY CASCADE")

    q = ProcrastinateTaskQueue(queue_database, default_retries=2, job_timeout_seconds=5)
    await q.open()
    try:
        yield q
    finally:
        await q.close()


async def test_enqueue_run_and_status(queue: ProcrastinateTaskQueue) -> None:
    seen: list[dict] = []

    async def handler(payload):
        seen.append(payload)

    queue.register("test.echo", Queue.CHAT_FAST, handler)
    job_id = await queue.enqueue(
        JobSpec(task_name="test.echo", queue=Queue.CHAT_FAST, payload={"n": 1})
    )
    info = await queue.get(job_id)
    assert info is not None and info.status is JobStatus.PENDING and info.queue == "chat-fast"
    await queue.run_until_idle([Queue.CHAT_FAST], concurrency=1)
    assert seen == [{"n": 1}]
    info = await queue.get(job_id)
    assert info is not None and info.status is JobStatus.SUCCEEDED


async def test_queueing_lock_deduplicates(queue: ProcrastinateTaskQueue) -> None:
    async def handler(payload):
        return None

    queue.register("test.dedup", Queue.ARCHIVE, handler)
    a = await queue.enqueue(
        JobSpec(task_name="test.dedup", queue=Queue.ARCHIVE, idempotency_key="seg-1")
    )
    b = await queue.enqueue(
        JobSpec(task_name="test.dedup", queue=Queue.ARCHIVE, idempotency_key="seg-1")
    )
    assert not a.startswith("dedup:") and b == "dedup:seg-1"


async def test_failing_handler_is_retried_then_failed(queue: ProcrastinateTaskQueue) -> None:
    attempts = 0

    async def handler(payload):
        nonlocal attempts
        attempts += 1
        raise ValueError("always fails")

    queue.register("test.fail", Queue.MEMORY_EXTRACT, handler, retries=2)
    job_id = await queue.enqueue(JobSpec(task_name="test.fail", queue=Queue.MEMORY_EXTRACT))
    # retries are scheduled with backoff (1s + 2^n); run the worker until the job is terminal
    for _ in range(12):
        await queue.run_until_idle([Queue.MEMORY_EXTRACT], concurrency=1)
        info = await queue.get(job_id)
        if info and info.status is JobStatus.FAILED:
            break
        await asyncio.sleep(1)
    info = await queue.get(job_id)
    assert info is not None and info.status is JobStatus.FAILED
    assert attempts == 3  # 1 initial + 2 retries


async def test_a_job_that_gives_up_is_handed_to_its_give_up_handler(
    queue: ProcrastinateTaskQueue,
) -> None:
    """``on_give_up`` runs once, after the last attempt, with the payload and the error -
    the hook that marks a document whose index job gave up FAILED instead of STAGED forever."""
    attempts: list[int] = []
    gave_up: list[tuple[dict, str]] = []

    async def handler(payload):
        attempts.append(1)
        raise ValueError("embedding service down")

    async def give_up(payload, error):
        gave_up.append((payload, error))

    queue.register("test.index", Queue.EMBEDDING, handler, retries=1, on_give_up=give_up)
    job_id = await queue.enqueue(
        JobSpec(task_name="test.index", queue=Queue.EMBEDDING, payload={"document_id": "d1"})
    )
    for _ in range(12):
        await queue.run_until_idle([Queue.EMBEDDING], concurrency=1)
        info = await queue.get(job_id)
        if info and info.status is JobStatus.FAILED:
            break
        assert gave_up == [], "not while a retry is to come"
        await asyncio.sleep(1)
    assert len(attempts) == 2
    assert gave_up == [({"document_id": "d1"}, "ValueError: embedding service down")]


async def test_handler_timeout_is_enforced(queue: ProcrastinateTaskQueue) -> None:
    queue.job_timeout_seconds = 1

    async def handler(payload):
        await asyncio.sleep(5)

    queue.register("test.slow", Queue.SUMMARY, handler, retries=0)
    job_id = await queue.enqueue(JobSpec(task_name="test.slow", queue=Queue.SUMMARY))
    await queue.run_until_idle([Queue.SUMMARY], concurrency=1)
    info = await queue.get(job_id)
    assert info is not None and info.status is JobStatus.FAILED


async def test_ping(queue: ProcrastinateTaskQueue) -> None:
    assert await queue.ping() is True


async def test_a_pools_long_jobs_never_hold_another_pools_slots(
    queue: ProcrastinateTaskQueue,
) -> None:
    """The worker's pools (``worker_pools``). A graph backlog takes the graph pool's slots
    only: a reconcile job (a run's feedback, a learned skill), which ranks below enrichment,
    still runs at once. With the shared pool's slots taken by jobs that wait (a slow
    model's), an indexing job still runs at once. In one pool both waited for a slot, which
    on a CPU model took minutes."""
    release = asyncio.Event()
    started: list[str] = []
    ran: dict[str, asyncio.Event] = {"learn": asyncio.Event(), "index": asyncio.Event()}

    async def slow(payload):
        started.append(payload["n"])
        await release.wait()

    async def done(payload):
        ran[payload["n"]].set()

    async def until(count: int) -> None:
        for _ in range(100):
            if len(started) >= count:
                return
            await asyncio.sleep(0.1)

    queue.register("test.enrich", Queue.GRAPH, slow)
    queue.register("test.extract", Queue.CHAT_FAST, slow)
    queue.register("test.learn", Queue.RECONCILE, done)
    queue.register("test.index", Queue.EMBEDDING, done)
    for n in ("g1", "g2", "g3"):
        await queue.enqueue(JobSpec(task_name="test.enrich", queue=Queue.GRAPH, payload={"n": n}))
    pools = worker_pools(concurrency=1, indexing=1, graph=1)
    assert pools[0] == ([Queue.EMBEDDING], 1) and pools[1] == ([Queue.GRAPH], 1)
    assert Queue.GRAPH not in pools[2][0] and Queue.RECONCILE in pools[2][0]
    worker = asyncio.create_task(queue.run_pools(pools, shutdown_grace_seconds=5))
    try:
        await until(1)
        await asyncio.sleep(1)
        assert started == ["g1"], "the graph pool's one slot runs the backlog, one at a time"
        await queue.enqueue(
            JobSpec(task_name="test.learn", queue=Queue.RECONCILE, payload={"n": "learn"})
        )
        await asyncio.wait_for(ran["learn"].wait(), timeout=10)

        await queue.enqueue(
            JobSpec(task_name="test.extract", queue=Queue.CHAT_FAST, payload={"n": "x"})
        )
        await until(2)
        assert started == ["g1", "x"], "the shared pool's one slot is taken"
        await queue.enqueue(
            JobSpec(task_name="test.index", queue=Queue.EMBEDDING, payload={"n": "index"})
        )
        await asyncio.wait_for(ran["index"].wait(), timeout=10)
    finally:
        release.set()
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


#: The ``queue`` fixture opens its pool on the session's loop (``asyncio_default_fixture_loop_scope``):
#: a test that needs the pool to open more connections runs there too, or the pool's
#: background workers, which open them, never run.
ON_THE_POOLS_LOOP = pytest.mark.asyncio(loop_scope="session")


async def _until(check, seconds: float = 20) -> None:
    for _ in range(int(seconds * 10)):
        if check():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("not within the time")


def _db_retries() -> float:
    from memory_service.observability.metrics import REGISTRY

    found = REGISTRY.get_sample_value("memory_worker_db_retries_total", {"operation": "query"})
    return found or 0.0


@ON_THE_POOLS_LOOP
async def test_a_queue_pool_timeout_is_ridden_out_and_the_worker_goes_on(
    queue: ProcrastinateTaskQueue,
) -> None:
    """Every connection of the queue pool taken for longer than the pool's timeout - the
    database out of reach under memory pressure - and the worker's job fetch times out. It
    used to raise out of the worker, and the process ended; now it is retried with a growing
    pause, counted, and the next job runs once a connection is free again."""
    from memory_service.config.constants import DATABASE

    ran: list[int] = []

    async def job(payload):
        ran.append(payload["n"])

    queue.register("test.job", Queue.RECONCILE, job)
    worker = asyncio.create_task(
        queue.run_pools([([Queue.RECONCILE], 1)], shutdown_grace_seconds=5, db_outage_seconds=60)
    )
    try:
        await queue.enqueue(JobSpec(task_name="test.job", queue=Queue.RECONCILE, payload={"n": 1}))
        await _until(lambda: ran == [1])
        before = _db_retries()
        pool = queue.app.connector.pool  # type: ignore[attr-defined]
        async with contextlib.AsyncExitStack() as held:
            for _ in range(pool.max_size):
                await held.enter_async_context(pool.connection())
            await asyncio.sleep(DATABASE.pool_timeout_seconds + 4)
            assert not worker.done(), "the worker rides the outage out"
        assert _db_retries() > before, "each retry is counted"
        await queue.enqueue(JobSpec(task_name="test.job", queue=Queue.RECONCILE, payload={"n": 2}))
        await _until(lambda: ran == [1, 2])
        assert not worker.done()
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


@ON_THE_POOLS_LOOP
async def test_a_database_out_of_reach_past_the_budget_ends_the_worker_with_an_error(
    queue: ProcrastinateTaskQueue,
) -> None:
    """Past ``db_outage_seconds`` the pool timeout is raised: the worker process exits
    non-zero, for its supervisor to start a fresh one."""
    ran: list[bool] = []

    async def job(payload):
        ran.append(True)

    queue.register("test.job", Queue.RECONCILE, job)
    worker = asyncio.create_task(
        queue.run_pools([([Queue.RECONCILE], 1)], shutdown_grace_seconds=5, db_outage_seconds=1)
    )
    try:
        await queue.enqueue(JobSpec(task_name="test.job", queue=Queue.RECONCILE))
        await _until(lambda: ran == [True])  # the worker is up
        pool = queue.app.connector.pool  # type: ignore[attr-defined]
        async with contextlib.AsyncExitStack() as held:
            for _ in range(pool.max_size):
                await held.enter_async_context(pool.connection())
            with pytest.raises(Exception) as raised:  # noqa: PT011 - the connector's own error
                await asyncio.wait_for(worker, timeout=60)
        assert type(raised.value.__cause__).__name__ == "PoolTimeout", raised.value
    finally:
        worker.cancel()
        with contextlib.suppress(BaseException):
            await worker


async def test_a_worker_that_stops_unasked_ends_the_pools_with_an_error(
    queue: ProcrastinateTaskQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Procrastinate stops a worker whose heartbeat or deferrer failed and returns as if it
    had been asked to; the pools then raise instead of returning, so the process does not exit
    0 as a clean stop would."""
    from memory_service.adapters.tasks.procrastinate_queue import WorkerStoppedError

    async def stopped_on_its_own(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(queue, "run_worker", stopped_on_its_own)
    with pytest.raises(WorkerStoppedError, match="worker:reconcile"):
        await asyncio.wait_for(queue.run_pools([([Queue.RECONCILE], 1)]), timeout=10)


async def _stop_mid_job(queue_database: str, tmp_path, *, seconds: float, grace: float):
    """Start a worker on one slow job, SIGTERM it while the job runs; the job's final
    state, whether it finished, and how long the stop took."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    queue = ProcrastinateTaskQueue(queue_database, default_retries=2)

    async def never(payload):  # the worker in the subprocess runs it, not this one
        raise AssertionError

    queue.register("test.slow", Queue.CHAT_FAST, never)
    job_id = await queue.enqueue(JobSpec(task_name="test.slow", queue=Queue.CHAT_FAST))
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
        ),
    }
    marker = tmp_path / "finished"
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(Path(__file__).with_name("_graceful_worker.py")),
            queue_database,
            str(seconds),
            str(grace),
            str(marker),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), timeout=60)  # type: ignore[union-attr]
        assert line.startswith("TOOK")
        started = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=30)
        took = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0
    info = await queue.get(job_id)
    await queue.close()
    return info, marker.exists(), took


async def test_sigterm_lets_a_running_job_finish_inside_the_grace(
    queue: ProcrastinateTaskQueue, queue_database: str, tmp_path
) -> None:
    info, finished, took = await _stop_mid_job(queue_database, tmp_path, seconds=1.5, grace=20)
    assert finished, "the job was cut off instead of being allowed to finish"
    assert info is not None and info.status is JobStatus.SUCCEEDED
    assert took < 15


async def test_sigterm_releases_a_job_that_outlives_the_grace(
    queue: ProcrastinateTaskQueue, queue_database: str, tmp_path
) -> None:
    info, finished, took = await _stop_mid_job(queue_database, tmp_path, seconds=60, grace=0.5)
    assert not finished
    assert took < 15, "the worker waited past its grace"
    # aborted for the shutdown and put back to be retried, not left in `doing`
    assert info is not None and info.status is JobStatus.PENDING, info
