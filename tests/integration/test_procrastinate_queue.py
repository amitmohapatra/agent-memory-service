"""Real Procrastinate: enqueue, run, retry, queueing-lock dedup, status."""

from __future__ import annotations

import asyncio

import pytest

from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
from memory_service.domain.enums import JobStatus
from memory_service.ports.tasks import JobSpec, Queue
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
