"""The job worker's queue gauges read from the real tables (ADR 0031)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
from memory_service.adapters.tasks.queue_metrics import sample_once
from memory_service.observability.metrics import REGISTRY
from memory_service.ports.tasks import JobSpec, Queue
from tests.conftest import DB_URL
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


def _value(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


async def test_depth_lag_backlog_and_failures_are_read_from_postgres(container) -> None:
    queue = ProcrastinateTaskQueue(DB_URL.replace("postgresql+psycopg://", "postgresql://"))

    async def handler(payload):
        return None

    queue.register("test.metric", Queue.SUMMARY, handler)
    try:
        for n in range(3):
            await queue.enqueue(
                JobSpec(task_name="test.metric", queue=Queue.SUMMARY, payload={"n": n})
            )
    finally:
        await queue.close()
    async with container.database.engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE procrastinate_events SET at = now() - interval '90 seconds' "
                "WHERE type = 'deferred'"
            )
        )
        await conn.execute(
            text("UPDATE procrastinate_jobs SET status = 'failed' WHERE args->>'n' = '2'")
        )
        await conn.execute(
            text(
                "INSERT INTO job_outbox (task_name, queue, payload) "
                "VALUES ('test.metric', 'summary', '{}'), ('test.metric', 'summary', '{}')"
            )
        )

    await sample_once(container.database.engine)

    assert _value("memory_queue_depth", queue="summary") == 2
    lag = _value("memory_queue_oldest_lag_seconds", queue="summary")
    assert lag is not None and 85 <= lag <= 600
    assert _value("memory_jobs_failed_terminal", queue="summary") == 1
    assert _value("memory_queue_depth", queue="archive") == 0, "every queue has a series"
    assert _value("memory_outbox_backlog") == 2
    assert (_value("memory_worker_last_sample_timestamp_seconds") or 0) > 0
