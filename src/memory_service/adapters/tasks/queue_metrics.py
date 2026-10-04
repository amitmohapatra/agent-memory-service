"""The job worker's queue gauges, read from PostgreSQL on a timer.

What an operator needs to see from a worker is not in the worker's memory: how many jobs
wait per queue, how long the oldest due one has waited, how much of the outbox is committed
but not yet handed to the queue, and how many jobs gave up. Each is one grouped statement
over the queue's own tables, run every ``SAMPLE_SECONDS`` and set on the gauges the
worker's metrics port serves (``observability.metrics``). A failed read is logged and leaves
the previous values, and ``memory_worker_last_sample_timestamp_seconds`` stops moving -
which is the healthcheck's signal that the worker cannot reach its database.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import (
    jobs_failed_terminal,
    outbox_backlog,
    queue_depth,
    queue_oldest_lag_seconds,
    worker_last_sample_timestamp,
)
from memory_service.ports.tasks import Queue

log = get_logger(__name__)

SAMPLE_SECONDS: Final = 15.0

#: waiting jobs and the age of the oldest due one, per queue; a job's age is from when it
#: became due (``scheduled_at``) or, unscheduled, from its ``deferred`` event
_QUEUES: Final = text(
    """
    SELECT j.queue_name,
           count(*) AS depth,
           coalesce(max(extract(epoch FROM now() - coalesce(j.scheduled_at, e.at)))
                    FILTER (WHERE j.scheduled_at IS NULL OR j.scheduled_at <= now()), 0) AS lag
    FROM procrastinate_jobs j
    LEFT JOIN procrastinate_events e ON e.job_id = j.id AND e.type = 'deferred'
    WHERE j.status = 'todo'
    GROUP BY j.queue_name
    """
)
_FAILED: Final = text(
    "SELECT queue_name, count(*) FROM procrastinate_jobs WHERE status = 'failed' "
    "GROUP BY queue_name"
)
_OUTBOX: Final = text("SELECT count(*) FROM job_outbox WHERE dispatched_at IS NULL AND NOT dead")


async def sample_once(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        waiting = {str(row[0]): (int(row[1]), float(row[2])) for row in await conn.execute(_QUEUES)}
        failed = {str(row[0]): int(row[1]) for row in await conn.execute(_FAILED)}
        backlog = int((await conn.execute(_OUTBOX)).scalar_one())
    for name in {q.value for q in Queue} | set(waiting) | set(failed):
        depth, lag = waiting.get(name, (0, 0.0))
        queue_depth.labels(name).set(depth)
        queue_oldest_lag_seconds.labels(name).set(round(lag, 3))
        jobs_failed_terminal.labels(name).set(failed.get(name, 0))
    outbox_backlog.set(backlog)
    worker_last_sample_timestamp.set(time.time())


async def sample_forever(engine: AsyncEngine, *, every: float = SAMPLE_SECONDS) -> None:
    while True:
        try:
            await sample_once(engine)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a gauge must never take the worker down
            log.warning("worker.metrics_sample_failed", error=type(exc).__name__)
        await asyncio.sleep(every)
