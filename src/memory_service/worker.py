"""Background worker entrypoint (Procrastinate). Task handlers are registered by modules.

The worker stops the way an orchestrator asks it to: on SIGTERM or SIGINT it stops fetching
jobs, gives the running ones ``OVERLOAD.worker_shutdown_grace_seconds`` to finish, then
aborts what is left, which Procrastinate releases for a retry (every handler is idempotent),
and only then
closes the container - flushing what services buffered while the pool is still open. It used
to run with Procrastinate's signal handling switched off, so SIGTERM killed it mid-job and
the job sat in ``doing`` until the stalled-job reconciler noticed thirty seconds later.

It runs four pools (``application.container.worker_pools_for``, sized by
``ports.tasks.worker_pools``): chat's (half of ``tasks.worker_concurrency``), indexing's, the
graph queue's and the shared one, every pool's computing slots together about one per CPU,
with only the graph pool's slots on top when they wait on a model. Chat and indexing share
their slots by priority, chat first; a long parse or a graph backlog never holds one of them,
and with a model, chat jobs waiting on it never hold indexing's.

A database out of reach (no connection free in the queue pool, the LISTEN connection lost) is
ridden out for ``TASKS.worker_db_outage_seconds``, each retry logged and counted
(``memory_worker_db_retries_total``). Past that, or when a pool's worker stops on its own,
the process exits non-zero for its supervisor to restart it: under memory pressure one pool
timeout in a heartbeat stopped the worker, and the process exited as if asked to.

``tasks.metrics_port`` serves the worker's Prometheus series - queue depth, oldest-job lag,
outbox backlog, terminally failed jobs, running jobs, the last sample time - and is what the
compose healthcheck reads.
"""

from __future__ import annotations

import asyncio
import contextlib

from memory_service.__about__ import __version__
from memory_service.application.container import build_container, worker_pools_for
from memory_service.config.constants import OVERLOAD, SERVICE_NAME, TASKS
from memory_service.config.settings import get_settings
from memory_service.observability.logging import configure_logging, get_logger
from memory_service.observability.metrics import serve_worker_metrics
from memory_service.observability.tracing import configure_tracing

log = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.service.log_level, settings.service.log_json)
    configure_tracing(settings.otel_endpoint, SERVICE_NAME + "-worker", __version__)
    container = await build_container(settings, __version__, role="worker")
    if container.tasks is None:
        log.error("worker.no_task_queue")
        return
    sampler: asyncio.Task[None] | None = None
    if settings.tasks.metrics_port is not None:
        from memory_service.adapters.tasks.queue_metrics import sample_forever

        serve_worker_metrics(settings.tasks.metrics_port)
        sampler = asyncio.create_task(sample_forever(container.database.engine))
    try:
        pools = worker_pools_for(settings)
        log.info(
            "worker.started",
            concurrency=settings.tasks.worker_concurrency,
            pools={",".join(queues): slots for queues, slots in pools},
            metrics_port=settings.tasks.metrics_port,
        )
        await container.tasks.run_pools(
            pools,
            install_signal_handlers=True,
            shutdown_grace_seconds=OVERLOAD.worker_shutdown_grace_seconds,
            db_outage_seconds=TASKS.worker_db_outage_seconds,
        )
        log.info("worker.stopped")
    except Exception as exc:
        # non-zero exit, for the supervisor to restart the worker
        log.error("worker.failed", error=repr(exc))
        raise
    finally:
        if sampler is not None:
            sampler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler
        await container.close()
