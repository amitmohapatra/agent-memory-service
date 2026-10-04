"""Background worker entrypoint (Procrastinate). Task handlers are registered by modules.

The worker stops the way an orchestrator asks it to: on SIGTERM or SIGINT it stops fetching
jobs, gives the running ones ``OVERLOAD.worker_shutdown_grace_seconds`` to finish, then
aborts what is left, which Procrastinate releases for a retry (every handler is idempotent),
and only then
closes the container - flushing what services buffered while the pool is still open. It used
to run with Procrastinate's signal handling switched off, so SIGTERM killed it mid-job and
the job sat in ``doing`` until the stalled-job reconciler noticed thirty seconds later.

``tasks.metrics_port`` serves the worker's Prometheus series - queue depth, oldest-job lag,
outbox backlog, terminally failed jobs, running jobs, the last sample time - and is what the
compose healthcheck reads.
"""

from __future__ import annotations

import asyncio
import contextlib

from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.constants import OVERLOAD, SERVICE_NAME
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
        log.info(
            "worker.started",
            concurrency=settings.tasks.worker_concurrency,
            metrics_port=settings.tasks.metrics_port,
        )
        await container.tasks.run_worker(
            concurrency=settings.tasks.worker_concurrency,
            install_signal_handlers=True,
            shutdown_grace_seconds=OVERLOAD.worker_shutdown_grace_seconds,
        )
        log.info("worker.stopped")
    finally:
        if sampler is not None:
            sampler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler
        await container.close()
