"""Background worker entrypoint (Procrastinate). Task handlers are registered by modules."""

from __future__ import annotations

from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import get_settings
from memory_service.observability.logging import configure_logging, get_logger
from memory_service.observability.tracing import configure_tracing

log = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.service.log_level, settings.service.log_json)
    configure_tracing(settings.observability, settings.service.name + "-worker", __version__)
    container = await build_container(settings, __version__)
    if container.tasks is None:
        log.error("worker.no_task_queue")
        return
    try:
        log.info("worker.started", concurrency=settings.tasks.worker_concurrency)
        await container.tasks.run_worker(concurrency=settings.tasks.worker_concurrency)
    finally:
        await container.close()
