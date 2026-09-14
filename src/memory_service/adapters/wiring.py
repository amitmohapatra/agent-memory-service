"""Provider wiring. Grows milestone by milestone; M0 registers nothing but the registry itself."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory_service.observability.logging import get_logger

if TYPE_CHECKING:
    from memory_service.application.container import Container

log = get_logger(__name__)


async def wire_all(container: Container) -> None:
    settings = container.settings
    log.info(
        "wiring.start",
        environment=settings.service.environment,
        cache=settings.cache.provider,
        search=settings.search.provider,
        blob=settings.blob.provider,
        tasks=settings.tasks.provider,
        authorization=settings.authorization.provider,
        llm_enabled=settings.models.llm.enabled,
    )
    # M1+: database, tasks, cache, ... are attached here.
    log.info("wiring.done", dependencies=sorted(container.dependencies))
