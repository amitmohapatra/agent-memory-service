"""Provider wiring. Grows milestone by milestone."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory_service.application.container import Dependency
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
    await _wire_cache(container)
    await _wire_database(container)
    await _wire_tasks(container)
    _wire_uow(container)
    await _wire_authorization(container)
    _wire_services(container)
    log.info("wiring.done", dependencies=sorted(container.dependencies))


# ---------------------------------------------------------------------------
# M1 wiring
# ---------------------------------------------------------------------------


async def _wire_cache(container: Container) -> None:
    cfg = container.settings.cache
    if cfg.provider == "disabled":
        container.cache = None
        return
    if cfg.provider == "memory":
        from memory_service.adapters.cache.memory_cache import MemoryCache

        container.cache = MemoryCache()
    else:
        from memory_service.adapters.cache.redis_cache import RedisCache

        container.cache = RedisCache(cfg)
    cache = container.cache
    container.add_dependency(
        Dependency(name="cache", mandatory=False, ping=cache.ping, close=cache.close)
    )


async def _wire_database(container: Container) -> None:
    from memory_service.adapters.db.engine import Database

    db = Database(container.settings.database)
    container.database = db
    container.add_dependency(
        Dependency(name="postgres", mandatory=True, ping=db.ping, close=db.close)
    )


async def _wire_tasks(container: Container) -> None:
    cfg = container.settings.tasks
    if cfg.provider == "procrastinate":
        from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

        queue = ProcrastinateTaskQueue(
            container.settings.database.procrastinate_dsn,
            default_retries=cfg.default_retries,
            job_timeout_seconds=cfg.job_timeout_seconds,
        )
        container.tasks = queue
        container.add_dependency(
            Dependency(name="task_queue", mandatory=True, ping=queue.ping, close=queue.close)
        )
    elif cfg.provider == "inline":
        from memory_service.adapters.tasks.inline_queue import InlineTaskQueue

        container.tasks = InlineTaskQueue()
    else:
        from memory_service.adapters.tasks.inline_queue import RecordingTaskQueue

        container.tasks = RecordingTaskQueue()


def _wire_uow(container: Container) -> None:
    from memory_service.adapters.db.uow import OutboxRelay, SqlUnitOfWorkFactory

    relay = OutboxRelay(container.database.session_factory, container.tasks)
    container.services["outbox_relay"] = relay
    container.services["uow_factory"] = SqlUnitOfWorkFactory(
        container.database.session_factory, relay
    )


async def _wire_authorization(container: Container) -> None:
    cfg = container.settings.authorization
    if cfg.provider == "openfga":
        from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider

        provider = OpenFGAAuthorizationProvider(cfg)
        container.add_dependency(
            Dependency(name="openfga", mandatory=True, ping=provider.ping, close=provider.close)
        )
    else:
        from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider

        provider = MemoryAuthorizationProvider(max_listed_objects=cfg.max_listed_objects)
    container.authorization = provider


def _wire_services(container: Container) -> None:
    from memory_service.modules.auth.authentication import ServiceAuthenticator
    from memory_service.modules.authz.service import AuthorizationService
    from memory_service.modules.idempotency.service import IdempotencyService

    settings = container.settings
    container.services["idempotency"] = IdempotencyService(container.cache)
    from memory_service.adapters.auth.gcp_id_token import verify_google_id_token

    container.services["authenticator"] = ServiceAuthenticator(
        settings.authentication, gcp_verifier=verify_google_id_token
    )
    container.services["authz"] = AuthorizationService(
        container.authorization,
        container.cache,
        max_listed_objects=settings.authorization.max_listed_objects,
        cache_ttl_seconds=settings.cache.authz_ttl_seconds,
        decision_cache=settings.authorization.decision_cache,
    )
