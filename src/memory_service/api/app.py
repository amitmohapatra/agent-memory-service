"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from memory_service.__about__ import __version__
from memory_service.api.errors import install_error_handlers
from memory_service.api.middleware import CorrelationMiddleware, RateLimitMiddleware
from memory_service.api.openapi import custom_openapi
from memory_service.api.routers import ops
from memory_service.application.container import Container, build_container
from memory_service.config.settings import Settings, get_settings
from memory_service.observability.logging import configure_logging, get_logger
from memory_service.observability.tracing import configure_tracing

log = get_logger(__name__)


def create_app(settings: Settings | None = None, *, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.service.log_level, settings.service.log_json)
    configure_tracing(settings.observability, settings.service.name, __version__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or await build_container(settings, __version__)
        log.info("app.started", version=__version__, environment=settings.service.environment)
        try:
            yield
        finally:
            await app.state.container.close()
            log.info("app.stopped")

    app = FastAPI(
        title="Memory Service API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    # outermost first: correlation ids wrap everything, the rate limit sits inside them
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=settings.service.rate_limit_per_minute,
        burst=settings.service.rate_limit_burst,
    )
    app.add_middleware(CorrelationMiddleware, max_body_bytes=settings.service.max_body_bytes)
    install_error_handlers(app)
    app.include_router(ops.router)
    _include_v1_routers(app)

    if settings.observability.otel_enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health/live,health/ready,metrics")

    def _openapi() -> dict[str, Any]:
        return custom_openapi(app, version=__version__)

    app.openapi = _openapi  # type: ignore[method-assign]
    return app


def _include_v1_routers(app: FastAPI) -> None:
    """Public /v1 routers are registered here as milestones land."""
    from memory_service.api.routers import v1

    for router in v1.routers():
        app.include_router(router, prefix="/v1")
