"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from memory_service.__about__ import __version__
from memory_service.api.errors import install_error_handlers
from memory_service.api.middleware import CorrelationMiddleware, RateLimitMiddleware
from memory_service.api.openapi import custom_openapi
from memory_service.api.routers import ops
from memory_service.application.container import Container, Overrides, build_container
from memory_service.config import constants
from memory_service.config.settings import Settings, get_settings
from memory_service.observability.logging import configure_logging, get_logger
from memory_service.observability.tracing import configure_tracing

log = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
    overrides: Overrides | None = None,
) -> FastAPI:
    """``overrides`` swaps backing stores for in-process stand-ins (tests and benchmarks)."""
    settings = settings or get_settings()
    configure_logging(settings.service.log_level, settings.service.log_json)
    configure_tracing(settings.observability, constants.SERVICE_NAME, __version__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or await build_container(
            settings, __version__, overrides=overrides
        )
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
        # No default_response_class: naming one (ORJSONResponse, say) turns *off* FastAPI's
        # own fast path, which serialises a response model straight to JSON bytes through
        # pydantic's Rust core and never builds the intermediate dict. Both fastapi 0.141's
        # deprecation of ORJSONResponse and routing.py:719-740 say so; measured on a
        # context-bundle-shaped response, orjson-through-a-dict is the slower of the two.
    )
    app.state.settings = settings
    # Added innermost first, so the order a request meets them is the reverse: correlation
    # ids wrap everything, the rate limit sits inside them, and compression sits closest to
    # the route - it acts on what the route produced, not on a 413 or a 429 envelope.
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=settings.service.rate_limit_per_minute,
        burst=constants.RATE_LIMIT_BURST,
    )
    app.add_middleware(CorrelationMiddleware, max_body_bytes=constants.MAX_BODY_BYTES)
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
