"""Async SQLAlchemy engine + session factory (psycopg3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, exc, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memory_service.config.constants import DATABASE
from memory_service.config.settings import DatabaseSettings
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


def _is_dead(dbapi_connection: Any) -> bool:
    """Whether this pooled connection is already known to be unusable.

    psycopg knows its own connection is closed or broken without asking the server, so this
    is the part of ``pool_pre_ping`` that costs nothing: no round trip, no statement. What it
    cannot see is a connection the server, a proxy or an idle timeout closed while the pool
    held it and nobody has touched since - ``pool_recycle`` is what bounds that, by throwing
    a connection away before it gets old enough for anyone else to have closed it, which is
    why that window is shorter than the idle timeouts of the things that sit in between.
    """
    return bool(getattr(dbapi_connection, "closed", False)) or bool(
        getattr(dbapi_connection, "broken", False)
    )


class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self.engine: AsyncEngine = create_async_engine(
            settings.dsn,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=DATABASE.pool_timeout_seconds,
            # A pre-ping is a round trip on every checkout, and a request takes two or three
            # checkouts against a database that is on another host: at the target rate that
            # is milliseconds of pure latency to catch something a recycle window makes rare.
            pool_pre_ping=False,
            pool_recycle=DATABASE.pool_recycle_seconds,
            connect_args={
                "options": f"-c statement_timeout={DATABASE.statement_timeout_ms}",
                "connect_timeout": DATABASE.connect_timeout_seconds,
            },
        )

        @event.listens_for(self.engine.sync_engine, "checkout")
        def _discard_dead_connections(dbapi_connection, connection_record, connection_proxy):  # type: ignore[no-untyped-def]
            if _is_dead(dbapi_connection):
                # SQLAlchemy answers DisconnectionError here by discarding this connection
                # and checking out another one, once. That is the retry the pre-ping used to
                # buy, for the case that can be seen for free.
                raise exc.DisconnectionError("connection was closed while pooled")

        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        # Every statement carries the SQLAlchemy instrumentation's wrapper once this is
        # installed, whether or not anything is listening. Tracing is configured exactly when
        # an exporter is set (observability.otel_exporter != "none"), and that is what puts a
        # real TracerProvider in place of the API's no-op proxy, so the presence of one is
        # the same condition read where it can be seen from an adapter.
        if _tracing_is_on():
            try:
                from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

                SQLAlchemyInstrumentor().instrument(engine=self.engine.sync_engine)
            except Exception:
                log.debug("db.otel_instrumentation_unavailable")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def ping(self) -> bool:
        """Whether the database answers, within a bounded time.

        ``connect_timeout`` bounds opening a socket; this bounds the whole round trip,
        including a connection handed back from the pool that turns out to be dead and a
        server that accepts the query and never answers. A probe is idempotent, so a single
        operational failure - the dead-while-pooled connection, a server that has just come
        back - is retried once rather than reported as "not ready".
        """
        budget = DATABASE.connect_timeout_seconds + DATABASE.pool_timeout_seconds
        for attempt in (1, 2):
            try:
                async with asyncio.timeout(budget):
                    async with self.engine.connect() as conn:
                        await conn.execute(text("SELECT 1"))
                return True
            except exc.OperationalError:
                if attempt == 2:
                    return False
            except Exception:  # TimeoutError included; a probe never raises
                return False
        return False

    async def close(self) -> None:
        await self.engine.dispose()


def _tracing_is_on() -> bool:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    return isinstance(trace.get_tracer_provider(), TracerProvider)
