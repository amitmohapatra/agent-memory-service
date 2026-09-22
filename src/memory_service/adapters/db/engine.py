"""Async SQLAlchemy engine + session factory (psycopg3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
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


class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self.engine: AsyncEngine = create_async_engine(
            settings.dsn,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=DATABASE.pool_timeout_seconds,
            pool_pre_ping=True,
            connect_args={
                "options": f"-c statement_timeout={DATABASE.statement_timeout_ms}",
                "connect_timeout": DATABASE.connect_timeout_seconds,
            },
        )
        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
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
        server that accepts the query and never answers.
        """
        budget = DATABASE.connect_timeout_seconds + DATABASE.pool_timeout_seconds
        try:
            async with asyncio.timeout(budget):
                async with self.engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # TimeoutError included; a probe never raises
            return False

    async def close(self) -> None:
        await self.engine.dispose()
