"""Async SQLAlchemy engine + session factory (psycopg3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memory_service.config.settings import DatabaseSettings
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self.engine: AsyncEngine = create_async_engine(
            settings.url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_pre_ping=True,
            echo=settings.echo,
            connect_args={"options": f"-c statement_timeout={settings.statement_timeout_ms}"},
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
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self.engine.dispose()
