"""Integration fixtures: real PostgreSQL (migrated) and Redis when reachable, else skip."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from memory_service.__about__ import __version__
from memory_service.application.container import Container, build_container
from memory_service.config.settings import Settings

DB_URL = os.environ.get(
    "MEMORY__DATABASE__URL", "postgresql+psycopg://memory:memory@localhost:5432/memory"
)
REDIS_URL = os.environ.get("MEMORY__CACHE__URL", "redis://localhost:6379/0")

TABLES = [
    "graph_relations",
    "graph_entities",
    "memories",
    "context_edges",
    "chunks",
    "document_nodes",
    "document_versions",
    "file_staging",
    "documents",
    "archive_segments",
    "job_outbox",
    "idempotency_keys",
    "revisions",
    "observations",
    "turn_run_links",
    "agent_runs",
    "message_attachments",
    "message_versions",
    "messages",
    "turns",
    "sessions",
    "threads",
]


def _pg_reachable() -> bool:
    import psycopg

    try:
        with psycopg.connect(
            DB_URL.replace("postgresql+psycopg://", "postgresql://"), connect_timeout=2
        ):
            return True
    except Exception:
        return False


def _redis_reachable() -> bool:
    import redis

    try:
        return bool(redis.from_url(REDIS_URL, socket_connect_timeout=1).ping())
    except Exception:
        return False


from tests.conftest import PG_AVAILABLE  # noqa: E402
from tests.support_real import reset_real_backends  # noqa: E402

REDIS_AVAILABLE = _redis_reachable()

requires_pg = pytest.mark.skipif(
    not PG_AVAILABLE, reason="PostgreSQL not reachable at MEMORY__DATABASE__URL"
)
requires_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="Redis/Dragonfly not reachable at MEMORY__CACHE__URL"
)


def integration_settings(make_settings, **overrides):
    base = {
        "database": {"url": DB_URL},
        "tasks": {"provider": "memory"},
        "cache": {"provider": "memory"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return make_settings(**base)


@pytest.fixture
async def container(make_settings, tmp_path) -> AsyncIterator[Container]:
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings: Settings = integration_settings(
        make_settings, blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")}
    )
    c = await build_container(settings, __version__)
    async with c.database.engine.begin() as conn:
        await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
        await conn.execute(
            text("TRUNCATE procrastinate_jobs, procrastinate_events RESTART IDENTITY CASCADE")
        )
    await reset_real_backends(c)
    try:
        yield c
    finally:
        await c.close()


@pytest.fixture
def uow_factory(container: Container):
    return container.services["uow_factory"]
