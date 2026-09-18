"""Shared fixtures. Real backing stores are used when reachable; otherwise tests that need
them are skipped with a clear reason (never silently passed)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.config.settings import Settings, reset_settings_cache


def _test_settings(**overrides: object) -> Settings:
    base = {
        "service": {"environment": "test", "log_json": False, "log_level": "WARNING"},
        "authentication": {"mode": "trusted_dev", "trusted_dev_api_keys": ["test-key"]},
        "authorization": {"provider": "memory"},
        "cache": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
        "tasks": {"provider": "inline"},
        "models": {
            "embedding": {"provider": "hash", "dimension": 64},
            "reranker": {"provider": "lexical"},
            "nli": {"provider": "lexical"},
            "llm": {"enabled": False},
        },
        "documents": {"parser": "builtin"},
        "observability": {"otel_enabled": False},
        "database": {
            "url": os.environ.get(
                "MEMORY__DATABASE__URL", "postgresql+psycopg://memory:memory@localhost:5432/memory"
            )
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}  # type: ignore[dict-item]
        else:
            base[key] = value
    if os.environ.get("MEMORY_TEST_PROVIDERS") == "env":
        # Real-component runs: the environment's providers (weights, Qdrant server, cache,
        # OpenFGA, parser, LLM gateway) replace the hermetic stand-ins for these sections
        # only; tasks/blob stay test-local so drain() and tmp_path semantics hold.
        env_only = Settings().model_dump(exclude_unset=True)
        for section in ("models", "search", "cache", "authorization", "documents", "retrieval"):
            if isinstance(env_only.get(section), dict):
                base[section] = _deep_merge(base.get(section, {}), env_only[section])  # type: ignore[arg-type]
    return Settings(**base)  # type: ignore[arg-type]


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


#: The suite gets a database of its own.
#:
#: Tests truncate tables and hold locks; the dev stack's worker polls the same database and
#: claims jobs out from under them. Sharing "memory" with a running `docker compose` made the
#: suite fail in ways that had nothing to do with the code: a multi-agent test that runs in
#: 1.2 s idle timed out at 120 s waiting on a lock, and queue tests had their jobs stolen.
#: Point at the dev database explicitly with MEMORY__DATABASE__URL if that is what you want.
DB_URL = os.environ.get(
    "MEMORY__DATABASE__URL", "postgresql+psycopg://memory:memory@localhost:5432/memory_tests"
)


#: Where to connect to create the databases above; "postgres" always exists.
ADMIN_URL = os.environ.get(
    "MEMORY_TEST_ADMIN_URL", "postgresql://memory:memory@localhost:5432/postgres"
)


def pg_reachable() -> bool:
    """Whether the *server* is up. The suite's own databases are created on demand."""
    import psycopg

    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=2):
            return True
    except Exception:
        return False


PG_AVAILABLE = pg_reachable()

#: Queue tests get a database of their own.
#:
#: Procrastinate jobs are claimed by whichever worker polls first, so a queue test sharing
#: the application database with a running ``docker compose`` stack has its jobs stolen by
#: the dev worker — which does not know ``test.echo`` and fails it instantly. The tests then
#: pass only while the stack is *down*, which is the wrong way round. A separate database
#: no worker is pointed at makes them independent of what happens to be running.
QUEUE_DB_NAME = os.environ.get("MEMORY_TEST_QUEUE_DB", "memory_queue_tests")
QUEUE_DB_URL = DB_URL.rsplit("/", 1)[0] + "/" + QUEUE_DB_NAME


#: The same reasoning applies to the one test that kills a worker mid-job: it needs the
#: application tables *and* an uncontested queue, so it gets a full database of its own.
APP_DB_NAME = os.environ.get("MEMORY_TEST_APP_DB", "memory_failure_tests")
APP_DB_URL = DB_URL.rsplit("/", 1)[0] + "/" + APP_DB_NAME


def _create_database(name: str) -> None:
    """Create a database if it is not there yet. Safe to call from every session."""
    import psycopg

    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session")
def isolated_app_database() -> str:
    """A fully migrated application database no other process is polling."""
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL is not reachable")
    import asyncio

    from alembic import command
    from alembic.config import Config

    from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

    _create_database(APP_DB_NAME)
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", APP_DB_URL)
    command.upgrade(cfg, "head")

    async def _schema() -> None:
        q = ProcrastinateTaskQueue(APP_DB_URL.replace("postgresql+psycopg://", "postgresql://"))
        try:
            await q.ensure_schema()
        finally:
            await q.close()

    asyncio.run(_schema())
    return APP_DB_URL


@pytest.fixture(scope="session")
def queue_database() -> str:
    """A Procrastinate schema in a database of this suite's own. Returns its DSN."""
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL is not reachable")
    import asyncio

    from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

    _create_database(QUEUE_DB_NAME)
    dsn = QUEUE_DB_URL.replace("postgresql+psycopg://", "postgresql://")

    async def _schema() -> None:
        q = ProcrastinateTaskQueue(dsn)
        try:
            await q.ensure_schema()
        finally:
            await q.close()

    asyncio.run(_schema())
    return dsn


@pytest.fixture(scope="session", autouse=True)
def _migrated_database() -> None:
    """Apply Alembic + Procrastinate schemas once per session when PostgreSQL is reachable."""
    if not PG_AVAILABLE:
        return
    import asyncio

    from alembic import command
    from alembic.config import Config

    _create_database(DB_URL.rsplit("/", 1)[-1])
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", DB_URL)
    command.upgrade(cfg, "head")
    from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

    async def _schema() -> None:
        q = ProcrastinateTaskQueue(DB_URL.replace("postgresql+psycopg://", "postgresql://"))
        try:
            await q.ensure_schema()
        finally:
            await q.close()

    asyncio.run(_schema())


@pytest.fixture(autouse=True)
def _reset_settings() -> Iterator[None]:
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return _test_settings()


@pytest.fixture
def make_settings():
    return _test_settings


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
async def aclient(settings: Settings) -> AsyncIterator[object]:
    import httpx

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
