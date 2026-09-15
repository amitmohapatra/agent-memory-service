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


DB_URL = os.environ.get(
    "MEMORY__DATABASE__URL", "postgresql+psycopg://memory:memory@localhost:5432/memory"
)


def pg_reachable() -> bool:
    import psycopg

    try:
        with psycopg.connect(
            DB_URL.replace("postgresql+psycopg://", "postgresql://"), connect_timeout=2
        ):
            return True
    except Exception:
        return False


PG_AVAILABLE = pg_reachable()


@pytest.fixture(scope="session", autouse=True)
def _migrated_database() -> None:
    """Apply Alembic + Procrastinate schemas once per session when PostgreSQL is reachable."""
    if not PG_AVAILABLE:
        return
    import asyncio

    from alembic import command
    from alembic.config import Config

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
