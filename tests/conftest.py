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
    return Settings(**base)  # type: ignore[arg-type]


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
