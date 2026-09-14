from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from memory_service.api.app import create_app
from tests.conftest import PG_AVAILABLE

TABLES = [
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


@pytest.fixture
def app(make_settings, tmp_path):
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        tasks={"provider": "inline"},
        blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
    )
    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        container = app.state.container
        import asyncio

        async def _truncate() -> None:
            async with container.database.engine.begin() as conn:
                await conn.execute(
                    text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
                )

        c.portal.call(_truncate) if hasattr(c, "portal") else asyncio.run(_truncate())
        yield c


@pytest.fixture
def container(app, client):
    return app.state.container


def sdk_client(app, *, api_key: str = "test-key"):
    from universal_memory import MemoryClient

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://memory.test")
    return MemoryClient("http://memory.test", api_key=api_key, http_client=http)
