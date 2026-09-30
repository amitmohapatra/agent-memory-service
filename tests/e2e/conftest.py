from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from memory_service.api.app import create_app
from tests.conftest import PG_AVAILABLE, _test_overrides
from tests.support_real import reset_real_backends

TABLES = [
    # the platform layer (tests/agent), first: no table below references them
    "memory_reads",
    # No foreign key to tenants, so TRUNCATE ... CASCADE never reached it and its rows
    # outlived the test that wrote them: every feedback listing carried the previous test's
    # verdicts.
    "feedback",
    # Registered model keys: a tombstone keeps climbing its revision, so a test asserting
    # "this agent registers its first key" saw revision 4 from a previous run's rows.
    "agent_credentials",
    "llm_policies",
    "llm_usage_daily",
    "workspace_members",
    "workspaces",
    "api_keys",
    "tenants",
    "tool_invocations",
    "tool_stats",
    "approval_patterns",
    "procedures",
    "profile_blocks",
    "thread_summaries",
    "agent_tool_pulls",
    "prefetch_stats",
    "run_outcomes",
    "tools",
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
    "agent_runs",
    "message_attachments",
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
        blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
    )
    # The stand-ins are named in code, never in the environment: an in-process queue so
    # the flows drain inline, the hash encoder and lexical models so no weights are loaded,
    # and the real filesystem blob store under tmp_path so uploads land on disk.
    return create_app(settings, overrides=_test_overrides(tasks="inline", blob=None))


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        container = app.state.container
        import asyncio

        async def _truncate() -> None:
            async with container.database.engine.begin() as conn:
                # a reset, not a hot path: on a loaded host it may outlast the statement timeout
                await conn.execute(text("SET LOCAL statement_timeout = 0"))
                await conn.execute(
                    text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
                )
            await reset_real_backends(container)

        c.portal.call(_truncate) if hasattr(c, "portal") else asyncio.run(_truncate())
        yield c


@pytest.fixture
def container(app, client):
    return app.state.container


def sdk_client(app, *, api_key: str = "test-key"):
    from trellis.memory import MemoryClient

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://memory.test")
    return MemoryClient("http://memory.test", api_key=api_key, http_client=http)


def message_body(body: dict) -> dict:
    """``{"scope", role, content, ...}`` as the ``/v1/messages`` body: a one-message list."""
    body = dict(body)
    return {"scope": body.pop("scope"), "messages": [body]}


class OneAck:
    """A ``/v1/messages`` response for one message: ``json()`` is that message's ack."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.text = response.text

    def json(self):
        data = self.response.json()
        return data["messages"][0] if self.status_code == 202 else data


def post_message(client, headers: dict, body: dict) -> OneAck:
    """POST one message (``{"scope", role, content, ...}``) to ``/v1/messages``."""
    return OneAck(client.post("/v1/messages", headers=headers, json=message_body(body)))
