"""End-to-end: /v1/graph/query over HTTP and the SDK, and graph facts inside /v1/context."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import sdk_client

pytestmark = pytest.mark.e2e
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


def _scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def test_graph_query_and_context_facts(client) -> None:
    scope = _scope()
    msg = client.post(
        "/v1/messages",
        headers=H,
        json={"scope": scope, "role": "USER", "content": "report attached"},
    ).json()
    r = client.post(
        "/v1/files",
        headers=H,
        files={"file": ("acme_fy26_annual_report.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope), "message_id": msg["message_id"], "title": "ACME FY26"},
    )
    assert r.status_code == 202, r.text
    doc_id = r.json()["document_id"]
    r = client.post(
        "/v1/graph/query",
        headers=H,
        json={"scope": scope, "entities": ["Adjusted EBITDA"], "hops": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [m["canonical_name"] for m in body["matched"]] == ["adjusted ebitda"]
    assert body["visited"] > 1 and body["facts"]
    preds = {f["predicate"] for f in body["facts"]}
    assert {"defined_in", "mentioned_in", "co_occurs_with"} <= preds
    defined = next(f for f in body["facts"] if f["predicate"] == "defined_in")
    assert defined["document_id"] == doc_id and defined["evidence"][0]["page"] == 1
    assert defined["status"] == "CURRENT" and defined["relation_id"].startswith("rel_")
    # free-text resolution
    r = client.post(
        "/v1/graph/query",
        headers=H,
        json={"scope": scope, "query": "what is linked to the restructuring programme?"},
    )
    assert any(m["canonical_name"] == "restructuring programme" for m in r.json()["matched"])
    # other user: nothing (thread-scoped document)
    r = client.post(
        "/v1/graph/query",
        headers={**H, "X-Memory-User": "u2"},
        json={"scope": {}, "entities": ["Adjusted EBITDA"]},
    )
    assert r.status_code == 200 and r.json()["facts"] == [] and r.json()["matched"] == []
    # context bundle carries facts with relation citations and rendered "## Facts"
    bundle = client.post(
        "/v1/context",
        headers=H,
        json={"scope": scope, "query": "Why did Adjusted EBITDA increase despite lower revenue?"},
    ).json()
    assert bundle["graph_facts"] and bundle["graph_facts"][0]["citation"].startswith("relation_id:")
    assert "## Facts" in bundle["rendered"]
    pages = {k.get("page") for k in bundle["knowledge"]}
    assert {11, 14, 20} <= pages, pages
    assert (
        client.post("/v1/graph/query", headers=H, json={"scope": scope, "hops": 9}).status_code
        == 422
    )
    assert client.post("/v1/graph/query", json={"scope": scope}).status_code == 401


async def test_sdk_graph_temporal(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **_scope())
    await ctx.chat.user("kick-off")
    await ctx.observe("I work at ACME Corp.")
    answer = await ctx.graph.query(entities=["ACME Corp"])
    fact = next(f for f in answer.facts if f.predicate == "works_at")
    assert fact.subject == "user:u1" and fact.object.lower() == "acme corp" and fact.memory_id
    await ctx.observe("I work at Globex now.")
    now = await ctx.graph.query(entities=["ACME Corp", "Globex"])
    current = [f for f in now.facts if f.predicate == "works_at"]
    assert len(current) == 1 and current[0].object.lower() == "globex"
    past = await ctx.graph.query(
        entities=["ACME Corp"], as_of=datetime.now(UTC) - timedelta(seconds=30)
    )
    old = [f for f in past.facts if f.predicate == "works_at"]
    assert old and old[0].status == "SUPERSEDED" and old[0].object.lower() == "acme corp"
    await memory.aclose()
