"""End-to-end: /v1/graph/entities over HTTP and the SDK, and graph facts inside /v1/context."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import post_message, sdk_client

pytestmark = pytest.mark.e2e
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def _scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def test_graph_query_and_context_facts(client) -> None:
    scope = _scope()
    msg = post_message(
        client, H, {"scope": scope, "role": "USER", "content": "report attached"}
    ).json()
    r = client.post(
        "/v1/documents",
        headers=H,
        files={"file": ("acme_fy26_annual_report.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope), "message_id": msg["message_id"], "title": "ACME FY26"},
    )
    assert r.status_code == 202, r.text
    doc_id = r.json()["document_id"]
    found = client.get("/v1/graph/entities", headers=H, params={"q": "Adjusted EBITDA"})
    assert found.status_code == 200, found.text
    matched = found.json()["entities"]
    assert matched and matched[0]["canonical_name"] == "adjusted ebitda"
    r = client.get(f"/v1/graph/entities/{matched[0]['entity_id']}", headers=H, params={"depth": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    facts = body["relations"] + body["neighborhood"]["facts"]
    assert body["neighborhood"]["visited"] > 1 and facts
    preds = {f["predicate"] for f in facts}
    assert {"defined_in", "mentioned_in", "co_occurs_with"} <= preds
    defined = next(f for f in facts if f["predicate"] == "defined_in")
    assert defined["document_id"] == doc_id and defined["evidence"][0]["page"] == 1
    assert defined["status"] == "CURRENT" and defined["relation_id"].startswith("rel_")
    # free-text resolution
    r = client.get(
        "/v1/graph/entities",
        headers=H,
        params={"q": "what is linked to the restructuring programme?"},
    )
    assert any(e["canonical_name"] == "restructuring programme" for e in r.json()["entities"])
    # other user: nothing (thread-scoped document)
    r = client.get(
        "/v1/graph/entities",
        headers={**H, "X-Trellis-User": "u2"},
        params={"q": "Adjusted EBITDA"},
    )
    assert r.status_code == 200 and r.json()["entities"] == []
    # context bundle carries facts with relation citations and rendered "## Facts"
    bundle = client.post(
        "/v1/context",
        headers=H,
        json={
            "scope": scope,
            "query": "Why did Adjusted EBITDA increase despite lower revenue?",
            "format": "full",
        },
    ).json()
    fact = bundle["graph_facts"][0]
    assert fact["id"] and fact["subject"] and fact["predicate"] and fact["object"]
    assert 0.0 <= fact["relevance"] <= 1.0 and "rendered" not in bundle
    prompt = client.post(
        "/v1/context",
        headers=H,
        json={"scope": scope, "query": "Why did Adjusted EBITDA increase despite lower revenue?"},
    ).json()
    assert "## Facts" in prompt["rendered"]
    pages = {k.get("page") for k in bundle["knowledge"]}
    assert {11, 14, 20} <= pages, pages
    # layer restriction: structural edges only
    r = client.get(
        f"/v1/graph/entities/{matched[0]['entity_id']}",
        headers=H,
        params={"depth": 2, "layers": ["structural"]},
    )
    assert r.status_code == 200, r.text
    hood = r.json()["neighborhood"]["facts"]
    assert hood and {f["layer"] for f in hood} == {"structural"}
    # entity search and profile over HTTP, scoped by the same headers
    hits = client.get("/v1/graph/entities", headers=H, params={"q": "adjusted", "limit": 5})
    assert hits.status_code == 200, hits.text
    ebitda = next(e for e in hits.json()["entities"] if e["canonical_name"] == "adjusted ebitda")
    assert ebitda["summary"].startswith("Adjusted EBITDA")
    profile = client.get(f"/v1/graph/entities/{ebitda['entity_id']}", headers=H)
    assert profile.status_code == 200, profile.text
    assert profile.json()["relations"] and profile.json()["evidence"]
    other = {**H, "X-Trellis-User": "u2"}
    assert client.get(f"/v1/graph/entities/{ebitda['entity_id']}", headers=other).status_code == 404
    assert client.get("/v1/graph/entities", headers=other, params={"q": "adjusted"}).json() == {
        "entities": []
    }
    deep = client.get(f"/v1/graph/entities/{ebitda['entity_id']}", headers=H, params={"depth": 9})
    assert deep.status_code == 422
    assert client.get("/v1/graph/entities", params={"q": "adjusted"}).status_code == 401


async def test_sdk_graph_temporal(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **_scope())
    await ctx.history.add([("USER", "kick-off")])
    await ctx.history.add([("USER", "I work at ACME Corp.")])
    graph = ctx.advanced.graph
    [acme] = [e for e in await graph.entities("ACME Corp") if e.canonical_name == "acme corp"]
    answer = await graph.entity(acme.entity_id, depth=1)
    fact = next(f for f in answer.relations if f.predicate == "works_at")
    assert fact.object.lower() == "acme corp" and fact.memory_id
    await ctx.history.add([("USER", "I work at Globex now.")])
    [globex] = [e for e in await graph.entities("Globex") if e.canonical_name == "globex"]
    current = [
        f
        for entity in (acme, globex)
        for f in (await graph.entity(entity.entity_id)).relations
        if f.predicate == "works_at"
    ]
    assert len(current) == 1 and current[0].object.lower() == "globex"
    past = await graph.entity(
        acme.entity_id, depth=1, as_of=datetime.now(UTC) - timedelta(seconds=30)
    )
    old = [f for f in past.neighborhood.facts if f.predicate == "works_at"]
    assert old and old[0].status == "SUPERSEDED" and old[0].object.lower() == "acme corp"
    await memory.aclose()
