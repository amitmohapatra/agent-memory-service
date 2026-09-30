"""End-to-end: upload -> parse -> index -> /v1/recall and /v1/context over HTTP and the SDK."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import post_message, sdk_client

pytestmark = pytest.mark.e2e
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
Q = "Why did Adjusted EBITDA increase despite lower revenue?"


def _scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def _upload(client, scope: dict[str, str]) -> str:
    msg = post_message(
        client, H, {"scope": scope, "role": "USER", "content": "here is the FY26 report"}
    ).json()
    r = client.post(
        "/v1/documents",
        headers=H,
        files={"file": ("acme_fy26_annual_report.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope), "message_id": msg["message_id"], "title": "ACME FY26"},
    )
    assert r.status_code == 202, r.text
    ack = r.json()
    # inline task queue: parse + chained index already ran before the response
    for job_id in ack["job_ids"]:
        assert client.get(f"/v1/jobs/{job_id}", headers=H).json()["status"] == "SUCCEEDED"
    return ack["document_id"]


def test_recall_and_context_over_http(client) -> None:
    scope = _scope()
    doc_id = _upload(client, scope)

    r = client.post(
        "/v1/recall", headers=H, json={"scope": scope, "query": Q, "limit": 5, "debug": True}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_type"] == "DOCUMENT_MULTI_HOP"
    assert body["diagnostics"]["fused_candidates"] > 0
    assert 0 < len(body["items"]) <= 5
    top = body["items"][0]
    assert top["document_id"] == doc_id and top["page"] == 11
    assert "increased to EUR 98" in top["text"]
    assert top["citation"] == f"chunk_id:{top['id']}" and top["kind"] == "chunk"
    assert top["debug"]["representation"] == "CHUNK"
    assert set(top["debug"]["retrievers"]) <= {"fusion", "dense_en", "dense_ml", "bm25", "exact"}

    # exact identifier round-trip through the public API
    r = client.post(
        "/v1/recall",
        headers=H,
        json={"scope": scope, "query": f"open {top['id']}", "debug": True},
    )
    assert r.json()["query_type"] == "EXACT_IDENTIFIER"
    assert [x["id"] for x in r.json()["items"]] == [top["id"]]

    r = client.post(
        "/v1/context",
        headers=H,
        json={"scope": scope, "query": Q, "token_budget": 3000, "format": "full"},
    )
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert bundle["cache_hit"] is False and bundle["token_estimate"] <= 3000
    assert bundle["conversation"]["thread_id"] == scope["thread_id"]
    assert "here is the FY26 report" in bundle["conversation"]["rendered"]
    assert bundle["knowledge"][0]["item_id"] == top["id"]
    assert bundle["evidence"]["status"] == "COMPLETE"
    assert {"defined_by:Adjusted EBITDA", "footnote:3", "cross_reference:Section 8"} <= {
        name.rsplit(":", 1)[0] for name in bundle["evidence"]["required_groups"]
    }
    assert bundle["summaries"] and "## Summaries" in bundle["rendered"]
    assert bundle["thread_summary"] is None  # a short thread has no durable summary yet
    # recall exposes the same report; an unrelated question is INSUFFICIENT
    r = client.post("/v1/recall", headers=H, json={"scope": scope, "query": Q})
    assert r.json()["evidence"]["status"] == "COMPLETE"
    r = client.post(
        "/v1/context",
        headers=H,
        json={
            "scope": scope,
            "query": "Who won the 1998 football championship?",
            "format": "full",
        },
    )
    assert r.json()["evidence"]["status"] == "INSUFFICIENT"
    assert "## Evidence status\nINSUFFICIENT" in r.json()["rendered"]
    assert (
        "## Recent conversation" in bundle["rendered"]
        and "increased to EUR 98" in bundle["rendered"]
    )
    again = client.post(
        "/v1/context",
        headers=H,
        json={"scope": scope, "query": Q, "token_budget": 3000, "format": "full"},
    ).json()
    assert again["cache_hit"] is True

    # validation and authorization
    assert (
        client.post("/v1/recall", headers=H, json={"scope": scope, "query": ""}).status_code == 422
    )
    assert (
        client.post(
            "/v1/context", headers=H, json={"scope": scope, "query": Q, "token_budget": 10}
        ).status_code
        == 422
    )
    assert client.post("/v1/recall", json={"scope": scope, "query": Q}).status_code == 401
    # another user in the same tenant: the thread-scoped document is invisible
    other = client.post(
        "/v1/recall", headers={**H, "X-Trellis-User": "u2"}, json={"scope": {}, "query": Q}
    )
    assert other.status_code == 200 and other.json()["items"] == []
    # another tenant: nothing
    stranger = client.post(
        "/v1/recall", headers={**H, "X-Trellis-Tenant": "globex"}, json={"scope": {}, "query": Q}
    )
    assert stranger.status_code == 200 and stranger.json()["items"] == []


async def test_sdk_context_and_recall(app, client) -> None:
    scope = _scope()
    doc_id = _upload(client, scope)
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **scope)
    bundle = await ctx.context(Q, token_budget=4000, format="full")
    assert bundle.query_type == "DOCUMENT_MULTI_HOP" and bundle.knowledge
    assert bundle.knowledge[0].document_id == doc_id and bundle.knowledge[0].page == 11
    assert bundle.evidence.status == "COMPLETE" and bundle.token_estimate <= 4000
    assert "increased to EUR 98" in bundle.rendered
    assert bundle.evidence.required_groups and not bundle.evidence.missing_groups
    lenient = await ctx.context("Who won the 1998 football championship?", format="full")
    assert lenient.evidence.status == "INSUFFICIENT"
    items = await ctx.search("restructuring programme headcount", limit=3)
    assert 0 < len(items) <= 3 and items[0].citation.startswith("chunk_id:")
    assert any("headcount" in i.text for i in items)
    # a child agent inherits the user's access to the thread-scoped document
    agent = ctx.agent("analyst", agent_run_id=new_id("agent_run"))
    assert (await agent.search("Adjusted EBITDA", limit=2))[0].document_id == doc_id
    await memory.aclose()
