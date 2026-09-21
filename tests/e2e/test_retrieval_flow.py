"""End-to-end: upload -> parse -> index -> /v1/recall and /v1/context over HTTP and the SDK."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import sdk_client

pytestmark = pytest.mark.e2e
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
Q = "Why did Adjusted EBITDA increase despite lower revenue?"


def _scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def _upload(client, scope: dict[str, str]) -> str:
    msg = client.post(
        "/v1/messages",
        headers=H,
        json={"scope": scope, "role": "USER", "content": "here is the FY26 report"},
    ).json()
    r = client.post(
        "/v1/files",
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

    r = client.post("/v1/recall", headers=H, json={"scope": scope, "query": Q, "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_type"] == "DOCUMENT_MULTI_HOP"
    # Reranking is off by default: measured significantly worse than fusion alone on this
    # corpus (p=0.012, docs/MEASUREMENTS.md). The engine only sets this diagnostic when it
    # actually reranks, so its *absence* is the contract now — asserting that here is what
    # catches the flag being switched back on without a measurement to justify it.
    assert "reranked" not in body["diagnostics"]
    assert body["diagnostics"]["fused_candidates"] > 0
    assert 0 < len(body["results"]) <= 5
    top = body["results"][0]
    assert top["document_id"] == doc_id and top["page"] == 11
    assert "increased to EUR 98" in top["text"]
    assert top["citation"] == f"chunk_id:{top['item_id']}" and top["representation"] == "CHUNK"
    assert top["evidence"][0]["chunk_id"] == top["item_id"]
    assert set(top["retrievers"]) <= {"fusion", "dense", "bm25", "exact"}

    # exact identifier round-trip through the public API
    r = client.post(
        "/v1/recall", headers=H, json={"scope": scope, "query": f"open {top['item_id']}"}
    )
    assert r.json()["query_type"] == "EXACT_IDENTIFIER"
    assert [x["item_id"] for x in r.json()["results"]] == [top["item_id"]]

    r = client.post(
        "/v1/context", headers=H, json={"scope": scope, "query": Q, "token_budget": 3000}
    )
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert bundle["cache_hit"] is False and bundle["token_estimate"] <= 3000
    assert bundle["conversation"]["thread_id"] == scope["thread_id"]
    assert "here is the FY26 report" in bundle["conversation"]["rendered"]
    assert bundle["knowledge"][0]["item_id"] == top["item_id"]
    assert bundle["evidence"]["status"] == "COMPLETE"
    assert {"defined_by:Adjusted EBITDA", "footnote:3", "cross_reference:Section 8"} <= set(
        bundle["evidence"]["required_groups"]
    )
    assert bundle["summaries"] and "## Summaries" in bundle["rendered"]
    assert bundle["conversation"]["summary"] is None  # everything fits the window
    # recall exposes the same report; an unrelated question is INSUFFICIENT
    r = client.post("/v1/recall", headers=H, json={"scope": scope, "query": Q})
    assert r.json()["evidence"]["status"] == "COMPLETE"
    r = client.post(
        "/v1/context",
        headers=H,
        json={"scope": scope, "query": "Who won the 1998 football championship?"},
    )
    assert r.json()["evidence"]["status"] == "INSUFFICIENT"
    assert "## Evidence status\nINSUFFICIENT" in r.json()["rendered"]
    assert (
        "## Recent conversation" in bundle["rendered"]
        and "increased to EUR 98" in bundle["rendered"]
    )
    again = client.post(
        "/v1/context", headers=H, json={"scope": scope, "query": Q, "token_budget": 3000}
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
        "/v1/recall", headers={**H, "X-Memory-User": "u2"}, json={"scope": {}, "query": Q}
    )
    assert other.status_code == 200 and other.json()["results"] == []
    # another tenant: nothing
    stranger = client.post(
        "/v1/recall", headers={**H, "X-Memory-Tenant": "globex"}, json={"scope": {}, "query": Q}
    )
    assert stranger.status_code == 200 and stranger.json()["results"] == []


async def test_sdk_context_and_recall(app, client) -> None:
    scope = _scope()
    doc_id = _upload(client, scope)
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **scope)
    bundle = await ctx.context(Q, token_budget=4000)
    assert bundle.query_type == "DOCUMENT_MULTI_HOP" and bundle.knowledge
    assert bundle.knowledge[0].document_id == doc_id and bundle.knowledge[0].page == 11
    assert bundle.evidence.status == "COMPLETE" and bundle.token_estimate <= 4000
    assert "increased to EUR 98" in bundle.rendered
    assert bundle.evidence.required_groups and not bundle.evidence.missing_groups
    from universal_memory import InsufficientEvidence

    with pytest.raises(InsufficientEvidence) as exc:
        await ctx.context("Who won the 1998 football championship?", require_evidence=True)
    assert exc.value.code == "INSUFFICIENT_EVIDENCE"
    lenient = await ctx.context("Who won the 1998 football championship?")
    assert lenient.evidence.status == "INSUFFICIENT"
    items = await ctx.recall("restructuring programme headcount", limit=3)
    assert 0 < len(items) <= 3 and items[0].citation.startswith("chunk_id:")
    assert any("headcount" in i.text for i in items)
    # a child agent inherits the user's access to the thread-scoped document
    agent = ctx.agent("analyst", agent_run_id=new_id("agent_run"))
    assert (await agent.recall("Adjusted EBITDA", limit=2))[0].document_id == doc_id
    await memory.aclose()
