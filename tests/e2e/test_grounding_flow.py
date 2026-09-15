"""End-to-end: upload -> /v1/context with an answer to verify -> /v1/verify by bundle id,
items and query, over HTTP and the SDK; evidence never leaks across principals."""

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
GOOD = "Adjusted EBITDA increased to EUR 98 million from EUR 81 million, despite lower revenue."
BAD = "Adjusted EBITDA increased to EUR 150 million from EUR 81 million."


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
    return r.json()["document_id"]


def test_verify_over_http(client) -> None:
    scope = _scope()
    _upload(client, scope)
    # the bundle carries a handle and what was retrieved but not packed
    r = client.post(
        "/v1/context", headers=H, json={"scope": scope, "query": Q, "answer": f"{GOOD} {BAD}"}
    )
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert bundle["bundle_id"] and bundle["evidence"]["llm_tokens"] == 0
    assert isinstance(bundle["evidence"]["unused"], list)
    grounding = bundle["evidence"]["grounding"]
    assert [c["verdict"] for c in grounding["claims"]] == ["supported", "contradicted"]
    assert grounding["per_claim_hallucination_rate"] == 0.5
    assert grounding["representative"] is False and grounding["nli_provider"] == "lexical-nli-v1"
    assert grounding["claims"][0]["evidence_ids"][0] == bundle["knowledge"][0]["item_id"]
    assert "X-Memory-LLM-Tokens" not in r.headers  # no LLM configured: nothing to account
    # the plain bundle is unchanged (the report is attached to the response only)
    plain = client.post("/v1/context", headers=H, json={"scope": scope, "query": Q}).json()
    assert plain["evidence"]["grounding"] is None and plain["bundle_id"] == bundle["bundle_id"]

    # by bundle id, while cached
    r = client.post(
        "/v1/verify",
        headers=H,
        json={"scope": scope, "answer": GOOD, "bundle_id": bundle["bundle_id"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "bundle" and r.json()["per_claim_hallucination_rate"] == 0.0
    assert r.json()["evidence_count"] == len(bundle["knowledge"]) + len(bundle["summaries"]) + len(
        bundle["memories"]
    ) + len(bundle["graph_facts"])
    # by query: retrieval is re-run under the caller's scope
    r = client.post("/v1/verify", headers=H, json={"scope": scope, "answer": BAD, "query": Q})
    assert r.status_code == 200 and r.json()["source"] == "query"
    assert r.json()["claims"][0]["verdict"] == "contradicted"
    # by items: evidence the caller holds, with a citation that must match
    items = [
        {"item_id": k["item_id"], "text": k["text"], "citation": k["citation"]}
        for k in bundle["knowledge"][:2]
    ]
    r = client.post(
        "/v1/verify",
        headers=H,
        json={"scope": scope, "answer": f"{GOOD[:-1]} [1].", "items": items},
    )
    assert r.status_code == 200 and r.json()["source"] == "items"
    assert r.json()["claims"][0]["verdict"] == "supported"
    r = client.post(
        "/v1/verify",
        headers=H,
        json={"scope": scope, "answer": f"{GOOD[:-1]} [9].", "items": items},
    )
    assert r.json()["claims"][0]["verdict"] == "unsupported"
    assert r.json()["claims"][0]["method"] == "citation"

    # validation: exactly one evidence source; unknown bundle; error envelope
    for payload in (
        {"scope": scope, "answer": GOOD},
        {"scope": scope, "answer": GOOD, "query": Q, "items": items},
        {"scope": scope, "answer": GOOD, "query": Q, "unused": items},
        {"scope": scope, "answer": "", "query": Q},
    ):
        r = client.post("/v1/verify", headers=H, json=payload)
        assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION", payload
    r = client.post(
        "/v1/verify", headers=H, json={"scope": scope, "answer": GOOD, "bundle_id": "nope"}
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    assert (
        client.post("/v1/verify", json={"scope": scope, "answer": GOOD, "query": Q}).status_code
        == 401
    )

    # another user cannot verify against this user's document: no evidence, nothing supported
    other = client.post(
        "/v1/verify",
        headers={**H, "X-Memory-User": "u2"},
        json={"scope": {}, "answer": GOOD, "query": Q},
    )
    assert other.status_code == 200 and other.json()["evidence_count"] == 0
    assert other.json()["claims"][0]["verdict"] == "unsupported"
    # nor by the bundle handle of the other tenant
    r = client.post(
        "/v1/verify",
        headers={**H, "X-Memory-Tenant": "globex"},
        json={"scope": {}, "answer": GOOD, "bundle_id": bundle["bundle_id"]},
    )
    assert r.status_code == 404


async def test_verify_through_the_sdk(app, client) -> None:
    scope = _scope()
    _upload(client, scope)
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **scope)
    bundle = await ctx.context(Q)
    assert bundle.bundle_id and bundle.grounding is None
    report = await ctx.verify(f"{GOOD} {BAD}", bundle=bundle)
    assert [c.verdict for c in report.claims] == ["supported", "contradicted"]
    assert report.per_claim_hallucination_rate == 0.5 and report.grounded is False
    assert report.claims[1].evidence_ids == [bundle.knowledge[0].item_id]
    by_query = await ctx.verify(GOOD, query=Q)
    assert by_query.grounded and by_query.evidence_count > 0
    by_items = await ctx.verify(f"{GOOD[:-1]} [1].", items=bundle.knowledge[:1])
    assert by_items.claims[0].verdict == "supported" and by_items.claims[0].citations == ["1"]
    with pytest.raises(ValueError, match="bundle, items or a query"):
        await ctx.verify(GOOD)
    await memory.aclose()
