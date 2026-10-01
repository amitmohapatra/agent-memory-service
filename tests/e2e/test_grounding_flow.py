"""End-to-end: upload -> /v1/context -> /v1/verify by bundle id, over HTTP and the SDK;
evidence never leaks across principals."""

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
GOOD = "Adjusted EBITDA increased to EUR 98 million from EUR 81 million, despite lower revenue."
BAD = "Adjusted EBITDA increased to EUR 150 million from EUR 81 million."


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
    return r.json()["document_id"]


def test_verify_over_http(client) -> None:
    scope = _scope()
    _upload(client, scope)
    # the bundle carries a handle and what was retrieved but not packed
    r = client.post("/v1/context", headers=H, json={"scope": scope, "query": Q, "format": "full"})
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert bundle["bundle_id"] and isinstance(bundle["evidence"]["unused"], list)
    prompt = client.post("/v1/context", headers=H, json={"scope": scope, "query": Q}).json()
    assert prompt["bundle_id"] == bundle["bundle_id"] and prompt["rendered"]

    # the answer is verified against the bundle it was given
    r = client.post(
        "/v1/verify",
        headers=H,
        json={"scope": scope, "answer": f"{GOOD} {BAD}", "bundle_id": bundle["bundle_id"]},
    )
    assert r.status_code == 200, r.text
    report = r.json()
    assert [c["verdict"] for c in report["claims"]] == ["supported", "contradicted"]
    assert report["per_claim_hallucination_rate"] == 0.5
    assert report["representative"] is False and report["nli_provider"] == "lexical-nli-v2"
    assert report["claims"][0]["evidence_ids"][0] == bundle["knowledge"][0]["item_id"]
    assert "X-Trellis-LLM-Tokens" not in r.headers  # no LLM configured: nothing to account
    good = client.post(
        "/v1/verify",
        headers=H,
        json={"scope": scope, "answer": GOOD, "bundle_id": bundle["bundle_id"]},
    )
    assert good.status_code == 200 and good.json()["per_claim_hallucination_rate"] == 0.0

    # validation: a bundle is required; unknown bundle; problem details
    for payload in (
        {"scope": scope, "answer": GOOD},
        {"scope": scope, "answer": "", "bundle_id": bundle["bundle_id"]},
        {"scope": scope, "answer": GOOD, "bundle_id": bundle["bundle_id"], "query": Q},
    ):
        r = client.post("/v1/verify", headers=H, json=payload)
        assert r.status_code == 422 and r.json()["code"] == "VALIDATION", payload
    r = client.post(
        "/v1/verify", headers=H, json={"scope": scope, "answer": GOOD, "bundle_id": "nope"}
    )
    assert r.status_code == 404 and r.json()["code"] == "NOT_FOUND"
    unauthenticated = client.post(
        "/v1/verify", json={"scope": scope, "answer": GOOD, "bundle_id": bundle["bundle_id"]}
    )
    assert unauthenticated.status_code == 401

    # nor by the bundle handle of the other tenant
    r = client.post(
        "/v1/verify",
        headers={**H, "X-Trellis-Tenant": "globex"},
        json={"scope": {}, "answer": GOOD, "bundle_id": bundle["bundle_id"]},
    )
    assert r.status_code == 404


async def test_verify_through_the_sdk(app, client) -> None:
    scope = _scope()
    _upload(client, scope)
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **scope)
    bundle = await ctx.context(Q, format="full")
    assert bundle.bundle_id
    report = await ctx.verify(f"{GOOD} {BAD}", bundle_id=bundle.bundle_id)
    assert [c.verdict for c in report.claims] == ["supported", "contradicted"]
    assert report.per_claim_hallucination_rate == 0.5 and report.grounded is False
    assert report.claims[1].evidence_ids == [bundle.knowledge[0].item_id]
    good = await ctx.verify(GOOD, bundle_id=bundle.bundle_id)
    assert good.grounded and good.evidence_count > 0
    await memory.aclose()
