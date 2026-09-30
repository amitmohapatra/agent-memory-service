"""The per-turn verbs on a bound context: what each one sends and what it returns."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from trellis.memory import MemoryClient, ToolHints

BASE = "http://memory.test"


@pytest.fixture
def ctx():
    client = MemoryClient(BASE, api_key="k", max_retries=0)
    return client.bind(tenant_id="acme", user_id="u1", thread_id="thr_1").agent(
        "buyer", agent_run_id="run_1"
    )


def _body(route: respx.Route) -> dict:
    return json.loads(route.calls.last.request.content)


_BUNDLE = {
    "query": "q",
    "query_type": "GENERAL_SEMANTIC",
    "conversation": {"thread_id": "thr_1"},
    "evidence": {"status": "COMPLETE"},
    "token_budget": 2000,
    "token_estimate": 10,
    "rendered": "## Profile\n...",
    "revision": 42,
    "profile": [{"block": "user", "text": "Prefers email.", "version": 3}],
    "thread_summary": {"text": "Asked for a PO.", "covers_to_sequence": 20, "version": 1},
    "procedures": [{"id": "prc_1", "title": "Order", "steps": [], "success_rate": 1.0}],
    "tools": {"candidates": [{"name": "erp-create_po", "score": 0.9}], "next": "erp-create_po"},
}


@respx.mock
async def test_context_asks_for_tools_and_a_delta_and_reads_every_section(ctx) -> None:
    route = respx.post(f"{BASE}/v1/context").respond(200, json=_BUNDLE)
    bundle = await ctx.context(
        "order paper", token_budget=2000, tools={"available": ["erp-create_po"]}, since_revision=7
    )
    body = _body(route)
    assert body["tools"] == {"available": ["erp-create_po"], "k": 8}
    assert body["since_revision"] == 7 and body["token_budget"] == 2000
    assert bundle.revision == 42 and bundle.profile[0].block == "user"
    assert bundle.thread_summary is not None and bundle.thread_summary.covers_to_sequence == 20
    assert bundle.procedures[0].id == "prc_1"
    assert isinstance(bundle.tools, ToolHints) and bundle.tools.next == "erp-create_po"


@respx.mock
async def test_tool_hints_names_the_available_tools(ctx) -> None:
    route = respx.post(f"{BASE}/v1/tools/hints").respond(
        200,
        json={
            "candidates": [{"name": "erp-create_po", "score": 0.8, "why": "worked before"}],
            "prefill": {"supplier": {"tool": "erp-create_po", "value": "Acme", "source": "graph"}},
            "missing": [{"tool": "erp-create_po", "arg": "qty", "question": "How many?"}],
        },
    )
    hints = await ctx.tool_hints("order paper", available=["erp-create_po"], k=3)
    assert _body(route)["available"] == ["erp-create_po"] and _body(route)["k"] == 3
    assert hints.prefill["supplier"].value == "Acme" and hints.missing[0].arg == "qty"


@respx.mock
async def test_agent_tools_are_listed_and_called_in_scope(ctx) -> None:
    respx.get(f"{BASE}/v1/agent-tools").respond(
        200,
        json={"tools": [{"name": "memory_search", "description": "d", "input_schema": {}}]},
    )
    call = respx.post(f"{BASE}/v1/agent-tools/memory_search").respond(
        200, json={"result": [{"id": "mem_1"}]}
    )
    tools = await ctx.agent_tools()
    assert [t.name for t in tools] == ["memory_search"]
    assert await ctx.call_agent_tool("memory_search", {"query": "paper"}) == [{"id": "mem_1"}]
    body = _body(call)
    assert body["args"] == {"query": "paper"} and body["scope"]["agent_run_id"] == "run_1"


@respx.mock
async def test_record_tool_and_outcome_are_the_run_s(ctx) -> None:
    record = respx.post(f"{BASE}/v1/tools/invocations").respond(
        202, json={"invocation_id": "tiv_1", "step": 0, "args_hash": "h", "recorded": True}
    )
    outcome = respx.post(f"{BASE}/v1/runs/run_1/outcome").respond(
        200, json={"run_id": "run_1", "success": True, "source": "explicit"}
    )
    await ctx.record_tool("erp-get_stock", {"sku": "A4"}, status="cancelled", task=None, step=0)
    assert _body(record)["status"] == "cancelled" and _body(record)["task"] == ""
    assert (await ctx.outcome(success=True, note="ok")).source == "explicit"
    assert outcome.called
    with pytest.raises(ValueError, match="needs a run"):
        await ctx.derive(agent_run_id=None).outcome(success=True)


@respx.mock
async def test_profile_is_listed_set_and_edited(ctx) -> None:
    block = {"block": "user", "text": "Prefers email.", "version": 2}
    respx.get(f"{BASE}/v1/profile").respond(200, json={"blocks": [block]})
    put = respx.put(f"{BASE}/v1/profile/user").respond(200, json=block)
    patch = respx.patch(f"{BASE}/v1/profile/user").respond(200, json={**block, "version": 3})
    assert (await ctx.profile())[0].text == "Prefers email."
    await ctx.profile.set("user", "Prefers email.")
    assert _body(put)["text"] == "Prefers email."
    edited = await ctx.profile.edit("user", "email", "phone")
    assert _body(patch)["old"] == "email" and edited.version == 3


@respx.mock
async def test_summary_is_none_until_the_thread_has_one(ctx) -> None:
    route = respx.get(f"{BASE}/v1/threads/thr_1/summary")
    route.side_effect = [
        httpx.Response(404, json={"code": "NOT_FOUND", "detail": "no summary", "status": 404}),
        httpx.Response(200, json={"text": "s", "covers_to_sequence": 20, "version": 1}),
    ]
    assert await ctx.summary() is None
    summary = await ctx.summary()
    assert summary is not None and summary.version == 1
    assert await ctx.derive(thread_id=None).summary() is None


class _ContractsFeedback(BaseModel):
    """The shape of ``trellis.contracts.Feedback``, as the harness hands it over."""

    feedback_id: str = "fb_1"
    tenant_id: str = "acme"
    target_kind: str = "tool_call"
    target_id: str = "call_1"
    verdict: str = "approve"
    metadata: dict = {"tool": "erp-create_po"}


@respx.mock
async def test_feedback_takes_a_contracts_record_or_its_fields(ctx) -> None:
    stored = {
        "feedback_id": "fb_1",
        "tenant_id": "acme",
        "target_kind": "tool_call",
        "target_id": "call_1",
        "verdict": "approve",
        "created_at": "2026-09-30T10:00:00Z",
    }
    route = respx.post(f"{BASE}/v1/feedback").respond(201, json=stored)
    listed = respx.get(f"{BASE}/v1/feedback").respond(200, json={"feedback": [stored]})
    await ctx.feedback(_ContractsFeedback())
    assert _body(route)["metadata"] == {"tool": "erp-create_po"}
    await ctx.feedback("run", "run_1", "confirm", score=0.9)
    assert _body(route)["agent_run_id"] == "run_1" and _body(route)["score"] == 0.9
    assert [f.feedback_id for f in await ctx.feedback.list_for("tool_call", "call_1")] == ["fb_1"]
    assert listed.calls.last.request.url.params["target_kind"] == "tool_call"
    with pytest.raises(ValueError):
        await ctx.feedback("run", "run_1")


@respx.mock
async def test_the_catalog_and_the_agent_s_model_key_are_advanced(ctx) -> None:
    listed = respx.get(f"{BASE}/v1/tools").respond(
        200,
        json={"tools": [{"tool_id": "tool_1", "name": "erp-get_stock", "side_effects": "read"}]},
    )
    put = respx.put(f"{BASE}/v1/tools/catalog").respond(200, json={"tools": []})
    key = respx.put(f"{BASE}/v1/agents/model-key").respond(
        200, json={"registered": True, "revoked": False, "revision": 1}
    )
    suggestions = respx.get(f"{BASE}/v1/tools/approval-suggestions").respond(
        200, json={"suggestions": []}
    )
    entries = await ctx.advanced.tools.catalog(names=["erp-get_stock"])
    assert entries[0].side_effects == "read"
    assert listed.calls.last.request.url.params.get_list("names") == ["erp-get_stock"]
    await ctx.advanced.tools.put_catalog([{"name": "erp-get_stock", "side_effects": "read"}])
    assert _body(put)["tools"][0]["name"] == "erp-get_stock"
    await ctx.advanced.model_keys.set("vk-1", idempotency_key="model-key:buyer")
    assert key.calls.last.request.headers["Idempotency-Key"] == "model-key:buyer"
    assert await ctx.advanced.tools.approval_suggestions() == [] and suggestions.called
