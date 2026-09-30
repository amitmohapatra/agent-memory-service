"""The per-turn verbs on a bound context: what each one sends and what it returns."""

from __future__ import annotations

import json

import pytest
import respx
from pydantic import BaseModel

from trellis.memory import ContextBundle, MemoryClient, Message, PromptContext, ToolHints

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
    "bundle_id": "b1",
    "conversation": {"thread_id": "thr_1"},
    "evidence": {"status": "COMPLETE"},
    "token_budget": 2000,
    "token_estimate": 10,
    "rendered": "## Profile\n...",
    "handles": {"m1": "mem_1"},
    "profile": [{"block": "user", "text": "Prefers email.", "version": 3}],
    "thread_summary": {"text": "Asked for a PO.", "covers_to_sequence": 20, "version": 1},
    "procedures": [{"id": "prc_1", "title": "Order", "steps": [], "success_rate": 1.0}],
    "tools": {"candidates": [{"name": "erp-create_po", "score": 0.9}], "next": "erp-create_po"},
}


@respx.mock
async def test_context_is_the_prompt_by_default(ctx) -> None:
    route = respx.post(f"{BASE}/v1/context").respond(
        200,
        json={
            "rendered": "## Memories\n- [m1] Prefers email.",
            "bundle_id": "b1",
            "token_estimate": 12,
            "tool_candidates": ["erp-create_po"],
        },
    )
    pushed = await ctx.context(
        "order paper", token_budget=2000, tools=["erp-create_po"], window=False
    )
    body = _body(route)
    assert body["tools"] == {"available": ["erp-create_po"], "k": 8}
    assert body["format"] == "prompt" and body["window"] is False and body["token_budget"] == 2000
    assert "use_llm" not in body and "since_revision" not in body
    assert isinstance(pushed, PromptContext)
    assert pushed.bundle_id == "b1" and pushed.tool_candidates == ["erp-create_po"]


@respx.mock
async def test_the_full_bundle_reads_every_section(ctx) -> None:
    route = respx.post(f"{BASE}/v1/context").respond(200, json=_BUNDLE)
    bundle = await ctx.context("order paper", format="full", debug=True)
    assert _body(route)["format"] == "full" and _body(route)["debug"] is True
    assert isinstance(bundle, ContextBundle) and bundle.handles == {"m1": "mem_1"}
    assert bundle.profile[0].block == "user"
    assert bundle.thread_summary is not None and bundle.thread_summary.covers_to_sequence == 20
    assert bundle.procedures[0].id == "prc_1"
    assert isinstance(bundle.tools, ToolHints) and bundle.tools.next == "erp-create_po"


@respx.mock
async def test_tool_hints_names_the_available_tools(ctx) -> None:
    route = respx.post(f"{BASE}/v1/tools/hints").respond(
        200,
        json={
            "candidates": [{"name": "erp-create_po", "score": 0.8, "why": "worked before"}],
            "prefill": {
                "erp-create_po.supplier": {
                    "tool": "erp-create_po",
                    "value": "Acme",
                    "source": "graph",
                }
            },
            "missing": [{"tool": "erp-create_po", "arg": "qty", "question": "How many?"}],
        },
    )
    hints = await ctx.tool_hints("order paper", available=["erp-create_po"], k=3)
    assert _body(route)["available"] == ["erp-create_po"] and _body(route)["k"] == 3
    assert hints.prefill["erp-create_po.supplier"].value == "Acme"
    assert hints.missing[0].arg == "qty"


@respx.mock
async def test_agent_tools_are_listed_and_called_in_scope_with_the_toolbox(ctx) -> None:
    respx.get(f"{BASE}/v1/agent-tools").respond(
        200,
        json={"tools": [{"name": "memory_search", "description": "d", "input_schema": {}}]},
    )
    call = respx.post(f"{BASE}/v1/agent-tools/memory_search").respond(
        200, json={"result": [{"id": "mem_1"}]}
    )
    search = respx.post(f"{BASE}/v1/agent-tools/tool_search").respond(
        200, json={"result": {"next": "erp-create_po"}}
    )
    tools = await ctx.agent_tools()
    assert [t.name for t in tools] == ["memory_search"]
    assert await ctx.call_agent_tool("memory_search", {"query": "paper"}) == [{"id": "mem_1"}]
    body = _body(call)
    assert body["args"] == {"query": "paper"} and body["scope"]["agent_run_id"] == "run_1"
    assert "toolbox" not in body
    await ctx.call_agent_tool("tool_search", {"task": "order"}, toolbox=["erp-create_po"])
    assert _body(search)["toolbox"] == ["erp-create_po"]
    assert _body(search)["args"] == {"task": "order"}


@respx.mock
async def test_record_tool_is_the_run_s(ctx) -> None:
    record = respx.post(f"{BASE}/v1/tools/invocations").respond(
        202, json={"invocation_id": "tiv_1", "step": 0, "args_hash": "h", "recorded": True}
    )
    await ctx.record_tool("erp-get_stock", {"sku": "A4"}, status="cancelled", task=None, step=0)
    assert _body(record)["status"] == "cancelled" and _body(record)["task"] == ""
    assert _body(record)["scope"]["agent_run_id"] == "run_1"


@respx.mock
async def test_profile_is_listed_and_edited(ctx) -> None:
    block = {"block": "user", "text": "Prefers email.", "version": 2}
    respx.get(f"{BASE}/v1/profile").respond(200, json={"blocks": [block]})
    patch = respx.patch(url__regex=rf"{BASE}/v1/profile/user.*").respond(
        200, json={**block, "version": 3}
    )
    assert (await ctx.profile())[0].text == "Prefers email."
    edited = await ctx.profile.edit("user", "phone", old="email")
    assert _body(patch)["old"] == "email" and _body(patch)["new"] == "phone"
    assert "source_query" not in _body(patch) and edited.version == 3
    await ctx.profile.edit("user", "Prefers email.")
    assert _body(patch)["old"] == "" and _body(patch)["new"] == "Prefers email."
    await ctx.profile.edit("user.suppliers", source_query="Which suppliers do we use?")
    assert _body(patch) == {
        "scope": _body(patch)["scope"],
        "old": "",
        "source_query": "Which suppliers do we use?",
    }
    await ctx.profile.edit("user.suppliers", source_query=None)
    assert _body(patch)["source_query"] is None


@respx.mock
async def test_history_reads_and_appends_the_thread(ctx) -> None:
    listed = respx.get(f"{BASE}/v1/threads/thr_1/messages").respond(
        200,
        json={
            "thread_id": "thr_1",
            "messages": [
                {
                    "message_id": "m1",
                    "role": "USER",
                    "kind": "VISIBLE",
                    "sequence": 1,
                    "content": "hi",
                }
            ],
        },
    )
    ack = {
        "message_id": "msg_1",
        "thread_id": "thr_1",
        "session_id": "s",
        "turn_id": "t",
        "sequence": 1,
    }
    added = respx.post(f"{BASE}/v1/messages").respond(
        202, json={"messages": [ack, {**ack, "message_id": "msg_2", "sequence": 2}]}
    )
    thread = respx.get(f"{BASE}/v1/threads/thr_1").respond(
        200,
        json={
            "thread_id": "thr_1",
            "tenant_id": "acme",
            "summary": {"text": "s", "covers_to_sequence": 20, "version": 1},
        },
    )
    titled = respx.patch(f"{BASE}/v1/threads/thr_1").respond(
        200, json={"thread_id": "thr_1", "tenant_id": "acme", "title": "Paper"}
    )
    assert [m.content for m in await ctx.history(limit=5)] == ["hi"]
    assert listed.calls.last.request.url.params["limit"] == "5"
    acks = await ctx.history.add(
        [("USER", "order paper"), Message(role="EVENT", content="stock checked")],
        idempotency_key="run_1:msgs:0",
    )
    assert [a.message_id for a in acks] == ["msg_1", "msg_2"]
    body = _body(added)
    assert [m["role"] for m in body["messages"]] == ["USER", "EVENT"]
    assert added.calls.last.request.headers["Idempotency-Key"] == "run_1:msgs:0"
    info = await ctx.history.thread()
    assert thread.called and info.summary is not None and info.summary.version == 1
    await ctx.history.update(title="Paper")
    assert _body(titled)["title"] == "Paper"


@respx.mock
async def test_a_run_without_a_thread_reads_the_thread_named_by_the_run(ctx) -> None:
    unthreaded = ctx.client.bind(tenant_id="acme", user_id="u1").agent("buyer", agent_run_id="r9")
    route = respx.get(f"{BASE}/v1/threads/r9/messages").respond(
        200, json={"thread_id": "r9", "messages": []}
    )
    assert await unthreaded.history() == [] and route.called


@respx.mock
async def test_verify_sends_the_bundle_and_reports_the_judge_s_feedback(ctx) -> None:
    route = respx.post(f"{BASE}/v1/verify").respond(
        200,
        json={
            "claims": [],
            "supported": 2,
            "unsupported": 1,
            "per_claim_hallucination_rate": 1 / 3,
            "feedback_id": "fb_judge",
        },
    )
    report = await ctx.verify("Paper ordered [m1].", bundle_id="b1")
    assert _body(route)["bundle_id"] == "b1" and "run_id" not in _body(route)
    assert report.feedback_id == "fb_judge" and report.score == pytest.approx(2 / 3)


@respx.mock
async def test_search_returns_items_and_sends_the_time_window(ctx) -> None:
    from datetime import UTC, datetime

    route = respx.post(f"{BASE}/v1/recall").respond(
        200,
        json={
            "items": [
                {
                    "id": "mem_1",
                    "kind": "memory",
                    "text": "Prefers email.",
                    "observed_on": "2026-09-01",
                    "citation": "memory_id:mem_1",
                }
            ]
        },
    )
    items = await ctx.search(
        "contact", kinds=["memory", "message"], time_from=datetime(2026, 9, 1, tzinfo=UTC)
    )
    assert items[0].id == "mem_1" and items[0].observed_on == "2026-09-01"
    body = _body(route)
    assert body["kinds"] == ["memory", "message"] and body["time_from"].startswith("2026-09-01")


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
    await ctx.feedback("run", "run_1", "confirm", score=0.9, source="system")
    assert _body(route)["agent_run_id"] == "run_1" and _body(route)["score"] == 0.9
    assert _body(route)["source"] == "system"
    assert [f.feedback_id for f in await ctx.feedback.list_for("tool_call", "call_1")] == ["fb_1"]
    assert listed.calls.last.request.url.params["target_kind"] == "tool_call"
    with pytest.raises(ValueError):
        await ctx.feedback("run", "run_1")


@respx.mock
async def test_the_catalog_and_the_agent_s_model_key_are_advanced(ctx) -> None:
    listed = respx.get(f"{BASE}/v1/tools").respond(
        200,
        json={
            "tools": [
                {
                    "tool_id": "tool_1",
                    "name": "erp-get_stock",
                    "annotations": {"readOnlyHint": True},
                    "risk": "read",
                }
            ]
        },
    )
    put = respx.put(f"{BASE}/v1/tools/catalog").respond(200, json={"tools": []})
    key = respx.put(f"{BASE}/v1/agents/model-key").respond(
        200, json={"registered": True, "revoked": False, "revision": 1}
    )
    suggestions = respx.get(f"{BASE}/v1/tools/approval-suggestions").respond(
        200, json={"suggestions": []}
    )
    accepted = respx.post(f"{BASE}/v1/tools/approval-suggestions/s1/accept").respond(
        200, json={"tool_id": "tool_2", "name": "erp-create_po", "approve_when": "amount > 1"}
    )
    entries = await ctx.advanced.tools.catalog(names=["erp-get_stock"])
    assert entries[0].risk == "read" and entries[0].annotations == {"readOnlyHint": True}
    assert listed.calls.last.request.url.params.get_list("names") == ["erp-get_stock"]
    await ctx.advanced.tools.put_catalog([{"name": "erp-get_stock", "side_effects": "read"}])
    assert _body(put)["tools"][0]["name"] == "erp-get_stock"
    await ctx.advanced.model_keys.set("vk-1", idempotency_key="model-key:buyer")
    assert key.calls.last.request.headers["Idempotency-Key"] == "model-key:buyer"
    assert await ctx.advanced.tools.approval_suggestions() == [] and suggestions.called
    assert (await ctx.advanced.tools.accept_suggestion("s1")).approve_when == "amount > 1"
    assert accepted.called
