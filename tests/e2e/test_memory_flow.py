"""End-to-end memory intelligence: chat messages and explicit observations become memories
that show up in /v1/recall, /v1/context and the SDK; forget removes them everywhere."""

from __future__ import annotations

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import sdk_client

pytestmark = pytest.mark.e2e
H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


def _scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def test_messages_and_observations_become_memories(client) -> None:
    scope = _scope()
    # a chat message is an observation too (inline queue: processed before the response)
    r = client.post(
        "/v1/messages",
        headers=H,
        json={
            "scope": scope,
            "role": "USER",
            "content": "My timezone is Europe/Berlin and I prefer concise answers.",
        },
    )
    assert r.status_code == 202, r.text
    # an explicit observation with a decision
    r = client.post(
        "/v1/observations",
        headers=H,
        json={
            "scope": scope,
            "kind": "DECISION",
            "content": "We decided to use PostgreSQL as the canonical store.",
        },
    )
    assert r.status_code == 202, r.text
    ack = r.json()
    assert ack["observation_id"].startswith("obs_") and ack["job_ids"]
    # identical retry -> replayed acknowledgement, no second processing
    again = client.post(
        "/v1/observations",
        headers=H,
        json={
            "scope": scope,
            "kind": "DECISION",
            "content": "We decided to use PostgreSQL as the canonical store.",
        },
    )
    assert again.json()["observation_id"] == ack["observation_id"]
    assert again.headers.get("Idempotent-Replayed") == "true"

    mems = client.get("/v1/memories", headers=H, params={"thread_id": scope["thread_id"]}).json()[
        "memories"
    ]
    by_pred = {m["predicate"]: m for m in mems}
    assert {"timezone", "prefers", "decided"} <= set(by_pred)
    tz = by_pred["timezone"]
    assert tz["memory_type"] == "USER" and tz["visibility"] == "USER"
    assert tz["temporal_status"] == "CURRENT" and tz["evidence"][0]["source_type"] == "message"
    assert (
        by_pred["decided"]["visibility"] == "THREAD"
        and by_pred["decided"]["category"] == "decision"
    )
    one = client.get(f"/v1/memories/{tz['memory_id']}", headers=H)
    assert one.status_code == 200 and one.json()["object"] == "europe/berlin"
    # not visible to another user (403, existence not revealed as 404 either way)
    assert (
        client.get(
            f"/v1/memories/{tz['memory_id']}", headers={**H, "X-Memory-User": "u2"}
        ).status_code
        == 403
    )
    assert client.get("/v1/memories/mem_nope", headers=H).status_code == 404

    # recall: memory-only and mixed; context bundle carries memories
    r = client.post(
        "/v1/recall",
        headers=H,
        json={"scope": scope, "query": "what is my timezone?", "kinds": ["memory"]},
    )
    assert r.status_code == 200 and r.json()["query_type"] == "USER_MEMORY"
    items = r.json()["results"]
    assert items and items[0]["representation"] == "MEMORY"
    assert items[0]["citation"] == f"memory_id:{items[0]['item_id']}"
    assert any("Europe/Berlin" in i["text"] for i in items)
    bundle = client.post(
        "/v1/context", headers=H, json={"scope": scope, "query": "which store did we decide on?"}
    ).json()
    assert bundle["memories"] and "## Memories" in bundle["rendered"]
    assert any("PostgreSQL" in m["text"] for m in bundle["memories"])

    # forget: gone from list, get, recall and context (cache invalidated by revision bump)
    r = client.delete(f"/v1/memories/{tz['memory_id']}", headers=H)
    assert r.status_code == 204
    assert client.get(f"/v1/memories/{tz['memory_id']}", headers=H).status_code == 404
    r = client.post(
        "/v1/recall",
        headers=H,
        json={"scope": scope, "query": "what is my timezone?", "kinds": ["memory"]},
    )
    assert not any("Europe/Berlin" in i["text"] for i in r.json()["results"])
    assert (
        client.delete(
            f"/v1/memories/{by_pred['decided']['memory_id']}", headers={**H, "X-Memory-User": "u2"}
        ).status_code
        == 403
    )
    # validation
    assert (
        client.post("/v1/observations", headers=H, json={"scope": scope, "content": ""}).status_code
        == 422
    )
    assert client.post("/v1/observations", json={"scope": scope, "content": "x"}).status_code == 401


async def test_sdk_remember_recall_forget(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **_scope())
    ack = await ctx.observe("I work at ACME Corp and my favourite editor is neovim.")
    assert ack.observation_id.startswith("obs_")
    # remember() = observe with expert hints
    await ctx.remember("Always answer in British English.", memory_type="PREFERENCE")
    items = await ctx.recall("favourite editor", kinds=["memory"], limit=5)
    assert items and any("neovim" in i.text for i in items)
    fav = next(i for i in items if "neovim" in i.text)
    got = await ctx.get_memory(fav.item_id)
    assert got.memory_id == fav.item_id and got.memory_type == "PREFERENCE"
    assert got.visibility == "USER" and got.lifetime == "LONG_TERM"
    await ctx.forget(fav.item_id)
    assert not any(
        "neovim" in i.text for i in await ctx.recall("favourite editor", kinds=["memory"])
    )
    bundle = await ctx.context("how should I phrase the answer?")
    assert any("British English" in m.text for m in bundle.memories)
    await memory.aclose()


async def test_sdk_agent_handoff_and_shared_findings(app, client) -> None:
    memory = sdk_client(app)
    user = memory.bind(tenant_id="acme", user_id="u1", agent_group_id="crew", **_scope())
    await user.chat.user("Please prepare the FY26 brief.")
    planner = user.agent("planner")
    writer = planner.agent("writer")  # child run: reads the planner's hand-off context
    await planner.observe("Plan: split the brief into revenue and cost.", kind="AGENT_RESULT")
    q = "plan for the brief sections"
    assert any("Plan:" in i.text for i in await writer.recall(q, kinds=["memory"]))
    assert not any("Plan:" in i.text for i in await user.recall(q, kinds=["memory"]))
    assert not any(
        "Plan:" in i.text for i in await user.agent("intern").recall(q, kinds=["memory"])
    )
    # explicit sharing with the agent group; a second agent corroborates
    fact = "Revenue was EUR 412 million in FY26."
    await planner.remember(fact, memory_type="SHARED", visibility="AGENT_GROUP")
    await writer.remember(fact, memory_type="SHARED", visibility="AGENT_GROUP")
    auditor = user.agent("auditor")
    items = await auditor.recall("FY26 revenue", kinds=["memory"])
    hit = next(i for i in items if "412" in i.text)
    got = await auditor.get_memory(hit.item_id)
    assert got.visibility == "AGENT_GROUP" and got.reinforcement_count == 2
    assert got.owner_principal == "agent:planner" and got.contributors == ["agent:writer"]
    # a conflicting single-valued fact from another agent is kept, linked, never overwritten
    await planner.remember("My manager is Dana.", memory_type="SHARED", visibility="AGENT_GROUP")
    await auditor.remember("My manager is Lee.", memory_type="SHARED", visibility="AGENT_GROUP")
    managers = [
        i
        for i in await writer.recall("who is my manager?", kinds=["memory"])
        if "manager" in i.text
    ]
    assert len(managers) == 2
    linked = [await writer.get_memory(i.item_id) for i in managers]
    assert any(m.contradicts for m in linked)
    await memory.aclose()
