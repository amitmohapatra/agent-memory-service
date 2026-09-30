"""End-to-end memory intelligence: chat messages and events become memories
that show up in /v1/recall, /v1/context and the SDK; forget removes them everywhere."""

from __future__ import annotations

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import post_message, sdk_client

pytestmark = pytest.mark.e2e
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def _scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def test_messages_and_events_become_memories(client) -> None:
    scope = _scope()
    # a chat message is an observation too (inline queue: processed before the response)
    r = post_message(
        client,
        H,
        {
            "scope": scope,
            "role": "USER",
            "content": "My timezone is Europe/Berlin and I prefer concise answers.",
        },
    )
    assert r.status_code == 202, r.text
    # an event the service learns from: a decision
    event = {
        "scope": scope,
        "role": "EVENT",
        "content": "We decided to use PostgreSQL as the canonical store.",
    }
    r = post_message(client, H, event)
    assert r.status_code == 202, r.text
    ack = r.json()
    assert ack["message_id"] and ack["job_ids"]
    # identical retry -> replayed acknowledgement, no second processing
    again = post_message(client, H, event)
    assert again.json()["message_id"] == ack["message_id"]
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
    # A memory is listed as soon as it is stored, but it is only retrievable once the
    # memory.index job has run. Without a receipt on the row there is no way over the public
    # API to tell those two moments apart, so a caller polling the list either papers over
    # the window with a fixed sleep or reads "stored" as "searchable" and is wrong. The value
    # may legitimately be null here - the job need not have drained - but the FIELD must be
    # there on both reads, which share memory_to_api.
    assert "indexed_at" in one.json()
    assert all("indexed_at" in m for m in mems)
    # not visible to another user (403, existence not revealed as 404 either way)
    assert (
        client.get(
            f"/v1/memories/{tz['memory_id']}", headers={**H, "X-Trellis-User": "u2"}
        ).status_code
        == 403
    )
    assert client.get("/v1/memories/mem_nope", headers=H).status_code == 404

    # recall: memory-only and mixed; context bundle carries memories
    r = client.post(
        "/v1/recall",
        headers=H,
        json={
            "scope": scope,
            "query": "what is my timezone?",
            "kinds": ["memory"],
            "debug": True,
        },
    )
    assert r.status_code == 200 and r.json()["query_type"] == "USER_MEMORY"
    items = r.json()["items"]
    assert items and items[0]["kind"] == "memory"
    assert items[0]["citation"] == f"memory_id:{items[0]['id']}"
    assert any("Europe/Berlin" in i["text"] for i in items)
    bundle = client.post(
        "/v1/context",
        headers=H,
        json={"scope": scope, "query": "which store did we decide on?", "format": "full"},
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
    # ...by item_id, not by substring. One sentence becomes three memories here: the
    # extracted fact ("My timezone is Europe/Berlin"), the preference beside it, and the
    # verbatim turn that both were read out of, which still contains the words. Asserting
    # the string was absent asserted that forgetting a fact also unsays the sentence it came
    # from, which is a different promise and not one this endpoint makes.
    assert tz["memory_id"] not in {i["id"] for i in r.json()["items"]}
    assert any("Europe/Berlin" in i["text"] for i in r.json()["items"]), (
        "the verbatim turn is still there - forgetting the fact does not retract the message"
    )
    assert (
        client.delete(
            f"/v1/memories/{by_pred['decided']['memory_id']}", headers={**H, "X-Trellis-User": "u2"}
        ).status_code
        == 403
    )
    # validation
    empty = {"scope": scope, "role": "EVENT", "content": ""}
    assert post_message(client, H, empty).status_code == 422
    assert post_message(client, {}, {**empty, "content": "x"}).status_code == 401


async def test_sdk_remember_recall_forget(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(tenant_id="acme", user_id="u1", **_scope())
    [ack] = await ctx.history.add(
        [("USER", "I work at ACME Corp and my favourite editor is neovim.")]
    )
    assert ack.message_id and ack.job_ids
    # remember() stores what it is given, as given
    await ctx.remember("Always answer in British English.", memory_type="PREFERENCE")
    items = await ctx.search("favourite editor", kinds=["memory"], limit=5)
    assert items and any("neovim" in i.text for i in items)
    fav = next(i for i in items if "neovim" in i.text)
    got = await ctx.advanced.memories.get(fav.id)
    assert got.memory_id == fav.id and got.memory_type == "PREFERENCE"
    assert got.visibility == "USER" and got.lifetime == "LONG_TERM"
    await ctx.forget(fav.id)
    assert not any(
        "neovim" in i.text for i in await ctx.search("favourite editor", kinds=["memory"])
    )
    bundle = await ctx.context("how should I phrase the answer?", format="full")
    assert any("British English" in m.text for m in bundle.memories)
    await memory.aclose()


async def test_sdk_agent_handoff_and_shared_findings(app, client) -> None:
    memory = sdk_client(app)
    user = memory.bind(tenant_id="acme", user_id="u1", agent_group_id="crew", **_scope())
    await user.history.add([("USER", "Please prepare the FY26 brief.")])
    planner = user.agent("planner")
    writer = planner.agent("writer")  # child run: reads the planner's hand-off context
    await planner.remember("Plan: split the brief into revenue and cost.", visibility="RUN")
    q = "plan for the brief sections"
    assert any("Plan:" in i.text for i in await writer.search(q, kinds=["memory"]))
    assert not any("Plan:" in i.text for i in await user.search(q, kinds=["memory"]))
    assert not any(
        "Plan:" in i.text for i in await user.agent("intern").search(q, kinds=["memory"])
    )
    # explicit sharing with the agent group: every agent of the group reads it
    fact = "Revenue was EUR 412 million in FY26."
    await planner.remember(fact, memory_type="SHARED", visibility="AGENT_GROUP")
    auditor = user.agent("auditor")
    items = await auditor.search("FY26 revenue", kinds=["memory"])
    hit = next(i for i in items if "412" in i.text)
    got = await auditor.advanced.memories.get(hit.id)
    assert got.visibility == "AGENT_GROUP"
    # Bound to the user the agent runs for: agent_id is unauthenticated request body.
    assert got.owner_principal == "agent:u1/planner"
    await memory.aclose()


def test_two_tenants_using_the_same_identifiers_share_nothing(client) -> None:
    """The collision case: same user id, same thread id, different tenant.

    Identifiers are chosen by callers, so two tenants naming a thread ``thr_1`` and a user
    ``u1`` is ordinary, not adversarial. Every audience key is tenant-prefixed for that
    reason (``thread:acme/thr_1``, never ``thread:thr_1``) and ``allows()`` compares the
    tenant before it looks at a single key, so a collision cannot resolve into a match.

    Covered exhaustively over visibilities in tests/security/test_isolation.py; this is the
    same property end to end, through the API, and across the graph as well as the memories
    - an org-wide knowledge graph is the place where a tenant-blind key would hurt most.
    """
    acme = {**H, "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
    globex = {**H, "X-Trellis-Tenant": "globex", "X-Trellis-User": "u1"}
    scope = _scope()

    decision = "We decided to acquire Initech for 40 million."
    event = {"scope": scope, "role": "EVENT", "content": decision}
    assert post_message(client, acme, event).status_code == 202

    q = {"scope": scope, "query": "what did we decide about acquiring?", "kinds": ["memory"]}
    assert client.post("/v1/recall", headers=acme, json=q).json()["items"], "own tenant reads"
    assert client.post("/v1/recall", headers=globex, json=q).json()["items"] == []

    listed = client.get("/v1/memories", headers=globex, params={"thread_id": scope["thread_id"]})
    assert listed.status_code == 200 and listed.json()["memories"] == []

    g = {"q": "Initech", "thread_id": scope["thread_id"]}
    ours = client.get("/v1/graph/entities", headers=acme, params=g).json()["entities"]
    assert ours, "own tenant"
    theirs = client.get("/v1/graph/entities", headers=globex, params=g).json()
    assert theirs["entities"] == []
    assert (
        client.get(f"/v1/graph/entities/{ours[0]['entity_id']}", headers=globex).status_code == 404
    )
