"""End-to-end chat flows through the public API and the Python SDK."""

from __future__ import annotations

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import post_message, sdk_client
from trellis.memory import AuthorizationError, NotFoundError, ValidationError

pytestmark = pytest.mark.e2e

H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def _scope(**kw):
    return {
        "thread_id": kw.get("thread", new_id("thread")),
        "session_id": kw.get("session", new_id("session")),
        "turn_id": kw.get("turn", new_id("turn")),
    }


def test_new_chat_next_turn_reopen(client, container) -> None:
    scope = _scope()
    r = post_message(client, H, {"scope": scope, "role": "USER", "content": "hello"})
    assert r.status_code == 202, r.text
    ack = r.json()
    assert (
        ack["sequence"] == 1 and len(ack["job_ids"]) == 2 and ack["job_ids"][0].startswith("obx_")
    )
    # jobs ran inline after commit
    job = client.get(f"/v1/jobs/{ack['job_ids'][0]}", headers=H).json()
    assert job["status"] == "SUCCEEDED" and job["task_name"] == "memory.process_observation"
    r = post_message(client, H, {"scope": scope, "role": "ASSISTANT", "content": "hi!"})
    assert r.json()["sequence"] == 2
    # next question, same session, new turn
    scope2 = {**scope, "turn_id": new_id("turn")}
    assert (
        post_message(client, H, {"scope": scope2, "role": "USER", "content": "more"}).json()[
            "sequence"
        ]
        == 3
    )
    # reopen later: same thread, new session, new turn
    scope3 = {
        "thread_id": scope["thread_id"],
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    assert (
        post_message(client, H, {"scope": scope3, "role": "USER", "content": "back again"}).json()[
            "sequence"
        ]
        == 4
    )
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["sequence"] for m in hist["messages"]] == [1, 2, 3, 4]
    assert [m["role"] for m in hist["messages"]] == ["USER", "ASSISTANT", "USER", "USER"]
    thread = client.get(f"/v1/threads/{scope['thread_id']}", headers=H).json()
    assert thread["owner_user_id"] == "u1" and thread["revision"] >= 4
    # paging backwards
    page = client.get(
        f"/v1/threads/{scope['thread_id']}/messages", headers=H, params={"limit": 2}
    ).json()
    assert [m["sequence"] for m in page["messages"]] == [3, 4] and page["next_before_sequence"] == 3
    page2 = client.get(
        f"/v1/threads/{scope['thread_id']}/messages",
        headers=H,
        params={"limit": 2, "before_sequence": 3},
    ).json()
    assert [m["sequence"] for m in page2["messages"]] == [1, 2]


def test_idempotent_retry_returns_same_ack(client) -> None:
    scope = _scope()
    body = {"scope": scope, "role": "USER", "content": "same"}
    first = post_message(client, {**H, "Idempotency-Key": "k-1"}, body)
    second = post_message(client, {**H, "Idempotency-Key": "k-1"}, body)
    assert first.status_code == 202 and second.status_code == 202
    assert first.json() == second.json() and second.headers.get("Idempotent-Replayed") == "true"
    conflict = post_message(
        client, {**H, "Idempotency-Key": "k-1"}, {**body, "content": "different"}
    )
    assert conflict.status_code == 409 and conflict.json()["code"] == "CONFLICT"
    # without a header the server derives the key from lineage + content: a network retry of
    # the same request never duplicates, while a distinct key is a distinct request
    a = post_message(client, H, {**body, "content": "retry me"})
    b = post_message(client, H, {**body, "content": "retry me"})
    assert a.json()["message_id"] == b.json()["message_id"]
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in hist["messages"]] == ["same", "retry me"]


def test_internal_messages_do_not_pollute_visible_history(client, container) -> None:
    scope = _scope()
    post_message(client, H, {"scope": scope, "role": "USER", "content": "plan my trip"})
    agent_scope = {
        **scope,
        "agent_id": "planner",
        "agent_run_id": new_id("agent_run"),
        "agent_group_id": "crew",
    }
    r = post_message(
        client,
        H,
        {
            "scope": agent_scope,
            "role": "AGENT",
            "kind": "INTERNAL",
            "content": "tool: search flights",
        },
    )
    assert r.status_code == 202, r.text
    child_scope = {
        **agent_scope,
        "agent_id": "flights",
        "agent_run_id": new_id("agent_run"),
        "parent_agent_run_id": agent_scope["agent_run_id"],
    }
    assert (
        post_message(
            client,
            H,
            {
                "scope": child_scope,
                "role": "TOOL",
                "kind": "INTERNAL",
                "content": "3 flights found",
            },
        ).status_code
        == 202
    )
    post_message(client, H, {"scope": scope, "role": "ASSISTANT", "content": "Here is your trip"})
    visible = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()["messages"]
    assert [m["role"] for m in visible] == ["USER", "ASSISTANT"]
    everything = client.get(
        f"/v1/threads/{scope['thread_id']}/messages", headers=H, params={"include_internal": True}
    ).json()["messages"]
    assert [m["kind"] for m in everything] == ["VISIBLE", "INTERNAL", "INTERNAL", "VISIBLE"]
    assert everything[1]["agent_run_id"] == agent_scope["agent_run_id"]
    # lineage rows exist and are linked to the single turn

    async def _runs():
        async with container.services["uow_factory"]() as uow:
            ids = (agent_scope["agent_run_id"], child_scope["agent_run_id"])
            return [await uow.agent_runs.get("acme", i) for i in ids]

    runs = [r for r in client.portal.call(_runs) if r is not None]
    assert {r.turn_id for r in runs} == {scope["turn_id"]}
    assert {r.agent_id for r in runs} == {"planner", "flights"}
    assert {r.parent_agent_run_id for r in runs} == {None, agent_scope["agent_run_id"]}
    # visible AGENT role is rejected
    bad = post_message(client, H, {"scope": scope, "role": "AGENT", "content": "x"})
    assert bad.status_code == 422


def test_authorization_boundaries(client) -> None:
    scope = _scope()
    post_message(client, H, {"scope": scope, "role": "USER", "content": "private"})
    other_user = {**H, "X-Trellis-User": "u2"}
    assert client.get(f"/v1/threads/{scope['thread_id']}", headers=other_user).status_code == 403
    assert (
        client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=other_user).status_code
        == 403
    )
    r = post_message(client, other_user, {"scope": scope, "role": "USER", "content": "hijack"})
    assert r.status_code == 403 and r.json()["code"] == "SCOPE_DENIED"
    other_tenant = {**H, "X-Trellis-Tenant": "globex"}
    assert client.get(f"/v1/threads/{scope['thread_id']}", headers=other_tenant).status_code == 404
    # agent acting for the owner may write internal messages to the thread
    agent_scope = {**scope, "agent_id": "helper", "agent_run_id": new_id("agent_run")}
    assert (
        post_message(
            client,
            H,
            {"scope": agent_scope, "role": "AGENT", "kind": "INTERNAL", "content": "note"},
        ).status_code
        == 202
    )


def test_cache_outage_does_not_affect_correctness(client, container) -> None:
    scope = _scope()
    post_message(client, H, {"scope": scope, "role": "USER", "content": "one"})
    container.cache.available = False
    r = post_message(client, H, {"scope": scope, "role": "ASSISTANT", "content": "two"})
    assert r.status_code == 202
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in hist["messages"]] == ["one", "two"]
    container.cache.available = True
    # cache is repopulated lazily and stays consistent with the database
    r = post_message(client, H, {"scope": scope, "role": "USER", "content": "three"})
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in hist["messages"]] == ["one", "two", "three"]


def test_validation_errors(client) -> None:
    r = post_message(client, H, {"scope": {}, "role": "USER", "content": "x"})
    assert r.status_code == 422, "a message needs a thread"
    r = post_message(client, H, {"scope": _scope(), "role": "USER", "content": "x", "unknown": 1})
    assert r.status_code == 422 and r.json()["code"] == "VALIDATION"
    assert client.get("/v1/jobs/obx_999999", headers=H).status_code == 404
    assert client.get("/v1/jobs/nope", headers=H).status_code == 404


def test_patching_a_thread_creates_it_and_converges(client) -> None:
    thread_id = new_id("thread")
    body = {"scope": {}, "title": "Q3"}
    a = client.patch(f"/v1/threads/{thread_id}", headers=H, json=body)
    b = client.patch(f"/v1/threads/{thread_id}", headers=H, json=body)
    assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
    assert a.json()["thread_id"] == b.json()["thread_id"] == thread_id
    assert client.get(f"/v1/threads/{thread_id}", headers=H).json()["title"] == "Q3"
    assert client.delete(f"/v1/threads/{thread_id}", headers=H).status_code == 204
    assert client.get(f"/v1/threads/{thread_id}", headers=H).status_code == 404


async def test_sdk_ninety_percent_path(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(
        tenant_id="acme",
        user_id="u1",
        thread_id=new_id("thread"),
        session_id=new_id("session"),
        turn_id=new_id("turn"),
    )
    [ack] = await ctx.history.add([("USER", "What changed in EBITDA?")])
    assert ack.sequence == 1 and ack.job_ids
    job = await ctx.advanced.job(ack.job_ids[0])
    assert job.status == "SUCCEEDED"
    await ctx.history.add([("ASSISTANT", "EBITDA rose because of restructuring savings.")])
    history = await ctx.history()
    assert [m.role for m in history] == ["USER", "ASSISTANT"]
    thread = await ctx.history.thread()
    assert thread.thread_id == ctx.scope.thread_id
    # child agent context records internal lineage without touching the visible chat
    agent = ctx.agent("research")
    await agent.history.add(
        [{"role": "AGENT", "kind": "INTERNAL", "content": "searched 3 filings"}]
    )
    assert len(await ctx.history()) == 2
    assert len(await ctx.history(include_internal=True)) == 3
    # another user is denied; a missing thread is NotFound
    other = memory.bind(tenant_id="acme", user_id="u2", thread_id=ctx.scope.thread_id)
    with pytest.raises(AuthorizationError):
        await other.history.thread()
    missing = memory.bind(tenant_id="acme", user_id="u1", thread_id=new_id("thread"))
    with pytest.raises(NotFoundError):
        await missing.history.thread()
    with pytest.raises(ValidationError):
        await memory.bind(tenant_id="acme", user_id="u1").history.add([("USER", "no lineage")])
    await memory.aclose()


def test_a_message_with_only_a_thread_gets_its_session_and_turn(client) -> None:
    thread = new_id("thread")
    scope = {"thread_id": thread}
    asked = post_message(client, H, {"scope": scope, "role": "USER", "content": "ok"})
    assert asked.status_code == 202, asked.text
    answered = post_message(client, H, {"scope": scope, "role": "ASSISTANT", "content": "ok"})
    assert answered.json()["turn_id"] == asked.json()["turn_id"]
    assert answered.json()["session_id"] == asked.json()["session_id"]
    # the same content again, with no turn and no key, is a retry of the same message...
    again = post_message(client, H, {"scope": scope, "role": "USER", "content": "ok"})
    assert again.json()["message_id"] == asked.json()["message_id"]
    # ...unless the client says when it happened
    later = post_message(
        client,
        H,
        {
            "scope": scope,
            "role": "USER",
            "content": "ok",
            "occurred_at": "2026-09-30T10:00:00Z",
        },
    )
    assert later.json()["message_id"] != asked.json()["message_id"]
    assert later.json()["turn_id"] != asked.json()["turn_id"], "a question opens a new turn"


def test_remember_refuses_what_it_cannot_store(client) -> None:
    for body in (
        {"content": "x", "lifetime": "EPHEMERAL"},
        {"content": "x", "memory_type": "CUSTOM"},
        {"content": "x", "valid_from": "2026-09-30T00:00:00Z", "valid_to": "2026-09-01T00:00:00Z"},
        {"content": ""},
    ):
        r = client.post("/v1/memories", headers=H, json={"scope": {}, **body})
        assert r.status_code == 422, (body, r.text)


def test_a_runs_messages_without_a_thread_go_to_the_runs_own_conversation(client) -> None:
    """A run is a conversation of its own when nobody named one: its messages go to the
    thread named by its ``agent_run_id``, and read back from there. With neither a thread
    nor a run there is nowhere to put them."""
    run_id = new_id("agent_run")
    scope = {"agent_id": "planner", "agent_run_id": run_id}
    sent = post_message(client, H, {"scope": scope, "role": "USER", "content": "plan a trip"})
    assert sent.status_code == 202, sent.text
    assert sent.json()["thread_id"] == run_id
    listed = client.get(f"/v1/threads/{run_id}/messages", headers=H)
    assert listed.status_code == 200, listed.text
    assert [m["content"] for m in listed.json()["messages"]] == ["plan a trip"]
    nowhere = post_message(
        client, H, {"scope": {"agent_id": "planner"}, "role": "USER", "content": "x"}
    )
    assert nowhere.status_code == 422, nowhere.text
    assert "thread_id" in nowhere.json()["detail"]


def test_a_profile_edit_must_change_something_and_old_needs_new(client) -> None:
    """``PATCH /v1/profile/{block}`` with nothing to change, or text to replace with
    nothing, is refused before anything is read or written."""
    scope = {"thread_id": new_id("thread")}
    for body, said in (
        ({}, "send new, source_query, or both"),
        ({"old": "Berlin", "source_query": "where do I live?"}, "old needs new"),
    ):
        refused = client.patch("/v1/profile/user", headers=H, json={"scope": scope, **body})
        assert refused.status_code == 422, refused.text
        assert said in refused.text


def test_a_tools_approval_rule_must_parse_and_empty_removes_it(client) -> None:
    """``approve_when`` is checked when the catalog entry is written: a rule that does not
    parse is a 422, and an empty string clears the rule."""

    def put(rule: str) -> object:
        entry = {"name": "billing.refund", "description": "refund", "approve_when": rule}
        return client.put("/v1/tools/catalog", headers=H, json={"tools": [entry]})

    assert put("amount >=").status_code == 422
    kept = put("amount >= 1000")
    assert kept.status_code == 200, kept.text
    assert kept.json()["tools"][0]["approve_when"] == "amount >= 1000"
    cleared = put("")
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["tools"][0]["approve_when"] is None


def test_an_internal_message_read_by_id_is_refused_to_a_person_who_did_not_write_it(
    client,
) -> None:
    """Internal execution is not chat: a person reading the thread sees its visible
    messages by id, but an agent's internal step answers ``403 SCOPE_DENIED``."""
    scope = _scope()
    said = post_message(client, H, {"scope": scope, "role": "USER", "content": "plan my trip"})
    agent_scope = {**scope, "agent_id": "planner", "agent_run_id": new_id("agent_run")}
    step = post_message(
        client,
        H,
        {"scope": agent_scope, "role": "AGENT", "kind": "INTERNAL", "content": "tool: search"},
    )
    assert step.status_code == 202, step.text
    visible = client.get(f"/v1/messages/{said.json()['message_id']}", headers=H)
    assert visible.status_code == 200, visible.text
    internal = client.get(f"/v1/messages/{step.json()['message_id']}", headers=H)
    assert internal.status_code == 403, internal.text
    assert internal.json()["code"] == "SCOPE_DENIED"


def test_a_session_or_turn_of_another_conversation_is_refused(client) -> None:
    """Lineage stays a tree: a session belongs to one thread and a turn to one session, and
    a message that names them across is refused rather than filed under both."""
    first = _scope()
    assert (
        post_message(client, H, {"scope": first, "role": "USER", "content": "a"}).status_code == 202
    )
    other_thread = {**first, "thread_id": new_id("thread")}
    crossed = post_message(client, H, {"scope": other_thread, "role": "USER", "content": "b"})
    assert crossed.status_code == 422, crossed.text
    assert "session_id belongs to a different thread" in crossed.text
    other_session = {**first, "session_id": new_id("session")}
    crossed = post_message(client, H, {"scope": other_session, "role": "USER", "content": "c"})
    assert crossed.status_code == 422, crossed.text
    assert "turn_id belongs to a different session" in crossed.text


def test_an_imported_message_is_stored_once_per_source_id(client) -> None:
    """A re-import of the same ``(source_system, source_message_id)`` is deduplicated: the
    first message's acknowledgement comes back, marked so, and nothing is stored twice."""
    scope = {"thread_id": new_id("thread")}
    imported = {"source_system": "slack", "source_message_id": "C1-1700000000.1"}
    first = post_message(
        client, H, {"scope": scope, "role": "USER", "content": "hello", **imported}
    )
    assert first.status_code == 202, first.text
    again = post_message(
        client,
        H,
        {"scope": scope, "role": "USER", "content": "hello (edited)", **imported},
    )
    assert again.status_code == 202, again.text
    assert again.json()["message_id"] == first.json()["message_id"]
    assert again.json()["deduplicated"] is True and not first.json()["deduplicated"]
    listed = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in listed["messages"]] == ["hello"]


def test_an_internal_agent_message_needs_the_agent(client) -> None:
    scope = _scope()
    refused = post_message(
        client, H, {"scope": scope, "role": "AGENT", "kind": "INTERNAL", "content": "step"}
    )
    assert refused.status_code == 422, refused.text
    assert "require agent_id" in refused.text
