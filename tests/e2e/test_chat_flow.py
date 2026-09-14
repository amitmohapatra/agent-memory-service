"""End-to-end chat flows through the public API and the Python SDK."""

from __future__ import annotations

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import sdk_client
from universal_memory import AuthorizationError, NotFoundError, ValidationError

pytestmark = pytest.mark.e2e

H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


def _scope(**kw):
    return {
        "thread_id": kw.get("thread", new_id("thread")),
        "session_id": kw.get("session", new_id("session")),
        "turn_id": kw.get("turn", new_id("turn")),
    }


def test_new_chat_next_turn_reopen(client, container) -> None:
    scope = _scope()
    r = client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "USER", "content": "hello"}
    )
    assert r.status_code == 202, r.text
    ack = r.json()
    assert (
        ack["sequence"] == 1 and len(ack["job_ids"]) == 2 and ack["job_ids"][0].startswith("obx_")
    )
    # jobs ran inline after commit
    job = client.get(f"/v1/jobs/{ack['job_ids'][0]}", headers=H).json()
    assert job["status"] == "SUCCEEDED" and job["task_name"] == "memory.process_observation"
    r = client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "ASSISTANT", "content": "hi!"}
    )
    assert r.json()["sequence"] == 2
    # next question, same session, new turn
    scope2 = {**scope, "turn_id": new_id("turn")}
    assert (
        client.post(
            "/v1/messages", headers=H, json={"scope": scope2, "role": "USER", "content": "more"}
        ).json()["sequence"]
        == 3
    )
    # reopen later: same thread, new session, new turn
    scope3 = {
        "thread_id": scope["thread_id"],
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    assert (
        client.post(
            "/v1/messages",
            headers=H,
            json={"scope": scope3, "role": "USER", "content": "back again"},
        ).json()["sequence"]
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
    first = client.post("/v1/messages", headers={**H, "Idempotency-Key": "k-1"}, json=body)
    second = client.post("/v1/messages", headers={**H, "Idempotency-Key": "k-1"}, json=body)
    assert first.status_code == 202 and second.status_code == 202
    assert first.json() == second.json() and second.headers.get("Idempotent-Replayed") == "true"
    conflict = client.post(
        "/v1/messages",
        headers={**H, "Idempotency-Key": "k-1"},
        json={**body, "content": "different"},
    )
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "CONFLICT"
    # without a header the server derives the key from lineage + content: a network retry of
    # the same request never duplicates, while a distinct key is a distinct request
    a = client.post("/v1/messages", headers=H, json={**body, "content": "retry me"})
    b = client.post("/v1/messages", headers=H, json={**body, "content": "retry me"})
    assert a.json()["message_id"] == b.json()["message_id"]
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in hist["messages"]] == ["same", "retry me"]


def test_internal_messages_do_not_pollute_visible_history(client, container) -> None:
    scope = _scope()
    client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "USER", "content": "plan my trip"}
    )
    agent_scope = {
        **scope,
        "agent_id": "planner",
        "agent_run_id": new_id("agent_run"),
        "agent_group_id": "crew",
    }
    r = client.post(
        "/v1/messages",
        headers=H,
        json={
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
        client.post(
            "/v1/messages",
            headers=H,
            json={
                "scope": child_scope,
                "role": "TOOL",
                "kind": "INTERNAL",
                "content": "3 flights found",
            },
        ).status_code
        == 202
    )
    client.post(
        "/v1/messages",
        headers=H,
        json={"scope": scope, "role": "ASSISTANT", "content": "Here is your trip"},
    )
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
            return await uow.agent_runs.list_for_turn("acme", scope["turn_id"])

    runs = client.portal.call(_runs)
    assert {r.agent_id for r in runs} == {"planner", "flights"}
    assert {r.parent_agent_run_id for r in runs} == {None, agent_scope["agent_run_id"]}
    # visible AGENT role is rejected
    bad = client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "AGENT", "content": "x"}
    )
    assert bad.status_code == 422


def test_authorization_boundaries(client) -> None:
    scope = _scope()
    client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "USER", "content": "private"}
    )
    other_user = {**H, "X-Memory-User": "u2"}
    assert client.get(f"/v1/threads/{scope['thread_id']}", headers=other_user).status_code == 403
    assert (
        client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=other_user).status_code
        == 403
    )
    r = client.post(
        "/v1/messages",
        headers=other_user,
        json={"scope": scope, "role": "USER", "content": "hijack"},
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "SCOPE_DENIED"
    other_tenant = {**H, "X-Memory-Tenant": "globex"}
    assert client.get(f"/v1/threads/{scope['thread_id']}", headers=other_tenant).status_code == 404
    # agent acting for the owner may write internal messages to the thread
    agent_scope = {**scope, "agent_id": "helper", "agent_run_id": new_id("agent_run")}
    assert (
        client.post(
            "/v1/messages",
            headers=H,
            json={"scope": agent_scope, "role": "AGENT", "kind": "INTERNAL", "content": "note"},
        ).status_code
        == 202
    )


def test_cache_outage_does_not_affect_correctness(client, container) -> None:
    scope = _scope()
    client.post("/v1/messages", headers=H, json={"scope": scope, "role": "USER", "content": "one"})
    container.cache.available = False
    r = client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "ASSISTANT", "content": "two"}
    )
    assert r.status_code == 202
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in hist["messages"]] == ["one", "two"]
    container.cache.available = True
    # cache is repopulated lazily and stays consistent with the database
    r = client.post(
        "/v1/messages", headers=H, json={"scope": scope, "role": "USER", "content": "three"}
    )
    hist = client.get(f"/v1/threads/{scope['thread_id']}/messages", headers=H).json()
    assert [m["content"] for m in hist["messages"]] == ["one", "two", "three"]


def test_validation_errors(client) -> None:
    r = client.post(
        "/v1/messages",
        headers=H,
        json={"scope": {"thread_id": new_id("thread")}, "role": "USER", "content": "x"},
    )
    assert r.status_code == 422
    r = client.post(
        "/v1/messages",
        headers=H,
        json={"scope": _scope(), "role": "USER", "content": "x", "unknown": 1},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION"
    assert client.get("/v1/jobs/obx_999999", headers=H).status_code == 404
    assert client.get("/v1/jobs/nope", headers=H).status_code == 404


def test_create_thread_endpoint_is_idempotent(client) -> None:
    body = {"scope": {}, "thread_id": new_id("thread"), "title": "Q3"}
    a = client.post("/v1/threads", headers={**H, "Idempotency-Key": "t-1"}, json=body)
    b = client.post("/v1/threads", headers={**H, "Idempotency-Key": "t-1"}, json=body)
    assert a.status_code == 201 and b.status_code == 201 and a.json() == b.json()
    assert client.get(f"/v1/threads/{body['thread_id']}", headers=H).json()["title"] == "Q3"
    assert client.delete(f"/v1/threads/{body['thread_id']}", headers=H).status_code == 204
    assert client.get(f"/v1/threads/{body['thread_id']}", headers=H).status_code == 404


async def test_sdk_ninety_percent_path(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(
        tenant_id="acme",
        user_id="u1",
        thread_id=new_id("thread"),
        session_id=new_id("session"),
        turn_id=new_id("turn"),
    )
    ack = await ctx.chat.user("What changed in EBITDA?")
    assert ack.sequence == 1 and ack.job_ids
    job = await ctx.job(ack.job_ids[0])
    assert job.status == "SUCCEEDED"
    await ctx.chat.assistant("EBITDA rose because of restructuring savings.")
    history = await ctx.chat.history()
    assert [m.role for m in history] == ["USER", "ASSISTANT"]
    thread = await ctx.chat.thread()
    assert thread.thread_id == ctx.scope.thread_id
    # child agent context records internal lineage without touching the visible chat
    agent = ctx.agent("research")
    await agent.chat.internal("searched 3 filings")
    assert len(await ctx.chat.history()) == 2
    assert len(await ctx.chat.history(include_internal=True)) == 3
    # another user is denied; a missing thread is NotFound
    other = memory.bind(tenant_id="acme", user_id="u2", thread_id=ctx.scope.thread_id)
    with pytest.raises(AuthorizationError):
        await other.chat.thread()
    missing = memory.bind(tenant_id="acme", user_id="u1", thread_id=new_id("thread"))
    with pytest.raises(NotFoundError):
        await missing.chat.thread()
    with pytest.raises(ValidationError):
        await memory.bind(tenant_id="acme", user_id="u1").chat.user("no lineage")
    await memory.aclose()
