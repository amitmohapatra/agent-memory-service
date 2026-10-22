# ruff: noqa: RUF001 - literal multilingual fixtures.
"""The conversation window in the messages form (``window_format=messages``): the thread's own
messages back, verbatim and in order, tool calls with their results, the moment they are
written - with no worker running - for a caller that replays them as its own conversation."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.config.constants import CONTEXT
from memory_service.domain.ids import new_id
from tests.e2e.conftest import sdk_client

pytestmark = pytest.mark.e2e

H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
#: The section the rendered form shows the window in.
WINDOW_SECTION = "## Recent conversation"


@pytest.fixture
def stopped_worker(make_settings, make_overrides) -> Iterator[TestClient]:
    """The API with a queue nothing drains: no job a message queues ever runs, as with the
    worker down."""
    app = create_app(make_settings(), overrides=make_overrides(tasks="memory"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _post(client: TestClient, thread: str, *messages: dict[str, Any]) -> list[dict[str, Any]]:
    r = client.post(
        "/v1/messages", headers=H, json={"scope": {"thread_id": thread}, "messages": messages}
    )
    assert r.status_code == 202, r.text
    return r.json()["messages"]


def _context(client: TestClient, thread: str, query: str, **extra: Any) -> dict[str, Any]:
    body = {"scope": {"thread_id": thread}, "query": query, "window_format": "messages"} | extra
    r = client.post("/v1/context", headers=H, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _said(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m["role"], m["text"]) for m in body.get("conversation", {}).get("messages", [])]


def test_a_turn_is_in_the_next_window_at_once_verbatim_and_in_order(stopped_worker) -> None:
    thread = new_id("thread")
    turns = [
        ("Hi, my order number is QX-4471. Please keep it in mind.", "Noted: QX-4471."),
        ("Also, I prefer deliveries on Tuesdays.", "Tuesdays it is."),
    ]
    for question, answer in turns:
        _post(
            stopped_worker,
            thread,
            {"role": "USER", "content": question},
            {"role": "ASSISTANT", "content": answer},
        )
    body = _context(stopped_worker, thread, "What is my order number?")
    assert _said(body) == [(r, t) for q, a in turns for r, t in (("USER", q), ("ASSISTANT", a))]
    assert body["conversation"]["thread_id"] == thread
    # replayed by the caller, so not rendered as well
    assert WINDOW_SECTION not in body["rendered"]
    # the rendered form is as it was
    rendered = stopped_worker.post(
        "/v1/context",
        headers=H,
        json={"scope": {"thread_id": thread}, "query": "What is my order number?"},
    ).json()
    assert WINDOW_SECTION in rendered["rendered"] and "conversation" not in rendered
    assert "QX-4471. Please keep it in mind." in rendered["rendered"]


def test_a_turn_written_after_a_context_is_in_the_next_one(stopped_worker) -> None:
    """The bundle cache follows the thread: a cached window is never served stale."""
    thread = new_id("thread")
    _post(stopped_worker, thread, {"role": "USER", "content": "first"})
    assert _said(_context(stopped_worker, thread, "anything?")) == [("USER", "first")]
    _post(stopped_worker, thread, {"role": "ASSISTANT", "content": "second"})
    assert _said(_context(stopped_worker, thread, "anything?")) == [
        ("USER", "first"),
        ("ASSISTANT", "second"),
    ]


def test_tool_calls_come_back_with_their_results(stopped_worker) -> None:
    thread = new_id("thread")
    calls = [
        {"id": "call_1", "name": "get_order", "arguments": {"order_id": "QX-4471"}},
        {"id": "call_2", "name": "get_eta", "arguments": {"order_id": "QX-4471", "days": 2}},
    ]
    _post(
        stopped_worker,
        thread,
        {"role": "USER", "content": "Where is order QX-4471?"},
        {"role": "ASSISTANT", "content": "", "tool_calls": calls},
        {
            "role": "TOOL",
            "kind": "INTERNAL",
            "content": '{"status": "shipped"}',
            "tool_call_id": "call_1",
        },
        {
            "role": "TOOL",
            "kind": "INTERNAL",
            "content": '{"eta": "Tuesday"}',
            "tool_call_id": "call_2",
        },
        {"role": "ASSISTANT", "content": "Shipped; it arrives Tuesday."},
    )
    messages = _context(stopped_worker, thread, "When does it arrive?")["conversation"]["messages"]
    assert [m["role"] for m in messages] == ["USER", "ASSISTANT", "TOOL", "TOOL", "ASSISTANT"]
    assert messages[1]["tool_calls"] == calls and "tool_call_id" not in messages[1]
    assert [m.get("tool_call_id") for m in messages[2:4]] == ["call_1", "call_2"]
    assert messages[2]["text"] == '{"status": "shipped"}'
    assert "tool_calls" not in messages[0] and "tool_calls" not in messages[4]
    # the visible history is still the chat: the call's message, not its results
    listed = stopped_worker.get(f"/v1/threads/{thread}/messages", headers=H).json()["messages"]
    assert [m["role"] for m in listed] == ["USER", "ASSISTANT", "ASSISTANT"]
    assert listed[1]["tool_calls"] == calls
    internal = stopped_worker.get(
        f"/v1/threads/{thread}/messages", headers=H, params={"include_internal": True}
    ).json()["messages"]
    assert [m.get("tool_call_id") for m in internal if m["role"] == "TOOL"] == ["call_1", "call_2"]


def test_multilingual_turns_are_given_back_as_written(stopped_worker) -> None:
    thread = new_id("thread")
    texts = [
        ("USER", "我的订单号是 QX-4471，请记住。"),
        ("ASSISTANT", "好的，已记录 QX-4471。"),
        ("USER", "Je préfère être livré le mardi — merci !"),
        ("ASSISTANT", "تم، سنوصل يوم الثلاثاء."),
        ("USER", "मेरी टीम में 14 सीटें हैं।"),
    ]
    _post(stopped_worker, thread, *({"role": r, "content": t} for r, t in texts))
    assert _said(_context(stopped_worker, thread, "订单号是多少？")) == texts


def test_an_empty_thread_has_no_window(stopped_worker) -> None:
    body = _context(stopped_worker, new_id("thread"), "What did I say?")
    assert "conversation" not in body and WINDOW_SECTION not in body["rendered"]


def test_a_long_thread_keeps_its_newest_turns_within_the_budget(stopped_worker) -> None:
    thread = new_id("thread")
    turns = 12
    for n in range(turns):
        _post(
            stopped_worker,
            thread,
            {"role": "USER", "content": f"question {n}: " + "word " * 120},
            {"role": "ASSISTANT", "content": f"answer {n}: " + "word " * 120},
        )
    said = _said(_context(stopped_worker, thread, "what was the last answer?"))
    assert 0 < len(said) < 2 * turns
    assert said[0][0] == "USER" and said[-1][1].startswith(f"answer {turns - 1}:")
    # the newest, contiguous: no turn skipped
    first = int(said[0][1].split(":")[0].removeprefix("question "))
    assert [t.split(":")[0] for _, t in said] == [
        f"{kind} {n}" for n in range(first, turns) for kind in ("question", "answer")
    ]
    assert len(said) * 120 // 4 <= CONTEXT.conversation_token_budget * 2


def test_tool_fields_belong_to_their_roles(stopped_worker) -> None:
    thread = new_id("thread")
    for wrong in (
        {"role": "USER", "content": "hi", "tool_calls": [{"id": "c", "name": "t"}]},
        {"role": "ASSISTANT", "content": "hi", "tool_call_id": "c"},
    ):
        r = stopped_worker.post(
            "/v1/messages", headers=H, json={"scope": {"thread_id": thread}, "messages": [wrong]}
        )
        assert r.status_code == 422 and r.json()["code"] == "VALIDATION", r.text
    # nothing was stored
    assert "conversation" not in _context(stopped_worker, thread, "hi?")


def test_concurrent_turns_on_one_thread_stay_whole(make_settings, make_overrides) -> None:
    """Two runs answering at once on the same thread: each turn's batch is appended whole
    (one transaction under the thread's lock), so neither interleaves the other."""
    app = create_app(make_settings(), overrides=make_overrides(tasks="memory"))
    thread = new_id("thread")

    async def run() -> list[tuple[str, str]]:
        async with app.router.lifespan_context(app):
            client = sdk_client(app)
            async with client:
                ctx = client.bind(tenant_id="acme", user_id="u1", thread_id=thread)
                await asyncio.gather(
                    *(
                        ctx.history.add([("USER", f"q{n}"), ("ASSISTANT", f"a{n}")])
                        for n in range(6)
                    )
                )
                found = await ctx.context("what was said?", window_format="messages")
        assert found.conversation is not None
        return [(m.role, m.text) for m in found.conversation.messages]

    said = asyncio.run(run())
    assert len(said) == 12
    for user, assistant in zip(said[::2], said[1::2], strict=True):
        assert user[0] == "USER" and assistant == ("ASSISTANT", "a" + user[1][1:])


def test_results_of_different_calls_are_different_messages(stopped_worker) -> None:
    """Two requests alike but for the call their result answers are two messages, not a
    retry of one."""
    thread = new_id("thread")
    _post(
        stopped_worker,
        thread,
        {"role": "USER", "content": "check both"},
        {"role": "ASSISTANT", "content": "", "tool_calls": [{"id": "c1", "name": "t"}]},
    )
    for call in ("c1", "c2"):
        [ack] = _post(
            stopped_worker,
            thread,
            {"role": "TOOL", "kind": "INTERNAL", "content": "ok", "tool_call_id": call},
        )
        assert not ack["deduplicated"]
    internal = stopped_worker.get(
        f"/v1/threads/{thread}/messages", headers=H, params={"include_internal": True}
    ).json()["messages"]
    assert [m.get("tool_call_id") for m in internal if m["role"] == "TOOL"] == ["c1", "c2"]
