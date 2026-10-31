"""The conversation window in the messages form, through the API on PostgreSQL: the caller's
``conversation_token_budget``, turns from the store in layers (``conversation_max_messages``
counts turns) - each turn's steps list, its text, the newest turn whole when it fits - the
newest turn's text and steps always, ``conversation.over_budget`` when they alone are over the
budget, the run-record facts a caller writes carried by the steps list, and the evidence
status labelled as long-term memory's.

The reproduction behind it: an agent asked "where did you get that?" after an answer drawn from
one ~10.9k-character tool result was replayed the bare answer - the window stopped at the
call-and-result unit and dropped the turn's question, the call and the result."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from memory_service.domain.ids import new_id
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]

H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
Q1 = "What did ACME close at yesterday?"
ANSWER = "ACME closed at 123.45 USD."
SOURCE = "Nasdaq Last Sale feed"
#: the reproduction's quote: the headline fields, then ~10k characters of intraday ticks
QUOTE = (
    f'{{"symbol": "ACME", "close": 123.45, "source": "{SOURCE}", "ticks": ['
    + ", ".join(
        f'{{"t": "15:{m:02d}:{s:02d}", "p": 120.5}}' for m in range(60) for s in range(0, 60, 12)
    )
    + "]}"
)


@pytest.fixture
def api(make_settings, make_overrides) -> Iterator[TestClient]:
    """The API with a queue nothing drains (the window is read from the store)."""
    app = create_app(make_settings(), overrides=make_overrides(tasks="memory"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _post(client: TestClient, thread: str, *messages: dict[str, Any]) -> None:
    r = client.post(
        "/v1/messages", headers=H, json={"scope": {"thread_id": thread}, "messages": messages}
    )
    assert r.status_code == 202, r.text


def _turn(question: str, results: list[str], answer: str) -> list[dict[str, Any]]:
    """A question, one ASSISTANT message per call (as a ReAct loop makes them) with its
    result, and the answer."""
    out: list[dict[str, Any]] = [{"role": "USER", "content": question}]
    for n, result in enumerate(results):
        call = f"call_{uuid.uuid4().hex[:12]}_{n}"
        out += [
            {
                "role": "ASSISTANT",
                "content": "",
                "tool_calls": [{"id": call, "name": "stock_quote", "arguments": {"n": n}}],
            },
            {"role": "TOOL", "kind": "INTERNAL", "content": result, "tool_call_id": call},
        ]
    return [*out, {"role": "ASSISTANT", "content": answer}]


def _context(client: TestClient, thread: str, query: str, **extra: Any) -> dict[str, Any]:
    body = {"scope": {"thread_id": thread}, "query": query, "window_format": "messages"} | extra
    r = client.post("/v1/context", headers=H, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    return body.get("conversation", {}).get("messages", [])


def _said(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m["role"], m["text"]) for m in _messages(body)]


def _pairs_whole(body: dict[str, Any]) -> None:
    messages = _messages(body)
    calls = {c["id"] for m in messages for c in m.get("tool_calls", [])}
    assert calls == {m["tool_call_id"] for m in messages if m["role"] == "TOOL"}


RECORD = "## Run record\n"


def _steps(body: dict[str, Any]) -> list[str]:
    """The steps lists of the context's run record (none is ever a replayed message)."""
    assert not [t for _, t in _said(body) if t.startswith("For ")]
    rendered = body["rendered"]
    if RECORD not in rendered:
        return []
    section = rendered.split(RECORD, 1)[1].split("\n## ", 1)[0]
    return section.split("\n\n")[1:]  # after the note saying who wrote it


def test_a_10k_character_tool_result_comes_back_with_its_question_and_call(api) -> None:
    thread = new_id("thread")
    _post(api, thread, *_turn(Q1, [QUOTE], ANSWER))
    # the follow-up's context, before its question is written: an older client (no budget)
    # gets the question, the steps list and the answer - the result does not fit 2000
    body = _context(api, thread, "Where did you get that answer?")
    assert _said(body)[0] == ("USER", Q1) and _said(body)[-1] == ("ASSISTANT", ANSWER)
    assert _steps(body) == [f'For "{Q1}" (1 call):\n1. stock_quote(n=0)']
    assert "over_budget" not in body["conversation"]
    _pairs_whole(body)
    # a caller that sizes the window for its model: the result too, nothing to trim
    sized = _context(api, thread, "Where did you get that answer?", conversation_token_budget=8000)
    assert [r for r, _ in _said(sized)] == ["USER", "ASSISTANT", "TOOL", "ASSISTANT"]
    assert SOURCE in _said(sized)[2][1] and "over_budget" not in sized["conversation"]
    _pairs_whole(sized)
    # the question written first, then asked: the earlier turn as its text and steps list
    _post(api, thread, {"role": "USER", "content": "Where did you get that answer?"})
    both = _context(api, thread, "Where did you get that answer?", conversation_token_budget=8000)
    assert [r for r, _ in _said(both)] == ["USER", "ASSISTANT", "USER"]


def test_twelve_calls_in_one_turn_are_read_and_listed(api) -> None:
    """26 messages: the read counted messages (20) and lost the turn's question and its
    first three calls."""
    thread = new_id("thread")
    _post(
        api,
        thread,
        *_turn("Compare the twelve stores.", [f"store {n}: 4" for n in range(12)], "S4."),
    )
    body = _context(api, thread, "Which steps did you take?")
    assert _said(body)[0] == ("USER", "Compare the twelve stores.")
    assert sum(len(m.get("tool_calls", [])) for m in _messages(body)) == 12
    assert _steps(body)[0].splitlines()[-1] == "12. stock_quote(n=11)"
    assert "over_budget" not in body["conversation"]
    _pairs_whole(body)


def test_a_newest_turn_over_the_budget_keeps_its_steps_and_is_flagged(api) -> None:
    thread = new_id("thread")
    _post(api, thread, *_turn("What is my plan?", [], "Pro."))
    _post(api, thread, *_turn(Q1 + " please" * 400, ["q" * 3000, "r" * 3000], ANSWER))
    body = _context(api, thread, "Why is it accurate?", conversation_token_budget=500)
    assert [r for r, _ in _said(body)] == ["USER", "ASSISTANT"]
    assert len(_steps(body)) == 1 and body["conversation"]["over_budget"] is True


def test_a_chat_without_calls_is_given_as_before(api) -> None:
    """No budget named, a plain chat within 20 messages: every message, oldest first."""
    thread = new_id("thread")
    turns = [_turn("Hi, my order number is QX-4471.", [], "Noted."), _turn("Thanks!", [], "Sure.")]
    for turn in turns:
        _post(api, thread, *turn)
    body = _context(api, thread, "What is my order number?")
    assert _said(body) == [(m["role"], m["content"]) for t in turns for m in t]
    assert "over_budget" not in body["conversation"]


def test_older_turns_give_their_steps_never_their_results(api) -> None:
    thread = new_id("thread")
    _post(api, thread, *_turn("oldest question", ["x" * 8000], "old answer"))
    _post(api, thread, *_turn("middle question", ["m" * 400], "middle answer"))
    _post(api, thread, *_turn("newest question", [], "new answer"))
    body = _context(api, thread, "and then?", conversation_token_budget=400)
    said = _said(body)
    assert said[0] == ("USER", "oldest question") and said[-1] == ("ASSISTANT", "new answer")
    assert len(_steps(body)) == 2 and all(r != "TOOL" for r, _ in said)
    assert "over_budget" not in body["conversation"]


def test_the_run_record_a_caller_writes_is_in_the_steps_list(api) -> None:
    thread = new_id("thread")
    turn = _turn(Q1, ['{"close": 123.45}'], ANSWER)
    turn[2]["custom_metadata"] = {"run_record": {"status": "ok", "seconds": 0.2, "sources": SOURCE}}
    turn[-1]["custom_metadata"] = {"run_record": {"status": "SUCCESS", "seconds": 4.0}}
    _post(api, thread, *turn)
    body = _context(api, thread, "Where is that from?")
    assert _steps(body) == [
        f'For "{Q1}" (1 call):\n1. stock_quote(n=0): ok, 0.2 s, source: {SOURCE}\n'
        "The run: status SUCCESS, 4.0 s."
    ]


def test_the_window_reaches_back_twenty_turns(api) -> None:
    thread = new_id("thread")
    for n in range(23):
        _post(api, thread, *_turn(f"q{n}", [], f"a{n}"))
    body = _context(api, thread, "and?", conversation_token_budget=100_000)
    assert [t for r, t in _said(body) if r == "USER"] == [f"q{n}" for n in range(3, 23)]


def test_the_budget_is_bounded_and_optional(api) -> None:
    thread = new_id("thread")
    _post(api, thread, *_turn("q", [], "a"))
    for wrong in (99, 1_000_001, "lots"):
        r = api.post(
            "/v1/context",
            headers=H,
            json={
                "scope": {"thread_id": thread},
                "query": "q?",
                "conversation_token_budget": wrong,
            },
        )
        assert r.status_code == 422 and r.json()["code"] == "VALIDATION", r.text
    # the rendered form takes it too
    rendered = api.post(
        "/v1/context",
        headers=H,
        json={"scope": {"thread_id": thread}, "query": "q?", "conversation_token_budget": 100},
    ).json()
    assert "## Recent conversation" in rendered["rendered"] and "conversation" not in rendered


def test_the_evidence_status_is_labelled_as_long_term_memorys(api) -> None:
    """Nothing in memory, the answer in the conversation: the status says INSUFFICIENT about
    memory, under a heading that says so, and the window still carries the turn."""
    thread = new_id("thread")
    _post(api, thread, *_turn(Q1, [QUOTE], ANSWER))
    body = _context(api, thread, "Where did you get that answer?", conversation_token_budget=8000)
    assert body["evidence_status"] == "INSUFFICIENT"
    assert "## Evidence status (long-term memory)\nINSUFFICIENT" in body["rendered"]
    assert SOURCE in _said(body)[2][1]


@pytest.fixture
def draining(make_settings, make_overrides) -> Iterator[TestClient]:
    """The API with its jobs run inline: a memory is indexed when it is remembered."""
    app = create_app(make_settings(), overrides=make_overrides(tasks="inline"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_a_large_window_does_not_crowd_the_memories_out(draining) -> None:
    """The messages form is the caller's to replay: a window over the context's budget (a
    caller's large one, or a newest turn given whole) takes the default conversation share of
    ``token_budget`` and no more."""
    user = {**H, "X-Trellis-User": f"u-{uuid.uuid4().hex[:8]}"}
    r = draining.post(
        "/v1/memories",
        headers={**user, "Idempotency-Key": uuid.uuid4().hex},
        json={"scope": {}, "content": "I prefer metric units for every measurement."},
    )
    assert r.status_code == 201, r.text
    thread = new_id("thread")
    r = draining.post(
        "/v1/messages",
        headers=user,
        json={"scope": {"thread_id": thread}, "messages": _turn(Q1, ["z " * 15_000], ANSWER)},
    )
    assert r.status_code == 202, r.text
    for extra in ({}, {"conversation_token_budget": 50_000}):
        r = draining.post(
            "/v1/context",
            headers=user,
            json={
                "scope": {"thread_id": thread},
                "query": "Which measurement units do I prefer?",
                "window_format": "messages",
                "token_budget": 4000,
                **extra,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("memory_ids"), (extra, body)
        assert "metric units" in body["rendered"]
        assert _said(body)[0] == ("USER", Q1)
