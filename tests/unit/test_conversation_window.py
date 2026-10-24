"""The messages form of the conversation window (``modules/context/window.py``): verbatim, in
order, a tool call never parted from its results, cut from the newest unit back."""

from __future__ import annotations

import itertools

from memory_service.domain.context_bundle import ContextBundle, ConversationWindow, EvidenceReport
from memory_service.domain.conversation import Message, MessageToolCall
from memory_service.domain.enums import (
    EvidenceStatus,
    MessageKind,
    MessageRole,
    QueryType,
    WindowFormat,
)
from memory_service.domain.ids import content_hash
from memory_service.modules.context.window import (
    MESSAGE_OVERHEAD_TOKENS,
    message_tokens,
    replay_window,
    replayable,
    units,
)

_seq = itertools.count(1)


def _m(
    role: MessageRole,
    content: str = "",
    *,
    calls: tuple[str, ...] = (),
    answers: str | None = None,
    kind: MessageKind | None = None,
) -> Message:
    n = next(_seq)
    internal = role in (MessageRole.TOOL, MessageRole.AGENT, MessageRole.EVENT)
    return Message(
        message_id=f"m{n}",
        tenant_id="t",
        thread_id="thr",
        session_id="s",
        turn_id="trn",
        sequence=n,
        role=role,
        kind=kind or (MessageKind.INTERNAL if internal else MessageKind.VISIBLE),
        content=content,
        content_hash=content_hash(content),
        author_principal="agent:a",
        tool_calls=[MessageToolCall(id=c, name="lookup", arguments={"q": c}) for c in calls],
        tool_call_id=answers,
    )


U, A, T = MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL


def _said(window: ConversationWindow) -> list[tuple[str, str]]:
    return [(m.role, m.text) for m in window.messages]


def test_the_window_is_the_thread_verbatim_and_in_order() -> None:
    msgs = [_m(U, "Hi, I'm Inês — QX-4471."), _m(A, "Noted."), _m(U, "Tuesdays?"), _m(A, "Yes.")]
    w = replay_window("thr", msgs, 2000)
    assert _said(w) == [(m.role.value, m.content) for m in msgs]
    assert w.message_ids == [m.message_id for m in msgs]
    assert w.form is WindowFormat.MESSAGES and w.thread_id == "thr"
    assert w.token_estimate == sum(message_tokens(m) for m in msgs)


def test_an_empty_thread_is_an_empty_window() -> None:
    w = replay_window("thr", [], 2000)
    assert w.messages == [] and w.token_estimate == 0 and w.form is WindowFormat.MESSAGES


def test_a_tool_call_comes_with_its_results() -> None:
    msgs = [
        _m(U, "where is it?"),
        _m(A, calls=("c1", "c2")),
        _m(T, "shipped", answers="c1"),
        _m(T, "tuesday", answers="c2"),
        _m(A, "Shipped, Tuesday."),
    ]
    w = replay_window("thr", msgs, 2000)
    assert [m.role for m in w.messages] == ["USER", "ASSISTANT", "TOOL", "TOOL", "ASSISTANT"]
    assert [c.id for c in w.messages[1].tool_calls] == ["c1", "c2"]
    assert w.messages[1].tool_calls[0].arguments == {"q": "c1"}
    assert [m.tool_call_id for m in w.messages[2:4]] == ["c1", "c2"]
    assert units(msgs) == [[msgs[0]], msgs[1:4], [msgs[4]]]


def test_the_budget_never_parts_a_call_from_its_results() -> None:
    call = [_m(U, "q"), _m(A, calls=("c1",)), _m(T, "r" * 400, answers="c1")]
    tail = [_m(A, "done"), _m(U, "next?"), _m(A, "ok")]
    budget = sum(message_tokens(m) for m in tail) + 5  # the call unit does not fit
    w = replay_window("thr", call + tail, budget)
    assert _said(w) == [("USER", "next?"), ("ASSISTANT", "ok")]  # starts on a USER message
    assert all(m.role != "TOOL" for m in w.messages)


def test_the_newest_unit_is_given_even_past_the_budget() -> None:
    msgs = [_m(U, "old"), _m(U, "x" * 4000)]
    w = replay_window("thr", msgs, 10)
    assert _said(w) == [("USER", "x" * 4000)]
    assert w.token_estimate == message_tokens(msgs[1]) > 10


def test_the_window_stops_at_the_first_turn_that_does_not_fit() -> None:
    msgs = [_m(U, "a"), _m(A, "b"), _m(U, "long " * 300), _m(A, "c"), _m(U, "d"), _m(A, "e")]
    keep = msgs[4:]
    w = replay_window("thr", msgs, sum(message_tokens(m) for m in [*keep, msgs[3]]) + 1)
    # "c" fits but its question does not, and nothing older is taken past the gap
    assert _said(w) == [("USER", "d"), ("ASSISTANT", "e")]


def test_a_window_with_no_user_message_keeps_what_fits() -> None:
    msgs = [_m(A, "one"), _m(A, "two")]
    assert _said(replay_window("thr", msgs, 2000)) == [("ASSISTANT", "one"), ("ASSISTANT", "two")]


def test_an_unanswered_call_is_not_replayed() -> None:
    """A run stopped between a call and its result: the call is dropped (a model API refuses
    it alone); what the message said stays."""
    spoke = _m(A, "Let me check.", calls=("c1",))
    silent = _m(A, calls=("c2",))
    w = replay_window("thr", [_m(U, "q"), spoke, _m(U, "q2"), silent], 2000)
    assert _said(w) == [("USER", "q"), ("ASSISTANT", "Let me check."), ("USER", "q2")]
    assert w.messages[1].tool_calls == []


def test_a_result_whose_call_was_cut_off_is_dropped() -> None:
    w = replay_window("thr", [_m(T, "orphan", answers="c0"), _m(U, "q"), _m(A, "a")], 2000)
    assert _said(w) == [("USER", "q"), ("ASSISTANT", "a")]


def test_a_result_of_no_call_in_the_window_is_dropped_wherever_it_is() -> None:
    msgs = [_m(U, "q"), _m(A, "a"), _m(T, "stray", answers="c9"), _m(A, "b")]
    assert _said(replay_window("thr", msgs, 2000)) == [
        ("USER", "q"),
        ("ASSISTANT", "a"),
        ("ASSISTANT", "b"),
    ]


def test_only_the_chat_and_the_tool_results_are_replayed() -> None:
    assert replayable(_m(U, "q")) and replayable(_m(T, "r", answers="c"))
    assert not replayable(_m(T, "r"))  # a tool result naming no call
    assert not replayable(_m(MessageRole.AGENT, "plan"))
    assert not replayable(_m(MessageRole.EVENT, "happened"))
    assert not replayable(_m(U, "note", kind=MessageKind.INTERNAL))


def test_a_message_costs_its_text_its_calls_and_the_overhead() -> None:
    plain = _m(U, "abcd" * 10)
    called = _m(A, "abcd" * 10, calls=("c1",))
    assert message_tokens(plain) == 10 + MESSAGE_OVERHEAD_TOKENS
    assert message_tokens(called) > message_tokens(plain)


def test_the_messages_form_is_not_rendered_but_still_dedupes_memories() -> None:
    msgs = [_m(U, "My order is QX-4471."), _m(A, "Noted.")]
    window = replay_window("thr", msgs, 2000)
    bundle = ContextBundle(
        query="order?",
        query_type=QueryType.USER_MEMORY,
        conversation=window,
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=2000,
        token_estimate=window.token_estimate,
    )
    assert "## Recent conversation" not in bundle.render()
    rendered = bundle.model_copy(
        update={"conversation": window.model_copy(update={"form": WindowFormat.RENDERED})}
    )
    assert "## Recent conversation\nUSER: My order is QX-4471." in rendered.render()
