"""The messages form of the conversation window (``modules/context/window.py``): verbatim, in
order, a tool call never parted from its results, cut from the newest turn back, whole turns
only, the newest always (flagged when it alone is over the budget)."""

from __future__ import annotations

import itertools

from memory_service.domain.context_bundle import (
    EVIDENCE_HEADING,
    ContextBundle,
    ConversationWindow,
    EvidenceReport,
)
from memory_service.domain.conversation import Message, MessageToolCall
from memory_service.domain.enums import (
    EvidenceStatus,
    MessageKind,
    MessageRole,
    QueryType,
    WindowFormat,
)
from memory_service.domain.ids import content_hash
from memory_service.modules.context.views import conversation_body
from memory_service.modules.context.window import (
    MESSAGE_OVERHEAD_TOKENS,
    message_tokens,
    replay_window,
    replayable,
    turns,
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


# -- whole turns: a follow-up about an answer needs the calls and results behind it ---------


def _turn(question: str, results: list[str], answer: str) -> list[Message]:
    """One turn: the question, a call per result (each its own ASSISTANT message, as a ReAct
    loop makes them) with its result, and the answer."""
    out = [_m(U, question)]
    for n, result in enumerate(results):
        call = f"{question[:8]}-{n}"
        out += [_m(A, calls=(call,)), _m(T, result, answers=call)]
    return [*out, _m(A, answer)]


def _cost(messages: list[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


def _whole_pairs(window: ConversationWindow) -> None:
    """Every call in ``window`` has its result in it, and every result its call."""
    calls = {c.id for m in window.messages for c in m.tool_calls}
    results = {m.tool_call_id for m in window.messages if m.role == "TOOL"}
    assert calls == results


def test_a_10k_character_tool_result_replays_with_its_question_and_call() -> None:
    """The reproduction: one quote of ~10.9k characters. The window used to stop at the
    call-and-result unit and replay the answer alone, so "where did you get that?" had
    nothing to go on."""
    result = '{"symbol": "ACME", "source": "Nasdaq Last Sale feed", "ticks": "' + "t" * 10_000
    turn = _turn("What did ACME close at yesterday?", [result], "ACME closed at 123.45 USD.")
    # the follow-up's context, before its own question is written: the turn is the newest
    w = replay_window("thr", turn, 2000)
    assert _said(w) == [(m.role.value, m.content) for m in turn]
    assert w.over_budget and w.token_estimate == _cost(turn) > 2000
    # the question written first: a budget the caller sized for its model keeps both turns
    follow_up = _m(U, "Where did you get that answer?")
    w = replay_window("thr", [*turn, follow_up], 8000)
    assert [m.message_id for m in w.messages] == [m.message_id for m in [*turn, follow_up]]
    assert not w.over_budget
    _whole_pairs(w)


def test_twelve_calls_in_one_turn_replay_whole() -> None:
    """26 messages: past a limit counted in messages (20), which cut the turn's question."""
    turn = _turn("Compare the twelve stores.", [f"store {n}: 4 units" for n in range(12)], "S4.")
    w = replay_window("thr", turn, 2000, max_turns=20)
    assert len(w.messages) == 26 and w.messages[0].role == "USER"
    assert sum(len(m.tool_calls) for m in w.messages) == 12
    assert not w.over_budget
    _whole_pairs(w)


def test_a_newest_turn_over_the_budget_is_kept_and_flagged() -> None:
    older = _turn("What is my plan?", [], "Pro.")
    newest = _turn("Quote ACME.", ["q" * 6000, "r" * 6000], "123.45")
    w = replay_window("thr", [*older, *newest], 500)
    assert [m.message_id for m in w.messages] == [m.message_id for m in newest]
    assert w.over_budget and w.token_estimate == _cost(newest)
    _whole_pairs(w)


def test_a_window_within_its_budget_is_not_flagged() -> None:
    turn = _turn("Quote ACME.", ["123.45"], "123.45")
    assert not replay_window("thr", turn, 2000).over_budget
    assert not replay_window("thr", [], 2000).over_budget


def test_an_older_client_gets_the_window_it_got_before() -> None:
    """No budget named (the service's 2000) and a newest turn that fits: the window the
    earlier build cut from the same messages (measured on origin/main f86a141, which agreed
    with this cut at every budget from the newest turn's cost up to 2100)."""
    stray = [_m(A, "a stray answer under the summary")]
    thread = stray + [
        m
        for n in range(8)
        for m in _turn(
            f"question {n}: " + "word " * 40,
            [f'{{"ok": true, "n": {n}}}'] if n % 3 == 0 else [],
            f"answer {n}: " + "word " * 40,
        )
    ]
    read = thread[-20:]  # what the read of 20 messages gave it

    def labels(budget: int) -> list[str]:
        w = replay_window("thr", read, budget, max_turns=20)
        assert not w.over_budget
        return [m.text.split(":")[0] if m.text and m.role != "TOOL" else m.role for m in w.messages]

    def turns_from(first: int) -> list[str]:
        out: list[str] = []
        for n in range(first, 8):
            out += [f"question {n}", *(["ASSISTANT", "TOOL"] if n % 3 == 0 else [])]
            out.append(f"answer {n}")
        return out

    assert labels(2000) == turns_from(1)
    assert labels(600) == turns_from(4)
    assert labels(300) == turns_from(6)


def test_a_call_and_its_result_are_never_parted_at_any_budget() -> None:
    thread = [
        *_turn("first", ["a" * 900, "b" * 50], "one"),
        *_turn("second", [], "two"),
        *_turn("third", ["c" * 300, "d" * 1200, "e" * 10], "three"),
        *_turn("fourth", ["f" * 400], "four"),
    ]
    user_ids = {m.message_id for m in thread if m.role is U}
    for budget in range(0, _cost(thread) + 50, 25):
        w = replay_window("thr", thread, budget)
        _whole_pairs(w)
        assert w.messages[0].message_id in user_ids  # whole turns: from a question
        assert w.messages[-1].text == "four"
        assert w.over_budget == (budget < _cost(thread[-4:]))


def test_turns_that_fit_are_kept_and_the_oldest_that_does_not_is_left_whole() -> None:
    oldest = _turn("oldest", ["x" * 4000], "old answer")
    middle = _turn("middle", ["m" * 200], "middle answer")
    newest = _turn("newest", [], "new answer")
    w = replay_window("thr", [*oldest, *middle, *newest], _cost([*middle, *newest]) + 10)
    assert [m.message_id for m in w.messages] == [m.message_id for m in [*middle, *newest]]
    assert not w.over_budget
    # nothing of the oldest turn: not its answer, not its call, not its result
    assert not {m.message_id for m in oldest} & set(w.message_ids)


def test_the_window_gives_at_most_max_turns_turns() -> None:
    thread = [m for n in range(25) for m in _turn(f"q{n}", [], f"a{n}")]
    w = replay_window("thr", thread, 100_000, max_turns=20)
    assert [m.text for m in w.messages if m.role == "USER"] == [f"q{n}" for n in range(5, 25)]
    assert replay_window("thr", thread, 100_000).message_ids == [m.message_id for m in thread]


def test_turns_group_whole_units_from_each_question() -> None:
    stray = _m(A, "before any question")
    first = _turn("q1", ["r1"], "a1")
    second = _turn("q2", [], "a2")
    pieces = units([stray, *first, *second])
    assert [[m for u in t for m in u] for t in turns(pieces)] == [[stray], first, second]


def test_the_flag_is_on_the_wire_only_when_set() -> None:
    newest = _turn("Quote ACME.", ["q" * 6000], "123.45")
    flagged = conversation_body(replay_window("thr", newest, 100))
    assert flagged is not None and flagged["over_budget"] is True
    plain = conversation_body(replay_window("thr", newest, 100_000))
    assert plain is not None and "over_budget" not in plain


def test_the_evidence_status_is_labelled_as_long_term_memorys() -> None:
    """INSUFFICIENT is about what memory holds, never the conversation: under a bare
    "Evidence status" it read as "nothing to answer from" to a model whose own conversation
    held the answer it was asked to explain."""
    turn = _turn("What did ACME close at?", ['{"source": "Nasdaq"}'], "123.45")
    bundle = ContextBundle(
        query="Where did you get that?",
        query_type=QueryType.CONVERSATION_HISTORY,
        conversation=replay_window("thr", turn, 2000),
        evidence=EvidenceReport(status=EvidenceStatus.INSUFFICIENT),
        token_budget=2000,
        token_estimate=0,
    )
    assert f"## {EVIDENCE_HEADING}\nINSUFFICIENT" in bundle.render()
    assert EVIDENCE_HEADING == "Evidence status (long-term memory)"
