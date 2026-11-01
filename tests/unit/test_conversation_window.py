"""The messages form of the conversation window (``modules/context/window.py``): in order, a
tool call never parted from its results, cut from the newest turn back in layers - every
turn's steps list, then its text, then the newest turn whole - the newest turn's text and steps
always (flagged when they alone are over the budget). The steps lists are the window's run
record, rendered in the context: never a message the model would read as its own."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

from memory_service.domain.context_bundle import (
    EVIDENCE_HEADING,
    RUN_RECORD_HEADING,
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
    RUN_RECORD,
    message_tokens,
    replay_window,
    replayable,
    steps_list,
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
    assert w.messages[4].text == "Shipped, Tuesday."  # the answer as the agent gave it
    assert w.record.startswith('For "where is it?" (2 calls):')  # the steps, in the record
    assert w.message_ids == [m.message_id for m in msgs]
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


def test_the_run_record_is_in_the_context_and_no_message_is_put_in_the_agent_s_mouth() -> None:
    """The steps list used to be replayed as an ASSISTANT message, and a model asked about it
    disowned it as a claim it had made up. It is the context's run record now, under a
    heading that says who wrote it; the agent's own answer stays its message."""
    turn = _turn("What did ACME close at?", ["123.45"], "123.45 USD.")
    window = replay_window("thr", [*turn, _m(U, "Where is that from?")], 100_000)
    assert [(m.role, m.text) for m in window.messages if m.role == "ASSISTANT"] == [
        ("ASSISTANT", "123.45 USD.")
    ]
    bundle = ContextBundle(
        query="source?",
        query_type=QueryType.USER_MEMORY,
        conversation=window,
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=2000,
        token_estimate=window.token_estimate,
    )
    rendered = bundle.render()
    assert f"## {RUN_RECORD_HEADING}\n" in rendered
    assert 'For "What did ACME close at?" (1 call):\n1. lookup(q="What did-0")' in rendered


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


def _steps(window: ConversationWindow) -> list[str]:
    return window.record.split("\n\n") if window.record else []


def test_a_10k_character_tool_result_replays_with_its_question_and_call() -> None:
    """The reproduction: one quote of ~10.9k characters. The window used to stop at the
    call-and-result unit and replay the answer alone, so "where did you get that?" had
    nothing to go on. Now the turn's question, answer and steps list always come back, and
    the result too when the caller's budget holds it."""
    result = '{"symbol": "ACME", "source": "Nasdaq Last Sale feed", "ticks": "' + "t" * 10_000
    turn = _turn("What did ACME close at yesterday?", [result], "ACME closed at 123.45 USD.")
    w = replay_window("thr", turn, 2000)  # too small for the result
    assert _said(w)[0] == ("USER", turn[0].content) and _said(w)[-1] == (
        "ASSISTANT",
        turn[-1].content,
    )
    assert all(m.role != "TOOL" and not m.tool_calls for m in w.messages)
    assert _steps(w) == [
        'For "What did ACME close at yesterday?" (1 call):\n1. lookup(q="What did-0")'
    ]
    assert not w.over_budget and w.token_estimate < 2000
    w = replay_window("thr", turn, 8000)  # sized for the model: kept whole
    assert [m.role for m in w.messages if m.role == "TOOL"] == ["TOOL"]
    assert not w.over_budget
    _whole_pairs(w)


def test_an_older_turn_never_replays_its_results() -> None:
    first = _turn("What did ACME close at?", ["123.45"], "123.45 USD.")
    second = _m(U, "And ZETA?")
    w = replay_window("thr", [*first, second], 100_000)
    assert [m.role for m in w.messages] == ["USER", "ASSISTANT", "USER"]
    assert not any(m.tool_calls for m in w.messages) and len(_steps(w)) == 1
    assert w.message_ids == [first[0].message_id, first[-1].message_id, second.message_id]


def test_twelve_calls_in_one_turn_list_twelve_steps() -> None:
    turn = _turn("Compare the twelve stores.", [f"store {n}: 4 units" for n in range(12)], "S4.")
    w = replay_window("thr", turn, 2000, max_turns=20)
    assert w.messages[0].role == "USER" and sum(len(m.tool_calls) for m in w.messages) == 12
    (listed,) = _steps(w)
    assert listed.splitlines()[0].endswith("(12 calls):") and listed.splitlines()[-1].startswith(
        "12. "
    )
    assert not w.over_budget
    _whole_pairs(w)


def test_a_newest_turn_over_the_budget_keeps_its_text_and_steps_and_is_flagged() -> None:
    older = _turn("What is my plan?", [], "Pro.")
    newest = _turn("Quote ACME " + "please " * 300, ["q" * 6000, "r" * 6000], "123.45")
    w = replay_window("thr", [*older, *newest], 50)
    assert _said(w)[0][1] == newest[0].content and _said(w)[-1] == ("ASSISTANT", "123.45")
    assert len(_steps(w)) == 1 and all(m.role != "TOOL" for m in w.messages)
    assert w.over_budget and w.token_estimate > 50


def test_a_window_within_its_budget_is_not_flagged() -> None:
    turn = _turn("Quote ACME.", ["123.45"], "123.45")
    assert not replay_window("thr", turn, 2000).over_budget
    assert not replay_window("thr", [], 2000).over_budget


def test_a_chat_without_calls_is_given_verbatim() -> None:
    """No calls, no steps list: the window an older client got for a plain chat."""
    thread = [
        m for n in range(8) for m in _turn(f"question {n}: " + "word " * 40, [], f"answer {n}")
    ]
    w = replay_window("thr", thread, 2000, max_turns=20)
    assert _said(w) == [(m.role.value, m.content) for m in thread[-len(w.messages) :]]
    assert not _steps(w) and w.messages[0].role == "USER"


def test_a_call_and_its_result_are_never_parted_at_any_budget() -> None:
    thread = [
        *_turn("first", ["a" * 900, "b" * 50], "one"),
        *_turn("second", [], "two"),
        *_turn("third", ["c" * 300, "d" * 1200, "e" * 10], "three"),
        *_turn("fourth", ["f" * 400], "four"),
    ]
    for budget in range(0, sum(message_tokens(m) for m in thread) + 50, 25):
        w = replay_window("thr", thread, budget)
        _whole_pairs(w)
        assert w.messages[-1].text == "four" and w.token_estimate <= max(budget, w.token_estimate)
        assert w.over_budget == (w.token_estimate > budget)


def test_short_of_space_results_go_first_and_steps_lists_last() -> None:
    oldest = _turn("oldest", ["x" * 400], "old answer " + "w " * 200)
    middle = _turn("middle", ["m" * 400], "middle answer " + "w " * 200)
    newest = _turn("newest", ["n" * 4000], "new answer")
    thread = [*oldest, *middle, *newest]
    roomy = replay_window("thr", thread, 100_000)
    assert len(_steps(roomy)) == 3 and any(m.role == "TOOL" for m in roomy.messages)
    tight = replay_window("thr", thread, 150)  # no room for the newest result, nor old texts
    assert len(_steps(tight)) == 3 and all(m.role != "TOOL" for m in tight.messages)
    assert [m.text for m in tight.messages if m.role == "USER"] == ["newest"]
    assert not tight.over_budget


def test_the_steps_list_carries_the_recorded_facts_and_nothing_else() -> None:
    turn = _turn("What did ACME close at?", ["123.45"], "123.45 USD.")
    turn[2].custom_metadata[RUN_RECORD] = {
        "status": "OK",
        "seconds": 0.04,
        "sources": "Nasdaq Last Sale feed, 2026-10-27",
    }
    turn[-1].custom_metadata[RUN_RECORD] = {
        "status": "SUCCESS",
        "seconds": 3.21,
        "approvals": "none",
    }
    assert steps_list(turns(units(turn))[0], NOW).splitlines() == [
        'For "What did ACME close at?" (1 call):',
        '1. lookup(q="What did-0"): OK, 0.0 s, source: Nasdaq Last Sale feed, 2026-10-27',
        "The run: status SUCCESS, 3.2 s, approvals: none.",
        "Evidence: 1 of 1 calls succeeded; 1 source: Nasdaq Last Sale feed, 2026-10-27 (named "
        "by 1 of 1). How sure, from this record: high.",
    ]
    bare = _turn("again?", ["x"], "y")
    assert (
        steps_list(turns(units(bare))[0], NOW) == 'For "again?" (1 call):\n1. lookup(q="again?-0")'
    )
    unstatused = _recorded(_turn("q", ["a"], "x"), {"sources": "Tape"})
    assert "no call status recorded; 1 source: Tape" in steps_list(turns(units(unstatused))[0])
    assert steps_list(turns(units(_turn("hi", [], "hello")))[0]) == ""


NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


def _recorded(turn: list[Message], *facts: dict[str, object]) -> list[Message]:
    """``turn`` with each tool result's run-record ``facts``, in order."""
    for result, said in zip([m for m in turn if m.role is T], facts, strict=True):
        result.custom_metadata[RUN_RECORD] = said
    return turn


def test_the_fetch_time_is_said_once_in_the_steps_list() -> None:
    turn = _recorded(
        _turn("ACME?", ["123.45"], "123.45"),
        {"status": "ok", "fetched": "2026-10-31T11:00:00+00:00", "max_age": 0},
    )
    record = steps_list(turns(units(turn))[0], NOW)
    assert '1. lookup(q="ACME?-0"): ok, fetched: 2026-10-31T11:00:00+00:00' in record
    assert record.count("2026-10-31T11:00:00") == 1
    assert "out of date now: lookup (fetch again for current values)" in record
    assert record.endswith("How sure, from this record: medium.")


def test_how_sure_is_graded_from_the_record_never_from_the_answer() -> None:
    sourced = {"status": "ok", "sources": "Tape", "fetched": "2026-10-31T11:59:00Z"}
    high = _recorded(_turn("q", ["a", "b"], "I am certain."), sourced, sourced | {"max_age": 600})
    assert steps_list(turns(units(high))[0], NOW).endswith(
        "Evidence: 2 of 2 calls succeeded; 1 source: Tape (named by 2 of 2). How sure, from "
        "this record: high."
    )
    half = _recorded(_turn("q", ["a", "b"], "I am certain."), sourced, {"status": "error"})
    assert "1 of 2 calls succeeded" in steps_list(turns(units(half))[0], NOW)
    assert steps_list(turns(units(half))[0], NOW).endswith("medium.")
    none = _recorded(_turn("q", ["a"], "I am certain."), {"status": "timeout"})
    assert steps_list(turns(units(none))[0], NOW).endswith("low.")
    failed = _recorded(_turn("q", ["a"], "ok"), sourced)
    failed[-1].custom_metadata[RUN_RECORD] = {"status": "FAILED"}
    assert steps_list(turns(units(failed))[0], NOW).endswith("low.")
    two = _recorded(_turn("q", ["a", "b"], "x"), sourced, sourced | {"sources": "Feed; Tape"})
    assert "2 sources: Tape (named by 2 of 2); Feed (named by 1 of 2)" in steps_list(
        turns(units(two))[0], NOW
    )
    odd = _recorded(_turn("q", ["a"], "x"), {"status": "ok", "fetched": "soon", "max_age": 1})
    assert "out of date" not in steps_list(turns(units(odd))[0], NOW)
    naive = _recorded(_turn("q", ["a"], "x"), {"fetched": "2026-10-31T10:00:00", "max_age": 60})
    assert "out of date now: lookup" in steps_list(turns(units(naive))[0], NOW)


def test_a_step_says_which_earlier_result_its_argument_came_from() -> None:
    first, second = _m(A, calls=("c1",)), _m(A, calls=("c2",))
    first.tool_calls[0].arguments["q"] = "ACME"
    second.tool_calls[0].arguments.update({"q": "SKU-77", "n": 1})
    turn = [_m(U, "stock of ACME?"), first, _m(T, '{"sku": "SKU-77"}', answers="c1")]
    turn += [second, _m(T, "7", answers="c2"), _m(A, "7")]
    lines = steps_list(turns(units(turn))[0], NOW).splitlines()
    assert lines[1] == '1. lookup(q="ACME")'
    assert lines[2] == '2. lookup(q="SKU-77", n=1): from step 1'


def test_each_window_message_carries_its_run_record() -> None:
    turn = _recorded(_turn("q", ["r"], "a"), {"status": "ok", "fetched": "2026-10-31T11:00Z"})
    window = replay_window("thr", turn, 100_000, now=NOW)
    (result,) = [m for m in window.messages if m.role == "TOOL"]
    assert result.run_record == {"status": "ok", "fetched": "2026-10-31T11:00Z"}
    body = conversation_body(window)
    assert body is not None
    assert [m.get("run_record") for m in body["messages"]] == [None, None, result.run_record, None]


def test_a_long_argument_is_shown_short() -> None:
    call = _m(A, calls=("c1",))
    call.tool_calls[0].arguments["q"] = "z" * 500
    (line,) = steps_list([[_m(U, "q")], [call, _m(T, "r", answers="c1")]]).splitlines()[1:]
    assert len(line) < 100 and line.endswith("…)")


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
    newest = _turn("Quote ACME " + "please " * 300, ["q" * 6000], "123.45")
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
