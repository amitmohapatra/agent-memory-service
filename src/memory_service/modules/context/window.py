"""The conversation window in the messages form (``WindowFormat.MESSAGES``): the thread's most
recent turns, in order, for a caller that replays them to its model as its own conversation -
the way a framework's checkpoint replays a thread.

What a model API accepts, and what a follow-up question needs, decide how the window is cut.
A tool call and its results are one unit: a call whose result is missing, or a result whose
call is, is refused by the APIs that take tool calls, so neither is ever left alone. A *turn*
is a USER message and every unit after it up to the next USER message - the question, the
calls made to answer it, their results and the answer.

A turn is given in layers, each cheaper than the one before and kept longer:

* its *steps list*: the turn's question, then a line per call (the tool, its arguments in
  short, and what the caller recorded of it: :data:`RUN_RECORD`) - "what did you do?" and
  "where is that from?" are about these. The steps lists are the window's run record
  (``ConversationWindow.record``), which the context renders under a heading saying who wrote
  it, never a message: replayed as an ASSISTANT message, a model read it as its own claim
  and disowned it. A chat template that does not render tool calls still renders the context;
* its *text*: the question and what the agent said, without the calls and their results;
* *whole*: the calls and their results too - the newest turn only, when it fits; an older
  result is never replayed (it would read as a fresh one).

The budget is spent from the newest turn back, layer by layer: every turn's steps list (a
turn without calls: its text) until the first that does not fit (a window has no gaps), then
the turns' text from the newest back, then the newest turn whole. Short of space, the full
results go first and the steps lists last. The newest turn's question, answer and steps list
are always given, even when they alone are over the budget: the window then says so
(``ConversationWindow.over_budget``) and the caller trims it to its model's context with its
own library (LangChain's ``trim_messages``). At most ``max_turns`` turns are given. Messages
before the window's first USER message (their question is older than the read, or under the
thread's summary) are a turn only when there is no other: a conversation that opens with an
answer reads as one whose question was lost.

A caller that keeps a run record writes its facts on the messages it records, under
``custom_metadata[RUN_RECORD]`` (plain values: custom metadata nests two levels at most): on a
tool result, ``status``, ``seconds``, ``sources`` (the result's sources, as text), ``fetched``
(when the call returned: ISO 8601, UTC) and ``max_age`` (the seconds its value stays current,
when the tool says); on the turn's last message, the run's ``status``, ``seconds`` and
``approvals`` (as text). The steps list carries what is there, and nothing that is not: the
fetch time is said there and nowhere else, and each window message carries its facts
(``WindowMessage.run_record``) for the caller to act on.

"Why" and "how sure" are the record's, never the model's own account. A step whose argument
is a value an earlier step's result returned says so (``from step 2``): the evidence of why it
was taken. The turn's *evidence* line (:func:`evidence`) counts the calls that succeeded, the
sources named and how many calls name each, and the fetch times against the tools' ages
(:data:`OUT_OF_DATE`), and grades them :data:`SURE`: ``high`` (every call succeeded, a source
named, nothing out of date), ``low`` (no call succeeded, or the run failed), else ``medium``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import orjson

from memory_service.domain.context_bundle import ConversationWindow, WindowMessage
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageKind, MessageRole, WindowFormat
from memory_service.modules.ingestion.hierarchy import estimate_tokens

#: What a message costs beyond its text, in tokens: its role and the chat template's markers.
MESSAGE_OVERHEAD_TOKENS: Final = 4
#: The ``custom_metadata`` key of the run-record facts a caller writes on what it records.
RUN_RECORD: Final = "run_record"
#: The most of one argument's value a steps list shows.
ARGUMENT_CHARS: Final = 60
#: The most of a turn's question a steps list shows.
QUESTION_CHARS: Final = 120
#: The fewest characters an argument value has for a steps list to say an earlier step's
#: result returned it (a shorter one, ``1`` or ``US``, matches by chance).
FROM_STEP_CHARS: Final = 3
#: The tool status of a call that succeeded, and the run statuses of a run that did not.
CALL_OK: Final = "ok"
RUN_FAILED: Final = frozenset({"FAILED", "CANCELLED", "TIMED_OUT", "REJECTED"})
#: The grades of the evidence line, from the record alone (the module says when each holds).
SURE: Final = ("high", "medium", "low")
#: What the evidence line says of values older than their tools' ``max_age``.
OUT_OF_DATE: Final = "out of date now: {tools} (fetch again for current values)"


def replayable(message: Message) -> bool:
    """Whether a stored message belongs in the messages form: the visible chat, and the
    results of the tool calls made in it (INTERNAL TOOL messages naming their call)."""
    if message.kind is MessageKind.VISIBLE:
        return True
    return message.role is MessageRole.TOOL and message.tool_call_id is not None


def message_tokens(message: Message) -> int:
    """What one message costs in a prompt (an estimate): its text, its calls' names and
    arguments, and :data:`MESSAGE_OVERHEAD_TOKENS`."""
    calls = sum(
        estimate_tokens(c.name) + estimate_tokens(orjson.dumps(c.arguments).decode())
        for c in message.tool_calls
    )
    return estimate_tokens(message.content) + calls + MESSAGE_OVERHEAD_TOKENS


def units(messages: Sequence[Message]) -> list[list[Message]]:
    """``messages`` (oldest first) cut into the pieces a window may start at: a message, or an
    ASSISTANT message with tool calls together with the results that follow it. A result whose
    call is not in ``messages`` (cut off by the read) is a unit of its own, dropped later."""
    out: list[list[Message]] = []
    open_calls: set[str] = set()
    for m in messages:
        if m.role is MessageRole.TOOL and m.tool_call_id in open_calls and out:
            out[-1].append(m)
            continue
        out.append([m])
        open_calls = {c.id for c in m.tool_calls}
    return out


def whole(unit: list[Message]) -> list[Message]:
    """``unit`` as a model API accepts it: as it is when every call it makes has its result;
    a call left unanswered (a run stopped between the call and its result) is not replayed -
    what the message said is, without its calls, and nothing at all when it said nothing. A
    result whose call was cut off by the read is dropped."""
    head = unit[0]
    if head.role is MessageRole.TOOL:
        return []
    answered = {m.tool_call_id for m in unit[1:]}
    if all(c.id in answered for c in head.tool_calls):
        return unit
    return [head.model_copy(update={"tool_calls": []})] if head.content else []


def turns(pieces: Sequence[list[Message]]) -> list[list[list[Message]]]:
    """``pieces`` (whole units, oldest first) grouped into turns: a USER message and every unit
    after it up to the next USER message. Units before the first USER message are a turn of
    their own, without its question."""
    out: list[list[list[Message]]] = []
    for unit in pieces:
        if not out or unit[0].role is MessageRole.USER:
            out.append([unit])
        else:
            out[-1].append(unit)
    return out


def from_a_question(found: list[list[list[Message]]]) -> list[list[list[Message]]]:
    """``found`` without a leading turn whose question is not in it, when another turn is."""
    if len(found) > 1 and found[0][0][0].role is not MessageRole.USER:
        return found[1:]
    return found


def _cut(text: str, most: int) -> str:
    return text if len(text) <= most else text[: most - 1] + "…"


def _short(value: object) -> str:
    return _cut(orjson.dumps(value).decode(), ARGUMENT_CHARS)


def _asked(turn: Sequence[list[Message]]) -> str:
    first = turn[0][0]
    if first.role is not MessageRole.USER or not first.content.strip():
        return "an answer whose question is not in the conversation"
    return '"' + _cut(" ".join(first.content.split()), QUESTION_CHARS) + '"'


def _facts(message: Message | None) -> dict[str, Any]:
    found = message.custom_metadata.get(RUN_RECORD) if message is not None else None
    return found if isinstance(found, dict) else {}


def _seconds(value: object) -> str | None:
    return f"{value:.1f} s" if isinstance(value, int | float) else None


def _said(label: str, value: object) -> str | None:
    return f"{label}: {value}" if isinstance(value, str) and value else None


def _from_step(arguments: dict[str, Any], earlier: Sequence[str]) -> str | None:
    """Which earlier step's result (``earlier``: their texts, in order) returned one of
    ``arguments``' values — the first, as ``from step N`` — or ``None``."""
    values = [str(v) for v in arguments.values() if len(str(v)) >= FROM_STEP_CHARS]
    for n, text in enumerate(earlier, start=1):
        if any(v in text for v in values):
            return f"from step {n}"
    return None


@dataclass(frozen=True)
class Step:
    """One call of a turn, with what the caller recorded of its result."""

    name: str
    arguments: dict[str, Any]
    facts: dict[str, Any]
    result: str

    def line(self, n: int, earlier: Sequence[str]) -> str:
        args = ", ".join(f"{k}={_short(v)}" for k, v in self.arguments.items())
        facts = self.facts
        said = [
            str(facts["status"]) if facts.get("status") else None,
            _seconds(facts.get("seconds")),
            _said("fetched", facts.get("fetched")),
            _said("source", facts.get("sources")),
            _from_step(self.arguments, earlier),
        ]
        told = ", ".join(x for x in said if x)
        return f"{n}. {self.name}({args})" + (f": {told}" if told else "")

    def out_of_date(self, now: datetime) -> bool:
        """Whether the value is older now than its tool's ``max_age`` (unknown: no)."""
        age, at = self.facts.get("max_age"), _when(self.facts.get("fetched"))
        if not isinstance(age, int | float) or at is None:
            return False
        return (now - at).total_seconds() > age


def _when(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        at = datetime.fromisoformat(value)
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def _steps(turn: Sequence[list[Message]]) -> list[Step]:
    results = {m.tool_call_id: m for unit in turn for m in unit[1:]}
    return [
        Step(
            call.name,
            call.arguments,
            _facts(results.get(call.id)),
            results[call.id].content if call.id in results else "",
        )
        for unit in turn
        for call in unit[0].tool_calls
    ]


def _sources(steps: Sequence[Step]) -> str:
    named: dict[str, int] = {}
    for s in steps:
        for source in str(s.facts.get("sources") or "").split("; "):
            if source:
                named[source] = named.get(source, 0) + 1
    if not named:
        return "no call named a source"
    each = "; ".join(f"{name} (named by {k} of {len(steps)})" for name, k in named.items())
    return f"{len(named)} {'source' if len(named) == 1 else 'sources'}: {each}"


def _grade(
    steps: Sequence[Step], run: dict[str, Any], *, ok: int, recorded: bool, stale: bool
) -> str:
    """:data:`SURE`'s grade of the evidence (the module says when each holds)."""
    if str(run.get("status", "")).upper() in RUN_FAILED or (recorded and ok == 0):
        return SURE[2]
    named = any(s.facts.get("sources") for s in steps)
    return SURE[0] if ok == len(steps) and named and not stale else SURE[1]


def evidence(steps: Sequence[Step], run: dict[str, Any], now: datetime) -> str | None:
    """The turn's evidence for "why" and "how sure", from its record alone: the calls that
    succeeded, the sources and how many calls name each, the values now out of date, and the
    grade they make (:data:`SURE`); ``None`` when the caller recorded nothing of the turn."""
    if not run and not any(s.facts for s in steps):
        return None
    ok = sum(1 for s in steps if str(s.facts.get("status", "")).lower() == CALL_OK)
    recorded = any(s.facts.get("status") for s in steps)
    stale = list(dict.fromkeys(s.name for s in steps if s.out_of_date(now)))
    sure = _grade(steps, run, ok=ok, recorded=recorded, stale=bool(stale))
    said = [
        f"{ok} of {len(steps)} calls succeeded" if recorded else "no call status recorded",
        _sources(steps),
        OUT_OF_DATE.format(tools=", ".join(stale)) if stale else None,
    ]
    return f"Evidence: {'; '.join(x for x in said if x)}. How sure, from this record: {sure}."


def steps_list(turn: Sequence[list[Message]], now: datetime | None = None) -> str:
    """The turn's calls as plain text under its question, one numbered line each (the tool, its
    arguments in short, and the facts the caller recorded on its result), the run's own
    recorded facts, then its :func:`evidence` as of ``now`` (default: the current time); empty
    for a turn that made no call."""
    steps = _steps(turn)
    if not steps:
        return ""
    lines = [s.line(n, [e.result for e in steps[: n - 1]]) for n, s in enumerate(steps, 1)]
    run = _facts(turn[-1][-1])
    said = [
        f"status {run['status']}" if run.get("status") else None,
        _seconds(run.get("seconds")),
        _said("approvals", run.get("approvals")),
    ]
    told = ", ".join(x for x in said if x)
    head = f"For {_asked(turn)} ({len(lines)} {'call' if len(lines) == 1 else 'calls'}):"
    sure = evidence(steps, run, now or datetime.now(UTC))
    return "\n".join(
        [head, *lines, *([f"The run: {told}."] if told else []), *([sure] if sure else [])]
    )


def _shown(m: Message, *, calls: bool) -> WindowMessage:
    return WindowMessage(
        message_id=m.message_id,
        role=m.role.value,
        text=m.content,
        tool_calls=list(m.tool_calls) if calls else [],
        tool_call_id=m.tool_call_id if calls else None,
        run_record=_facts(m) or None,
    )


@dataclass(frozen=True)
class Layers:
    """One turn's three forms (the module says how they are spent)."""

    #: its steps list (:func:`steps_list`); None for a turn that made no call
    steps: str | None
    text: list[Message]
    whole: list[Message]

    @classmethod
    def of(cls, turn: Sequence[list[Message]], now: datetime) -> Layers:
        said = [m for unit in turn for m in unit if m.role is not MessageRole.TOOL and m.content]
        return cls(steps_list(turn, now) or None, said, [m for unit in turn for m in unit])

    def cost(self, layer: str) -> int:
        steps = estimate_tokens(self.steps) if self.steps else 0
        text = sum(message_tokens(m.model_copy(update={"tool_calls": []})) for m in self.text)
        if layer == "steps":
            return steps
        if layer == "text":
            return text
        return sum(message_tokens(m) for m in self.whole) - text  # beyond the text

    def messages(self, *, text: bool, whole: bool) -> list[WindowMessage]:
        """The turn's messages as shown (its steps list is in the window's record)."""
        said = self.whole if whole else (self.text if text else [])
        return [_shown(m, calls=whole) for m in said]


def _cores(layers: Sequence[Layers], budget: int) -> tuple[int, int]:
    """How many turns, from the newest back, give their core (a steps list, or the text of a
    turn without calls) within ``budget`` — the newest always, its text too — and the tokens
    used."""
    newest = layers[0]
    used = newest.cost("steps") + newest.cost("text")
    kept = 1
    for turn in layers[1:]:
        core = turn.cost("steps") if turn.steps else turn.cost("text")
        if used + core > budget:
            break
        used, kept = used + core, kept + 1
    return kept, used


def _texts(layers: Sequence[Layers], kept: int, used: int, budget: int) -> tuple[int, int]:
    """How many of the ``kept`` turns, from the newest back, give their text too, and the
    tokens used then."""
    texts = 1
    for turn in layers[1:kept]:
        if turn.steps is not None:
            if used + turn.cost("text") > budget:
                break
            used += turn.cost("text")
        texts += 1
    return texts, used


def _shown_turns(layers: Sequence[Layers], texts: int, *, full: bool) -> list[WindowMessage]:
    """The messages of ``layers`` (newest first), oldest first: the newest ``texts`` turns'
    text, and the newest turn whole when ``full``."""
    return [
        m
        for n, turn in reversed(list(enumerate(layers)))
        for m in turn.messages(text=n < texts, whole=n == 0 and full)
    ]


def replay_window(
    thread_id: str,
    messages: Sequence[Message],
    token_budget: int,
    max_turns: int | None = None,
    now: datetime | None = None,
) -> ConversationWindow:
    """The most recent replayable turns of ``messages`` (oldest first) within
    ``token_budget``, at most ``max_turns`` of them, each in the layers that fit, oldest first
    (the module says how it is cut); the steps lists' evidence as of ``now`` (default: the
    current time)."""
    pieces = [u for u in map(whole, units([m for m in messages if replayable(m)])) if u]
    found = from_a_question(turns(pieces))
    if max_turns is not None:
        found = found[-max_turns:] if max_turns > 0 else []
    at = now or datetime.now(UTC)
    layers = [Layers.of(t, at) for t in reversed(found)]  # newest first
    if not layers:
        return ConversationWindow(thread_id=thread_id, form=WindowFormat.MESSAGES)
    kept, used = _cores(layers, token_budget)
    texts, used = _texts(layers, kept, used, token_budget)
    newest = layers[0]
    full = newest.steps is not None and used + newest.cost("whole") <= token_budget
    if full:
        used += newest.cost("whole")
    shown = _shown_turns(layers[:kept], texts, full=full)
    return ConversationWindow(
        thread_id=thread_id,
        message_ids=[m.message_id for m in shown],
        messages=shown,
        record="\n\n".join(t.steps for t in reversed(layers[:kept]) if t.steps),
        # what ``ContextBundle.redundant`` compares the memories with; never shown
        rendered="\n".join(f"{m.role}: {m.text}" for m in shown if m.text),
        token_estimate=used,
        form=WindowFormat.MESSAGES,
        over_budget=used > token_budget,
    )
