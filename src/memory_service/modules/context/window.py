"""The conversation window in the messages form (``WindowFormat.MESSAGES``): the thread's most
recent turns, verbatim and in order, for a caller that replays them to its model as its own
conversation - the way a framework's checkpoint replays a thread.

What a model API accepts, and what a follow-up question needs, decide how the window is cut.
A tool call and its results are one unit: a call whose result is missing, or a result whose
call is, is refused by the APIs that take tool calls, so neither is ever left alone. A *turn*
is a USER message and every unit after it up to the next USER message - the question, the
calls made to answer it, their results and the answer - and a turn is given whole or not at
all: "where did you get that?" is about the calls and results behind the answer, and an answer
replayed without them cannot be explained.

The budget is spent from the newest turn back and stops at the first turn that does not fit (a
window has no gaps). The newest turn is always given, from its USER message on, even when it
alone is over the budget: the window then says so (``ConversationWindow.over_budget``) and the
caller trims it to its model's context with its own library (LangChain's ``trim_messages``);
the service never shortens what was said. At most ``max_turns`` turns are given. Messages
before the window's first USER message (their question is older than the read, or under the
thread's summary) are a turn only when there is no other: a conversation that opens with an
answer reads as one whose question was lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import orjson

from memory_service.domain.context_bundle import ConversationWindow, WindowMessage
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageKind, MessageRole, WindowFormat
from memory_service.modules.ingestion.hierarchy import estimate_tokens

#: What a message costs beyond its text, in tokens: its role and the chat template's markers.
MESSAGE_OVERHEAD_TOKENS: Final = 4


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


def turn_tokens(turn: Sequence[list[Message]]) -> int:
    """What a whole turn costs in a prompt (an estimate)."""
    return sum(message_tokens(m) for unit in turn for m in unit)


def newest_turns_that_fit(
    found: Sequence[list[list[Message]]], token_budget: int
) -> list[list[list[Message]]]:
    """The newest turns of ``found`` (oldest first) that fit ``token_budget``, back to the
    first that does not; the newest always, whatever it costs."""
    chosen: list[list[list[Message]]] = []
    used = 0
    for turn in reversed(found):
        cost = turn_tokens(turn)
        if chosen and used + cost > token_budget:
            break
        chosen.append(turn)
        used += cost
    chosen.reverse()
    return chosen


def from_a_question(found: list[list[list[Message]]]) -> list[list[list[Message]]]:
    """``found`` without a leading turn whose question is not in it, when another turn is."""
    if len(found) > 1 and found[0][0][0].role is not MessageRole.USER:
        return found[1:]
    return found


def replay_window(
    thread_id: str,
    messages: Sequence[Message],
    token_budget: int,
    max_turns: int | None = None,
) -> ConversationWindow:
    """The most recent replayable turns of ``messages`` (oldest first) that fit
    ``token_budget``, whole turns only, at most ``max_turns`` of them, oldest first (the module
    says how it is cut)."""
    pieces = [u for u in map(whole, units([m for m in messages if replayable(m)])) if u]
    found = from_a_question(turns(pieces))
    if max_turns is not None:
        found = found[-max_turns:] if max_turns > 0 else []
    chosen = newest_turns_that_fit(found, token_budget)
    kept = [m for turn in chosen for unit in turn for m in unit]
    used = sum(message_tokens(m) for m in kept)
    return ConversationWindow(
        thread_id=thread_id,
        message_ids=[m.message_id for m in kept],
        messages=[
            WindowMessage(
                message_id=m.message_id,
                role=m.role.value,
                text=m.content,
                tool_calls=list(m.tool_calls),
                tool_call_id=m.tool_call_id,
            )
            for m in kept
        ],
        # what ``ContextBundle.redundant`` compares the memories with; never shown
        rendered="\n".join(f"{m.role.value}: {m.content}" for m in kept if m.content),
        token_estimate=used,
        form=WindowFormat.MESSAGES,
        over_budget=bool(chosen) and turn_tokens(chosen[-1]) > token_budget,
    )
