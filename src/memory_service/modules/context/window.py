"""The conversation window in the messages form (``WindowFormat.MESSAGES``): the thread's most
recent messages, verbatim and in order, for a caller that replays them to its model as its own
conversation — the way a framework's checkpoint replays a thread.

What a model API accepts decides how the window is cut. A tool call and its results are one
unit: a call whose result is missing, or a result whose call is, is refused by the APIs that
take tool calls, so neither is ever left alone at the edge. The budget is spent from the newest
unit back and stops at the first that does not fit (a window has no gaps); the newest unit is
always given, as the rendered form always gives the newest message. The window then starts on
a USER message when one is left in it: a conversation that opens with an answer reads as one
whose question was lost.
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


def newest_that_fit(pieces: Sequence[list[Message]], token_budget: int) -> list[list[Message]]:
    """The newest ``pieces`` (oldest first) that fit ``token_budget``, back to the first that
    does not; the newest always."""
    chosen: list[list[Message]] = []
    used = 0
    for unit in reversed(pieces):
        cost = sum(message_tokens(m) for m in unit)
        if chosen and used + cost > token_budget:
            break
        chosen.append(unit)
        used += cost
    chosen.reverse()
    return chosen


def from_a_question(chosen: list[list[Message]]) -> list[list[Message]]:
    """``chosen`` from its first USER message on, when it has one."""
    first = next((i for i, u in enumerate(chosen) if u[0].role is MessageRole.USER), 0)
    return chosen[first:]


def replay_window(
    thread_id: str, messages: Sequence[Message], token_budget: int
) -> ConversationWindow:
    """The most recent replayable messages of ``messages`` (oldest first) that fit
    ``token_budget``, whole units only, oldest first (the module says how it is cut)."""
    pieces = [u for u in map(whole, units([m for m in messages if replayable(m)])) if u]
    chosen = from_a_question(newest_that_fit(pieces, token_budget))
    kept = [m for u in chosen for m in u]
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
    )
