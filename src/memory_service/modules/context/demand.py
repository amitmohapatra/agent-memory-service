"""A context that carries what stands and leaves what is ranked to be asked for (ADR 0051).

A context is two kinds of thing. What stands for the person whatever is asked - their
profile, the rules and preferences that apply, the conversation, the tool guidance - and
what retrieval ranks for this one request: memories, earlier values, graph facts, document
passages and their summaries, what the assistant answered in earlier conversations. The
second kind is the one that goes wrong by being *near* (ADR 0048), and it is the one an
agent with a search tool can ask for itself, when the request needs it and in its own words.

A caller that searches on demand says so (``recall: false`` on ``/v1/context``): the build
is the same build - the standing sections read the request's vectors, and the evidence
report still says whether memory holds something for the request - and what was ranked is
left out of what is carried. Everything left out is one search away (``/v1/recall``, the
``memory_search`` tool: its kinds are these sections).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    UnusedEvidence,
    earlier_chats_section,
)
from memory_service.domain.graph import BOOKKEEPING_LAYERS, layer_for
from memory_service.domain.memory import KEPT_AS_SAID, unverified_representation
from memory_service.modules.context.standing import stated
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.retrieval.engine import UNUSED_MAX

#: The diagnostics key that says what a build left to be asked for.
NOTE: Final = "recall"
#: What retrieval ranks for the request, by the bundle's sections.
RANKED: Final = ("memories", "history", "graph_facts", "knowledge", "summaries")
#: What the assistant answered in earlier conversations: found by the request too.
SAID: Final = "earlier_chats"
#: A message kept as said: the turn itself, and the turn with what it replied to.
#: What a model read from a message (the memory's ``provider``); word patterns are ``native``.
READ_BY_MODEL: Final = "llm"
WORD_PATTERNS: Final = "native"
#: A profile block's ``field: value`` line (the ``user`` block): the value is the statement.
_FIELD: Final = re.compile(r"^[\w-]{1,40}: ")


def standing(bundle: ContextBundle) -> ContextBundle:
    """``bundle`` with what stands - the profile and its pages, the conversation and its
    summary, the learned skills and the tool hints - and without what retrieval ranked for
    the request (``RANKED``, ``SAID``).

    The evidence report keeps its status: it says what memory holds for the request, which
    is what tells a caller whether a search is worth making. The memories found join the
    evidence the bundle does not carry (``unused``), which an answer is still checked
    against. The diagnostics count what was left, by section."""
    left = {name: len(getattr(bundle, name)) for name in (*RANKED, SAID)}
    found = [
        UnusedEvidence(item_id=m.item_id, kind="memory", text=m.text)
        for m in bundle.memories
        if not unverified_representation(m.attributes)
    ]
    saved = sum(i.token_estimate for name in RANKED for i in getattr(bundle, name)) + sum(
        estimate_tokens(part) for part in earlier_chats_section(bundle.earlier_chats)
    )
    return bundle.model_copy(
        update={
            **{name: [] for name in (*RANKED, SAID)},
            "evidence": bundle.evidence.model_copy(
                update={"unused": [*bundle.evidence.unused, *found][:UNUSED_MAX]}
            ),
            "token_estimate": max(0, bundle.token_estimate - saved),
            "diagnostics": {
                **bundle.diagnostics,
                NOTE: {"carried": False, "left": {k: n for k, n in left.items() if n}},
            },
        }
    )


def said_once(bundle: ContextBundle) -> ContextBundle:
    """``bundle`` with each statement once. Left out: a memory a profile line states word for
    word (a rule or preference that applies is listed, dated, in the profile), with the message
    it was read from; what a model read from a message, or word patterns cut out of it, when
    the message is shown as said (``_said_again``); and the graph facts that say nothing new
    (``_facts_once``). Each memory left out stays evidence the bundle does not carry
    (``unused``), which an answer is still checked against."""
    shown = {
        said for block in bundle.profile for line in block.text.splitlines() for said in _says(line)
    }
    shown.discard("")
    twice = _said_again(bundle.memories, shown)
    bundle = _facts_once(
        bundle,
        {m.item_id for m in bundle.memories},
        [*shown, *(_plain(m.text) for m in bundle.memories)],
    )
    if not twice:
        return bundle
    gone = {m.item_id for m in twice}
    return bundle.model_copy(
        update={
            "memories": [m for m in bundle.memories if m.item_id not in gone],
            "evidence": bundle.evidence.model_copy(
                update={
                    "unused": [
                        *bundle.evidence.unused,
                        *(
                            UnusedEvidence(item_id=m.item_id, kind="memory", text=m.text)
                            for m in twice
                            if not unverified_representation(m.attributes)
                        ),
                    ][:UNUSED_MAX]
                }
            ),
            "token_estimate": max(0, bundle.token_estimate - sum(m.token_estimate for m in twice)),
        }
    )


def _says(line: str) -> tuple[str, ...]:
    """What a profile line states, plain: the line, and a ``field: value`` line's value."""
    text = stated(line)[0]
    field = _FIELD.match(text)
    return (_plain(text), _plain(text[field.end() :])) if field else (_plain(text),)


def _one_of_each_message(memories: Sequence[ContextItem]) -> set[str]:
    """The memories that say again what a shown message kept as said says: what a model read
    from it, and a sentence word patterns cut out of it. The turn holds every reading of it
    whole, with what was said around it, as ranking lets it stand for the statement
    (``retrieval.memory_ranking._standing_for``); LongMemEval: the round, found by its
    facts, is what is read. A reading of a message not shown as said stays."""

    as_said = {
        mid
        for m in memories
        if m.attributes.get("category") in KEPT_AS_SAID
        for mid in _messages(m)
    }
    return {
        m.item_id
        for m in memories
        if m.attributes.get("category") not in KEPT_AS_SAID
        and m.attributes.get("provider") in (READ_BY_MODEL, WORD_PATTERNS)
        and _messages(m)
        and _messages(m) <= as_said
    }


def _said_again(memories: Sequence[ContextItem], shown: set[str]) -> list[ContextItem]:
    """The memories that say again what the context already says: one a profile line states
    word for word, with the message it was read from (the turn says it again); and what is
    read from a message shown as said (``_one_of_each_message``)."""
    stated = {m.item_id for m in memories if _plain(m.text) in shown}
    stated_in = {mid for m in memories if m.item_id in stated for mid in _messages(m)}
    again = _one_of_each_message(memories)
    return [
        m for m in memories if m.item_id in stated or m.item_id in again or _said_in(m, stated_in)
    ]


def _said_in(memory: ContextItem, messages: set[str]) -> bool:
    """Whether ``memory`` is a message kept as said, and every message it keeps is one of
    ``messages``."""
    kept = _messages(memory)
    return memory.attributes.get("category") in KEPT_AS_SAID and bool(kept) and kept <= messages


def _messages(memory: ContextItem) -> set[str]:
    """The messages ``memory`` was read from or keeps."""
    return {e.message_id for e in memory.evidence if e.message_id}


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold().strip(" .!?;:,")


def _facts_once(bundle: ContextBundle, stated: set[str], lines: list[str]) -> ContextBundle:
    """``bundle`` without the graph facts that say nothing new: one read from a memory the
    context carries (``stated``: the ranked memories, those a profile line states among
    them), one whose words a line of it already holds (``lines``: a fact the graph built with
    no link back to its memory), and the graph's bookkeeping ("u1 mentions Dana": where a name
    appears, ``BOOKKEEPING_LAYERS``), as a search leaves it out
    (``retrieval.search._asked_for``)."""
    kept = [
        f
        for f in bundle.graph_facts
        if str(f.attributes.get("memory_id") or "") not in stated
        and layer_for(str(f.attributes.get("predicate") or "")) not in BOOKKEEPING_LAYERS
        and not _held(_plain(_DAY.sub("", f.text, count=1)), lines)
    ]
    if len(kept) == len(bundle.graph_facts):
        return bundle
    saved = sum(f.token_estimate for f in bundle.graph_facts) - sum(f.token_estimate for f in kept)
    return bundle.model_copy(
        update={"graph_facts": kept, "token_estimate": max(0, bundle.token_estimate - saved)}
    )


#: The day a fact's text opens with (``2026-11-28 coach tote is short ...``)
_DAY: Final = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+")
#: A fact shorter than this is held by a line only when equal: a word or two is in many lines
#: without being read from any of them (as ``context.check.PART_CHARS``).
PART_CHARS: Final = 16


def _held(fact: str, lines: list[str]) -> bool:
    return bool(fact) and any(
        fact == line or (len(fact) >= PART_CHARS and fact in line) for line in lines
    )
