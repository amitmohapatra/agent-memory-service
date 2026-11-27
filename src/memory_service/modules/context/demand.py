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

from typing import Final

from memory_service.domain.context_bundle import (
    ContextBundle,
    UnusedEvidence,
    earlier_chats_section,
)
from memory_service.domain.memory import unverified_representation
from memory_service.modules.context.standing import stated
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.retrieval.engine import UNUSED_MAX

#: The diagnostics key that says what a build left to be asked for.
NOTE: Final = "recall"
#: What retrieval ranks for the request, by the bundle's sections.
RANKED: Final = ("memories", "history", "graph_facts", "knowledge", "summaries")
#: What the assistant answered in earlier conversations: found by the request too.
SAID: Final = "earlier_chats"


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
    """``bundle`` without the memories a profile line already states, word for word (a rule
    or preference that applies is listed with its date in the profile, and was in the
    prompt a second time as a ranked memory). Each stays evidence the bundle does not carry
    (``unused``), which an answer is still checked against."""
    shown = {
        _plain(stated(line)[0]) for block in bundle.profile for line in block.text.splitlines()
    }
    shown.discard("")
    twice = [m for m in bundle.memories if _plain(m.text) in shown]
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


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold().strip(" .!?;:,")
