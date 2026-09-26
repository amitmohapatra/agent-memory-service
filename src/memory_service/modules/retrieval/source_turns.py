"""Promote an already-retrieved source turn before the candidate budget is cut.

No source is fetched here. Both candidates have passed the search store's scope filter,
and literal containment ensures that a truncated turn cannot replace a fact it lost.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_service.modules.retrieval.engine import Candidate


def _source(candidate: Candidate) -> tuple[str, str] | None:
    refs = candidate.payload.get("source_refs") or []
    if candidate.kind != "memory" or len(refs) != 1:
        return None
    ref = refs[0]
    if not ref.get("source_type") or not ref.get("source_id"):
        return None
    return ref["source_type"], ref["source_id"]


def promote_source_turns(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    turns: dict[tuple[str, str], Candidate] = {}
    normalized = {c.record_id: " ".join(c.text.lower().split()) for c in candidates}
    for candidate in candidates:
        source = _source(candidate)
        if source is not None and candidate.payload.get("predicate") == "said":
            previous = turns.get(source)
            if previous is None or len(candidate.text) > len(previous.text):
                turns[source] = candidate
    out: list[Candidate] = []
    seen: set[str] = set()
    promoted = 0
    for candidate in candidates:
        source = _source(candidate)
        parent = turns.get(source) if source is not None else None
        text = normalized[candidate.record_id]
        if (
            parent is not None
            and "exact" not in candidate.retrievers
            and parent.record_id != candidate.record_id
            and text
            and text in normalized[parent.record_id]
        ):
            chosen = replace(
                parent,
                payload=dict(parent.payload),
                score=candidate.score,
                retrievers=list(
                    dict.fromkeys([*candidate.retrievers, *parent.retrievers, "source_turn"])
                ),
                expanded_from=candidate.record_id,
                expansion_edge="SOURCE_TURN",
            )
        else:
            chosen = candidate
        if chosen.record_id not in seen:
            seen.add(chosen.record_id)
            out.append(chosen)
            promoted += chosen is not candidate
    return out, promoted
