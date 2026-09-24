"""One fact should not occupy two slots in a bundle.

Identical-text dedup collapses exact twins. It cannot see the shape this service produces:
since a turn is kept verbatim as well as extracted, one sentence yields BOTH "Melanie prefers
tea" and the turn "I prefer tea, and my name is Amit..." - different text, different hash, two
slots, one fact. A hundred-memory bundle can therefore carry closer to fifty distinct things.

That matters for every position argument made about this renderer. arXiv 2307.03172 models N
DISTINCT documents with one gold among them; it says nothing about a bundle where half the
slots repeat the other half. The remedy there is not a better position - it is fewer copies of
the same evidence competing for the positions available.

Off by default: it is an ablation knob, not a decision taken without measurement.
"""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval import engine as eng
from memory_service.modules.retrieval.engine import Candidate, _dedup

pytestmark = pytest.mark.unit

TURN = "I prefer tea, and my name is Amit, and I moved to Berlin last year."
FACT = "I prefer tea, and my name is Amit"


def _c(rid: str, text: str, score: float) -> Candidate:
    return Candidate(
        record_id=rid, kind="memory", text=text, score=score, retrievers=["fusion"], payload={}
    )


def test_on_by_default_a_subsumed_candidate_is_collapsed() -> None:
    """On by default now. Multi-hop needs evidence from two or more DIFFERENT turns, and a
    relevance-only head can be several phrasings of one of them: measured on the full set,
    only 51.2% of multi_hop gold evidence reaches the head against 71.4% for temporal.
    Dropping a candidate another already contains frees that slot for the missing hop."""
    out = _dedup([_c("a", TURN, 0.9), _c("b", FACT, 0.8)])
    assert {c.record_id for c in out} == {"a"}, "the turn already carries the fact"


def test_a_fact_already_carried_by_a_kept_turn_is_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eng, "COLLAPSE_SUBSUMED", True)
    out = _dedup([_c("a", TURN, 0.9), _c("b", FACT, 0.8)])
    assert [c.record_id for c in out] == ["a"], "the superset keeps the slot"
    assert "b" in out[0].payload["duplicates"], "the collapse is recorded, not silent"


def test_the_longer_arrival_does_not_evict_a_kept_shorter_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing an accepted candidate would reorder a ranking the caller is entitled to."""
    monkeypatch.setattr(eng, "COLLAPSE_SUBSUMED", True)
    out = _dedup([_c("b", FACT, 0.9), _c("a", TURN, 0.8)])
    assert [c.record_id for c in out] == ["b", "a"]


def test_short_texts_are_never_treated_as_subsumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """'tea' is inside half the corpus; containment that short is coincidence."""
    monkeypatch.setattr(eng, "COLLAPSE_SUBSUMED", True)
    out = _dedup([_c("a", TURN, 0.9), _c("b", "tea", 0.8)])
    assert {c.record_id for c in out} == {"a", "b"}


def test_unrelated_texts_are_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eng, "COLLAPSE_SUBSUMED", True)
    out = _dedup([_c("a", TURN, 0.9), _c("b", "The build server runs Ubuntu 22.04.", 0.8)])
    assert {c.record_id for c in out} == {"a", "b"}
