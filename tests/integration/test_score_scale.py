"""One comparable relevance, not three incompatible scales.

``score`` carries an RRF fusion score (~0.001..0.25) or a hardcoded 1.0 for exact identifier
hits, so clients compare ``relevance``: bounded, and an exact hit above every fusion item.
"""

from __future__ import annotations

from memory_service.modules.context.builder import _relevance
from memory_service.modules.retrieval.engine import Candidate


def _candidate(score: float, *, retrievers=("fusion",)):
    return Candidate(
        record_id="r", kind="memory", text="t", score=score, retrievers=list(retrievers)
    )


def test_an_exact_hit_is_certain() -> None:
    _, kind, relevance = _relevance(_candidate(1.0, retrievers=("exact",)))
    assert kind == "exact" and relevance == 1.0


def test_a_fusion_item_sits_below_an_exact_hit() -> None:
    """Fusion scores are rank aggregates; the number must not claim a confidence nobody
    measured, and an exact identifier hit outranks every one of them."""
    _, kind, fusion = _relevance(_candidate(0.25))
    assert kind == "fusion"
    assert fusion < _relevance(_candidate(1.0, retrievers=("exact",)))[2]


def test_relevance_is_always_in_range() -> None:
    for c in (
        _candidate(99.0),
        _candidate(1.0, retrievers=("exact",)),
    ):
        _, _, relevance = _relevance(c)
        assert 0.0 <= relevance <= 1.0


def test_fusion_ordering_is_preserved() -> None:
    tail = [_relevance(_candidate(s))[2] for s in (0.25, 0.10, 0.01)]
    assert tail == sorted(tail, reverse=True)


def test_a_bundle_item_carries_the_relevance_it_computed() -> None:
    """`_relevance` was correct and called by nothing but this file.

    `candidate_to_item` set `score` and left `relevance` and `score_kind` at their defaults,
    so every item on the wire reported `relevance: 0.0` and `score_kind: "fusion"` whatever
    produced it — including exact identifier hits, which are the one case the caller can
    trust absolutely. A client that sorted or thresholded on the field got nothing.
    """
    from memory_service.modules.context.builder import candidate_to_item

    exact = candidate_to_item(_candidate(1.0, retrievers=("exact",)))
    assert exact.score_kind == "exact"
    assert exact.relevance == 1.0, "an exact identifier hit is as certain as it gets"

    unjudged = candidate_to_item(_candidate(0.25))
    assert unjudged.score_kind == "fusion"
    assert 0.0 < unjudged.relevance <= 0.05, "a fusion score is not a probability"
    assert unjudged.relevance < exact.relevance
