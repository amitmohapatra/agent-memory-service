"""One comparable relevance, not three incompatible scales.

``score`` carried a cross-encoder logit (-11..+11) for items the reranker judged, an RRF
fusion score (~0.001..0.25) for everything past ``candidate_k``, and a hardcoded 1.0 for
exact identifier hits. Measured live on one response: rank 27 scored +0.067 while rank 1
scored -4.14, so any client that sorted or thresholded on it got the ranking backwards.
"""

from __future__ import annotations

import pytest

from memory_service.modules.context.builder import _relevance
from memory_service.modules.retrieval.engine import Candidate


def _candidate(score: float, *, rerank: float | None = None, retrievers=("fusion",)):
    c = Candidate(record_id="r", kind="memory", text="t", score=score,
                  retrievers=list(retrievers))
    c.rerank_score = rerank
    return c


def test_a_reranked_item_reports_its_probability() -> None:
    raw, kind, relevance = _relevance(_candidate(0.016, rerank=0.968))
    assert kind == "cross_encoder"
    assert raw == pytest.approx(0.968)
    assert relevance == pytest.approx(0.968)


def test_an_exact_hit_is_certain() -> None:
    _, kind, relevance = _relevance(_candidate(1.0, retrievers=("exact",)))
    assert kind == "exact" and relevance == 1.0


def test_the_unjudged_tail_sits_below_everything_judged() -> None:
    """They were never scored by the reranker and ranked below those that were. The number
    must not claim a confidence nobody measured."""
    _, kind, tail = _relevance(_candidate(0.25))
    assert kind == "fusion"
    _, _, worst_judged = _relevance(_candidate(0.0, rerank=0.06))
    assert tail < worst_judged, "a fusion tail item must not outrank a judged one"


def test_relevance_is_always_in_range() -> None:
    for c in (_candidate(0.016, rerank=1.4), _candidate(0.016, rerank=-0.2),
              _candidate(99.0), _candidate(1.0, retrievers=("exact",))):
        _, _, relevance = _relevance(c)
        assert 0.0 <= relevance <= 1.0


def test_ordering_is_preserved_within_each_kind() -> None:
    judged = [_relevance(_candidate(0.0, rerank=r))[2] for r in (0.9, 0.5, 0.1)]
    tail = [_relevance(_candidate(s))[2] for s in (0.25, 0.10, 0.01)]
    assert judged == sorted(judged, reverse=True)
    assert tail == sorted(tail, reverse=True)
    assert min(judged) > max(tail)
