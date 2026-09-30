"""What use teaches the service: a memory's standing in the ranking."""

from __future__ import annotations

import pytest

from memory_service.domain.learning import MAX_STANDING_SHIFT, standing_factor
from memory_service.modules.retrieval.engine import Candidate, by_standing


def test_a_memory_nobody_judged_ranks_where_fusion_put_it() -> None:
    assert standing_factor(None, None) == 1.0
    assert standing_factor(0.5, 1) == 1.0


def test_confidence_and_reinforcement_move_the_score_within_a_bound() -> None:
    assert standing_factor(0.9, 1) > 1.0 > standing_factor(0.1, 1)
    assert standing_factor(0.5, 8) > standing_factor(0.5, 2) > 1.0
    assert standing_factor(1.0, 10_000) == pytest.approx(1.0 + MAX_STANDING_SHIFT)
    assert standing_factor(0.0, 1) >= 1.0 - MAX_STANDING_SHIFT


def _memory(record_id: str, score: float, confidence: float, reinforcement: int) -> Candidate:
    return Candidate(
        record_id=record_id,
        kind="memory",
        text=record_id,
        score=score,
        payload={"confidence": confidence, "reinforcement": reinforcement},
    )


def test_standing_reorders_near_ties_and_never_outweighs_relevance() -> None:
    ranked = by_standing(
        [
            _memory("doubted", 0.100, 0.1, 1),
            _memory("confirmed", 0.099, 0.95, 4),
            _memory("far_ahead", 0.200, 0.05, 1),
        ]
    )
    assert [c.record_id for c in ranked] == ["far_ahead", "confirmed", "doubted"]
