"""The offline weight fit is arithmetic over a finished dump, so its arithmetic is pinned here.

The fit decides which weighting a benchmark arm is then run with, so an error in it does not
produce a wrong number - it produces a wrong EXPERIMENT, and the arm that follows looks like
a measurement of something nobody chose. Two properties carry the whole thing: a weight is
applied per arm at RRF's one-based rank, and every depth is a prefix of ONE fused order.
"""

from __future__ import annotations

import pytest
from benchmark.fit_rrf_weights import DEPTHS, coverage, fit, fuse, score

pytestmark = pytest.mark.unit


def test_a_weight_moves_an_arm_up_the_fusion() -> None:
    arms = {"dense_en": ["a", "b"], "bm25": ["b", "a"]}
    assert fuse(arms, {"dense_en": 1.0, "bm25": 1.0}, 1) == ["a", "b"]  # tie, id order
    assert fuse(arms, {"dense_en": 1.0, "bm25": 2.0}, 1) == ["b", "a"]
    # k flattens the difference between adjacent ranks, which is what a large k is for
    assert fuse(arms, {"dense_en": 1.0, "bm25": 1.01}, 60) == ["b", "a"]


def test_an_arm_the_weighting_does_not_name_keeps_its_weight() -> None:
    arms = {"dense_en": ["a"], "bm25": ["b"]}
    assert fuse(arms, {"dense_en": 2.0}, 1) == ["a", "b"]


def test_every_depth_is_a_prefix_of_one_fused_order() -> None:
    """score() fuses once per question; fusing per depth must give the same answer.

    This is the property that let the fit drop from 23 minutes to 9 on the ensemble dump.
    """
    questions = [
        {
            "arms": {"dense_en": [f"m{i}" for i in range(250)], "bm25": ["m240", "m3"]},
            "carriers": {"m3": ["t1"], "m240": ["t2"]},
            "gold": ["t1", "t2"],
        },
        {
            "arms": {"dense_en": ["m1"], "bm25": ["m2"]},
            "carriers": {"m9": ["t9"]},
            "gold": ["t9"],
        },
    ]
    weights = {"dense_en": 1.0, "bm25": 2.0}
    per_depth = {}
    for depth in DEPTHS:
        rows = [
            coverage(fuse(q["arms"], weights, 1), set(q["gold"]), q["carriers"], depth)
            for q in questions
        ]
        per_depth[str(depth)] = {
            "recall": round(sum(r for r, _ in rows) / len(rows), 4),
            "complete": round(sum(c for _, c in rows) / len(rows), 4),
        }
    assert score(questions, weights, 1) == per_depth


def test_the_fit_reports_the_equal_weighting_it_has_to_beat() -> None:
    """Every number measured before the fit was taken at equal weights and k=1, so the fit is
    only readable if it carries that row beside its winner."""
    questions = [
        {
            "arms": {"dense_en": ["x", "m1"], "bm25": ["m1", "x"]},
            "carriers": {"m1": ["t1"]},
            "gold": ["t1"],
        }
    ]
    out = fit(questions, target_depth=100)
    assert out["equal_weights_k1"]["k"] == 1
    assert set(out["equal_weights_k1"]["weights"].values()) == {1.0}
    assert out["arms"] == ["bm25", "dense_en"]
    assert set(out["gain_at_target"]) == {"recall", "complete"}
    # the winner is never worse than equal weights at the depth it was fitted for
    key = str(out["target_depth"])
    assert out["best"]["at"][key]["complete"] >= out["equal_weights_k1"]["at"][key]["complete"]
