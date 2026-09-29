"""The arithmetic of the offline reranking question, without a model or a store.

Three orders are compared over one candidate pool - the fused order the pipeline produced,
the cross-encoder's order, and the best order any reranker could produce - so the scoring has
to be exact about two things: an order is only credited for what its first ``depth`` entries
carry, and the oracle is a REORDERING of the pool rather than a better pool. If the oracle
were allowed to add a candidate, every reranker would look worse than it is and the ceiling
this phase quotes would be wrong.
"""

from __future__ import annotations

import pytest
from benchmark.rerank_offline import (
    check_fused,
    coverage,
    oracle_order,
    rows_of,
    sampled,
    separation,
    summarize,
)

pytestmark = pytest.mark.unit

CARRIERS = {"m1": ["t1"], "m9": ["t2"]}


def test_only_the_first_depth_entries_are_credited() -> None:
    order = ["m1", *[f"pad{i}" for i in range(9)], "m9"]
    scored = coverage(order, {"t1", "t2"}, CARRIERS)
    assert scored["10"] == {"recall": 0.5, "complete": False}
    assert scored["20"] == {"recall": 1.0, "complete": True}


def test_a_reordering_moves_recall_without_touching_the_pool() -> None:
    pool = ["m1", *[f"pad{i}" for i in range(9)], "m9"]
    lifted = oracle_order(pool, CARRIERS)
    assert sorted(lifted) == sorted(pool), "the ceiling must be a reordering, not a wider pool"
    assert lifted[:2] == ["m1", "m9"]
    assert coverage(lifted, {"t1", "t2"}, CARRIERS)["10"] == {"recall": 1.0, "complete": True}


def test_the_oracle_is_stable_within_each_group() -> None:
    pool = ["a", "m1", "b", "m9", "c"]
    assert oracle_order(pool, CARRIERS) == ["m1", "m9", "a", "b", "c"]


def test_a_question_with_no_carrier_scores_zero_not_a_crash() -> None:
    assert coverage(["x", "y"], {"t1"}, {})["10"] == {"recall": 0.0, "complete": False}


def test_only_answerable_rows_with_a_bundle_are_scored() -> None:
    dump = {
        "records": [
            {
                "conversation": 0,
                "question": "q",
                "category": "multi_hop",
                "gold": ["t1"],
                "arms": {"fused": ["m1"], "carriers": CARRIERS},
            },
            {  # adversarial: its correct answer is to abstain, so a rank over it is undefined
                "conversation": 0,
                "question": "q",
                "category": "adversarial",
                "gold": ["t1"],
                "arms": {"fused": ["m1"], "carriers": CARRIERS},
            },
            {"conversation": 0, "question": "q", "category": "single_hop", "gold": [], "arms": {}},
            {"conversation": 0, "question": "q", "category": "single_hop", "gold": ["t1"]},
        ]
    }
    rows = rows_of(dump)
    assert [row["category"] for row in rows] == ["multi_hop"]


def test_every_category_is_summarized_and_rolled_up() -> None:
    rows = [
        {"category": "multi_hop", "c": coverage(["m1"], {"t1"}, CARRIERS)},
        {"category": "single_hop", "c": coverage(["x"], {"t1"}, CARRIERS)},
    ]
    out = summarize(rows, "c")
    assert out["multi_hop"]["at"]["10"]["recall"] == 1.0
    assert out["single_hop"]["at"]["10"]["recall"] == 0.0
    assert out["all_answerable"] == {
        "questions": 2,
        "at": {
            "10": {"recall": 0.5, "complete": 0.5},
            "20": {"recall": 0.5, "complete": 0.5},
            "50": {"recall": 0.5, "complete": 0.5},
        },
    }


def test_the_recomputed_baseline_is_checked_against_the_artifact() -> None:
    """A drift between the dump and the published summary is a finding, not a rounding blip."""
    measured = {"multi_hop": {"questions": 2, "at": {"10": {"recall": 0.4, "complete": 0.2}}}}
    same = {"multi_hop": {"questions": 2, "at": {"10": {"recall": 0.4, "complete": 0.2}}}}
    drifted = {"multi_hop": {"questions": 2, "at": {"10": {"recall": 0.6, "complete": 0.2}}}}
    assert check_fused(measured, same)["max_abs_delta"] == 0.0
    assert check_fused(measured, drifted)["max_abs_delta"] == pytest.approx(0.2)
    assert check_fused(measured, None) == {"checked": False}


def test_the_raw_scores_tell_a_weak_teacher_from_a_broken_one() -> None:
    """A reordering that loses can mean two different things, and they are not both findings.

    A checkpoint whose head is not a ranking head still returns numbers, and in a recall table
    that looks exactly like a model that ranks this corpus badly. The score summary separates
    them: no separation and no spread is a broken instrument.
    """
    ranks_well = separation({"m1": 0.9, "m9": 0.8, "x": 0.1, "y": 0.2}, CARRIERS)
    assert ranks_well is not None
    assert ranks_well["separation"] == pytest.approx(0.7)
    assert ranks_well["spread"] == pytest.approx(0.8)

    emits_one_number = separation({"m1": 0.5, "x": 0.5}, CARRIERS)
    assert emits_one_number is not None
    assert emits_one_means_broken(emits_one_number)

    # nothing to compare when every candidate carries gold, or none does
    assert separation({"m1": 0.9}, CARRIERS) is None
    assert separation({"x": 0.9}, CARRIERS) is None


def emits_one_means_broken(summary: dict[str, float]) -> bool:
    return summary["separation"] == 0.0 and summary["spread"] == 0.0


def test_a_sample_is_deterministic_and_keeps_every_category() -> None:
    rows: list[dict[str, object]] = [{"category": "multi_hop", "n": i} for i in range(100)]
    rows += [{"category": "single_hop", "n": i} for i in range(300)]
    rows += [{"category": "open_domain", "n": i} for i in range(20)]

    assert len(sampled(rows, category=None, sample=0)) == 420, "sample 0 scores everything"

    drawn = sampled(rows, category=None, sample=42)
    assert drawn == sampled(rows, category=None, sample=42), "reproducible from the artifact"
    counts = {
        name: sum(row["category"] == name for row in drawn)
        for name in {str(row["category"]) for row in drawn}
    }
    assert set(counts) == {"multi_hop", "single_hop", "open_domain"}, "no category is dropped"
    assert 35 <= len(drawn) <= 50
    assert counts["single_hop"] > counts["multi_hop"] > counts["open_domain"]

    one = sampled(rows, category="multi_hop", sample=0)
    assert len(one) == 100 and {row["category"] for row in one} == {"multi_hop"}
