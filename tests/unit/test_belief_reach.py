"""Sizing the bucket a dated aggregate targets, before anyone measures a delta on it.

The fragmentation bucket is the question set where two or more of the memories carrying a
question's gold turns sit in the SAME multi-valued (subject, predicate) slot: the shape
`BeliefService` collapses into one dated line. Two rules decide whether the number means
anything - a single-valued predicate is not a slot to aggregate, and a question the arm
already answers completely at depth 10 is not addressable by rendering it differently.
"""

from __future__ import annotations

import pytest
from benchmark.belief_reach import fragmentation

pytestmark = pytest.mark.unit


def question(category: str, carriers: dict[str, list[str]], *, complete: bool) -> dict:
    return {
        "category": category,
        "gold": ["t1"],
        "arms": {"carriers": carriers, "fused": list(carriers)},
        "coverage": {"10": {"recall": 1.0 if complete else 0.0, "complete": complete}},
    }


def test_two_memories_in_one_multi_valued_slot_are_the_bucket() -> None:
    records = [question("multi_hop", {"a": ["t1"], "b": ["t1"]}, complete=False)]
    out = fragmentation(records, {"a": ("u", "said"), "b": ("u", "said")})
    assert out["questions"] == 1
    assert out["by_category"] == {"multi_hop": 1}
    assert out["still_incomplete_at_10"] == 1


def test_one_memory_per_slot_is_not_fragmentation() -> None:
    records = [question("multi_hop", {"a": ["t1"], "b": ["t1"]}, complete=False)]
    assert fragmentation(records, {"a": ("u", "said"), "b": ("u", "prefers")})["questions"] == 0


def test_a_single_valued_predicate_is_not_a_slot_to_aggregate() -> None:
    """One value is the whole fact there; gathering it changes nothing."""
    records = [question("multi_hop", {"a": ["t1"], "b": ["t1"]}, complete=False)]
    assert fragmentation(records, {"a": ("u", "name"), "b": ("u", "name")})["questions"] == 0


def test_what_the_arm_already_answers_is_counted_but_not_addressable() -> None:
    records = [
        question("multi_hop", {"a": ["t1"], "b": ["t1"]}, complete=True),
        question("single_hop", {"c": ["t1"], "d": ["t1"]}, complete=False),
    ]
    out = fragmentation(records, dict.fromkeys("abcd", ("u", "said")))
    assert out["questions"] == 2
    assert out["still_incomplete_at_10"] == 1
    assert out["still_incomplete_by_category"] == {"single_hop": 1}


def test_adversarial_and_unscored_questions_stay_out() -> None:
    records = [
        question("adversarial", {"a": ["t1"], "b": ["t1"]}, complete=False),
        {"category": "multi_hop", "gold": [], "arms": {"carriers": {}}},
        {"category": "multi_hop", "gold": ["t1"]},
    ]
    assert fragmentation(records, {"a": ("u", "said"), "b": ("u", "said")})["questions"] == 0


def test_a_carrier_the_corpus_does_not_know_is_counted_not_guessed() -> None:
    records = [question("multi_hop", {"a": ["t1"], "gone": ["t1"]}, complete=False)]
    out = fragmentation(records, {"a": ("u", "said")})
    assert out["carriers_absent_from_the_corpus"] == 1
    assert out["questions"] == 0, "an unknown predicate cannot be counted into a slot"
