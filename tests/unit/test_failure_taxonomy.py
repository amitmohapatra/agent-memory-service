"""The failure taxonomy: four defects hide behind one wrong-answer count.

Each class is fixed somewhere else entirely - the answer protocol, the precomputed
aggregates, the ranking, the retrieval - so a run that only reports "39 wrong" cannot aim
the next change. These tests pin the rules that separate them, including the one that is
easy to get backwards: a wrong answer that the lenient ruler accepts is a *partial* answer,
not a wrongly chosen one.
"""

from __future__ import annotations

from benchmark.failure_taxonomy import classify, taxonomy


def _record(**changes: object) -> dict:
    base = {
        "conversation": 0,
        "category": "single_hop",
        "question": "who owns the rollback plan?",
        "answer": "Priya",
        "hit": False,
        "evidence_hit": True,
        "evidence_all_hit": True,
        "judged": {"produced": "Priya", "abstained": False, "correct": False},
    }
    return {**base, **changes}


def test_a_missing_gold_turn_is_the_only_retrieval_failure() -> None:
    record = _record(evidence_all_hit=False, evidence_hit=False)
    assert classify(record, lenient_hit=True) == "evidence missing"


def test_an_abstention_with_the_evidence_present_is_the_answer_protocol() -> None:
    record = _record(judged={"produced": "I don't know.", "abstained": True})
    assert classify(record, lenient_hit=False) == "abstained"


def test_an_answer_the_lenient_ruler_accepts_is_partial_not_wrong() -> None:
    """Three of four items named: the memory was there, the enumeration was not finished."""
    record = _record(answer="pottery, camping, painting, swimming")
    assert classify(record, lenient_hit=True) == "partial"
    assert classify(record, lenient_hit=False) == "wrong instance"


def test_evidence_all_hit_is_preferred_over_the_max_over_turns_flag() -> None:
    """``evidence_hit`` is a max over gold turns, so it reads true when one of three is
    present; the taxonomy must use the all-turns flag when the run recorded it."""
    record = _record(evidence_hit=True, evidence_all_hit=False)
    assert classify(record, lenient_hit=None) == "evidence missing"


def test_a_run_without_a_lenient_sibling_says_so_instead_of_guessing() -> None:
    report = taxonomy({"records": [_record()]})
    assert report["lenient_compared"] is False
    assert report["by_class"] == {"wrong instance": 1}


def test_correct_and_adversarial_rows_are_not_failures() -> None:
    records = [
        _record(hit=True),
        _record(category="adversarial", hit=False),
        _record(judged={"produced": "I don't know.", "abstained": True}),
    ]
    report = taxonomy({"records": records}, {"records": []})
    assert report["wrong_answerable"] == 1
    assert report["by_class"] == {"abstained": 1}
    assert report["by_category_class"] == {"single_hop/abstained": 1}
