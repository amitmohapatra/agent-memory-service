"""An ordering metric has to move when the ordering changes. The old one could not.

``evidence_all_hit`` scores ``_content_tokens(bundle.render())`` - the whole bundle flattened
into one token set - so it is a question about the UNION of what was retrieved. Permuting the
union cannot change it. That is why every ordering arm taken so far returned an exact null:
``abl_control`` 0.7167, ``abl_subsume`` 0.7167, ``abl_share35`` 0.7124, ``abl_both`` 0.7124,
``abl_tail`` 0.7039, all with ``evidence_all_recall`` byte-identical at 0.9785 and all with
``judged: false``. The knobs were not weak; the instrument was blind by construction.

These tests pin the property that makes an ordering experiment possible at all: same
memories, different order, different number.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from benchmark.locomo import _evidence_ranks, _rank_metrics, _rank_summary

pytestmark = pytest.mark.unit


def _memory(item_id: str, text: str) -> SimpleNamespace:
    """The shape ``_memory_line`` reads: citation, text, and observed_at/subject attributes."""
    return SimpleNamespace(
        item_id=item_id,
        citation=item_id,
        text=text,
        attributes={"observed_at": "2023-06-20T00:00:00+00:00", "subject": "user:caroline"},
    )


GOLD = "caroline went to the networking event in june"
MEMORIES = [
    _memory("m1", "caroline repotted the fiddle leaf fig on the balcony"),
    _memory("m2", "caroline mentioned the price of oat milk"),
    _memory("m3", GOLD),
]


def test_the_rank_follows_the_order_the_memories_are_in() -> None:
    assert _evidence_ranks(MEMORIES, [GOLD]) == [2]
    assert _evidence_ranks(list(reversed(MEMORIES)), [GOLD]) == [0]


def test_evidence_absent_from_the_bundle_has_no_rank() -> None:
    """``None``, not zero: a missing item is a recall failure, not a badly ranked one."""
    assert _evidence_ranks(MEMORIES, ["an entirely unrelated sentence about tax returns"]) == [None]
    assert _evidence_ranks(MEMORIES, [""]) == [None]
    assert _evidence_ranks([], [GOLD]) == [None]


def test_mrr_and_head_membership_move_when_only_the_order_does() -> None:
    """The property the old metric lacked, stated directly."""
    last = _rank_metrics(_evidence_ranks(MEMORIES, [GOLD]), head=1)
    first = _rank_metrics(_evidence_ranks(list(reversed(MEMORIES)), [GOLD]), head=1)
    assert first["evidence_mrr"] > last["evidence_mrr"]
    assert first["evidence_in_head"] == 1.0
    assert last["evidence_in_head"] == 0.0


def test_a_multi_hop_question_is_bounded_by_its_last_evidence_item() -> None:
    """``evidence_worst_rank`` is to rank what ``evidence_min_overlap`` is to recall."""
    second = "caroline mentioned the price of oat milk"
    metrics = _rank_metrics(_evidence_ranks(MEMORIES, [GOLD, second]), head=10)
    assert metrics["evidence_worst_rank"] == 2

    # one item missing entirely: the worst rank is undefined, not the best of what was found
    partial = _rank_metrics(_evidence_ranks(MEMORIES, [GOLD, "unrelated tax return talk"]), head=10)
    assert partial["evidence_worst_rank"] is None
    assert partial["evidence_mrr"] > 0.0, "the item that WAS found still counts toward mrr"


def test_a_question_with_no_evidence_annotated_reports_nothing() -> None:
    assert _rank_metrics([], head=10) == {
        "evidence_mrr": None,
        "evidence_in_head": None,
        "evidence_worst_rank": None,
    }


def test_the_summary_excludes_adversarial_rows() -> None:
    """Adversarial questions have no gold evidence: a rank over them is undefined, not zero."""
    records = [
        {"category": "single_hop", "evidence_mrr": 1.0, "evidence_in_head": 1.0},
        {"category": "adversarial", "evidence_mrr": None, "evidence_in_head": None},
    ]
    assert _rank_summary(records) == {"evidence_mrr": 1.0, "evidence_in_head": 1.0}


def test_a_run_with_nothing_scorable_reports_none_rather_than_zero() -> None:
    """Zero would read as "the ranking is terrible"; None reads as "not measured"."""
    assert _rank_summary([{"category": "adversarial", "evidence_mrr": None}]) == {
        "evidence_mrr": None,
        "evidence_in_head": None,
    }
