"""The LoCoMo harness metrics that decide what gets built next.

These four numbers are the ones the roadmap is argued from, and until now every one of them
was computed by a throwaway script against a result file after the run. That is how a
question-level metric (``evidence_all_recall``: every gold turn clears the bar) came to be
set beside an item-level one (``evidence_reconstructed``: the fraction of turns that clear
it) as though the difference between them meant something.

``benchmark/`` is in no other test suite - a removed keyword argument there once survived 645
green unit tests while aborting every LoCoMo run at question zero - so the metrics that
decide the next build are pinned here.
"""

from __future__ import annotations

import pytest
from benchmark.locomo import (
    _evidence_ranks,
    _evidence_ranks_by_arm,
    _rank_metrics,
    _rank_summary,
)

from memory_service.domain.context_bundle import ContextItem

pytestmark = pytest.mark.unit

BERLIN = "Amit moved to Berlin in May 2025 to lead the platform team"
TEA = "Melanie prefers tea over coffee in the afternoon"
ABSENT = "Priya defended her thesis in Lisbon last November"


def _m(item_id: str, text: str, retrievers: list[str]) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation="MEMORY",
        text=text,
        citation=item_id,
        retrievers=retrievers,
    )


def test_an_arm_only_answers_for_the_memories_it_produced() -> None:
    """The question ``evidence_ranks`` cannot answer: WHICH arm carried the gold turn.

    A gold turn carried by a graph-only memory that never reaches the head is fusion
    discarding evidence that was retrieved. A gold turn no arm carries is retrieval never
    having found it. The fixes are opposite - tune fusion, or add a retrieval path - and
    choosing between them without separating the two is guessing.
    """
    memories = [_m("a", TEA, ["dense", "bm25"]), _m("b", BERLIN, ["graph"])]
    by_arm = _evidence_ranks_by_arm(memories, [BERLIN])

    assert by_arm["graph"] == [0], "graph carried it, and it is that arm's only memory"
    assert by_arm["dense"] == [None], "dense never produced a memory carrying this turn"
    assert by_arm["bm25"] == [None]


def test_the_rank_is_within_the_arm_not_within_the_bundle() -> None:
    """An arm-alone column means 'where would this have ranked if only this arm ran'.

    Reporting the global bundle position instead would make every arm look worse the more
    memories the OTHER arms contributed, which is the opposite of what the column is for.
    """
    memories = [
        _m("a", TEA, ["dense"]),
        _m("b", ABSENT, ["dense"]),
        _m("c", BERLIN, ["dense"]),
    ]
    assert _evidence_ranks(memories, [BERLIN]) == [2], "third in the bundle"
    assert _evidence_ranks_by_arm(memories, [BERLIN])["dense"] == [2]

    # same dense sublist, but two graph memories now sit in front of it
    memories = [_m("x", TEA, ["graph"]), _m("y", TEA, ["graph"]), *memories]
    assert _evidence_ranks(memories, [BERLIN]) == [4], "fifth in the bundle now"
    assert _evidence_ranks_by_arm(memories, [BERLIN])["dense"] == [2], "still third for dense"


def test_a_turn_no_arm_carries_is_none_everywhere() -> None:
    memories = [_m("a", TEA, ["dense"]), _m("b", BERLIN, ["graph", "bm25"])]
    by_arm = _evidence_ranks_by_arm(memories, [ABSENT])
    assert by_arm == {"bm25": [None], "dense": [None], "graph": [None]}


def test_an_arm_that_produced_nothing_is_absent_rather_than_empty() -> None:
    """Only arms that actually contributed to this bundle get a column - an arm that ran and
    returned nothing must not be reported as an arm that found nothing relevant."""
    memories = [_m("a", TEA, ["dense"])]
    assert set(_evidence_ranks_by_arm(memories, [TEA])) == {"dense"}


def test_complete_is_all_or_nothing_not_a_fraction() -> None:
    """``evidence_in_head`` is the FRACTION of gold turns that reached the head; a multi-hop
    question with one hop in and one hop out scores 0.5 there and is still unanswerable. The
    complete_* pair is the all-or-nothing reading, which is the one that bounds the answer."""
    both = _rank_metrics([0, 1], head=30)
    assert both["complete_in_candidates"] is True and both["complete_in_head"] is True

    one_buried = _rank_metrics([0, 44], head=30)
    assert one_buried["evidence_in_head"] == 0.5, "half the turns reached the head"
    assert one_buried["complete_in_candidates"] is True, "both were retrieved"
    assert one_buried["complete_in_head"] is False, "but the question is still short a hop"

    one_missing = _rank_metrics([0, None], head=30)
    assert one_missing["complete_in_candidates"] is False
    assert one_missing["complete_in_head"] is False


def test_the_summary_separates_retrieval_loss_from_selection_loss() -> None:
    """The distance from complete_in_candidates down to complete_in_head is everything
    selection costs; the distance from 1.0 down to complete_in_candidates is everything
    retrieval costs. Keeping both in the summary is what stopped the two being conflated."""
    records = [
        {"category": "multi_hop", **_rank_metrics([0, 1], head=30)},
        {"category": "multi_hop", **_rank_metrics([0, 44], head=30)},
        {"category": "single_hop", **_rank_metrics([0, None], head=30)},
        # adversarial rows have no gold evidence; a rank over them is undefined, not zero
        {"category": "adversarial", **_rank_metrics([], head=30)},
    ]
    for r in records:
        r.setdefault("evidence_reconstructed", None)
    out = _rank_summary(records)

    assert out["complete_evidence_in_candidates"] == 0.6667, "adversarial excluded"
    assert out["complete_evidence_in_head"] == 0.3333


def test_an_all_adversarial_run_reports_none_rather_than_zero() -> None:
    out = _rank_summary([{"category": "adversarial", **_rank_metrics([], head=30)}])
    assert out["complete_evidence_in_candidates"] is None
    assert out["complete_evidence_in_head"] is None
