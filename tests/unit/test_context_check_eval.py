"""The context check's test set (``benchmark/context_check.py``): its cases are sound - every
labelled item is one the check is asked about, on its own or with items of the same label -
and its scoring reads an answer as the labels mean it. No model: the judgement itself is what
a paid run of the benchmark measures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from benchmark.context_check import (
    LABELS,
    Oracle,
    _labelled_drop,
    bundle_of,
    load,
    score,
    summarise,
    unlisted,
)

from memory_service.modules.context.check import ContextCheck, listed

pytestmark = pytest.mark.unit

TODAY = datetime(2026, 11, 27, tzinfo=UTC)
CASES = load()


def test_the_set_covers_what_the_check_is_for() -> None:
    labels = [item["label"] for case in CASES for item in case["items"]]
    assert set(labels) == set(LABELS)
    whys = {item["why"] for case in CASES for item in case["items"]}
    # what must stay, and each kind of noise the check was built against
    assert {"applies", "asked", "default", "general_rule", "moment_valid"} <= whys
    assert {"pasted", "moment", "other_task", "unrelated", "stale", "near_miss"} <= whys
    assert {case["lang"] for case in CASES} >= {"en", "de", "es", "hi"}
    assert len({case["id"] for case in CASES}) == len(CASES) >= 30


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_every_labelled_item_is_one_the_check_is_asked_about(case: dict) -> None:
    """A ``keep`` sentence inside a ``drop`` text would be asked about once, as the text: no
    answer could score both right."""
    assert unlisted(case, TODAY) == []
    bundle, placed = bundle_of(case, TODAY)
    places = {place for entry in listed(bundle) for place in entry.places}
    assert all(place in places for _, theirs in placed for place in theirs)


async def test_an_answer_that_follows_the_labels_scores_no_error() -> None:
    judge = Oracle()
    check = ContextCheck(judge)  # type: ignore[arg-type]
    rows = []
    for case in CASES:
        bundle, placed = bundle_of(case, TODAY)
        judge.drop = _labelled_drop(bundle, placed)
        dropped, note = await check.read(case["request"], bundle)
        rows.append(score(case, placed, dropped, note))
    summary = summarise(rows)
    assert summary["outcomes"] == {"checked": len(CASES)}
    assert summary["wrongly_removed"]["rate"] == 0.0 and summary["wrongly_removed"]["n"] >= 50
    assert summary["noise_left"]["rate"] == 0.0 and summary["noise_left"]["n"] >= 150
    assert 0.5 < summary["chars_removed_share"] < 0.95


async def test_the_two_errors_are_counted_apart() -> None:
    case = next(c for c in CASES if c["id"] == "alloc-create")
    bundle, placed = bundle_of(case, TODAY)
    entries = listed(bundle)
    # an answer that removes everything: every keep item is wrongly removed, no noise is left
    everything = summarise([score(case, placed, entries, {"outcome": "checked", "ms": 1.0})])
    assert everything["wrongly_removed"]["rate"] == 1.0
    assert everything["noise_left"]["rate"] == 0.0
    # an answer that removes nothing, and a check that is off: nothing wrongly removed, all
    # the noise left
    nothing = summarise([score(case, placed, [], {})])
    assert nothing["outcomes"] == {"off": 1}
    assert nothing["wrongly_removed"]["rate"] == 0.0 and nothing["noise_left"]["rate"] == 1.0
    assert nothing["by_why"]["pasted"]["rate"] == 1.0 and nothing["latency_ms"] == {}


async def test_a_case_can_be_measured_at_a_real_stores_size() -> None:
    """A hundred unrelated memories more: each its own statement, each labelled, and the
    answer has room to name them all."""
    case = next(c for c in load(extra=100) if c["id"] == "alloc-create")
    assert unlisted(case, TODAY) == []
    bundle, placed = bundle_of(case, TODAY)
    assert len(listed(bundle)) >= 110
    judge = Oracle()
    judge.drop = _labelled_drop(bundle, placed)
    dropped, note = await ContextCheck(judge).read(case["request"], bundle)  # type: ignore[arg-type]
    summary = summarise([score(case, placed, dropped, note)])
    assert summary["wrongly_removed"]["rate"] == 0.0 and summary["noise_left"]["rate"] == 0.0
    assert summary["by_why"]["bulk"]["n"] == 100
