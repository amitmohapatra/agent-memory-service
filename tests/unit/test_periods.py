"""A question's period and a memory's days (``modules.retrieval.periods``)."""

from __future__ import annotations

from datetime import date

import pytest

from memory_service.modules.retrieval.periods import (
    memory_days,
    months_dated,
    query_periods,
    within,
)

pytestmark = pytest.mark.unit

NOW = date(2023, 5, 30)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("When did Melanie go camping in June?", [6]),
        ("What did Gina find on 1 February, 2023?", [(date(2023, 2, 1), date(2023, 2, 1))]),
        (
            "What setback did Melanie face in October 2023?",
            [(date(2023, 10, 1), date(2023, 10, 31))],
        ),
        (
            "How often has Melanie gone to the beach in 2023?",
            [(date(2023, 1, 1), date(2023, 12, 31))],
        ),
        ("What certification did I complete last month?", [(date(2023, 4, 1), date(2023, 5, 29))]),
        ("What may Caroline think about it?", []),  # the verb is not the month
        ("What did Caroline paint?", []),
    ],
)
def test_the_period_a_question_names(question: str, expected: list) -> None:
    assert query_periods(question, NOW) == expected


def test_a_memory_is_about_its_day_and_the_days_it_resolved_to() -> None:
    payload = {
        "observed_at": "2023-07-02T10:00:00+00:00",
        "dated_mentions": [
            {"text": "yesterday", "date": "2023-07-01"},
            {"text": "last week", "date": "2023-06-18..2023-06-24 (FY2023 W20)"},
            {"text": "broken", "date": "soon"},
        ],
    }
    days = memory_days(payload)
    assert days == [
        (date(2023, 7, 2), date(2023, 7, 2)),
        (date(2023, 7, 1), date(2023, 7, 1)),
        (date(2023, 6, 18), date(2023, 6, 24)),
    ]
    assert within([6], days) and within([(date(2023, 6, 20), date(2023, 6, 20))], days)
    assert not within([(date(2023, 9, 1), date(2023, 9, 30))], days)
    assert memory_days({}) == []
    assert months_dated(payload) == {5, 6}  # June (a range) and July, 0 = January
    assert months_dated({}) == frozenset()
    a_year = {"dated_mentions": [{"date": "2022-11-01..2023-12-31"}]}
    assert months_dated(a_year) == frozenset(range(12))
    over_new_year = {"dated_mentions": [{"date": "2022-12-20..2023-01-05"}]}
    assert months_dated(over_new_year) == {11, 0}
