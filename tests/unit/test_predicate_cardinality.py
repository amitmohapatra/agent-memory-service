"""Unit tests: how many values a predicate holds, and how several of them read as one block.

The cardinality rule decides two different things in two different layers - whether a new
value supersedes the old one on the write path, and whether several memories are gathered
into one line on the read path - so it is asserted here once, against the vocabulary both
layers import.
"""

from __future__ import annotations

import pytest

from memory_service.domain.memory import aggregate_statement, dated_statement
from memory_service.domain.predicates import (
    SINGLE_VALUED,
    is_multi_valued,
    is_single_valued,
    predicate_label,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "predicate",
    ["name", "lives_in", "works_at", "timezone", "manager", "birthday", "favourite"],
)
def test_a_slot_held_once_at_a_time_is_single_valued(predicate: str) -> None:
    assert is_single_valued(predicate)
    assert not is_multi_valued(predicate)


@pytest.mark.parametrize("predicate", ["favourite_colour", "favourite_food", "favourite_band"])
def test_every_favourite_category_is_single_valued_without_enumerating_it(predicate: str) -> None:
    # one favourite per category: the family is covered by its prefix, not by a list that
    # would have to grow every time an extractor invents a category
    assert predicate not in SINGLE_VALUED
    assert is_single_valued(predicate)


@pytest.mark.parametrize(
    "predicate", ["visited", "participated_in", "likes", "read", "met", "attended"]
)
def test_a_slot_whose_values_accumulate_is_multi_valued(predicate: str) -> None:
    assert is_multi_valued(predicate)
    assert not is_single_valued(predicate)


@pytest.mark.parametrize("predicate", [None, ""])
def test_an_absent_predicate_is_neither(predicate: str | None) -> None:
    # nothing to supersede and nothing to accumulate: such a memory keeps its own line
    assert not is_single_valued(predicate)
    assert not is_multi_valued(predicate)


def test_predicate_label_reads_as_prose() -> None:
    assert predicate_label("works_at") == "works at"
    assert predicate_label("visited") == "visited"


def test_dated_statement_keeps_the_day_beside_the_words() -> None:
    # "last Tuesday" only means something against the day it was said
    assert dated_statement("2023-05-08", "we met last Tuesday") == (
        "[observed 2023-05-08] we met last Tuesday"
    )


def test_aggregate_statement_lists_every_value_under_one_heading() -> None:
    block = aggregate_statement(
        "user:alice",
        "participated_in",
        [
            ("2023-05-02", "Alice ran the charity race."),
            ("2023-06-11", "Alice joined the hackathon."),
        ],
    )
    assert block.splitlines() == [
        "user:alice — participated in (source statements):",
        "[observed 2023-05-02] Alice ran the charity race.",
        "[observed 2023-06-11] Alice joined the hackathon.",
    ]


def test_aggregate_statement_collapses_duplicates_and_keeps_order() -> None:
    # the same statement retrieved twice is one statement; order is the caller's (oldest first)
    block = aggregate_statement(
        "user:alice",
        "visited",
        [("2023-05-02", "the Tate"), ("2023-05-02", "the Tate"), ("2023-01-09", "Kew")],
    )
    assert block.splitlines()[1:] == [
        "[observed 2023-05-02] the Tate",
        "[observed 2023-01-09] Kew",
    ]
