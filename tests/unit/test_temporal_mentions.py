"""A relative date is resolved against the day it was said, in the language it was said in,
and an absolute date or a bare number is left alone."""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.script import Script, detect_script
from memory_service.modules.memory.temporal import DatedMention, resolve_dated_mentions

pytestmark = pytest.mark.unit

#: a Monday
BASE = datetime(2023, 5, 8, 15, 30, tzinfo=UTC)


def _resolve(text: str) -> list[DatedMention]:
    return resolve_dated_mentions(text, base=BASE, script=detect_script(text))


def test_english_relative_expressions_resolve_against_the_base() -> None:
    found = {
        m.text.casefold(): m.date for m in _resolve("We met yesterday and again three days ago.")
    }
    assert found["yesterday"] == "2023-05-07"
    assert found["three days ago"] == "2023-05-05"


def test_last_and_next_resolve_and_absolute_dates_are_left_alone() -> None:
    found = {
        m.text.casefold(): m.date
        for m in _resolve("I moved last week; the lease started on 8 May 2023.")
    }
    assert found["last week"] == "2023-05-01..2023-05-07"  # the Monday-to-Sunday before
    assert not any("2023" in text for text in found), found
    assert _resolve("The invoice was for 98 million and 3 items.") == []


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("Wir haben uns gestern getroffen.", "gestern", "2023-05-07"),
        ("Nos vimos ayer en la oficina.", "ayer", "2023-05-07"),
        ("Nos vimos hace 2 días.", "hace 2 días", "2023-05-06"),
        ("Мы встречались вчера.", "вчера", "2023-05-07"),
        ("我们昨天见过面。", "昨天", "2023-05-07"),
        ("Ne-am întâlnit ieri.", "ieri", "2023-05-07"),
    ],
)
def test_the_languages_the_service_is_measured_in(text: str, phrase: str, expected: str) -> None:
    found = {m.text.casefold(): m.date for m in _resolve(text)}
    assert found.get(phrase.casefold()) == expected, found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # BASE is Monday 2023-05-08
        ("I went there last Friday.", {"last friday": "2023-05-05"}),
        ("The show is next Monday.", {"next monday": "2023-05-15"}),
        ("We went camping last weekend.", {"last weekend": "2023-05-06..2023-05-07"}),
        ("Are you free this weekend?", {"this weekend": "2023-05-13..2023-05-14"}),
        ("I painted it this morning.", {"this morning": "2023-05-08"}),
        ("I started last month.", {"last month": "2023-04-01..2023-04-30"}),
        ("We moved last year.", {"last year": "2022-01-01..2022-12-31"}),
        ("I signed up a few days ago.", {"a few days ago": "2023-05-02..2023-05-06"}),
        ("We met a couple of weeks ago.", {"a couple of weeks ago": "2023-04-17..2023-04-28"}),
        # past or coming is not in the words: left unresolved rather than guessed
        ("We met on Friday.", {}),
        ("Let's go next weekend.", {}),
        ("I swam a lot last summer.", {}),
    ],
)
def test_english_weekdays_weekends_and_periods_resolve_by_rule(
    text: str, expected: dict[str, str]
) -> None:
    assert {m.text.casefold(): m.date for m in _resolve(text)} == expected


def test_a_script_without_a_parser_and_an_empty_text_resolve_nothing() -> None:
    assert resolve_dated_mentions("ᚠᚢᚦ", base=BASE, script=Script.OTHER) == []
    assert resolve_dated_mentions("   ", base=BASE, script=Script.LATIN) == []
    assert resolve_dated_mentions("42", base=BASE, script=Script.NONE) == []


def test_mentions_are_bounded_and_deduplicated() -> None:
    text = (
        "We met yesterday. She called yesterday too. He moved last month. They left last year. "
        "The lease started three weeks ago. I resigned last week. We fly tomorrow."
    )
    found = resolve_dated_mentions(text, base=BASE, script=Script.LATIN, limit=4)
    assert len(found) == 4 and len({m.text.casefold() for m in found}) == 4
    assert found[0].text.casefold() == "yesterday"


def test_a_weekday_phrase_is_resolved_by_rule_and_never_guessed() -> None:
    """``last Tuesday`` is resolved by an English rule, not by dateparser's ``absolute-time``
    parser, which reads the word "we" as a Wednesday (ADR 0024, decision 7). A bare weekday
    stays unresolved: whether "on Friday" is past or coming is not in the words."""
    assert {m.text.casefold(): m.date for m in _resolve("we met last Tuesday")} == {
        "last tuesday": "2023-05-02"
    }
    assert _resolve("I saw her on Friday.") == []
    found = {m.text.casefold(): m.date for m in _resolve("We met last Tuesday, three days ago.")}
    assert found == {"last tuesday": "2023-05-02", "three days ago": "2023-05-05"}


def test_the_renderer_prints_the_resolved_date_beside_the_text() -> None:
    from memory_service.domain.context_bundle import ContextItem, _memory_line
    from memory_service.domain.enums import Representation

    text = "We met three days ago at the office."
    # the pair the resolver actually produces for this text and this base
    assert _resolve(text) == [DatedMention(text="three days ago", date="2023-05-05")]
    item = ContextItem(
        item_id="mem_1",
        representation=Representation.MEMORY,
        text=text,
        citation="memory_id:mem_1",
        attributes={
            "observed_at": "2023-05-08T15:30:00+00:00",
            "subject": "user:caroline",
            "dated_mentions": [m.as_dict() for m in _resolve(text)],
        },
    )
    assert _memory_line(item, "m1") == (
        "- [m1] 2023-05-08 Mon caroline: We met three days ago at the office. "
        "(three days ago = 2023-05-05)"
    )
    assert _memory_line(item, "m1", body="(see Most relevant)").endswith("(see Most relevant)")
