"""The retail fiscal calendar (``domain.fiscal``): NRF 4-5-4 boundaries, 53-week years, and
fiscal phrases resolved against the day a memory was said."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from memory_service.domain.fiscal import FiscalCalendar, resolve_fiscal_mentions
from memory_service.modules.memory.temporal import dated_mentions

pytestmark = pytest.mark.unit

NRF = FiscalCalendar()
SAID = datetime(2026, 9, 2, 15, 0, tzinfo=UTC)  # a Wednesday in FY2026 week 31


@pytest.mark.parametrize(
    ("year", "start", "end", "weeks"),
    [
        (2023, date(2023, 1, 29), date(2024, 2, 3), 53),  # NRF's last 53-week year
        (2024, date(2024, 2, 4), date(2025, 2, 1), 52),
        (2025, date(2025, 2, 2), date(2026, 1, 31), 52),
        (2026, date(2026, 2, 1), date(2027, 1, 30), 52),
    ],
)
def test_the_year_ends_on_the_saturday_nearest_the_end_of_january(
    year: int, start: date, end: date, weeks: int
) -> None:
    assert NRF.year(year) == (start, end)
    assert NRF.weeks_in(year) == weeks
    assert end.weekday() == 5 and start.weekday() == 6


def test_periods_follow_the_pattern_and_a_53rd_week_joins_the_last_one() -> None:
    assert NRF.period_weeks(2026)[:3] == [4, 5, 4]
    assert FiscalCalendar(pattern="445").period_weeks(2026)[:3] == [4, 4, 5]
    assert NRF.period_weeks(2023)[-1] == 5
    assert NRF.period(2026, 1) == (date(2026, 2, 1), date(2026, 2, 28))
    assert NRF.period(2026, 2) == (date(2026, 3, 1), date(2026, 4, 4))
    assert NRF.quarter(2026, 3) == (date(2026, 8, 2), date(2026, 10, 31))
    assert NRF.position(SAID.date()).week == 31


def test_a_year_named_by_its_end_and_a_december_year_end() -> None:
    by_end = FiscalCalendar(named_by="end")
    assert by_end.year(2026) == NRF.year(2025)
    december = FiscalCalendar(end_month=12)
    start, end = december.year(2026)
    # the Saturday nearest 31 December 2026, a Thursday, is 2 January 2027
    assert (start, end) == (date(2026, 1, 4), date(2027, 1, 2))


def _resolved(text: str) -> dict[str, str]:
    return {
        m.text: m.as_dict()["date"] for m in resolve_fiscal_mentions(text, base=SAID, calendar=NRF)
    }


def test_planning_shorthand_resolves_to_the_days_it_covers() -> None:
    got = _resolved("LW sell-through was below LY; wk 32 and Q3 are the plan, FW26 too")
    assert got == {
        "LW": "2026-08-23..2026-08-29 (FY2026 W30)",
        "LY": "2025-02-02..2026-01-31 (FY2025)",
        "wk 32": "2026-09-06..2026-09-12 (FY2026 W32)",
        "Q3": "2026-08-02..2026-10-31 (FY2026 Q3)",
        "FW26": "2026-08-02..2027-01-30 (FY2026 fall)",
    }


def test_relative_phrases_step_across_year_boundaries() -> None:
    early = datetime(2026, 2, 3, tzinfo=UTC)  # FY2026 week 1, period 1
    got = {
        m.text: m.label
        for m in resolve_fiscal_mentions(
            "last month, last quarter, last season and last week", base=early, calendar=NRF
        )
    }
    assert got == {
        "last month": "FY2025 P12",
        "last quarter": "FY2025 Q4",
        "last season": "FY2025 fall",
        "last week": "FY2025 W52",
    }


def test_text_without_fiscal_shorthand_resolves_nothing() -> None:
    assert _resolved("We met at the cafe and talked about the new store layout.") == {}
    assert _resolved("week 60 of the project") == {}  # no such fiscal week


def test_a_fiscal_phrase_replaces_the_calendar_reading_of_the_same_words() -> None:
    mentions = dated_mentions(
        "Markdowns started last week. We met three days ago.", base=SAID, fiscal=NRF
    )
    by_text = {m["text"]: m["date"] for m in mentions}
    assert by_text["last week"] == "2026-08-23..2026-08-29 (FY2026 W30)"
    assert by_text["three days ago"] == "2026-08-30"
    plain = dated_mentions("Markdowns started last week.", base=SAID)
    # without a retail calendar, the calendar week (Monday to Sunday) before
    assert [m["text"] for m in plain] == ["last week"]
    assert plain[0]["date"] == "2026-08-24..2026-08-30"
