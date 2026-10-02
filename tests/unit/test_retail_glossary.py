"""Retail shorthand searched with its expansion (``domain.glossary``), and the switch that
turns it on for a deployment (``RetailSettings``)."""

from __future__ import annotations

import pytest

from memory_service.application.container import Overrides, Tuning
from memory_service.domain.fiscal import FiscalCalendar, parse_calendar
from memory_service.domain.glossary import MAX_ADDED, expand

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("WOS for dept 12 vs LY", "WOS for dept 12 vs LY (weeks of supply; last year)"),
        ("what was the sell-through on denim", "what was the sell-through on denim (ST)"),
        ("ST% and MD depth", "ST% and MD depth (sell-through; markdown)"),
        ("sell thru by store", "sell thru by store (ST)"),
        ("Weeks of supply (WOS) by store", "Weeks of supply (WOS) by store"),
        ("oh, the cafe was lovely", "oh, the cafe was lovely"),
    ],
)
def test_one_side_of_a_shorthand_brings_the_other(query: str, expected: str) -> None:
    assert expand(query) == expected


def test_a_list_of_acronyms_cannot_grow_a_query_without_bound() -> None:
    added = expand("WOS WOC DOS GMROI AUR AUC ASP OTB IMU MMU").split("(", 1)[1]
    assert added.count(";") == MAX_ADDED - 1


def test_a_retail_calendar_switches_the_calendar_and_the_glossary_on() -> None:
    off = Tuning.resolve(Overrides(), None)
    assert off.memory_intelligence.fiscal_calendar is None
    assert off.retrieval.retail_glossary is False
    on = Tuning.resolve(Overrides(), "445")
    assert on.memory_intelligence.fiscal_calendar == FiscalCalendar(pattern="445")
    assert on.retrieval.retail_glossary is True


@pytest.mark.parametrize(
    ("spec", "calendar"),
    [
        ("454", FiscalCalendar()),
        ("445-12", FiscalCalendar(pattern="445", end_month=12)),
        ("454-01-end", FiscalCalendar(named_by="end")),
        ("544-end", FiscalCalendar(pattern="544", named_by="end")),
    ],
)
def test_a_calendar_spec_names_pattern_year_end_and_naming(
    spec: str, calendar: FiscalCalendar
) -> None:
    assert parse_calendar(spec) == calendar


@pytest.mark.parametrize("spec", ["455", "454-13", "454-01-middle", "454-1-2-3"])
def test_a_malformed_calendar_spec_is_refused(spec: str) -> None:
    with pytest.raises(ValueError, match="not a fiscal calendar"):
        parse_calendar(spec)
