"""A retail fiscal calendar, and the fiscal phrases a memory names resolved against it.

Merchandise planners do not say "the week of 25 August"; they say "last week", "LY", "wk 32",
"P5", "Q3", "FW26", "this season". Each is a range of days in the retailer's own calendar -
weeks that run Sunday to Saturday, months of four or five weeks, a year that ends on the
Saturday nearest the end of January - and a reader handed the line months later cannot do
that arithmetic. As with ``modules.memory.temporal``, the phrase is resolved once at ingest
against the moment it was said and rendered beside the text:
``last week = 2026-08-23..2026-08-29 (FY2026 W30)``.

The default is the NRF 4-5-4 calendar most US retailers report in; the pattern, the month the
year ends in and the naming year are settings (``FiscalCalendarSettings``). Nothing is
inferred from the text: a phrase is resolved only when it is unambiguous retail shorthand,
and a phrase that names no fiscal unit is left to the calendar-date resolver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cache
from typing import Literal

Pattern = Literal["454", "445", "544"]
WEEKS: dict[str, tuple[int, int, int]] = {"454": (4, 5, 4), "445": (4, 4, 5), "544": (5, 4, 4)}
#: Saturday, the last day of a retail week (``date.weekday()``)
_SATURDAY = 5


@dataclass(frozen=True)
class FiscalCalendar:
    """A 52/53-week retail calendar.

    ``end_month`` is the month the year ends in (1 = January, NRF); the year ends on the
    Saturday nearest that month's last day. ``named_by`` says which calendar year names a
    fiscal year: ``start`` (NRF: FY2026 runs Feb 2026 - Jan 2027) or ``end``.
    """

    pattern: Pattern = "454"
    end_month: int = 1
    named_by: Literal["start", "end"] = "start"

    def year_end(self, fiscal_year: int) -> date:
        # named by its start, a year that ends before December ends in the next calendar year
        calendar_year = fiscal_year + (self.named_by == "start" and self.end_month != 12)
        last = _month_end(calendar_year, self.end_month)
        # the Saturday nearest the month's last day (at most three days either side)
        delta = (_SATURDAY - last.weekday()) % 7
        return last + timedelta(days=delta) if delta <= 3 else last - timedelta(days=7 - delta)

    def year_start(self, fiscal_year: int) -> date:
        return self.year_end(fiscal_year - 1) + timedelta(days=1)

    def year_of(self, day: date) -> int:
        guess = day.year if self.named_by == "end" else day.year - 1
        for fiscal_year in (guess - 1, guess, guess + 1, guess + 2):
            if self.year_start(fiscal_year) <= day <= self.year_end(fiscal_year):
                return fiscal_year
        raise ValueError(f"no fiscal year holds {day}")  # pragma: no cover - unreachable

    def weeks_in(self, fiscal_year: int) -> int:
        return ((self.year_end(fiscal_year) - self.year_start(fiscal_year)).days + 1) // 7

    def week(self, fiscal_year: int, number: int) -> tuple[date, date]:
        start = self.year_start(fiscal_year) + timedelta(weeks=number - 1)
        return start, start + timedelta(days=6)

    def period_weeks(self, fiscal_year: int) -> list[int]:
        """Weeks in each of the twelve periods; a 53rd week joins the last period."""
        weeks = list(WEEKS[self.pattern]) * 4
        if self.weeks_in(fiscal_year) == 53:
            weeks[-1] += 1
        return weeks

    def period(self, fiscal_year: int, number: int) -> tuple[date, date]:
        weeks = self.period_weeks(fiscal_year)
        first = sum(weeks[: number - 1]) + 1
        return self.week(fiscal_year, first)[0], self.week(
            fiscal_year, first + weeks[number - 1] - 1
        )[1]

    def quarter(self, fiscal_year: int, number: int) -> tuple[date, date]:
        return self.period(fiscal_year, 3 * number - 2)[0], self.period(fiscal_year, 3 * number)[1]

    def season(self, fiscal_year: int, half: int) -> tuple[date, date]:
        """Spring is the first half (Q1-Q2), fall the second (Q3-Q4)."""
        return self.quarter(fiscal_year, 2 * half - 1)[0], self.quarter(fiscal_year, 2 * half)[1]

    def year(self, fiscal_year: int) -> tuple[date, date]:
        return self.year_start(fiscal_year), self.year_end(fiscal_year)

    def position(self, day: date) -> FiscalPosition:
        fiscal_year = self.year_of(day)
        week = (day - self.year_start(fiscal_year)).days // 7 + 1
        weeks = self.period_weeks(fiscal_year)
        period, seen = 1, weeks[0]
        while week > seen:
            period += 1
            seen += weeks[period - 1]
        return FiscalPosition(fiscal_year, week, period, (period - 1) // 3 + 1)


def parse_calendar(spec: str) -> FiscalCalendar:
    """``454`` (NRF), ``445-12``, ``454-01-end``: the pattern, then optionally the month the
    year ends in and ``end`` when a year is named by the calendar year it ends in."""
    parts = spec.strip().lower().split("-")
    if parts[0] not in WEEKS or len(parts) > 3:
        raise ValueError(f"not a fiscal calendar: {spec!r} (e.g. 454, 445-12, 454-01-end)")
    month = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
    named = parts[-1] if parts[-1] in ("start", "end") and len(parts) > 1 else "start"
    if not 1 <= month <= 12 or (len(parts) == 3 and named != parts[2]):
        raise ValueError(f"not a fiscal calendar: {spec!r} (e.g. 454, 445-12, 454-01-end)")
    return FiscalCalendar(pattern=parts[0], end_month=month, named_by=named)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FiscalPosition:
    year: int
    week: int
    period: int
    quarter: int

    @property
    def half(self) -> int:
        return 1 if self.quarter <= 2 else 2


@dataclass(frozen=True)
class FiscalMention:
    """A fiscal phrase as written, and the days it covers."""

    text: str
    start: date
    end: date
    label: str

    def as_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "date": f"{self.start.isoformat()}..{self.end.isoformat()} ({self.label})",
        }


#: phrases kept per text, as for calendar dates
MAX_MENTIONS = 6
_YEAR = r"(?:fy\s?)?'?(\d{2}|\d{4})"
_SEASONS = {"spring": 1, "sp": 1, "ss": 1, "fall": 2, "fa": 2, "fw": 2, "aw": 2, "autumn": 2}


@cache
def _patterns() -> list[tuple[re.Pattern[str], str]]:
    rel = r"(last|this|next|previous|prior|current)"
    rules = [
        (rf"\b{rel}\s+(fiscal\s+)?(week|month|period|quarter|season|year)\b", "relative"),
        (r"\b(lw|tw|lm|lq|ly|ty|lytd|ytd|qtd|mtd|wtd)\b", "short"),
        (rf"\b(?:fiscal\s+)?(?:week|wk|w)\s?(\d{{1,2}})(?:\s+(?:of\s+)?{_YEAR})?\b", "week"),
        (rf"\b(?:fiscal\s+)?(?:period|p)\s?(\d{{1,2}})(?:\s+(?:of\s+)?{_YEAR})?\b", "period"),
        (rf"\b(?:{_YEAR}\s+)?q([1-4])(?:\s+{_YEAR})?\b", "quarter"),
        (r"\b(spring|fall|autumn|ss|sp|fw|aw|fa)\s?'?(\d{2}|\d{4})\b", "season"),
        (r"\b(?:fy|fiscal\s+(?:year\s+)?)'?(\d{2}|\d{4})\b", "year"),
    ]
    return [(re.compile(expression, re.IGNORECASE), kind) for expression, kind in rules]


def resolve_fiscal_mentions(
    text: str, *, base: datetime, calendar: FiscalCalendar, limit: int = MAX_MENTIONS
) -> list[FiscalMention]:
    """The fiscal phrases ``text`` names, resolved against the day ``base`` falls on."""
    now = calendar.position(base.date())
    out: list[FiscalMention] = []
    taken: list[tuple[int, int]] = []
    for pattern, kind in _patterns():
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            resolved = _resolve(kind, match, now, calendar)
            if resolved is None:
                continue
            start, end, label = resolved
            taken.append((match.start(), match.end()))
            out.append(FiscalMention(match.group(0).strip(), start, end, label))
    out.sort(key=lambda m: text.find(m.text))
    return out[:limit]


Resolved = tuple[date, date, str] | None


def _full_year(raw: str | None, current: int) -> int:
    if raw is None:
        return current
    value = int(raw)
    return value if value >= 1000 else 2000 + value


def _week(g: tuple[str | None, ...], now: FiscalPosition, cal: FiscalCalendar) -> Resolved:
    number, year = int(g[0] or 0), _full_year(g[1], now.year)
    if not 1 <= number <= cal.weeks_in(year):
        return None
    return (*cal.week(year, number), f"FY{year} W{number}")


def _period(g: tuple[str | None, ...], now: FiscalPosition, cal: FiscalCalendar) -> Resolved:
    number, year = int(g[0] or 0), _full_year(g[1], now.year)
    return (*cal.period(year, number), f"FY{year} P{number}") if 1 <= number <= 12 else None


def _quarter(g: tuple[str | None, ...], now: FiscalPosition, cal: FiscalCalendar) -> Resolved:
    number, year = int(g[1] or 0), _full_year(g[0] or g[2], now.year)
    return (*cal.quarter(year, number), f"FY{year} Q{number}")


def _season(g: tuple[str | None, ...], now: FiscalPosition, cal: FiscalCalendar) -> Resolved:
    half, year = _SEASONS[(g[0] or "").casefold()], _full_year(g[1], now.year)
    return (*cal.season(year, half), f"FY{year} {'spring' if half == 1 else 'fall'}")


def _year(g: tuple[str | None, ...], now: FiscalPosition, cal: FiscalCalendar) -> Resolved:
    year = _full_year(g[0], now.year)
    return (*cal.year(year), f"FY{year}")


def _resolve(
    kind: str, match: re.Match[str], now: FiscalPosition, calendar: FiscalCalendar
) -> Resolved:
    g = match.groups()
    if kind == "relative":
        return _relative((g[0] or "").casefold(), (g[2] or "").casefold(), now, calendar)
    if kind == "short":
        return _short((g[0] or "").casefold(), now, calendar)
    return _BY_KIND[kind](g, now, calendar)


_BY_KIND = {"week": _week, "period": _period, "quarter": _quarter, "season": _season, "year": _year}


def _relative(which: str, unit: str, now: FiscalPosition, calendar: FiscalCalendar) -> Resolved:
    step = {"last": -1, "previous": -1, "prior": -1, "this": 0, "current": 0, "next": 1}[which]
    if unit == "week":
        start = calendar.week(now.year, now.week)[0] + timedelta(weeks=step)
        p = calendar.position(start)
        return start, start + timedelta(days=6), f"FY{p.year} W{p.week}"
    if unit in ("month", "period"):
        year, number = _shift(now.year, now.period, step, 12)
        return (*calendar.period(year, number), f"FY{year} P{number}")
    if unit == "quarter":
        year, number = _shift(now.year, now.quarter, step, 4)
        return (*calendar.quarter(year, number), f"FY{year} Q{number}")
    if unit == "season":
        year, half = _shift(now.year, now.half, step, 2)
        return (*calendar.season(year, half), f"FY{year} {'spring' if half == 1 else 'fall'}")
    return (*calendar.year(now.year + step), f"FY{now.year + step}")


def _short(word: str, now: FiscalPosition, calendar: FiscalCalendar) -> Resolved:
    today_week = calendar.week(now.year, now.week)
    if word in ("lw", "tw", "wtd"):
        return _relative("last" if word == "lw" else "this", "week", now, calendar)
    if word in ("lm", "mtd"):
        return _relative("last" if word == "lm" else "this", "month", now, calendar)
    if word in ("lq", "qtd"):
        return _relative("last" if word == "lq" else "this", "quarter", now, calendar)
    if word in ("ly", "ty"):
        return _relative("last" if word == "ly" else "this", "year", now, calendar)
    if word == "ytd":
        return calendar.year_start(now.year), today_week[1], f"FY{now.year} W1-W{now.week}"
    # lytd: last year to the same week
    last = now.year - 1
    week = min(now.week, calendar.weeks_in(last))
    return calendar.year_start(last), calendar.week(last, week)[1], f"FY{last} W1-W{week}"


def _shift(year: int, number: int, step: int, per_year: int) -> tuple[int, int]:
    index = year * per_year + (number - 1) + step
    return index // per_year, index % per_year + 1


def _month_end(year: int, month: int) -> date:
    following = date(year + (month == 12), month % 12 + 1, 1)
    return following - timedelta(days=1)
