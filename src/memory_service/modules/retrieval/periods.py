"""The period a question names, and the days a memory is about (ADR 0026, period rule).

"When did Melanie go camping in June?", "What did Gina find on 1 February, 2023?", "What
setback did Melanie face in October 2023?" - about one LoCoMo question in seven names the
period its answer lies in, and nothing in the fusion reads it: a turn from June ranks no
higher than one from December. A question's period is parsed here by rule (an explicit
date, a month with or without its year, a year, or a period relative to the moment asked);
a memory is about the day it was said and the days its relative expressions resolved to at
ingest (``dated_mentions``). The ranking lifts the memories that fall in the period.

No model: a month name, a date and "last month" are read by pattern, as the calendar-date
resolver reads them at ingest.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Any

Period = tuple[date, date]
#: a month named without a year: that month of any year
AnyYearMonth = int

_MONTHS = {name.lower(): n for n, name in enumerate(calendar.month_name) if name}
#: capitalised, so the verb "may" is not May
_MONTH = "|".join(name for name in calendar.month_name if name)
_DATED = re.compile(
    rf"\b(\d{{1,2}})?\s*({_MONTH})\s*(\d{{1,2}})?(?:st|nd|rd|th)?,?\s*((?:19|20)\d{{2}})?\b"
)
_YEAR = re.compile(r"\b(?:in|during|of|throughout)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_AGO = re.compile(
    r"\b(a|an|one|two|three|four|five|six|few|couple of) (day|week|month|year)s? ago\b",
    re.IGNORECASE,
)
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_NUMBERS |= {"few": 3, "couple of": 2}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _month(year: int, month: int) -> Period:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def query_periods(query: str, now: date) -> list[Period | AnyYearMonth]:
    """The periods ``query`` names; a month named without a year is that month number."""
    out: list[Period | AnyYearMonth] = []
    for m in _DATED.finditer(query):
        month = _MONTHS[m.group(2).lower()]
        day, year = m.group(1) or m.group(3), m.group(4)
        if year and day and 1 <= int(day) <= calendar.monthrange(int(year), month)[1]:
            on = date(int(year), month, int(day))
            out.append((on, on))
        elif year:
            out.append(_month(int(year), month))
        else:
            out.append(month)
    out += [(date(int(y), 1, 1), date(int(y), 12, 31)) for y in _YEAR.findall(query)]
    return out + _relative(query.lower(), now)


def _relative(text: str, now: date) -> list[Period | AnyYearMonth]:
    out: list[Period | AnyYearMonth] = []
    if re.search(r"\b(last|past) week(end)?\b", text):
        out.append((now - timedelta(days=13), now - timedelta(days=1)))
    if re.search(r"\b(last|past) month\b", text):
        prior = now.replace(day=1) - timedelta(days=1)
        out.append((_month(prior.year, prior.month)[0], now - timedelta(days=1)))
    if re.search(r"\b(past|last) (few|couple of|two|three) months\b", text):
        out.append((now - timedelta(days=100), now))
    if re.search(r"\bthis year\b", text):
        out.append((date(now.year, 1, 1), now))
    if re.search(r"\blast year\b", text):
        out.append((date(now.year - 1, 1, 1), date(now.year - 1, 12, 31)))
    for m in _AGO.finditer(text):
        days = _UNIT_DAYS[m.group(2).lower()] * _NUMBERS[m.group(1).lower()]
        centre, slack = now - timedelta(days=days), timedelta(days=max(2, days // 4))
        out.append((centre - slack, centre + slack))
    return out


def memory_days(payload: dict[str, Any]) -> list[Period]:
    """The days a memory is about: the day it was observed, and every day or range its
    relative expressions resolved to at ingest."""
    out: list[Period] = []
    said = _date(str(payload.get("observed_at") or "")[:10])
    if said is not None:
        out.append((said, said))
    for mention in payload.get("dated_mentions") or ():
        if not isinstance(mention, dict):
            continue
        value = str(mention.get("date") or "").split(" ", 1)[0]
        start, _, end = value.partition("..")
        first, last = _date(start), _date(end or start)
        if first is not None and last is not None:
            out.append((first, last))
    return out


def within(periods: list[Period | AnyYearMonth], days: list[Period]) -> bool:
    """Whether any day range of a memory meets any period of a question."""
    for period in periods:
        for first, last in days:
            if isinstance(period, int):
                if period in (first.month, last.month):
                    return True
            elif first <= period[1] and period[0] <= last:
                return True
    return False


def _date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
