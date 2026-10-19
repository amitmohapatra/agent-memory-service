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
#: "summer 2022", "the winter of 2021": a season of a named year, read for the northern
#: hemisphere (meteorological seasons; winter is that year's December, January and February)
_SEASON = re.compile(
    r"\b(spring|summer|fall|autumn|winter)(?:\s+of)?\s+((?:19|20)\d{2})\b", re.IGNORECASE
)
_SEASON_MONTHS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11)}

#: The same reading in the other pack languages (ADR 0024): month names, a date written the
#: language's way ("10. April 2023", "10 de abril de 2023", "10 أبريل 2023", "10 अप्रैल
#: 2023"), a year, and the commonest relative periods. Arabic month names double as words
#: ("مارس" is also "he practised"), so one counts only beside a number or after "في"/"شهر".
FOREIGN_MONTHS: dict[str, dict[str, int]] = {
    "de": dict(
        zip(
            [
                "januar",
                "februar",
                "märz",
                "april",
                "mai",
                "juni",
                "juli",
                "august",
                "september",
                "oktober",
                "november",
                "dezember",
            ],
            range(1, 13),
            strict=True,
        )
    ),
    "es": dict(
        zip(
            [
                "enero",
                "febrero",
                "marzo",
                "abril",
                "mayo",
                "junio",
                "julio",
                "agosto",
                "septiembre",
                "octubre",
                "noviembre",
                "diciembre",
            ],
            range(1, 13),
            strict=True,
        )
    )
    | {"setiembre": 9},
    "ar": dict(
        zip(
            [
                "يناير",
                "فبراير",
                "مارس",
                "أبريل",
                "مايو",
                "يونيو",
                "يوليو",
                "أغسطس",
                "سبتمبر",
                "أكتوبر",
                "نوفمبر",
                "ديسمبر",
            ],
            range(1, 13),
            strict=True,
        )
    )
    | {"ابريل": 4, "إبريل": 4, "اغسطس": 8, "اكتوبر": 10},
    "hi": dict(
        zip(
            [
                "जनवरी",
                "फ़रवरी",
                "मार्च",
                "अप्रैल",
                "मई",
                "जून",
                "जुलाई",
                "अगस्त",
                "सितंबर",
                "अक्टूबर",
                "नवंबर",
                "दिसंबर",
            ],
            range(1, 13),
            strict=True,
        )
    )
    | {"फरवरी": 2, "सितम्बर": 9, "नवम्बर": 11, "दिसम्बर": 12},
}
_FOREIGN_MONTH = re.compile(
    r"(?<![^\W\d_])(\d{1,2})?\.?\s*(?:de\s+)?("
    + "|".join(
        sorted({re.escape(m) for months in FOREIGN_MONTHS.values() for m in months}, key=len)[::-1]
    )
    + r")(?![^\W\d_])(?:\s*(?:de|del)?\s*((?:19|20)\d{2}))?",
    re.IGNORECASE,
)
_ARABIC_MONTH_CONTEXT = re.compile(r"(?:في|شهر)\s*$")
#: "im Jahr 2023", "en 2023", "عام 2023", "2023 में"
_FOREIGN_YEAR = re.compile(
    r"(?:\b(?:im\s+jahr|in|seit|en|el\s+año|durante)\s+|(?:عام|سنة|في)\s+)((?:19|20)\d{2})\b"
    r"|\b((?:19|20)\d{2})\s+(?:में|के\s+दौरान)",
    re.IGNORECASE,
)
_FOREIGN_RELATIVE: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), unit)
    for pattern, unit in (
        (r"\b(?:letzte|vergangene)n?\s+woche(?:nende)?\b", "week"),
        (r"\b(?:la\s+semana|el\s+fin\s+de\s+semana)\s+pasad[oa]\b", "week"),
        (r"الأسبوع\s+الماضي|عطلة\s+نهاية\s+الأسبوع\s+الماضية", "week"),
        (r"पिछले\s+(?:हफ़्ते|हफ्ते|सप्ताह)", "week"),
        (r"\b(?:letzte|vergangene)n?\s+monat\b", "month"),
        (r"\bel\s+mes\s+pasado\b", "month"),
        (r"الشهر\s+الماضي", "month"),
        (r"पिछले\s+महीने", "month"),
        (r"\b(?:letzte|vergangene)s\s+jahr\b", "year"),
        (r"\bel\s+año\s+pasado\b", "year"),
        (r"العام\s+الماضي|السنة\s+الماضية", "year"),
        (r"पिछले\s+साल", "year"),
    )
)


def _month(year: int, month: int) -> Period:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def query_periods(query: str, now: date) -> list[Period | AnyYearMonth]:
    """The periods ``query`` names; a month named without a year is that month number.

    A relative period is read from the moment asked, unless the question names a day of its
    own: "last weekend before April 10, 2023" is the weekend before that day, not before
    today."""
    out: list[Period | AnyYearMonth] = []
    days: list[date] = []
    for m in _DATED.finditer(query):
        month = _MONTHS[m.group(2).lower()]
        day, year = m.group(1) or m.group(3), m.group(4)
        if year and day and 1 <= int(day) <= calendar.monthrange(int(year), month)[1]:
            on = date(int(year), month, int(day))
            out.append((on, on))
            days.append(on)
        elif year:
            out.append(_month(int(year), month))
        else:
            out.append(month)
    out += [(date(int(y), 1, 1), date(int(y), 12, 31)) for y in _YEAR.findall(query)]
    for m in _SEASON.finditer(query):
        season, year = m.group(1).lower(), int(m.group(2))
        if season == "winter":
            out += [(date(year, 1, 1), date(year, 2, calendar.monthrange(year, 2)[1]))]
            out += [_month(year, 12)]
        else:
            first, last = _SEASON_MONTHS[season]
            out.append((_month(year, first)[0], _month(year, last)[1]))
    foreign, foreign_days = _foreign(query)
    out += foreign
    days += foreign_days
    anchor = days[0] if len(set(days)) == 1 else now
    return out + _relative(query.lower(), anchor)


def _foreign(query: str) -> tuple[list[Period | AnyYearMonth], list[date]]:
    """Dates, months and years named in German, Spanish, Arabic or Hindi."""
    out: list[Period | AnyYearMonth] = []
    days: list[date] = []
    if query.isascii():
        return out, days
    lowered = query.lower()
    for m in _FOREIGN_MONTH.finditer(lowered):
        name = m.group(2)
        month = next(months[name] for months in FOREIGN_MONTHS.values() if name in months)
        day, year = m.group(1), m.group(3)
        if (
            name in FOREIGN_MONTHS["ar"]
            and not (day or year)
            and not _ARABIC_MONTH_CONTEXT.search(lowered[: m.start(2)])
        ):
            continue
        if year and day and 1 <= int(day) <= calendar.monthrange(int(year), month)[1]:
            on = date(int(year), month, int(day))
            out.append((on, on))
            days.append(on)
        elif year:
            out.append(_month(int(year), month))
        else:
            out.append(month)
    for m in _FOREIGN_YEAR.finditer(lowered):
        year = int(m.group(1) or m.group(2))
        if not any(isinstance(p, tuple) and p[0].year == year for p in out):
            out.append((date(year, 1, 1), date(year, 12, 31)))
    return out, days


def _relative(text: str, now: date) -> list[Period | AnyYearMonth]:
    out: list[Period | AnyYearMonth] = []
    week = re.search(r"\b(last|past) week(end)?\b", text)
    month = re.search(r"\b(last|past) month\b", text)
    year = re.search(r"\blast year\b", text)
    for pattern, unit in _FOREIGN_RELATIVE:
        if pattern.search(text):
            week = week or (unit == "week")
            month = month or (unit == "month")
            year = year or (unit == "year")
    if week:
        out.append((now - timedelta(days=13), now - timedelta(days=1)))
    if month:
        prior = now.replace(day=1) - timedelta(days=1)
        out.append((_month(prior.year, prior.month)[0], now - timedelta(days=1)))
    if re.search(r"\b(past|last) (few|couple of|two|three) months\b", text):
        out.append((now - timedelta(days=100), now))
    if re.search(r"\bthis year\b", text):
        out.append((date(now.year, 1, 1), now))
    if year:
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


def months_dated(payload: dict[str, Any]) -> frozenset[int]:
    """The months (0 = January) a memory is dated in: ``memory_days``, every month of a
    range included (all twelve for a range of a year or more)."""
    out: set[int] = set()
    for first, last in memory_days(payload):
        span = (last.year - first.year) * 12 + last.month - first.month
        out.update((first.month - 1 + k) % 12 for k in range(min(span, 11) + 1))
    return frozenset(out)


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
