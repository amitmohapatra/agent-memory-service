"""Relative dates a memory names, resolved against the moment it was observed.

"We met three days ago" is only a fact together with the date it was said on, and a reader
handed the line six weeks later has to do that arithmetic itself - which is the step LoCoMo's
temporal questions measure and the step a model is worst at. The observation carries the
date (``occurred_at`` becomes the memory's ``observed_at``), so the phrase is resolved once,
at ingest, and rendered beside the text: ``three days ago (2023-05-05)``.

``dateparser`` (BSD-3, 200 locales) does the resolving, with the observation's own moment as
``RELATIVE_BASE`` and only its relative-time parser enabled, so an absolute date in the text
("8 May 2023") is left as it is and a bare number is never read as a day. The cue words that
decide whether a text is worth parsing at all come from dateparser's own locale data, so the
pre-check and the parser cannot disagree about what a relative expression looks like.

**Weekday-relative phrases are deliberately NOT resolved.** ``relative-time`` covers named
offsets (``yesterday``, ``tomorrow``) and counted ones (``three days ago``, ``last week``) in
every language below, but not ``last Tuesday``. Reaching those needs dateparser's
``absolute-time`` parser, and measured on this base it resolves the weekday correctly while
also reading the word "we" as a Wednesday (``we`` -> 2023-05-03) and annotating the absolute
dates this module exists to leave alone. An unresolved phrase costs the reader one inference;
a phrase resolved to the wrong day silently corrupts the answer, so the parser stays narrow.
Same rule as the measurement protocol: unmeasured, never wrong.

Those that are unambiguous in English are resolved by rule instead (``_english``): "last
Friday" (the most recent Friday before the day it was said), "next Tuesday", "last weekend",
"this weekend", "this morning", "tonight", and the vague counts ("a few days ago", "a couple of
weeks ago") as the range they allow. A named period is a range, not a day: "last week" is the
Monday-to-Sunday before, "last month" the calendar month, "last year" the calendar year
(``2023-06-01..2023-06-30``), which is what a question naming that period is matched against
(``retrieval.periods``). A bare weekday ("on Friday"), "next weekend" and the seasons stay
unresolved: past or coming, and which hemisphere, is not in the words.

Nothing here is generated text: the phrase is the memory's own words and the date is
arithmetic on the observation's timestamp.

Cost, measured on this host (4 cores): 0.84 ms when the cue pre-check rejects the text (the
common case) and 6-27 ms when it parses, on the ingest path only. The read path never calls
this; the resolved pairs are already in the payload.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cache

from memory_service.domain.fiscal import FiscalCalendar, resolve_fiscal_mentions
from memory_service.domain.script import Script, detect_script

#: The dateparser languages tried for a script: the twelve the service is measured in, plus
#: the other languages commonly written in the same script. Fewer languages is faster and
#: less ambiguous, so the list is per script rather than every locale dateparser knows.
LANGUAGES: dict[Script, tuple[str, ...]] = {
    Script.LATIN: ("en", "de", "es", "fr", "pt", "it", "ro", "tr", "vi", "nl"),
    Script.CYRILLIC: ("ru", "uk", "bg"),
    Script.GREEK: ("el",),
    Script.ARABIC: ("ar", "fa"),
    Script.HEBREW: ("he",),
    Script.DEVANAGARI: ("hi",),
    Script.BENGALI: ("bn",),
    Script.THAI: ("th",),
    Script.HAN: ("zh",),
    Script.KANA: ("ja",),
    Script.HANGUL: ("ko",),
}
#: relative expressions kept per text; a memory is one to three sentences
MAX_MENTIONS = 6


@dataclass(frozen=True)
class DatedMention:
    """A relative expression as written, and the calendar date it resolves to."""

    text: str
    date: str

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "date": self.date}


@dataclass(frozen=True)
class _Cues:
    phrases: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]
    #: the connectives dateparser skips ("and", "on", "de"): a found phrase may start or
    #: end with one, and the mention rendered beside the text should not
    skip: frozenset[str]

    def found_in(self, text: str) -> bool:
        lowered = text.casefold()
        return any(phrase in lowered for phrase in self.phrases) or any(
            pattern.search(lowered) for pattern in self.patterns
        )


@cache
def _cues(languages: tuple[str, ...]) -> _Cues:
    """Every relative expression dateparser knows for these languages: the named ones
    (``yesterday``, ``last week``, ``今天``) and the numeric patterns (``3 days ago``)."""
    from dateparser.languages.loader import default_loader

    phrases: set[str] = set()
    patterns: list[re.Pattern[str]] = []
    skip: set[str] = set()
    for language in languages:
        info = default_loader.get_locale(language).info
        phrases.update(word.casefold() for word in info.get("ago") or ())
        for words in (info.get("relative-type") or {}).values():
            phrases.update(word.casefold() for word in words)
        for expressions in (info.get("relative-type-regex") or {}).values():
            patterns.extend(re.compile(expression, re.IGNORECASE) for expression in expressions)
        skip.update(word.casefold() for word in info.get("skip") or () if word.strip().isalpha())
    return _Cues(tuple(sorted(phrases, key=len, reverse=True)), tuple(patterns), frozenset(skip))


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_VAGUE_DAYS = {  # how far back "a few days ago" and the like may reach: (most, least) days
    ("few", "days"): (6, 2),
    ("couple", "days"): (3, 2),
    ("few", "weeks"): (35, 14),
    ("couple", "weeks"): (21, 10),
    ("few", "months"): (150, 45),
    ("couple", "months"): (90, 45),
}
_ENGLISH = re.compile(
    r"\b(?:(?P<wd_dir>last|next)\s+(?P<wd>" + "|".join(_WEEKDAYS) + r")"
    r"|(?P<we_dir>last|this)\s+weekend"
    r"|(?P<today>this\s+(?:morning|afternoon|evening)|tonight|earlier\s+today)"
    r"|(?P<p_dir>last|next)\s+(?P<period>week|month|year)"
    r"|(?:a\s+)?(?P<vague>few|couple(?:\s+of)?|several)\s+(?P<unit>days|weeks|months)\s+ago)\b",
    re.IGNORECASE,
)


def _span(first: date, last: date) -> str:
    return first.isoformat() if first == last else f"{first.isoformat()}..{last.isoformat()}"


def _weekday(m: re.Match[str], day: date) -> str:
    wanted = _WEEKDAYS.index(m.group("wd").lower())
    if m.group("wd_dir").lower() == "last":
        at = day - timedelta(days=(day.weekday() - wanted) % 7 or 7)
    else:
        at = day + timedelta(days=(wanted - day.weekday()) % 7 or 7)
    return _span(at, at)


def _weekend(m: re.Match[str], day: date) -> str:
    if m.group("we_dir").lower() == "this":
        ahead = day.weekday() < 5
        saturday = (
            day + timedelta(days=5 - day.weekday())
            if ahead
            else day - timedelta(days=day.weekday() - 5)
        )
    elif day.weekday() >= 5:
        saturday = day - timedelta(days=day.weekday() - 5 + 7)
    else:
        saturday = day - timedelta(days=day.weekday() + 2)
    return _span(saturday, saturday + timedelta(days=1))


def _period(m: re.Match[str], day: date) -> str:
    step = -1 if m.group("p_dir").lower() == "last" else 1
    period = m.group("period").lower()
    if period == "week":
        monday = day - timedelta(days=day.weekday()) + timedelta(weeks=step)
        return _span(monday, monday + timedelta(days=6))
    if period == "month":
        month = day.month - 1 + step
        year, month = day.year + month // 12, month % 12 + 1
        return _span(date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
    return _span(date(day.year + step, 1, 1), date(day.year + step, 12, 31))


def _vague(m: re.Match[str], day: date) -> str:
    vague = m.group("vague").lower()
    word = "few" if vague == "several" else vague.split()[0]
    most, least = _VAGUE_DAYS[(word, m.group("unit").lower())]
    return _span(day - timedelta(days=most), day - timedelta(days=least))


_RULES = (
    ("wd", _weekday),
    ("we_dir", _weekend),
    ("today", lambda _m, day: _span(day, day)),
    ("period", _period),
    ("vague", _vague),
)


def _english(text: str, base: datetime) -> list[DatedMention]:
    """The English relative expressions dateparser's relative-time parser leaves out, and
    the named periods as ranges (module docstring)."""
    day = base.date()
    out: list[DatedMention] = []
    for m in _ENGLISH.finditer(text):
        rule = next(fn for group, fn in _RULES if m.group(group))
        out.append(DatedMention(text=m.group(0), date=rule(m, day)))
    return out


def resolve_dated_mentions(
    text: str, *, base: datetime, script: Script, limit: int = MAX_MENTIONS
) -> list[DatedMention]:
    """The relative dates ``text`` names, resolved against ``base``, in the order the text
    names them; empty for a script with no parser, a text with no relative cue, or a text
    that names none."""
    languages = LANGUAGES.get(script)
    if languages is None or not text.strip():
        return []
    out = _english(text, base) if script is Script.LATIN else []
    seen = {m.text.casefold() for m in out}
    if _cues(languages).found_in(text):
        from dateparser.search import search_dates

        settings = {
            "PARSERS": ["relative-time"],
            "RELATIVE_BASE": base.replace(tzinfo=None),
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }
        ruled = list(seen)
        for phrase, when in search_dates(text, languages=list(languages), settings=settings) or ():
            cleaned = _trimmed(phrase, _cues(languages).skip).casefold()
            # a phrase the English rules already read (as a range) is not read again as a day
            if not cleaned or cleaned in seen or any(cleaned in r or r in cleaned for r in ruled):
                continue
            seen.add(cleaned)
            out.append(
                DatedMention(
                    text=_trimmed(phrase, _cues(languages).skip), date=when.date().isoformat()
                )
            )
    lowered = text.casefold()
    out.sort(key=lambda m: lowered.find(m.text.casefold()))
    return out[:limit]


def _trimmed(phrase: str, skip: frozenset[str]) -> str:
    """The phrase without the connectives the search swept up around it.

    The search reports the words around an expression that *could* have belonged to a date
    ("yesterday and", "on last week"); the mention rendered beside the text should be the
    expression itself. Only dateparser's own skip words are removed, so "hace 2 días" keeps
    its "hace" and "three days ago" keeps its "ago".
    """
    words = phrase.split()
    while words and words[-1].casefold().strip(".,;:") in skip:
        words.pop()
    while words and words[0].casefold().strip(".,;:") in skip:
        words.pop(0)
    return " ".join(words)


def dated_mentions(
    text: str, *, base: datetime, fiscal: FiscalCalendar | None = None
) -> list[dict[str, str]]:
    """Every relative date ``text`` names, as the payload stores them: the fiscal phrases
    first when the deployment has a retail calendar ("last week" is then a fiscal week, not
    seven days back), then the calendar-date phrases they do not already cover."""
    out = (
        [m.as_dict() for m in resolve_fiscal_mentions(text, base=base, calendar=fiscal)]
        if fiscal is not None
        else []
    )
    covered = {m["text"].casefold() for m in out}
    for mention in resolve_dated_mentions(text, base=base, script=detect_script(text)):
        if mention.text.casefold() not in covered and len(out) < MAX_MENTIONS:
            out.append(mention.as_dict())
    return out
