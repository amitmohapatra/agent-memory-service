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

Nothing here is generated text: the phrase is the memory's own words and the date is
arithmetic on the observation's timestamp.

Cost, measured on this host (4 cores): 0.84 ms when the cue pre-check rejects the text (the
common case) and 6-27 ms when it parses, on the ingest path only. The read path never calls
this; the resolved pairs are already in the payload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
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


def resolve_dated_mentions(
    text: str, *, base: datetime, script: Script, limit: int = MAX_MENTIONS
) -> list[DatedMention]:
    """The relative dates ``text`` names, resolved against ``base``; empty for a script
    with no parser, a text with no relative cue, or a text that names none."""
    languages = LANGUAGES.get(script)
    if languages is None or not text.strip() or not _cues(languages).found_in(text):
        return []
    from dateparser.search import search_dates

    settings = {
        "PARSERS": ["relative-time"],
        "RELATIVE_BASE": base.replace(tzinfo=None),
        "PREFER_DATES_FROM": "past",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    found = search_dates(text, languages=list(languages), settings=settings)
    out: list[DatedMention] = []
    seen: set[str] = set()
    for phrase, when in found or ():
        cleaned = _trimmed(phrase, _cues(languages).skip)
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        out.append(DatedMention(text=cleaned, date=when.date().isoformat()))
        if len(out) >= limit:
            break
    return out


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
