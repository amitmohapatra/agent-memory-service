"""The Unicode script a text is written in, decided the same way every time.

Two things read it. The retrieval engine prunes the English specialist's prefetch on any
query that is not Latin script, so a Cyrillic or Thai question costs one dense encode and
two prefetches instead of two and three. And every indexed record carries its script as an
indexed payload field, so a corpus can be filtered and measured per script.

It is not a language detector and does not try to be one: the script is a property of the
code points, read from their Unicode names (``LATIN SMALL LETTER A``, ``CYRILLIC SMALL
LETTER A``, ``CJK UNIFIED IDEOGRAPH-6211``), so the answer needs no model, no table beyond
this file, and cannot drift between two runs. Mixed text takes its dominant script by
letter count; a tie goes to the earlier member of ``Script``.
"""

from __future__ import annotations

import unicodedata
from enum import StrEnum
from functools import cache


class Script(StrEnum):
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    GREEK = "greek"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"
    THAI = "thai"
    #: CJK unified ideographs (Chinese, and the kanji of Japanese text)
    HAN = "han"
    #: hiragana and katakana
    KANA = "kana"
    HANGUL = "hangul"
    #: letters of a script this module does not name
    OTHER = "other"
    #: no letters at all (numbers, punctuation, an empty string)
    NONE = "none"


#: The first word of a letter's Unicode name, for the scripts named above.
_BY_NAME: dict[str, Script] = {
    "LATIN": Script.LATIN,
    "CYRILLIC": Script.CYRILLIC,
    "GREEK": Script.GREEK,
    "ARABIC": Script.ARABIC,
    "HEBREW": Script.HEBREW,
    "DEVANAGARI": Script.DEVANAGARI,
    "BENGALI": Script.BENGALI,
    "THAI": Script.THAI,
    "CJK": Script.HAN,
    "HIRAGANA": Script.KANA,
    "KATAKANA": Script.KANA,
    "HANGUL": Script.HANGUL,
}
#: Width variants carry the script as their second word (``FULLWIDTH LATIN SMALL LETTER A``).
_WIDTH_PREFIXES = frozenset({"FULLWIDTH", "HALFWIDTH"})
_ORDER = {member: position for position, member in enumerate(Script)}
#: Letters examined before the answer is settled: a chunk is 2,000 characters at most, and
#: a text that has not declared its script in 512 letters does not have one.
SAMPLE_LETTERS = 512


@cache
def script_of(char: str) -> Script:
    """The script of one letter; ``OTHER`` for a letter of an unnamed script.

    Cached on the character: a pure function of one code point, and a text reuses its
    alphabet on nearly every letter. ``unicodedata.name`` plus the split is the whole cost of
    ``detect_script``, and skipping it for a repeat letter takes a 2,000-character chunk from
    1.20 ms to 0.28 ms on this host - paid once per indexed record. The cache is bounded by
    the distinct letters the corpus actually contains, each mapping to one enum member.
    """
    words = unicodedata.name(char, "").split()
    if words and words[0] in _WIDTH_PREFIXES:
        words = words[1:]
    return _BY_NAME.get(words[0], Script.OTHER) if words else Script.OTHER


def detect_script(text: str, *, sample: int = SAMPLE_LETTERS) -> Script:
    """The dominant script of ``text`` by letter count over its first ``sample`` letters."""
    counts: dict[Script, int] = {}
    seen = 0
    for char in text:
        if unicodedata.category(char)[0] != "L":
            continue
        found = script_of(char)
        counts[found] = counts.get(found, 0) + 1
        seen += 1
        if seen >= sample:
            break
    if not counts:
        return Script.NONE
    return max(counts, key=lambda found: (counts[found], -_ORDER[found]))
