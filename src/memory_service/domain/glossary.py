"""Retail planning shorthand, searched with its expansion.

A planner writes "WOS for dept 12 vs LY" and a policy document says "weeks of supply"; a
memory says "sell-through" and the question asks for "ST%". No encoder knows that an acronym
of a trade's own jargon means the phrase it abbreviates, and BM25 cannot match one to the
other at all. So a query that names one side is searched with the other side too:
"WOS for dept 12 vs LY (weeks of supply; last year)".

Only unambiguous shorthand is listed. An acronym matches as an upper-case word ("OH" is on
hand, "oh" is not); a phrase matches in any case. Nothing is inferred - the expansion is a
fixed table, the same for every retailer (``RetailSettings.glossary``), kept as data in the
retail vocabulary pack.
"""

from __future__ import annotations

import re
from functools import cache

from memory_service.domain.subjects import pack_aliases

#: acronym -> the phrase it abbreviates: the upper-case forms of the retail vocabulary pack
#: (``domain/vocabulary/retail.json``), which the subject matcher reads too
RETAIL: dict[str, str] = {
    form: canonical
    for canonical, forms in pack_aliases("retail").items()
    for form in forms
    if form.isupper() and form.isalnum()
}
#: additions kept per query, so a list of acronyms cannot double a query's length
MAX_ADDED = 6


_Rule = tuple[re.Pattern[str], re.Pattern[str], str, str]


@cache
def _compiled(table: tuple[tuple[str, str], ...]) -> list[_Rule]:
    out: list[_Rule] = []
    for short, long in table:
        acronym = re.compile(rf"(?<![\w-]){re.escape(short)}%?(?![\w-])")
        body = re.escape(long).replace(r"\-", "[- ]?").replace("through", "thr(?:ough|u)")
        phrase = re.compile(rf"\b{body}\b", re.IGNORECASE)
        out.append((acronym, phrase, short, long))
    return out


def expand(query: str, table: dict[str, str] | None = None) -> str:
    """``query`` with the other side of every shorthand it names appended; unchanged when it
    names none."""
    added: list[str] = []
    for acronym, phrase, short, long in _compiled(tuple((table or RETAIL).items())):
        if len(added) >= MAX_ADDED:
            break
        has_short, has_long = bool(acronym.search(query)), bool(phrase.search(query))
        if has_short and not has_long:
            added.append(long)
        elif has_long and not has_short:
            added.append(short)
    return f"{query} ({'; '.join(added)})" if added else query
