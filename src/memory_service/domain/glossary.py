"""Retail planning shorthand, searched with its expansion.

A planner writes "WOS for dept 12 vs LY" and a policy document says "weeks of supply"; a
memory says "sell-through" and the question asks for "ST%". No encoder knows that an acronym
of a trade's own jargon means the phrase it abbreviates, and BM25 cannot match one to the
other at all. So a query that names one side is searched with the other side too:
"WOS for dept 12 vs LY (weeks of supply; last year)".

Only unambiguous shorthand is listed. An acronym matches as an upper-case word ("OH" is on
hand, "oh" is not); a phrase matches in any case. Nothing is inferred - the expansion is a
fixed table, the same for every retailer (``RetailSettings.glossary``).
"""

from __future__ import annotations

import re
from functools import cache

#: acronym -> the phrase it abbreviates
RETAIL: dict[str, str] = {
    "WOS": "weeks of supply",
    "WOC": "weeks of cover",
    "DOS": "days of supply",
    "ST": "sell-through",
    "GMROI": "gross margin return on investment",
    "GM": "gross margin",
    "AUR": "average unit retail",
    "AUC": "average unit cost",
    "ASP": "average selling price",
    "OTB": "open to buy",
    "MD": "markdown",
    "IMU": "initial markup",
    "MMU": "maintained markup",
    "SKU": "stock keeping unit",
    "UPC": "universal product code",
    "PO": "purchase order",
    "DC": "distribution center",
    "OH": "on hand",
    "OO": "on order",
    "BOP": "beginning of period",
    "EOP": "end of period",
    "ROS": "rate of sale",
    "OOS": "out of stock",
    "MOQ": "minimum order quantity",
    "UPT": "units per transaction",
    "ATV": "average transaction value",
    "LFL": "like for like",
    "POS": "point of sale",
    "BOGO": "buy one get one",
    "BTS": "back to school",
    "PLC": "product life cycle",
    "LY": "last year",
    "TY": "this year",
    "LW": "last week",
    "YTD": "year to date",
    "QTD": "quarter to date",
    "MTD": "month to date",
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
