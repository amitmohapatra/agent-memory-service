"""Task patterns (TOOL_MEMORY.md §30.9 step 4).

"update quote Q-1183 with EMEA price for SKU-22" and "update quote Q-9006 with APAC price for
SKU-7" are the same task with different values. Replacing the values with typed placeholders
gives a pattern that groups the runs whose statistics may legitimately be pooled:

    update quote {id} with {entity} price for {id}

The typing is deterministic and reuses the ingestion entity extractor, so no model is needed
and the same text always yields the same pattern.
"""

from __future__ import annotations

import re

from memory_service.modules.ingestion.context_graph import extract_entities

MAX_PATTERN_CHARS = 300

_MONEY = re.compile(r"\b(?:EUR|USD|GBP|CHF|JPY|INR|CAD|AUD)\s?[\d,.]+\b|\B[$€£]\s?[\d,.]+\b")
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(?:Q[1-4]|FY)\s?\d{2,4}\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_URL = re.compile(r"\bhttps?://\S+")
_IDENT = re.compile(r"\b(?=[\w-]*\d)[A-Za-z][\w-]{2,}\b")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_WS = re.compile(r"\s+")


#: The typed shapes a task's values are replaced by, most specific first so an email is not
#: first mangled into an identifier, and an identifier (``INV-2201``) not into a name plus a
#: number. Entities (names the extractor finds) come between the identifiers and the numbers.
_SHAPES_BEFORE_ENTITIES = (
    ("url", _URL),
    ("email", _EMAIL),
    ("money", _MONEY),
    ("date", _DATE),
    ("id", _IDENT),
)
_SHAPES_AFTER_ENTITIES = (("num", _NUMBER),)


def _typed(task: str) -> tuple[str, list[tuple[str, str]]]:
    """The task with each value replaced by its typed placeholder, and the values found, as
    ``(kind, value)`` in the order they were replaced."""
    text = task.strip()[: MAX_PATTERN_CHARS * 2]
    slots: list[tuple[str, str]] = []

    def replace(kind: str, regex: re.Pattern[str], source: str) -> str:
        def keep(match: re.Match[str]) -> str:
            slots.append((kind, match.group(0)))
            return "{" + kind + "}"

        return regex.sub(keep, source)

    for kind, regex in _SHAPES_BEFORE_ENTITIES:
        text = replace(kind, regex, text)
    for name in extract_entities(text, max_entities=12):
        if len(name) >= 3 and not name.startswith("{"):
            text = replace("entity", re.compile(rf"\b{re.escape(name)}\b"), text)
    for kind, regex in _SHAPES_AFTER_ENTITIES:
        text = replace(kind, regex, text)
    return text, slots


def task_pattern(task: str, *, max_chars: int = MAX_PATTERN_CHARS) -> str:
    """Typed-placeholder form of a task description."""
    if not task or not task.strip():
        return ""
    text, _ = _typed(task)
    return _WS.sub(" ", text).strip().casefold()[:max_chars]


def task_slots(task: str) -> list[tuple[str, str]]:
    """The values a task names, with their kind (url, email, money, date, entity, id, num),
    in the order the task names them: what a tool argument may be filled from."""
    if not task or not task.strip():
        return []
    return sorted(_typed(task)[1], key=lambda slot: task.find(slot[1]))


def similarity(left: str, right: str) -> float:
    """Token Jaccard between two patterns. Used to reuse a near-identical pattern's history
    when an exact pattern has no runs yet; deliberately simple and explainable."""
    a = set(left.split())
    b = set(right.split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
