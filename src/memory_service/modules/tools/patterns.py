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


def task_pattern(task: str, *, max_chars: int = MAX_PATTERN_CHARS) -> str:
    """Typed-placeholder form of a task description. Order matters: the most specific shapes
    are replaced first so an email is not first mangled into an identifier."""
    if not task or not task.strip():
        return ""
    text = task.strip()[: max_chars * 2]
    text = _URL.sub("{url}", text)
    text = _EMAIL.sub("{email}", text)
    text = _MONEY.sub("{money}", text)
    text = _DATE.sub("{date}", text)
    for name in extract_entities(text, max_entities=12):
        if len(name) < 3 or name.startswith("{"):
            continue
        text = re.sub(rf"\b{re.escape(name)}\b", "{entity}", text)
    text = _IDENT.sub("{id}", text)
    text = _NUMBER.sub("{num}", text)
    text = _WS.sub(" ", text).strip().casefold()
    return text[:max_chars]


def similarity(left: str, right: str) -> float:
    """Token Jaccard between two patterns. Used to reuse a near-identical pattern's history
    when an exact pattern has no runs yet; deliberately simple and explainable."""
    a = set(left.split())
    b = set(right.split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_match(pattern: str, candidates: list[str], *, threshold: float = 0.6) -> str | None:
    """The stored pattern closest to ``pattern``, or ``None`` when nothing is close enough.
    An exact match always wins."""
    if not pattern:
        return None
    if pattern in candidates:
        return pattern
    scored = [(similarity(pattern, c), c) for c in candidates]
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    if scored and scored[0][0] >= threshold:
        return scored[0][1]
    return None
