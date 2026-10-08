"""Statement-labeller evaluation: macro-F1 per language and per kind (ADR 0037).

Each labelled item is one user sentence and the ``StatementKind`` it should get, or ``NONE``
for a question, a greeting or a one-off request. ``macro_f1`` is the mean F1 over the six
kinds - a NONE item labelled as a kind is a false positive of that kind, a kind labelled
NONE a false negative - and ``macro_f1_with_none`` adds the NONE class to the mean.

Latency is what the write path pays: every item labelled as an observation of its own (the
lexicon, then the NLI head for what it left open), so the per-sentence percentiles include
the batch of hypotheses a single open sentence costs.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memory_service.domain.enums import StatementKind
from memory_service.modules.memory.statements import StatementLabeller

NONE = "NONE"
KINDS: tuple[str, ...] = tuple(k.value for k in StatementKind)
CLASSES: tuple[str, ...] = (*KINDS, NONE)
GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "eval" / "golden" / "statement_kinds.json"
#: sentences a language model wrote to order, cleaned against the guidelines (``blind1`` the
#: generalisation dev set, ``blind2`` held out and scored once)
BLIND = GOLDEN.with_name("statement_kinds_blind.json")

# What the extractor did before the labeller (``modules/memory/native.py`` up to ADR 0037):
# a question or an acknowledgement stored nothing, its standing-rule pattern stored a rule,
# anything else was stored as it always was - a fact. The baseline every set is scored against.
_QUESTION = re.compile(
    r"[?\uff1f\u061f]\s*$|^(?:what|why|how|when|where|who|can you|could you|do you)\b", re.I
)
_ACKNOWLEDGEMENT = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay|sure|great|cool|yes|no|got it|sounds good)\b[.!]?$",
    re.IGNORECASE,
)
_STANDING_RULE = re.compile(
    r"^(?:please\s+)?(?:(?:always|never)\b|(?:don't|do not)\s+ever\b|"
    r"(?:from now on|going forward|in (?:the )?future)\b[,:]?)\s*(.+)",
    re.IGNORECASE,
)


def load(path: Path = GOLDEN) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_kind(sentence: str) -> str:
    """The kind the extractor before the labeller effectively gave a sentence."""
    s = sentence.strip()
    if _QUESTION.search(s) or _ACKNOWLEDGEMENT.match(s):
        return NONE
    return StatementKind.RULE.value if _STANDING_RULE.match(s) else StatementKind.FACT.value


def evaluate_baseline(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``baseline_kind`` scored like the labeller: overall, per language and per kind."""
    rows_by_lang: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        rows_by_lang[item["lang"]].append((item["kind"], baseline_kind(item["text"])))
    every = [row for rows in rows_by_lang.values() for row in rows]
    return {
        **_summary(every),
        "per_language": {lang: _summary(rows) for lang, rows in sorted(rows_by_lang.items())},
    }


def not_below(labeller: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Where the labeller scores below the baseline: a kind's F1 (over every language) or a
    language's macro-F1. Empty when it is nowhere worse."""
    worse = [
        f"{k}: {labeller['per_kind_f1'][k]} < {v}"
        for k, v in baseline["per_kind_f1"].items()
        if labeller["per_kind_f1"][k] < v
    ]
    return worse + [
        f"{lang}: {labeller['per_language'][lang]['macro_f1']} < {v['macro_f1']}"
        for lang, v in baseline["per_language"].items()
        if labeller["per_language"][lang]["macro_f1"] < v["macro_f1"]
    ]


def f1_table(rows: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Per-class F1 over ``(gold, predicted)`` pairs."""
    out: dict[str, float] = {}
    for k in CLASSES:
        tp = sum(1 for g, p in rows if g == k and p == k)
        fp = sum(1 for g, p in rows if g != k and p == k)
        fn = sum(1 for g, p in rows if g == k and p != k)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[k] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def _summary(rows: Sequence[tuple[str, str]]) -> dict[str, Any]:
    table = f1_table(rows)
    present = [k for k in KINDS if any(g == k for g, _ in rows)]
    return {
        "n": len(rows),
        "macro_f1": round(statistics.fmean(table[k] for k in present), 4) if present else None,
        "macro_f1_with_none": round(statistics.fmean(table.values()), 4),
        "accuracy": round(sum(g == p for g, p in rows) / len(rows), 4) if rows else None,
        "per_kind_f1": {k: round(v, 4) for k, v in table.items()},
    }


async def evaluate(labeller: StatementLabeller, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Label every item on its own and score it; see the module docstring."""
    rows_by_lang: dict[str, list[tuple[str, str]]] = defaultdict(list)
    sources: Counter[str] = Counter()
    latency_ms: list[float] = []
    errors: list[dict[str, str]] = []
    rule_on_none = 0
    for item in items:
        started = time.perf_counter()
        [label] = await labeller.label([item["text"]])
        latency_ms.append((time.perf_counter() - started) * 1000)
        predicted = label.kind.value if label.kind else NONE
        sources[label.source] += 1
        rows_by_lang[item["lang"]].append((item["kind"], predicted))
        if item["kind"] == NONE and predicted in (
            StatementKind.RULE.value,
            StatementKind.CONDITIONAL_RULE.value,
        ):
            rule_on_none += 1
        if predicted != item["kind"]:
            errors.append(
                {
                    "id": item.get("id", ""),
                    "lang": item["lang"],
                    "gold": item["kind"],
                    "predicted": predicted,
                    "source": label.source,
                    "text": item["text"],
                }
            )
    every = [row for rows in rows_by_lang.values() for row in rows]
    ordered = sorted(latency_ms)
    return {
        **_summary(every),
        "per_language": {lang: _summary(rows) for lang, rows in sorted(rows_by_lang.items())},
        #: questions, greetings and one-off requests stored as a standing rule (must be 0)
        "rules_on_none": rule_on_none,
        "sources": dict(sources),
        "latency_ms_per_sentence": {
            "p50": round(ordered[len(ordered) // 2], 3) if ordered else None,
            "p95": round(ordered[int(len(ordered) * 0.95)], 3) if ordered else None,
            "mean": round(statistics.fmean(ordered), 3) if ordered else None,
        },
        "errors": errors,
    }
