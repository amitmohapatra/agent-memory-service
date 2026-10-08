"""Same-subject evaluation on labelled pairs (``tests/eval/golden/subject_pairs.json``).

Every pair goes through the service's own ``SubjectMatcher`` exactly as consolidation calls
it: the first subject is a new candidate's, the second a stored memory's, and the pair's
``context`` (tenant text that may define an abbreviation) is that memory's content - the
text the matcher learns from. SAME is a merge; POSSIBLE is routed to the conflict
adjudicator when one is configured and kept apart otherwise; DIFFERENT is kept apart.

The false-merge rate is the share of ``different`` pairs called SAME; on the hard negatives
(differing identifiers, units, dates, names that share words) it must be 0.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.subjects import SubjectVerdict
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.subjects import SubjectMatcher
from memory_service.ports.intelligence import MemoryCandidate

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
_EVIDENCE = EvidenceRef(
    source_type=EvidenceSource.MESSAGE, source_id="msg_pair", observed_at=datetime.now(UTC)
)


@dataclass(frozen=True)
class SubjectPairCase:
    id: str
    a: str
    b: str
    label: str  # same | different
    lang: str
    domain: str
    kind: str
    hard_negative: bool
    split: str
    context: str | None = None


def load_subject_pairs(path: Path) -> list[SubjectPairCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SubjectPairCase(**p) for p in raw["pairs"]]


def _candidate(subject: str, content: str) -> MemoryCandidate:
    return MemoryCandidate(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        lifetime=Lifetime.LONG_TERM,
        subject=subject,
        predicate="is",
        evidence=[_EVIDENCE],
    )


async def judge(
    matcher: SubjectMatcher, case: SubjectPairCase, *, dense: bool
) -> tuple[SubjectVerdict, str]:
    """The matcher's verdict on one labelled pair (``dense``: with the encoder's say)."""
    cand = _candidate(case.a, case.a)
    stored = build_memory(_candidate(case.b, case.context or case.b), CTX, now=datetime.now(UTC))
    pairs = matcher.pairs(cand, [stored])
    if dense:
        pairs = await matcher.with_vectors(cand, [stored], pairs)
    verdict = pairs[stored.memory_id].subject
    return verdict.verdict, verdict.reason


def score(rows: Iterable[tuple[SubjectPairCase, SubjectVerdict]]) -> dict[str, Any]:
    """Merge precision/recall/F1 (SAME is a merge), the false-merge rate overall and on the
    hard negatives, and how much reaches the adjudicator (SAME or POSSIBLE)."""
    tp = fp = fn = tn = hard = hard_fm = routed_same = routed_diff = 0
    same_n = diff_n = 0
    for case, verdict in rows:
        merged = verdict is SubjectVerdict.SAME
        routed = verdict is not SubjectVerdict.DIFFERENT
        if case.label == "same":
            same_n += 1
            tp += merged
            fn += not merged
            routed_same += routed
        else:
            diff_n += 1
            fp += merged
            tn += not merged
            routed_diff += routed
            if case.hard_negative:
                hard += 1
                hard_fm += merged
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pairs": same_n + diff_n,
        "same": same_n,
        "different": diff_n,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merges": fp,
        "false_merge_rate": round(fp / diff_n, 4) if diff_n else 0.0,
        "hard_negatives": hard,
        "hard_negative_false_merges": hard_fm,
        "hard_negative_false_merge_rate": round(hard_fm / hard, 4) if hard else 0.0,
        #: of the same-subject pairs, the share SAME or POSSIBLE (what the model can still win)
        "routed_recall": round(routed_same / same_n, 4) if same_n else 0.0,
        #: of the different pairs, the share POSSIBLE: model calls spent on a "no"
        "routed_different_rate": round(routed_diff / diff_n, 4) if diff_n else 0.0,
    }


async def evaluate_subject_pairs(
    matcher: SubjectMatcher, cases: list[SubjectPairCase], *, dense: bool = False
) -> dict[str, Any]:
    judged = [(case, *(await judge(matcher, case, dense=dense))) for case in cases]
    rows = [(case, verdict) for case, verdict, _ in judged]
    by: dict[str, dict[str, list[tuple[SubjectPairCase, SubjectVerdict]]]] = {
        "lang": defaultdict(list),
        "kind": defaultdict(list),
        "split": defaultdict(list),
        "domain": defaultdict(list),
    }
    for case, verdict in rows:
        for key in by:
            by[key][getattr(case, key)].append((case, verdict))
    return {
        **score(rows),
        **{
            f"by_{key}": {k: score(v) for k, v in sorted(groups.items())}
            for key, groups in by.items()
        },
        "errors": [
            {"id": c.id, "a": c.a, "b": c.b, "label": c.label, "verdict": v.value, "reason": r}
            for c, v, r in judged
            if (c.label == "same") != (v is SubjectVerdict.SAME)
        ],
    }
