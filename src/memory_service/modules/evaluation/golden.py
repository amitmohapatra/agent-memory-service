"""Golden-set evaluation for retrieval: Recall@k and Evidence-Group Recall.

A golden question lists *evidence groups*; each group is a list of alternative matchers
(document alias + page + substring). A group is satisfied when any retrieved chunk in the
top-k matches any alternative. Metrics:

* ``recall_at_k``: satisfied groups / required groups, averaged over questions (per-question
  partial credit — this is the "critical Recall@20" gate metric).
* ``evidence_group_recall``: fraction of questions whose groups are *all* satisfied within k
  (all-or-nothing — a multi-hop answer with one missing hop is wrong).

Both are 1.00 gates for ``critical`` questions. The evaluator is provider-agnostic: it only
needs the retrieved candidates' payload (document_id, page, text).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Matcher:
    document: str
    page: int | None
    contains: str

    def matches(self, *, document_alias: str | None, page: int | None, text: str) -> bool:
        if document_alias != self.document:
            return False
        if self.page is not None and page != self.page:
            return False
        return self.contains in text


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    query: str
    required_groups: dict[str, list[Matcher]]
    critical: bool = True
    query_type: str | None = None


@dataclass
class GoldenSet:
    name: str
    documents: dict[str, str]  # alias -> fixture filename
    questions: list[GoldenQuestion]

    @classmethod
    def load(cls, path: Path) -> GoldenSet:
        raw = json.loads(path.read_text(encoding="utf-8"))
        questions = [
            GoldenQuestion(
                id=q["id"],
                query=q["query"],
                critical=bool(q.get("critical", True)),
                query_type=q.get("query_type"),
                required_groups={
                    name: [
                        Matcher(m["document"], m.get("page"), m["contains"]) for m in alternatives
                    ]
                    for name, alternatives in q["required_groups"].items()
                },
            )
            for q in raw["questions"]
        ]
        return cls(name=raw["name"], documents=dict(raw["documents"]), questions=questions)


@dataclass
class RetrievedChunk:
    document_alias: str | None
    page: int | None
    text: str
    record_id: str = ""


@dataclass
class QuestionResult:
    id: str
    critical: bool
    query_type_expected: str | None
    query_type_observed: str | None
    satisfied: list[str]
    missing: list[str]
    first_rank: dict[str, int | None] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        total = len(self.satisfied) + len(self.missing)
        return len(self.satisfied) / total if total else 1.0

    @property
    def complete(self) -> bool:
        return not self.missing


def evaluate_question(
    q: GoldenQuestion, retrieved: Sequence[RetrievedChunk], *, k: int, observed_type: str | None
) -> QuestionResult:
    top = list(retrieved[:k])
    satisfied: list[str] = []
    missing: list[str] = []
    first_rank: dict[str, int | None] = {}
    for name, alternatives in q.required_groups.items():
        rank = next(
            (
                i + 1
                for i, c in enumerate(top)
                if any(
                    m.matches(document_alias=c.document_alias, page=c.page, text=c.text)
                    for m in alternatives
                )
            ),
            None,
        )
        first_rank[name] = rank
        (satisfied if rank is not None else missing).append(name)
    return QuestionResult(
        id=q.id,
        critical=q.critical,
        query_type_expected=q.query_type,
        query_type_observed=observed_type,
        satisfied=satisfied,
        missing=missing,
        first_rank=first_rank,
    )


def summarize(results: Sequence[QuestionResult], *, k: int) -> dict[str, Any]:
    critical = [r for r in results if r.critical]
    all_q = list(results)

    def _recall(rs: Sequence[QuestionResult]) -> float:
        return round(sum(r.recall for r in rs) / len(rs), 4) if rs else 1.0

    def _group_recall(rs: Sequence[QuestionResult]) -> float:
        return round(sum(1 for r in rs if r.complete) / len(rs), 4) if rs else 1.0

    routing = [r for r in all_q if r.query_type_expected]
    routing_acc = (
        round(
            sum(1 for r in routing if r.query_type_expected == r.query_type_observed)
            / len(routing),
            4,
        )
        if routing
        else None
    )
    return {
        "k": k,
        "questions": len(all_q),
        "critical_questions": len(critical),
        "critical_recall_at_k": _recall(critical),
        "critical_evidence_group_recall": _group_recall(critical),
        "recall_at_k": _recall(all_q),
        "evidence_group_recall": _group_recall(all_q),
        "routing_accuracy": routing_acc,
        "per_question": [
            {
                "id": r.id,
                "critical": r.critical,
                "recall": round(r.recall, 4),
                "complete": r.complete,
                "missing": r.missing,
                "first_rank": r.first_rank,
                "query_type": {
                    "expected": r.query_type_expected,
                    "observed": r.query_type_observed,
                },
            }
            for r in all_q
        ],
    }


def alias_for(document_id: str | None, aliases: Mapping[str, str]) -> str | None:
    """Map a document_id back to its golden alias (aliases: document_id -> alias)."""
    return aliases.get(document_id or "")
