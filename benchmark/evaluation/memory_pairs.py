"""Consolidation evaluation on labelled observation pairs (false-merge gate).

Each pair is run through a provider *without persistence*: the existing observation is
extracted/classified/built into memories, the incoming one is consolidated against them,
and the first non-CREATE outcome is compared with the label. ``false_merge_rate`` is the
share of ``distinct`` pairs that were merged/reinforced/superseded — the release gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, ObservationKind
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.modules.memory.pipeline import build_memory
from memory_service.ports.intelligence import MemoryIntelligenceProvider

_MERGING = {
    DedupDecision.REINFORCE,
    DedupDecision.MERGE,
    DedupDecision.SUPERSEDE,
    DedupDecision.UPDATE,
}


@dataclass(frozen=True)
class Pair:
    id: str
    existing: str
    incoming: str
    expected: (
        str  # reinforce | supersede | distinct | reinforce_or_distinct | supersede_or_distinct
    )


def load_pairs(path: Path) -> list[Pair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Pair(p["id"], p["existing"], p["incoming"], p["expected"]) for p in raw["pairs"]]


def _observation(text: str, ctx: MemoryExecutionContext) -> Observation:
    return Observation(
        tenant_id=ctx.tenant_id,
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        thread_id=ctx.thread_id,
        principal_id=ctx.principal_id,
    )


async def evaluate_pairs(
    provider: MemoryIntelligenceProvider, pairs: list[Pair], ctx: MemoryExecutionContext
) -> dict[str, Any]:
    now = datetime.now(UTC)
    per_pair: list[dict[str, Any]] = []
    false_merges = 0
    distinct_total = 0
    missed_merges = 0
    merge_total = 0
    for pair in pairs:
        existing_cands = [
            await provider.classify(c, ctx)
            for c in await provider.extract(_observation(pair.existing, ctx), ctx)
        ]
        existing = [build_memory(c, ctx, now=now) for c in existing_cands]
        incoming = [
            await provider.classify(c, ctx)
            for c in await provider.extract(_observation(pair.incoming, ctx), ctx)
        ]
        decisions: list[str] = []
        for cand in incoming:
            outcome = await provider.consolidate(cand, existing, ctx)
            decisions.append(outcome.decision.value)
        merged = any(DedupDecision(d) in _MERGING for d in decisions)
        superseded = DedupDecision.SUPERSEDE.value in decisions
        reinforced = any(
            d in (DedupDecision.REINFORCE.value, DedupDecision.MERGE.value) for d in decisions
        )
        ok: bool
        if pair.expected == "distinct":
            distinct_total += 1
            ok = not merged
            if merged:
                false_merges += 1
        elif pair.expected == "reinforce":
            merge_total += 1
            ok = reinforced
            if not ok:
                missed_merges += 1
        elif pair.expected == "supersede":
            merge_total += 1
            ok = superseded
            if not ok:
                missed_merges += 1
        elif pair.expected == "reinforce_or_distinct":
            ok = not superseded
        elif pair.expected == "supersede_or_distinct":
            ok = not reinforced
        else:
            raise ValueError(pair.expected)
        per_pair.append(
            {
                "id": pair.id,
                "expected": pair.expected,
                "decisions": decisions,
                "extracted": [len(existing_cands), len(incoming)],
                "ok": ok,
            }
        )
    return {
        "pairs": len(pairs),
        "distinct_pairs": distinct_total,
        "false_merges": false_merges,
        "false_merge_rate": round(false_merges / distinct_total, 4) if distinct_total else 0.0,
        "merge_pairs": merge_total,
        "missed_merges": missed_merges,
        "dedup_recall": round(1 - missed_merges / merge_total, 4) if merge_total else 1.0,
        "failures": [p["id"] for p in per_pair if not p["ok"]],
        "per_pair": per_pair,
    }
