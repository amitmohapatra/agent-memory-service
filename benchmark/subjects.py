"""Same-subject matching: accuracy on the labelled pair set and the cost on the write path.

    uv run python -m benchmark.subjects            # words only, plus the encoder when present

Writes ``benchmark/results/subject_matching.json``:

* ``words`` - the matcher as every write without the conflict adjudicator runs it;
* ``words+encoder`` - with the multilingual encoder's cosine (subject against subject) for
  the pairs the words leave undecided, as a write with the adjudicator runs it; the
  threshold (``DENSE_POSSIBLE``) is read off the dev half and reported on both halves;
* ``rapidfuzz`` - when the package is importable, its ratio in place of the one-edit typo
  rule, to decide whether the dependency earns its place (it is not a dependency);
* timings - one comparison (parse + compare, cold and warm) and one statement's
  consolidation against 20 stored memories, p50/p95 in milliseconds.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.common import provenance, write_result
from benchmark.evaluation.subject_pairs import (
    CTX,
    SubjectPairCase,
    evaluate_subject_pairs,
    judge,
    load_subject_pairs,
)
from benchmark.retrieval import _pct
from memory_service.config.constants import FROZEN_MODELS, MemoryIntelligenceSettings
from memory_service.domain import subjects as domain
from memory_service.domain.enums import ObservationKind
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.domain.subjects import SubjectVerdict
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.subjects import SubjectMatcher

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "tests" / "eval" / "golden" / "subject_pairs.json"

#: what a retail tenant's memory holds: facts about named things, the user's own facts,
#: decisions and the turns themselves
STORED = [
    "Forklift 4 is in aisle 3.",
    "Forklift 3 is due for service on Friday.",
    "Warehouse 13 is closed for inventory.",
    "SKU-1001 costs 4.99 USD.",
    "SKU-1010 is out of stock in store 12.",
    "Acme Logistics delivers to DC 3 on Mondays.",
    "Acme Foods is our dairy supplier.",
    "PO 4471 was approved by Dana.",
    "Store 12 manager is Priya Sharma.",
    "Hazardous materials are stored in cage 2.",
    "My name is Amit.",
    "My timezone is Europe/Berlin.",
    "I prefer concise answers with tables.",
    "We decided to use cross-docking for produce.",
    "Remind me to call Acme Logistics by Friday.",
    "Yesterday the cooler in store 12 failed.",
    "The planogram reset for aisle 7 starts next week.",
    "Global Freight Solutions handles our imports.",
    "Dock 7 is reserved for returns.",
    "Truck 22 arrives at 6am.",
]
INCOMING = [
    "Forklift #4 is in aisle 5.",
    "forklift 4 is in aisle 3",
    "Warehouse 3 is open again.",
    "SKU 1001 costs 5.49 USD.",
    "Acme Logistics GmbH delivers to distribution center 3 on Tuesdays.",
    "PO4471 was cancelled.",
    "Hazmat must be stored in cage 4.",
    "My timezone is America/New_York.",
    "I prefer short answers with bullet points.",
    "We decided to use cross-docking for frozen goods.",
    "Store 12 manager is John Miller.",
    "GFS handles our exports.",
    "Truck 23 arrives at 7am.",
    "The cooler in store 12 was repaired.",
    "Dock 17 is reserved for inbound pallets.",
]


async def _cosine_threshold(matcher: SubjectMatcher, dev: list[SubjectPairCase]) -> dict[str, Any]:
    """The cosine between the subjects of dev pairs the words call different (and do not
    block), by label: the threshold that best separates them (Youden's J)."""
    assert matcher.embedding is not None
    soft: list[tuple[str, float]] = []
    for case in dev:
        verdict, reason = await judge(SubjectMatcher(), case, dense=False)
        if verdict is SubjectVerdict.DIFFERENT and reason.startswith("names differ"):
            vectors = await matcher._encode([case.a, case.b])
            soft.append((case.label, _cos(vectors[case.a], vectors[case.b])))
    same = sorted(c for label, c in soft if label == "same")
    diff = sorted(c for label, c in soft if label != "same")
    best = (0.0, domain.DENSE_POSSIBLE)
    for t in sorted({round(c, 2) for _, c in soft}):
        j = sum(c >= t for c in same) / max(1, len(same)) - sum(c >= t for c in diff) / max(
            1, len(diff)
        )
        if j > best[0]:
            best = (j, t)
    return {
        "soft_pairs": len(soft),
        "same_cosines": [round(c, 3) for c in same],
        "different_cosines": [round(c, 3) for c in diff],
        "youden_j": round(best[0], 3),
        "chosen_threshold": best[1],
        "shipped_threshold": domain.DENSE_POSSIBLE,
    }


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _rapidfuzz(cases: list[SubjectPairCase]) -> dict[str, Any] | None:
    """The one-edit typo rule against rapidfuzz's normalised ratio on the same word sets."""
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]
    except ImportError:
        return None
    out: dict[str, Any] = {}
    for threshold in (80, 85, 90, 95):
        same_hit = same_n = diff_hit = diff_n = 0
        for case in cases:
            vocab = domain.vocabulary().with_aliases(domain.defined_aliases(case.context or ""))
            a, b = domain.parse(case.a, vocab), domain.parse(case.b, vocab)
            if a.ids != b.ids:
                continue
            only_a, only_b = a.tokens - b.tokens, b.tokens - a.tokens
            if not only_a or not only_b:
                continue
            native = domain._spelling_variants(only_a, only_b)
            fuzzy = fuzz.ratio(" ".join(sorted(only_a)), " ".join(sorted(only_b))) >= threshold
            if case.label == "same":
                same_n += 1
                same_hit += fuzzy
            else:
                diff_n += 1
                diff_hit += fuzzy
            out.setdefault("native_edit1", {"same": 0, "different": 0})
            out["native_edit1"]["same" if case.label == "same" else "different"] += native
        out[f"ratio>={threshold}"] = {
            "same_routed": same_hit,
            "same_pairs": same_n,
            "different_routed": diff_hit,
            "different_pairs": diff_n,
        }
    return out


def _obs(text: str) -> Observation:
    return Observation(
        tenant_id=CTX.tenant_id,
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        user_id=CTX.user_id,
        workspace_id=CTX.workspace_id,
        principal_id=CTX.principal_id,
    )


async def _statements(provider: NativeMemoryIntelligence, rounds: int) -> dict[str, Any]:
    """One statement's consolidation against the 20 stored memories, ms per statement."""
    now = datetime.now(UTC)
    stored = []
    for text in STORED:
        for cand in await provider.extract(_obs(text), CTX):
            stored.append(build_memory(await provider.classify(cand, CTX), CTX, now=now))
    incoming = [
        await provider.classify(c, CTX)
        for t in INCOMING
        for c in await provider.extract(_obs(t), CTX)
    ]
    times: list[float] = []
    matcher_times: list[float] = []
    decisions: dict[str, int] = {}
    for _ in range(rounds):
        for cand in incoming:
            t0 = time.perf_counter()
            outcome = await provider.consolidate(cand, stored, CTX)
            times.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            provider.subjects.pairs(cand, stored)
            matcher_times.append((time.perf_counter() - t0) * 1000)
            decisions[outcome.decision.value] = decisions.get(outcome.decision.value, 0) + 1
    return {
        "stored_memories": len(stored),
        "statements": len(incoming),
        "rounds": rounds,
        "consolidate_ms": {"p50": _pct(times, 50), "p95": _pct(times, 95)},
        "subject_matcher_ms": {"p50": _pct(matcher_times, 50), "p95": _pct(matcher_times, 95)},
        "decisions": decisions,
    }


def _comparisons(cases: list[SubjectPairCase], rounds: int) -> dict[str, Any]:
    cold: list[float] = []
    warm: list[float] = []
    vocab = domain.vocabulary()
    for _ in range(rounds):
        domain._parse.cache_clear()
        domain._scan.cache_clear()
        for case in cases:
            t0 = time.perf_counter()
            domain.compare(domain.parse(case.a, vocab), domain.parse(case.b, vocab))
            cold.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            domain.compare(domain.parse(case.a, vocab), domain.parse(case.b, vocab))
            warm.append((time.perf_counter() - t0) * 1000)
    return {
        "comparisons": len(cold),
        "cold_ms": {"p50": _pct(cold, 50), "p95": _pct(cold, 95)},
        "warm_ms": {"p50": round(statistics.median(warm), 4), "p95": _pct(warm, 95)},
    }


async def main(rounds: int, *, encoder: bool) -> dict[str, Any]:
    cases = load_subject_pairs(PAIRS)
    dev = [c for c in cases if c.split == "dev"]
    test = [c for c in cases if c.split == "test"]
    out: dict[str, Any] = {"pairs": len(cases), "dev": len(dev), "test": len(test)}
    words = SubjectMatcher()
    out["words"] = await evaluate_subject_pairs(words, cases)
    out["rapidfuzz"] = _rapidfuzz(cases)
    out["comparison_timing"] = _comparisons(cases, rounds)
    settings = MemoryIntelligenceSettings()
    out["statement_timing"] = {
        "words": await _statements(NativeMemoryIntelligence(settings), rounds)
    }
    if encoder:
        from memory_service.adapters.models.embeddings import load_dense

        dense = load_dense(FROZEN_MODELS.dense_ml)
        matcher = SubjectMatcher(dense)
        out["encoder"] = dense.fingerprint()
        out["dense_threshold"] = await _cosine_threshold(matcher, dev)
        full = await evaluate_subject_pairs(matcher, cases, dense=True)
        out["words+encoder"] = {
            **full,
            "dev": (await evaluate_subject_pairs(matcher, dev, dense=True))["routed_recall"],
            "test": (await evaluate_subject_pairs(matcher, test, dense=True))["routed_recall"],
        }
        timed: list[float] = []
        for _ in range(rounds):
            fresh = SubjectMatcher(dense)  # an empty vector cache: every subject encoded
            for case in cases[:50]:
                t0 = time.perf_counter()
                await judge(fresh, case, dense=True)
                timed.append((time.perf_counter() - t0) * 1000)
        out["comparison_timing"]["with_encoder_cold_ms"] = {
            "p50": _pct(timed, 50),
            "p95": _pct(timed, 95),
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--no-encoder", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(main(args.rounds, encoder=not args.no_encoder))
    path = write_result("subject_matching.json", {**result, "provenance": provenance()})
    print(path)
    for name in ("words", "words+encoder"):
        if name in result:
            r = result[name]
            print(
                name,
                {k: r[k] for k in ("precision", "recall", "f1", "false_merges", "routed_recall")},
            )
    print(result["comparison_timing"], result["statement_timing"])
