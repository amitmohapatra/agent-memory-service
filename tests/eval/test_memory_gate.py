"""False-merge gate (M7): the native consolidator must not merge distinct memories.
Writes ``benchmark/results/memory_gate.json`` for ``release_gate``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.common import RESULTS, provenance

from memory_service.config.settings import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.evaluation.memory_pairs import evaluate_pairs, load_pairs
from memory_service.modules.memory.native import NativeMemoryIntelligence

pytestmark = pytest.mark.eval

PAIRS = Path(__file__).resolve().parent / "golden" / "memory_pairs.json"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1")


async def test_false_merge_rate_within_budget() -> None:
    settings = MemoryIntelligenceSettings()
    provider = NativeMemoryIntelligence(settings)
    report = await evaluate_pairs(provider, load_pairs(PAIRS), CTX)
    out = {
        "gate": "memory",
        "provider": provider.info.name,
        "golden_set": "memory_consolidation_pairs",
        "threshold": settings.false_merge_rate_max,
        **report,
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "memory_gate.json").write_text(json.dumps(out, indent=2) + "\n")
    bad = [p for p in report["per_pair"] if not p["ok"]]
    assert report["false_merge_rate"] <= settings.false_merge_rate_max, bad
    # dedup recall is informational for the gate but must not silently collapse
    assert report["dedup_recall"] >= 0.9, bad
    assert not bad, bad
