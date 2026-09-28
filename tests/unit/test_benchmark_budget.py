"""The budget guard refuses what the phase cap or the key's floor forbids, and the ledger
turns two readings into a spend."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmark.budget import Budget, Ledger, gateway_root, guard

pytestmark = pytest.mark.unit


def test_the_phase_cap_and_the_floor_both_refuse() -> None:
    key = Budget(key_name="k", max_limit_usd=10.0, current_usage_usd=3.0)
    assert guard(key, projected_usd=0.5, phase_spent_usd=1.0)[0]
    refused, reason = guard(key, projected_usd=0.5, phase_spent_usd=5.8)
    assert not refused and "phase cap" in reason
    refused, reason = guard(key, projected_usd=5.5, phase_spent_usd=0.0)
    assert not refused and "floor" in reason
    with pytest.raises(ValueError):
        guard(key, projected_usd=-1.0, phase_spent_usd=0.0)


def test_the_ledger_records_the_phase_start_once_and_the_spend_since(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.json")
    first = ledger.checkpoint("start", Budget("k", 10.0, 0.004))
    assert first["phase_spent_usd"] == 0.0 and ledger.phase_start == 0.004
    later = Ledger(tmp_path / "budget.json")
    entry = later.checkpoint("after A0", Budget("k", 10.0, 0.154))
    assert entry["phase_spent_usd"] == pytest.approx(0.15)
    assert later.phase_start == 0.004 and len(later.data["checkpoints"]) == 2


def test_the_gateway_root_drops_the_inference_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_GATEWAY_URL", raising=False)
    monkeypatch.setenv("BIFROST_URL", "http://host.docker.internal:8091/v1")
    assert gateway_root() == "http://host.docker.internal:8091"
    monkeypatch.setenv("BENCH_GATEWAY_URL", "http://localhost:8091/")
    assert gateway_root() == "http://localhost:8091"
