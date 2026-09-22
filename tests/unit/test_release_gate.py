"""The release gate evaluator: missing evidence fails, every hard gate is checked, and a
PASS with stand-in providers always carries the representativeness caveat."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_service.tools import release_gate

pytestmark = pytest.mark.unit

GOOD = {
    "durability.json": {"acknowledged_data_loss": 0},
    "security.json": {
        "cross_tenant_unauthorized": 0,
        "cross_user_unauthorized": 0,
        "private_agent_leakage": 0,
    },
    "retrieval_gate.json": {
        "k": 20,
        "critical_recall_at_k": 1.0,
        "critical_evidence_group_recall": 1.0,
        "representative": False,
        "embedding": "hash-v1-d64",
        "reranker": "lexical-v1",
    },
    "memory_gate.json": {"false_merge_rate": 0.0},
    "kg_gate.json": {
        "fact_recall": 1.0,
        "false_facts": 0,
        "noise_entities": 0,
        "query_hit_rate": 1.0,
    },
    "performance.json": {
        "chat_accept_p95_ms": 30,
        "cached_context_p95_ms": 5,
        "recall_p95_ms": 70,
        "context_bundle_p95_ms": 80,
        "file_accept_p95_ms": 25,
        "transport": "in-process ASGI",
        "providers": {"embedding": "hash-v1-d64", "representative": False},
    },
    "failure_injection.json": dict.fromkeys(
        ("worker_kill", "cache_flush", "blob_outage", "search_rebuild", "authz_denial"), "pass"
    ),
    "tool_gate.json": {
        "suggestion_hit_rate": 1.0,
        "next_step_hit_rate": 1.0,
        "plan_validity": 1.0,
        "isolation_violations": 0,
        "undeclared_tool_suggestions": 0,
    },
    "tests.json": {"failed": 0, "errors": 0, "total": 260},
}
NETWORK = {
    "durability_network.json": {
        "transport": "tcp",
        "acknowledged_data_loss": 0,
        "worker_kills_injected": 3,
        "recovery": {"timed_out": False},
    },
    "performance_network.json": {
        "chat_accept_p95_ms": 40,
        "cached_context_p95_ms": 8,
        "recall_p95_ms": 90,
        "context_bundle_p95_ms": 120,
        "file_accept_p95_ms": 35,
        "transport": "tcp",
        "providers": {
            "embedding": "sentence_transformers:granite",
            "search": "qdrant",
            "representative": True,
        },
    },
}


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(release_gate, "RESULTS", tmp_path)
    for name, payload in GOOD.items():
        (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


def _write(results: Path, name: str, payload: dict) -> None:
    (results / name).write_text(json.dumps(payload))


def test_pass_with_caveats(results: Path) -> None:
    ok, failures, notes = release_gate.evaluate_with_notes()
    assert ok and failures == []
    assert len(notes) == 4
    assert sum("NOT representative" in n for n in notes) == 2
    assert any("latency measured in-process only" in n for n in notes)
    assert any("chaos measured in-process only" in n for n in notes)


def test_network_artifacts_replace_the_in_process_only_caveats(results: Path) -> None:
    for name, payload in NETWORK.items():
        _write(results, name, payload)
    ok, failures, notes = release_gate.evaluate_with_notes()
    assert ok and failures == []
    assert not any("in-process only" in n for n in notes)
    assert len(notes) == 2 and all("NOT representative" in n for n in notes)


def test_network_artifacts_with_stand_in_providers_keep_the_provider_caveat(results: Path) -> None:
    for name, payload in NETWORK.items():
        _write(results, name, payload)
    perf = {
        **NETWORK["performance_network.json"],
        "providers": {"embedding": "hash:hash", "representative": False},
    }
    _write(results, "performance_network.json", perf)
    ok, _, notes = release_gate.evaluate_with_notes()
    assert ok
    assert sum("latency measured over tcp" in n and "NOT representative" in n for n in notes) == 1
    assert not any("in-process only" in n for n in notes)


@pytest.mark.parametrize(
    ("name", "patch", "needle"),
    [
        (
            "durability_network.json",
            {"acknowledged_data_loss": 2},
            "data loss over the network = 2",
        ),
        ("durability_network.json", {"recovery": {"timed_out": True}}, "recovery timed out"),
        ("performance_network.json", {"recall_p95_ms": 301}, "recall_p95_ms (network) = 301"),
        (
            "performance_network.json",
            {"chat_accept_p95_ms": None},
            "chat_accept_p95_ms (network) not measured",
        ),
    ],
)
def test_network_gates_use_the_same_thresholds(
    results: Path, name: str, patch: dict, needle: str
) -> None:
    for artifact, payload in NETWORK.items():
        _write(results, artifact, payload)
    _write(results, name, {**NETWORK[name], **patch})
    ok, failures, _ = release_gate.evaluate_with_notes()
    assert not ok and any(needle in f for f in failures), failures


def test_missing_evidence_is_a_failed_gate(results: Path) -> None:
    (results / "durability.json").unlink()
    ok, failures, _ = release_gate.evaluate_with_notes()
    assert not ok and any("durability.json missing" in f for f in failures)


@pytest.mark.parametrize(
    ("name", "patch", "needle"),
    [
        ("durability.json", {"acknowledged_data_loss": 1}, "acknowledged data loss = 1"),
        ("security.json", {"private_agent_leakage": 1}, "private_agent_leakage = 1"),
        ("retrieval_gate.json", {"critical_recall_at_k": 0.99}, "critical Recall@20"),
        ("retrieval_gate.json", {"k": 10}, "critical Recall@20"),
        ("retrieval_gate.json", {"critical_evidence_group_recall": 0.5}, "Evidence-Group"),
        ("memory_gate.json", {"false_merge_rate": 0.02}, "false merge rate"),
        ("kg_gate.json", {"fact_recall": 0.9}, "KG fact recall"),
        ("kg_gate.json", {"false_facts": 1}, "KG false facts"),
        ("kg_gate.json", {"query_hit_rate": 0.5}, "KG query hit rate"),
        ("tool_gate.json", {"suggestion_hit_rate": 0.5}, "tool suggestion_hit_rate"),
        ("tool_gate.json", {"next_step_hit_rate": 0.1}, "tool next_step_hit_rate"),
        ("tool_gate.json", {"plan_validity": 0.9}, "tool plan_validity"),
        ("tool_gate.json", {"isolation_violations": 2}, "tool isolation_violations"),
        ("performance.json", {"recall_p95_ms": 301}, "recall_p95_ms = 301"),
        ("performance.json", {"file_accept_p95_ms": None}, "file_accept_p95_ms not measured"),
        ("failure_injection.json", {"worker_kill": "fail"}, "worker_kill = fail"),
        ("tests.json", {"failed": 1}, "tests failed=1"),
        ("tests.json", {"total": 0}, "records no tests"),
    ],
)
def test_each_gate_blocks(results: Path, name: str, patch: dict, needle: str) -> None:
    data = {**GOOD[name], **patch}
    (results / name).write_text(json.dumps(data))
    ok, failures, _ = release_gate.evaluate_with_notes()
    assert not ok and any(needle in f for f in failures), failures


def test_representative_evidence_has_no_caveat(results: Path) -> None:
    for name in ("retrieval_gate.json", "performance.json"):
        data = json.loads((results / name).read_text())
        if name == "retrieval_gate.json":
            data["representative"] = True
        else:
            data["providers"]["representative"] = True
        (results / name).write_text(json.dumps(data))
    for name, payload in NETWORK.items():
        _write(results, name, payload)
    ok, _, notes = release_gate.evaluate_with_notes()
    assert ok and notes == []


def test_a_retired_measurement_is_not_silently_ignored() -> None:
    """`cache_violations` was retired on 2026-09-22 with an amendment to ADR 0018.

    The gate is fail-closed by design — ADR 0015: "Gates are never downgraded to make a
    build pass. Missing evidence == failed gate" — and it was doing exactly that: the
    output-cache read path (`/v1/tools/lookup`) was removed in 0035987, the producer stopped
    emitting the key, and `make gates` went red on `tool cache_violations = None`.

    Deleting the check to go green is the one thing ADR 0015 forbids, so the metric was
    retired in the ADR first. This test exists so that the *next* retirement has to do the
    same: if a key comes back into the gate without evidence behind it, the block above
    catches it; if this one quietly reappears here without an ADR, that is the same mistake
    in reverse.
    """
    gate = Path("src/memory_service/tools/release_gate.py").read_text()
    adr = Path("docs/adr/0018-tool-memory.md").read_text()
    assert (
        "cache_violations"
        not in gate.split("# cache_violations retired")[-1].split("for key in")[1]
    )
    assert "the cache-violation gate is retired" in adr, (
        "the retirement must stay recorded in ADR 0018, or the gate change has no basis"
    )
