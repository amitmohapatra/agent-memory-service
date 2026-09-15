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
    "tests.json": {"failed": 0, "errors": 0, "total": 260},
}


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(release_gate, "RESULTS", tmp_path)
    for name, payload in GOOD.items():
        (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


def test_pass_with_caveats(results: Path) -> None:
    ok, failures, notes = release_gate.evaluate_with_notes()
    assert ok and failures == []
    assert len(notes) == 2 and all("NOT representative" in n for n in notes)


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
    ok, _, notes = release_gate.evaluate_with_notes()
    assert ok and notes == []
