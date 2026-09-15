"""Release gate evaluator.

Reads the latest benchmark/eval artifacts under ``benchmark/results/`` and the test
results summary, then fails when any hard gate is violated:

    acknowledged data loss            = 0
    unauthorized retrieval            = 0
    critical Recall@K                 = 1.00
    critical Evidence-Group Recall    = 1.00
    false-merge rate                 <= configured threshold
    p95 latency                      <= configured budget
    failure-recovery suite            passes

Gates are never downgraded to make a build pass. Missing evidence == failed gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from memory_service.config.settings import Settings

RESULTS = Path("benchmark/results")


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def representativeness(results: dict[str, dict[str, Any] | None]) -> list[str]:
    """Caveats that do not fail the gate but must be stated with any PASS: evidence
    produced with stand-in providers bounds the service logic, not a deployment."""
    notes: list[str] = []
    retrieval = results.get("retrieval_gate")
    if retrieval is not None and not retrieval.get("representative", False):
        notes.append(
            f"retrieval quality measured with embedding={retrieval.get('embedding')} "
            f"reranker={retrieval.get('reranker')}: NOT representative of production models"
        )
    perf = results.get("performance")
    if perf is not None and not (perf.get("providers") or {}).get("representative", False):
        notes.append(
            f"latency measured over {perf.get('transport', 'unknown transport')} with "
            f"providers={ {k: v for k, v in (perf.get('providers') or {}).items() if k != 'representative'} }: "
            "NOT representative of a deployed instance"
        )
    return notes


def evaluate(settings: Settings | None = None) -> tuple[bool, list[str]]:
    ok, failures, _ = evaluate_with_notes(settings)
    return ok, failures


def evaluate_with_notes(settings: Settings | None = None) -> tuple[bool, list[str], list[str]]:
    settings = settings or Settings()
    failures: list[str] = []

    durability = _load("durability.json")
    if durability is None:
        failures.append("durability.json missing (acknowledged-data-loss gate has no evidence)")
    elif durability.get("acknowledged_data_loss", 1) != 0:
        failures.append(
            f"acknowledged data loss = {durability.get('acknowledged_data_loss')} (must be 0)"
        )

    security = _load("security.json")
    if security is None:
        failures.append("security.json missing (unauthorized-retrieval gate has no evidence)")
    else:
        for key in (
            "cross_tenant_unauthorized",
            "cross_user_unauthorized",
            "private_agent_leakage",
        ):
            if security.get(key, 1) != 0:
                failures.append(f"{key} = {security.get(key)} (must be 0)")

    retrieval = _load("retrieval_gate.json")
    if retrieval is None:
        failures.append("retrieval_gate.json missing (critical recall gates have no evidence)")
    else:
        k = settings.evaluation.critical_recall_k
        if retrieval.get("critical_recall_at_k", 0.0) < 1.0 or retrieval.get("k") != k:
            failures.append(
                f"critical Recall@{k} = {retrieval.get('critical_recall_at_k')} at k={retrieval.get('k')} (must be 1.00 at k={k})"
            )
        if retrieval.get("critical_evidence_group_recall", 0.0) < 1.0:
            failures.append(
                f"critical Evidence-Group Recall = {retrieval.get('critical_evidence_group_recall')} (must be 1.00)"
            )

    memory = _load("memory_gate.json")
    if memory is None:
        failures.append("memory_gate.json missing (false-merge gate has no evidence)")
    elif memory.get("false_merge_rate", 1.0) > settings.memory_intelligence.false_merge_rate_max:
        failures.append(
            f"false merge rate {memory.get('false_merge_rate')} > {settings.memory_intelligence.false_merge_rate_max}"
        )

    perf = _load("performance.json")
    if perf is None:
        failures.append("performance.json missing (p95 gate has no evidence)")
    else:
        budgets = {
            "chat_accept_p95_ms": settings.budgets.chat_accept_p95_ms,
            "cached_context_p95_ms": settings.budgets.cached_context_p95_ms,
            "recall_p95_ms": settings.budgets.recall_p95_ms,
            "context_bundle_p95_ms": settings.budgets.context_bundle_p95_ms,
            "file_accept_p95_ms": settings.budgets.file_accept_p95_ms,
        }
        for key, budget in budgets.items():
            observed = perf.get(key)
            if observed is None:
                failures.append(f"{key} not measured")
            elif observed > budget:
                failures.append(f"{key} = {observed} > budget {budget}")

    recovery = _load("failure_injection.json")
    if recovery is None:
        failures.append("failure_injection.json missing (recovery gate has no evidence)")
    else:
        for key in ("worker_kill", "cache_flush", "blob_outage", "search_rebuild", "authz_denial"):
            if recovery.get(key) != "pass":
                failures.append(f"failure injection {key} = {recovery.get(key)} (must be pass)")

    tests = _load("tests.json")
    if tests is None:
        failures.append("tests.json missing (all-tests-pass gate has no evidence)")
    elif tests.get("failed", 1) != 0 or tests.get("errors", 1) != 0:
        failures.append(f"tests failed={tests.get('failed')} errors={tests.get('errors')}")
    elif tests.get("total", 0) == 0:
        failures.append("tests.json records no tests")

    notes = representativeness({"retrieval_gate": retrieval, "performance": perf})
    return (not failures, failures, notes)


def main() -> int:
    ok, failures, notes = evaluate_with_notes()
    if ok:
        sys.stdout.write("RELEASE GATE: PASS\n")
    else:
        sys.stdout.write("RELEASE GATE: FAIL\n")
        for f in failures:
            sys.stdout.write(f"  - {f}\n")
    for n in notes:
        sys.stdout.write(f"  ! {n}\n")
    if ok and notes:
        sys.stdout.write(
            "  Gates pass for the service logic in this environment; production readiness "
            "additionally requires the same gates with representative providers.\n"
        )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
