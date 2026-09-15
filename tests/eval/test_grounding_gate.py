"""Grounding gate (change 20): precision/recall of the cascade per verdict over the labelled
answer/evidence pairs in ``golden/grounding_claims.json``. Writes
``benchmark/results/grounding_gate.json`` for the release gate.

With the lexical stand-in the deterministic parts must be perfect (citation validation,
decomposition, verbatim support, contradictions on numbers/negation, the unused scan) and
the file says ``representative: false``. The DeBERTa run (``models`` marker, real weights)
asserts the quality thresholds below and overwrites the file with ``representative: true``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from benchmark.common import RESULTS, provenance

from memory_service.adapters.models.nli import LexicalNLI
from memory_service.config.settings import NLISettings
from memory_service.modules.grounding.cascade import Evidence, GroundingCascade
from memory_service.ports.models import NLIProvider

SUPPORTED_PRECISION_MIN = 0.9
CONTRADICTED_RECALL_MIN = 0.9
DETERMINISTIC_CATEGORIES = ("citation", "number", "negation", "verbatim", "unused", "structure")
VERDICTS = ("supported", "unsupported", "contradicted", "borderline")

pytestmark = pytest.mark.eval

GOLDEN = Path(__file__).resolve().parent / "golden" / "grounding_claims.json"


def _load() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _evidence(golden: dict[str, Any], keys: list[str]) -> list[Evidence]:
    return [Evidence(item_id=k, text=golden["evidence"][k], kind="chunk") for k in keys]


async def run_gate(nli: NLIProvider) -> dict[str, Any]:
    golden = _load()
    cascade = GroundingCascade(nli, settings=NLISettings(provider="lexical"))
    rows: list[dict[str, Any]] = []
    for case in golden["cases"]:
        report = await cascade.verify(
            case["answer"],
            _evidence(golden, case["evidence"]),
            unused=_evidence(golden, case.get("unused", [])),
        )
        expected = case["expected"] if isinstance(case["expected"], list) else [case["expected"]]
        predicted = [c.verdict for c in report.claims]
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected": expected,
                "predicted": predicted,
                "methods": [c.method for c in report.claims],
                "ok": predicted == expected,
            }
        )
    per_verdict: dict[str, dict[str, Any]] = {}
    for v in VERDICTS:
        tp = fp = fn = 0
        for r in rows:
            pairs = list(zip(r["expected"], r["predicted"], strict=False))
            tp += sum(1 for e, p in pairs if e == v and p == v)
            fp += sum(1 for e, p in pairs if e != v and p == v)
            fn += sum(1 for e, p in pairs if e == v and p != v)
            fn += sum(1 for e in r["expected"][len(r["predicted"]) :] if e == v)
        per_verdict[v] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    categories: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = categories.setdefault(r["category"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(r["ok"])
    return {
        "gate": "grounding",
        "golden_set": golden["name"],
        "nli_provider": nli.fingerprint(),
        "representative": nli.representative,
        "cases": len(rows),
        "claims": sum(len(r["expected"]) for r in rows),
        "per_verdict": per_verdict,
        "categories": categories,
        "failures": [f"{r['id']}: expected {r['expected']}, got {r['predicted']}" for r in rows if not r["ok"]],
        "thresholds": {
            "supported_precision_min": SUPPORTED_PRECISION_MIN,
            "contradicted_recall_min": CONTRADICTED_RECALL_MIN,
            "deterministic_categories": list(DETERMINISTIC_CATEGORIES),
        },
        "rows": rows,
    }


def _write(report: dict[str, Any], note: str | None) -> None:
    out = {**report, "provenance": provenance()}
    if note:
        out["note"] = note
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "grounding_gate.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str) + "\n"
    )


def _deterministic_failures(report: dict[str, Any]) -> list[str]:
    return [
        f"{r['id']}: expected {r['expected']}, got {r['predicted']}"
        for r in report["rows"]
        if r["category"] in DETERMINISTIC_CATEGORIES and not r["ok"]
    ]


async def test_grounding_gate_with_lexical_stand_in() -> None:
    golden = _load()
    assert len(golden["cases"]) >= 30
    report = await run_gate(LexicalNLI())
    _write(
        report,
        "lexical NLI is a deterministic stand-in; only the deterministic categories are "
        "asserted and the quality numbers are not representative",
    )
    assert report["representative"] is False
    bad = _deterministic_failures(report)
    assert not bad, bad
    # the stand-in never promotes a wrong claim to supported: precision stays perfect
    assert report["per_verdict"]["supported"]["precision"] == 1.0, report["per_verdict"]
    assert report["per_verdict"]["contradicted"]["precision"] == 1.0, report["per_verdict"]


@pytest.mark.models
async def test_grounding_gate_with_deberta() -> None:
    root = os.environ.get("MEMORY_MODELS_DIR")
    if not root:
        pytest.skip("MEMORY_MODELS_DIR not set")
    path = Path(root) / "deberta-v3-base-mnli-fever-anli"
    if not path.exists():
        pytest.skip(f"{path} not present")
    from memory_service.adapters.models.nli import TransformersNLI

    report = await run_gate(TransformersNLI(NLISettings(model_path=str(path))))
    _write(report, None)
    assert report["representative"] is True
    bad = [
        f"{r['id']}: expected {r['expected']}, got {r['predicted']}"
        for r in report["rows"]
        if r["category"] in ("citation", "structure") and not r["ok"]
    ]
    assert not bad, bad
    supported = report["per_verdict"]["supported"]
    contradicted = report["per_verdict"]["contradicted"]
    assert supported["precision"] is not None and supported["precision"] >= SUPPORTED_PRECISION_MIN, (
        report["failures"]
    )
    assert contradicted["recall"] is not None and contradicted["recall"] >= CONTRADICTED_RECALL_MIN, (
        report["failures"]
    )
