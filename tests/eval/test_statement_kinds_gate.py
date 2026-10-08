"""Statement-labeller gate (ADR 0036): macro-F1 over the labelled sentences in
``golden/statement_kinds.json``, per language and per kind, and over the two blind sets in
``golden/statement_kinds_blind.json`` beside the extractor before it (reported). Writes
``benchmark/results/statement_kinds_gate.json`` for the release gate.

Without the ``models`` marker it measures the lexicon alone (the file says
``representative: false``: the service always runs with the NLI head). The frozen-head run
scores the same items through the full labeller and overwrites the file with
``representative: true``. Both hold the same bars: English macro-F1 >= 0.85 on the held-out
half, and no question, greeting or one-off request is ever stored as a standing rule.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from benchmark.common import RESULTS, provenance
from benchmark.evaluation.statement_kinds import BLIND, evaluate, evaluate_baseline, load

from memory_service.config.constants import FROZEN_MODELS
from memory_service.modules.memory.statements import StatementLabeller

pytestmark = pytest.mark.eval

ENGLISH_MACRO_F1_MIN = 0.85


async def _run(labeller: StatementLabeller, *, representative: bool) -> dict[str, Any]:
    golden, blind = load(), load(BLIND)
    report = {
        "gate": "statement_kinds",
        "golden_set": "statement_kinds_v1",
        "representative": representative,
        "nli_provider": labeller.nli.fingerprint() if labeller.nli is not None else None,
        "threshold": {"english_test_macro_f1": ENGLISH_MACRO_F1_MIN, "rules_on_none": 0},
        "dev": await evaluate(labeller, golden["dev"]),
        "test": await evaluate(labeller, golden["test"]),
        # sentences a language model wrote to order, cleaned against the guidelines only, and
        # what the extractor before the labeller made of them (a fact, a question, or a rule
        # by its English standing-rule pattern)
        "blind": {
            name: {
                "role": blind[name]["role"],
                "noise_rate": blind[name]["cleaning"]["noise_rate"],
                "labeller": await evaluate(labeller, blind[name]["items"]),
                "baseline": evaluate_baseline(blind[name]["items"]),
            }
            for name in ("blind1", "blind2")
        },
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "statement_kinds_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def _assert_bars(report: dict[str, Any]) -> None:
    test = report["test"]
    english = test["per_language"]["en"]["macro_f1"]
    assert english >= ENGLISH_MACRO_F1_MIN, test["errors"]
    assert report["dev"]["rules_on_none"] == 0 and test["rules_on_none"] == 0, test["errors"]


async def test_lexicon_statement_kinds() -> None:
    _assert_bars(await _run(StatementLabeller(), representative=False))


@pytest.mark.models
async def test_frozen_head_statement_kinds() -> None:
    from tests.support_models import requires_onnxruntime, requires_weights

    requires_onnxruntime()
    weights = requires_weights(FROZEN_MODELS.nli.local_dir) / FROZEN_MODELS.nli.local_dir
    from memory_service.adapters.models.onnx_nli import OnnxNLI

    nli = OnnxNLI(FROZEN_MODELS.nli.model_copy(update={"model_path": str(weights)}))
    try:
        _assert_bars(await _run(StatementLabeller(nli=nli), representative=True))
    finally:
        nli.close()
