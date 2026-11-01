"""Tool-hint confidence on the frozen encoders (``models``): a task a catalog tool answers keeps a
confident hint, one no tool is about (a greeting, the weather on Mars, a question about the
user) gets none at or above ``TOOL_ANSWERS_CONFIDENCE`` (the harness's ``NARROW_FLOOR`` is
the same 0.5), and the candidates keep the fusion's order.

``golden/tool_hint_confidence.json``: 94 tasks in en/de/es/hi/ar over a 30-tool catalog the
harness's benchmarks use, labelled match, near miss or unrelated. ``tool_cosine`` was chosen
on the dev half; the bars below are the held-out test half's, as measured. Before is the same
search read as it was (every hit a match). Writes ``benchmark/results/tool_hint_confidence.json``.
"""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from benchmark.common import RESULTS

from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.adapters.search.qdrant_store import QdrantSearchStore
from memory_service.config.constants import FROZEN_MODELS, local_model_path
from memory_service.config.settings import SearchSettings
from memory_service.domain.context_bundle import TOOL_ANSWERS_CONFIDENCE
from memory_service.domain.script import Script
from memory_service.domain.tools import ToolDescriptor
from memory_service.modules.rag.indexer import Indexer
from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
from memory_service.modules.tools.hints import _candidates
from memory_service.modules.tools.index import ToolIndex, ToolMatch
from memory_service.ports.search import VectorName
from tests.support_models import NO_WEIGHTS, requires_onnxruntime

pytestmark = [pytest.mark.eval, pytest.mark.models]

DATA = Path(__file__).resolve().parent / "golden" / "tool_hint_confidence.json"
K = 5
#: the held-out half, as measured (19 match, 11 near miss, 15 unrelated)
ABSTAIN_PRECISION_MIN = 0.94
UNRELATED_ABSTAINED_MIN = 1.0
FALSE_ABSTAIN_MAX = 0.06


@pytest.fixture(scope="module")
async def index():
    requires_onnxruntime()
    for spec in (FROZEN_MODELS.dense, FROZEN_MODELS.dense_ml):
        if local_model_path(spec.local_dir) is None and spec.model_path is None:
            pytest.skip(f"{NO_WEIGHTS} ({spec.local_dir} missing)")
    from memory_service.adapters.models.embeddings import load_dense

    english, multilingual = load_dense(FROZEN_MODELS.dense), load_dense(FROZEN_MODELS.dense_ml)
    # the deployment's spaces: the English one for Latin script, the multilingual for every query
    spaces = DenseSpaces(
        [
            DenseSpace(VectorName.DENSE_EN, english, query_scripts=frozenset({Script.LATIN})),
            DenseSpace(VectorName.DENSE_ML, multilingual),
        ]
    )
    store = QdrantSearchStore(SearchSettings(), local_path=":memory:")
    tools = ToolIndex(None, Indexer(None, store, spaces, Bm25SparseEncoder(), None), store)  # type: ignore[arg-type]
    await tools.ensure()
    catalog = [ToolDescriptor(tenant_id="t", **c) for c in _data()["catalog"]]
    await store.upsert(await tools._records(catalog))
    yield tools
    english.close()
    multilingual.close()


def _data() -> dict[str, Any]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def _auroc(scores: list[float], positive: list[bool]) -> float:
    s, y = np.asarray(scores), np.asarray(positive)
    pos, neg = s[y][:, None], s[~y][None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def _ece(confidence: list[float], right: list[bool], bins: int = 10) -> float:
    c, y = np.asarray(confidence), np.asarray(right, dtype=float)
    at = np.minimum((c * bins).astype(int), bins - 1)
    return float(
        sum((at == b).mean() * abs(c[at == b].mean() - y[at == b].mean()) for b in set(at))
    )


def _report(rows: list[dict[str, Any]], arm: str) -> dict[str, float | None]:
    best = [r[arm]["best"] for r in rows]
    match = [r["kind"] == "match" for r in rows]
    abstained = [b < TOOL_ANSWERS_CONFIDENCE for b in best]

    def share(kind: str) -> float:
        of = [a for a, r in zip(abstained, rows, strict=True) if r["kind"] == kind]
        return round(sum(of) / len(of), 3)

    held = [m for a, m in zip(abstained, match, strict=True) if a]
    return {
        "auroc_any_tool_fits": round(_auroc(best, match), 3),
        "ece_first_candidate": round(
            _ece([r[arm]["first"] for r in rows], [r["first_right"] for r in rows]), 3
        ),
        "abstain_precision": round(held.count(False) / len(held), 3) if held else None,
        "abstained_unrelated": share("unrelated"),
        "abstained_near_miss": share("near_miss"),
        "abstained_match": share("match"),
    }


async def test_a_task_no_tool_is_about_gets_no_confident_hint(index: ToolIndex) -> None:
    rows = []
    for task in _data()["tasks"]:
        found = await index.search("t", None, task["task"], limit=32)
        before = [ToolMatch(m.name, m.relevance) for m in found]  # every hit a match, as it was
        arms = {}
        for arm, hits in (("before", before), ("after", found)):
            ranked = _candidates(hits, None, None, {}, None, K)
            confidences = [c.confidence for c in ranked] or [0.0]
            arms[arm] = {"best": max(confidences), "first": confidences[0], "order": ranked}
        # the fusion orders the candidates, whatever their confidence
        assert [c.name for c in arms["after"]["order"]] == [c.name for c in arms["before"]["order"]]
        first = arms["after"]["order"][0].name if arms["after"]["order"] else None
        rows.append({**task, **arms, "first_right": first in task["tools"]})
    report: dict[str, Any] = {
        split: {
            arm: _report([r for r in rows if r["split"] == split], arm)
            for arm in ("before", "after")
        }
        for split in ("dev", "test")
    }
    report["tool_cosine"] = FROZEN_MODELS.dense_ml.tool_cosine
    report["unrelated_best_after"] = {
        r["task"]: r["after"]["best"] for r in rows if r["kind"] == "unrelated"
    }
    (RESULTS / "tool_hint_confidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    test = report["test"]["after"]
    assert test["abstained_unrelated"] >= UNRELATED_ABSTAINED_MIN, report["unrelated_best_after"]
    assert test["abstain_precision"] >= ABSTAIN_PRECISION_MIN, test
    assert test["abstained_match"] <= FALSE_ABSTAIN_MAX, test
    # before, no task was ever told no tool fits it
    assert report["test"]["before"]["abstained_unrelated"] == 0.0
