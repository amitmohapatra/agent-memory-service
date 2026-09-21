"""The ratchet over the external retrieval baseline.

Every other gate in this directory scores against fixtures written in this repo, which is why
they all read 1.0: they encode current behaviour and pass by construction. This one reads the
result of ``make bench-external`` — BEIR SciFact, a corpus nobody here chose — and fails when
retrieval gets worse than the recorded floor.

The floor is deliberately below the measured value. It is a regression alarm, not a target:
raising it after a genuine improvement is the intended way to use it, and lowering it needs a
reason in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULT = Path(__file__).resolve().parents[2] / "benchmark" / "results" / "external_retrieval.json"

#: Floors, set under the last measured run. Comparable only to runs at the same corpus size.
MIN_RECALL_AT_10 = 0.70
MIN_NDCG_AT_10 = 0.55


def _result() -> dict:
    if not RESULT.is_file():
        pytest.skip(f"{RESULT.name} not produced yet — run `make bench-external`")
    return json.loads(RESULT.read_text())


def test_the_recorded_baseline_was_measured_with_real_models() -> None:
    """A result from the hash stand-in is noise; gating on it would be theatre."""
    result = _result()
    if not result.get("representative"):
        pytest.skip(
            f"last run used the {result.get('embedding_provider')!r} stand-in embedding; "
            "re-run `make bench-external` for a gateable number"
        )
    assert result["dataset"]["queries"] > 0


def test_retrieval_has_not_regressed_on_an_external_corpus() -> None:
    result = _result()
    if not result.get("representative"):
        pytest.skip("not a representative run")
    k = result["k"]
    recall, ndcg = result[f"recall_at_{k}"], result[f"ndcg_at_{k}"]
    assert recall >= MIN_RECALL_AT_10, (
        f"recall@{k} fell to {recall} (floor {MIN_RECALL_AT_10}) on "
        f"{result['dataset']['name']} over {result['dataset']['corpus_size']} documents"
    )
    assert ndcg >= MIN_NDCG_AT_10, f"nDCG@{k} fell to {ndcg} (floor {MIN_NDCG_AT_10})"


def test_the_harness_mapped_its_results_back_to_the_corpus() -> None:
    """The failure that nearly shipped: candidates were read for a ``document_id`` attribute
    that lives in the payload, so nothing mapped and the score was a confident 0.0. A
    benchmark that cannot find its own documents must not be reported as a low score."""
    result = _result()
    mapped, _, returned = result.get("candidates_mapped_to_corpus", "0/0").partition("/")
    if int(returned or 0) == 0:
        pytest.skip("no candidates returned to map")
    assert int(mapped) > 0, "the benchmark mapped none of its hits back to corpus documents"
