"""Gold paragraph identities, empty queries and language pairing cannot inflate recall."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from benchmark.multilingual_dense import fuse, metrics, ranking, record
from benchmark.multilingual_sparse import SparseCorpus, evaluate

from memory_service.config.constants import DenseModel

pytestmark = pytest.mark.unit


def test_rankings_measure_document_identity_not_answer_word_overlap():
    docs = ["The red telescope is in Delhi.", "The blue telescope is in Paris."]
    result = evaluate(docs, [{"id": "q", "query": "red telescope Delhi", "gold": 0}])
    assert result["recall"]["1"] == 1 and result["mrr_at_10"] == 1
    result = evaluate(docs, [{"id": "q", "query": "red telescope Delhi", "gold": 1}])
    assert result["recall"]["1"] == 0


def test_empty_query_is_a_miss_instead_of_a_lucky_tie():
    assert SparseCorpus(["a", "b"]).retrieve("the and") == ([], True)
    result = evaluate(["a"], [{"id": "q", "query": "the and", "gold": 0}])
    assert result["recall"]["10"] == 0 and result["empty_queries"] == 1


def test_dense_ties_and_fusion_have_stable_document_identity():
    found = ranking([1.0, 0.0], np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert found == [0, 1, 2]
    assert fuse([2, 0], [1, 0]) == [0, 1, 2]


def test_dense_metrics_cut_at_ten_before_counting_recall():
    rows = []
    record(rows, {"id": "outside", "gold": 10}, list(range(12)))
    record(rows, {"id": "inside", "gold": 0}, list(range(12)))
    assert metrics(rows)["recall"]["10"] == 0.5
    assert metrics(rows)["mrr_at_10"] == 0.5


async def test_resume_rejects_a_different_query_limit_before_loading_a_model(tmp_path, monkeypatch):
    from benchmark.multilingual_dense import run

    spec = DenseModel()
    spec_path, output = tmp_path / "model.json", tmp_path / "report.json"
    spec_path.write_text(spec.model_dump_json())
    (tmp_path / "manifest.json").write_text("{}")
    output.write_text(json.dumps({"spec": spec.model_dump(), "dataset": {}, "languages": {}}))

    def unexpected_load(spec):
        pytest.fail("Resume mismatch must fail before loading model weights")

    monkeypatch.setattr("benchmark.multilingual_dense.load_dense", unexpected_load)
    args = SimpleNamespace(
        spec=spec_path, data=tmp_path, output=output, resume=True, limit=100, languages="en"
    )
    with pytest.raises(ValueError, match="same model specification and dataset"):
        await run(args)
