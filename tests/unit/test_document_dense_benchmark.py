"""Offline vector reuse must bind weights and source bytes, never hide a stale corpus."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from benchmark.document_dense import manifest, validate_vectors, vectors

from memory_service.config.constants import DenseModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_invalid_vectors_cannot_generate_a_benchmark_score(invalid):
    with pytest.raises(ValueError, match="finite"):
        validate_vectors(np.asarray([[invalid, 0.0]]), np.asarray([[0.0, 1.0]]), 1, 1, 2)


def test_document_vector_manifest_changes_when_weights_or_source_change(tmp_path):
    for name in (
        "onnx/model.onnx",
        "tokenizer.json",
        "config.json",
        "1_Pooling/config.json",
        "data.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    spec = DenseModel(model_path=str(tmp_path))
    before = manifest(spec, tmp_path / "data.json")
    (tmp_path / "onnx/model.onnx").write_bytes(b"different weights")
    changed_model = manifest(spec, tmp_path / "data.json")
    assert changed_model != before
    (tmp_path / "data.json").write_text("changed corpus")
    assert manifest(spec, tmp_path / "data.json") != changed_model


async def test_document_vectors_are_reused_without_model_calls_and_wrong_shape_is_refused(
    tmp_path, monkeypatch
):
    model = SimpleNamespace(
        embed_documents=AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]]),
        embed_query=AsyncMock(return_value=[0.0, 1.0]),
        close=Mock(),
    )
    load = Mock(return_value=model)
    monkeypatch.setattr("benchmark.document_dense.embeddings.load_dense", load)
    path = tmp_path / "vectors.npz"
    spec = DenseModel(dimension=2)
    first = await vectors(spec, ["a", "b"], ["q"], path)
    cached = await vectors(spec, ["a", "b"], ["q"], path)
    np.testing.assert_array_equal(first[0], cached[0])
    np.testing.assert_array_equal(first[1], cached[1])
    assert cached[2] == {"reused_vectors": True}
    load.assert_called_once()
    model.close.assert_called_once()
    with pytest.raises(ValueError, match="shapes"):
        await vectors(spec, ["a"], ["q"], path)
