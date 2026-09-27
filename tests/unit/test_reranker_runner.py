"""The neural reranker uses the shared bounded CPU execution path."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from memory_service.adapters.models._runner import SerialRunner
from memory_service.adapters.models.rerankers import CrossEncoderReranker
from memory_service.config.constants import CrossEncoderModel
from tests.unit.test_model_runner import Occupancy

pytestmark = pytest.mark.unit


async def test_reranker_serializes_concurrent_predictions_and_preserves_ranking(monkeypatch):
    occupancy = Occupancy()
    configuration = {}

    def predict(pairs, **kwargs):
        occupancy()
        return [float(i) / len(pairs) for i in range(len(pairs))]

    def load(*args, **kwargs):
        configuration.update(kwargs)
        return SimpleNamespace(predict=predict)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=load),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float32="float32", nn=SimpleNamespace(Sigmoid=lambda: "sigmoid")),
    )
    model = CrossEncoderReranker(CrossEncoderModel(max_length=256, revision="fixture-revision"))
    try:
        assert configuration["max_length"] == 256
        assert configuration["revision"] == "fixture-revision"
        assert isinstance(model._runner, SerialRunner)
        results = await asyncio.gather(
            *(model.rerank("query", ["first", "second"], top_k=1) for _ in range(4))
        )
        assert all(result[0].index == 1 for result in results)
        assert occupancy.calls == 4 and occupancy.peak == 1
        assert await model.rerank("query", [], top_k=1) == []
    finally:
        model.close()
