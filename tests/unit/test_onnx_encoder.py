"""The ONNX encoder's arithmetic, against a fake session.

No weights, no onnxruntime, no torch: this machine has no wheel for any of them. What is
pinned here is everything that decides what a vector *is* — which tokens are padding, which
positions the mask hides, how the token vectors become one vector, whether it is unit
length, which inputs the graph is fed, and what the resulting vector space is called. A
mistake in any of those is silent: it produces vectors, and they are wrong.

Real weights are the contract test (``tests/contract/test_model_adapters.py``, marked
``models``), which asserts the same vectors as the torch runner.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from memory_service.adapters.models.embeddings import (
    OnnxEmbedding,
    _load_graph,
    _OnnxEncoder,
    _pad_token_id,
    _pooling_mode,
    onnx_fingerprint,
)
from memory_service.config.constants import DenseModel
from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.unit

PAD = 7


class FakeTokenizer:
    """One id per word, the id being the word's length, and no special tokens: padding and
    the attention mask are what these tests are about, not the vocabulary."""

    def encode_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        out = []
        for text in texts:
            ids = [len(word) for word in text.split()]
            out.append(SimpleNamespace(ids=ids, attention_mask=[1] * len(ids)))
        return out


class FakeSession:
    """Returns the hidden state it was given and remembers what it was fed."""

    def __init__(self, hidden: Any, inputs: tuple[str, ...] = ("input_ids", "attention_mask")):
        self._hidden = hidden
        self._inputs = inputs
        self.feeds: list[dict[str, Any]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self._inputs]

    def run(self, _outputs: Any, feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(feed)
        hidden = self._hidden(feed) if callable(self._hidden) else self._hidden
        return [np.asarray(hidden, dtype=np.float32)]


def encoder(
    hidden: Any,
    *,
    pooling: str = "cls",
    normalize: bool = False,
    inputs: tuple[str, ...] = ("input_ids", "attention_mask"),
) -> _OnnxEncoder:
    return _OnnxEncoder(
        FakeTokenizer(),
        FakeSession(hidden, inputs),
        pooling=pooling,
        dimension=2,
        pad_id=PAD,
        normalize=normalize,
    )


def test_cls_pooling_takes_the_first_position() -> None:
    enc = encoder([[[1.0, 0.0], [9.0, 9.0]], [[0.0, 1.0], [9.0, 9.0]]])
    assert enc.encode(["aa bb", "cc dd"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_mean_pooling_ignores_the_padded_positions() -> None:
    """The padded position carries a large value on purpose: if the mask were not applied
    the second vector would be 51, not 2."""
    enc = encoder([[[1.0, 1.0], [3.0, 3.0]], [[2.0, 2.0], [100.0, 100.0]]], pooling="mean")
    assert enc.encode(["aa bb", "cc"]) == [[2.0, 2.0], [2.0, 2.0]]


def test_vectors_are_unit_length_when_the_spec_normalises() -> None:
    enc = encoder([[[3.0, 4.0], [0.0, 0.0]]], normalize=True)
    assert enc.encode(["aa bb"]) == [pytest.approx([0.6, 0.8])]


def test_a_zero_vector_normalises_to_zero_rather_than_to_nan() -> None:
    enc = encoder([[[0.0, 0.0], [0.0, 0.0]]], normalize=True)
    assert enc.encode(["aa bb"]) == [[0.0, 0.0]]


def test_the_batch_is_padded_to_its_own_longest_member() -> None:
    """Padding a 3-token query to the model's 512 is forty times the arithmetic for the same
    answer, so the width is the batch's, and the filler is the checkpoint's pad id."""
    session = FakeSession(np.zeros((2, 3, 2), dtype=np.float32))
    enc = _OnnxEncoder(
        FakeTokenizer(), session, pooling="cls", dimension=2, pad_id=PAD, normalize=False
    )
    enc.encode(["a bb ccc", "dddd"])
    feed = session.feeds[0]
    assert feed["input_ids"].tolist() == [[1, 2, 3], [4, PAD, PAD]]
    assert feed["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert feed["input_ids"].dtype == np.int64


def test_an_empty_string_still_produces_a_row() -> None:
    enc = encoder(np.zeros((1, 1, 2), dtype=np.float32))
    assert enc.encode([""]) == [[0.0, 0.0]]


def test_the_session_is_fed_exactly_the_inputs_it_declares() -> None:
    """ModernBERT declares no token_type_ids; feeding them is an error, and omitting them
    from a graph that does declare them is another."""
    modern = FakeSession(np.zeros((1, 1, 2), dtype=np.float32))
    _OnnxEncoder(
        FakeTokenizer(), modern, pooling="cls", dimension=2, pad_id=PAD, normalize=False
    ).encode(["aa"])
    assert set(modern.feeds[0]) == {"input_ids", "attention_mask"}

    bert = FakeSession(
        np.zeros((1, 1, 2), dtype=np.float32),
        inputs=("input_ids", "attention_mask", "token_type_ids"),
    )
    _OnnxEncoder(
        FakeTokenizer(), bert, pooling="cls", dimension=2, pad_id=PAD, normalize=False
    ).encode(["aa"])
    assert bert.feeds[0]["token_type_ids"].tolist() == [[0]]


def test_an_input_with_no_source_is_refused_at_construction() -> None:
    session = FakeSession(np.zeros((1, 1, 2)), inputs=("input_ids", "past_key_values"))
    with pytest.raises(DependencyUnavailable, match="past_key_values"):
        _OnnxEncoder(
            FakeTokenizer(), session, pooling="cls", dimension=2, pad_id=PAD, normalize=False
        )


def test_the_fingerprint_names_the_graph_so_int8_cannot_share_a_collection() -> None:
    spec = DenseModel(runtime="onnx")
    assert onnx_fingerprint(spec) == "onnx-granite-embedding-small-english-r2-model-d384"
    int8 = spec.model_copy(update={"graph_file": "onnx/model_qint8.onnx"})
    assert onnx_fingerprint(int8) == "onnx-granite-embedding-small-english-r2-model_qint8-d384"
    assert onnx_fingerprint(int8) != onnx_fingerprint(spec)
    assert onnx_fingerprint(spec) != "st-granite-embedding-small-english-r2-torch-d384"


def test_the_fingerprint_carries_the_measured_width_not_the_declared_one() -> None:
    assert onnx_fingerprint(DenseModel(runtime="onnx"), 256).endswith("-d256")


# --- the adapter around the graph -------------------------------------------------------


def adapter(session: FakeSession, **spec_fields: Any) -> OnnxEmbedding:
    spec = DenseModel(runtime="onnx", dimension=2, **spec_fields)
    graph = _OnnxEncoder(
        FakeTokenizer(), session, pooling="cls", dimension=2, pad_id=PAD, normalize=True
    )
    return OnnxEmbedding(spec, encoder=graph)


async def test_documents_are_encoded_in_batches_of_the_frozen_size() -> None:
    session = FakeSession(lambda feed: np.ones((len(feed["input_ids"]), 1, 2), dtype=np.float32))
    emb = adapter(session, batch_size=2)
    try:
        vectors = await emb.embed_documents(["a", "b", "c", "d", "e"])
    finally:
        emb.close()
    assert len(vectors) == 5
    assert [len(feed["input_ids"]) for feed in session.feeds] == [2, 2, 1]


async def test_no_texts_never_reaches_the_session() -> None:
    session = FakeSession(np.zeros((1, 1, 2), dtype=np.float32))
    emb = adapter(session)
    try:
        assert await emb.embed_documents([]) == []
    finally:
        emb.close()
    assert session.feeds == []


async def test_two_queries_are_never_inside_the_model_at_once() -> None:
    """The point of the executor and the gate: three concurrent requests used to put three
    encodes inside a model that was itself fanning over every core."""
    occupancy = {"inside": 0, "peak": 0}
    lock = threading.Lock()

    def hidden(feed: dict[str, Any]) -> Any:
        with lock:
            occupancy["inside"] += 1
            occupancy["peak"] = max(occupancy["peak"], occupancy["inside"])
        time.sleep(0.02)
        with lock:
            occupancy["inside"] -= 1
        return np.ones((len(feed["input_ids"]), 1, 2), dtype=np.float32)

    emb = adapter(FakeSession(hidden))
    try:
        await asyncio.gather(*(emb.embed_query(f"query {i}") for i in range(5)))
    finally:
        emb.close()
    assert occupancy["peak"] == 1


async def test_the_query_vector_is_the_document_vector() -> None:
    session = FakeSession(lambda feed: np.ones((len(feed["input_ids"]), 1, 2), dtype=np.float32))
    emb = adapter(session)
    try:
        assert await emb.embed_query("aa bb") == (await emb.embed_documents(["aa bb"]))[0]
    finally:
        emb.close()


async def test_the_adapter_reports_the_graph_it_loaded() -> None:
    emb = adapter(FakeSession(np.zeros((1, 1, 2), dtype=np.float32)))
    try:
        assert emb.dimension == 2
        assert emb.threads == 2, "the frozen count, not torch's or ORT's default"
        assert emb.fingerprint().startswith("onnx-granite-embedding-small-english-r2-model-")
        assert emb.info.locality == "local"
    finally:
        emb.close()


# --- what is read off disk --------------------------------------------------------------


def write_pooling(directory: Path, **modes: Any) -> Path:
    (directory / "1_Pooling").mkdir(parents=True, exist_ok=True)
    (directory / "1_Pooling" / "config.json").write_text(json.dumps(modes), encoding="utf-8")
    return directory


def test_the_pooling_mode_comes_from_the_checkpoint(tmp_path: Path) -> None:
    """granite-embedding-small-english-r2 is CLS-pooled. Mean-pooling it would return
    plausible vectors and lose accuracy without raising anything, so it is read."""
    write_pooling(tmp_path, pooling_mode_cls_token=True, word_embedding_dimension=384)
    assert _pooling_mode(tmp_path) == ("cls", 384)
    write_pooling(tmp_path, pooling_mode_mean_tokens=True, pooling_mode_cls_token=False)
    assert _pooling_mode(tmp_path) == ("mean", None)


def test_a_pooling_mode_this_runner_does_not_implement_is_an_error(tmp_path: Path) -> None:
    write_pooling(tmp_path, pooling_mode_max_tokens=True)
    with pytest.raises(DependencyUnavailable, match="max_tokens"):
        _pooling_mode(tmp_path)
    write_pooling(tmp_path, pooling_mode_cls_token=True, pooling_mode_mean_tokens=True)
    with pytest.raises(DependencyUnavailable, match="pooling"):
        _pooling_mode(tmp_path)


def test_a_checkpoint_that_does_not_say_how_it_pools_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DependencyUnavailable, match="cannot be guessed"):
        _pooling_mode(tmp_path)


def test_the_pad_id_is_the_checkpoints_own(tmp_path: Path) -> None:
    """An id outside the vocabulary is an out-of-range gather, not harmless filler."""
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 50283}), encoding="utf-8")
    assert _pad_token_id(tmp_path, SimpleNamespace(token_to_id=lambda _t: None)) == 50283


def test_the_pad_id_falls_back_to_the_tokenizers_pad_token(tmp_path: Path) -> None:
    tokenizer = SimpleNamespace(token_to_id=lambda token: 3 if token == "[PAD]" else None)
    assert _pad_token_id(tmp_path, tokenizer) == 3


def test_a_missing_graph_names_the_command_that_writes_it(tmp_path: Path) -> None:
    spec = DenseModel(runtime="onnx", model_path=str(tmp_path))
    with pytest.raises(DependencyUnavailable, match="--export-onnx"):
        _load_graph(spec, threads=2)
