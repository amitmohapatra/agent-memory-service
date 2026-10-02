"""The tensor path of the ColBERT encoder, against a fake session and a word-level fake
tokenizer (as ``test_onnx_encoder``: the suite runs without the ``models`` extra): what is
fed, what is kept, what comes back. The real graph was compared with PyLate (ADR 0025)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from memory_service.adapters.models.late_interaction import ColbertGraph, HashLateInteraction
from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.unit

VOCAB = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[Q]": 3, "[D]": 4, ".": 5, "hello": 6, "world": 7}
CONFIG = {
    "do_lower_case": True,
    "query_length": 6,
    "document_length": 6,
    "query_prefix_id": 3,
    "document_prefix_id": 4,
    "pad_token_id": 0,
    "embedding_dim": 4,
    "skiplist_words": [".", "!"],
    "do_query_expansion": False,
}


class _Tokenizer:
    """Whitespace words to ids, wrapped in ``[CLS] ... [SEP]`` as a BERT tokenizer does."""

    def token_to_id(self, word: str) -> int | None:
        return VOCAB.get(word)

    def encode_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(ids=[1, *(VOCAB.get(w, 0) for w in text.split()), 2]) for text in texts
        ]


def _tokenizer() -> _Tokenizer:
    return _Tokenizer()


class _Session:
    """Each token's vector is ``(id + 1, 1, 0, ...)``, so the kept ids can be read back from
    the unit vectors as ``v[0] / v[1] - 1``."""

    def __init__(self, inputs: tuple[str, ...], width: int = 4) -> None:
        self._inputs = inputs
        self.width = width
        self.fed: list[dict] = []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self._inputs]

    def run(self, _outputs, feed):
        self.fed.append(feed)
        ids = feed["input_ids"].astype(np.float32)
        out = np.zeros((*ids.shape, self.width), dtype=np.float32)
        out[:, :, 0] = ids + 1
        out[:, :, 1] = 1
        return [out]


def test_a_query_gets_its_marker_after_cls_and_keeps_every_token() -> None:
    session = _Session(("input_ids", "attention_mask"))
    graph = ColbertGraph(_tokenizer(), session, CONFIG)
    [vectors] = graph.encode(["Hello ."], query=True)
    assert session.fed[0]["input_ids"].tolist() == [[1, 3, 6, 5, 2]]
    assert len(vectors) == 5  # punctuation kept in a query
    assert np.allclose(np.linalg.norm(np.asarray(vectors), axis=1), 1.0)


def test_a_document_drops_punctuation_and_is_cut_with_sep_last() -> None:
    session = _Session(("input_ids", "attention_mask"))
    graph = ColbertGraph(_tokenizer(), session, CONFIG)
    [vectors] = graph.encode(["hello . world hello world hello"], query=False)
    fed = session.fed[0]["input_ids"].tolist()[0]
    assert fed == [1, 4, 6, 5, 7, 2]  # length 6: [CLS] [D] four tokens with [SEP] last
    kept = [round(v[0] / v[1]) - 1 for v in vectors]
    assert kept == [1, 4, 6, 7, 2]  # "." skipped


def test_a_batch_is_padded_and_masked_per_text() -> None:
    session = _Session(("input_ids", "attention_mask"))
    graph = ColbertGraph(_tokenizer(), session, CONFIG)
    out = graph.encode(["hello", "hello world"], query=False)
    assert session.fed[0]["attention_mask"].tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert [len(v) for v in out] == [4, 5]


def test_a_graph_the_runner_cannot_feed_or_expand_is_refused() -> None:
    with pytest.raises(DependencyUnavailable):
        ColbertGraph(_tokenizer(), _Session(("input_ids", "token_type_ids")), CONFIG)
    with pytest.raises(DependencyUnavailable):
        ColbertGraph(_tokenizer(), _Session(("input_ids",)), {**CONFIG, "do_query_expansion": True})
    graph = ColbertGraph(_tokenizer(), _Session(("input_ids", "attention_mask"), width=3), CONFIG)
    with pytest.raises(ValueError, match="unexpected shape"):
        graph.encode(["hello"], query=True)


async def test_the_hash_stand_in_is_deterministic_and_unit_length() -> None:
    late = HashLateInteraction(dimension=8)
    first, again = (
        await late.embed_documents(["Hello world"]),
        await late.embed_query("hello world"),
    )
    assert first[0] == again and len(again) == 2
    assert np.allclose(np.linalg.norm(np.asarray(again), axis=1), 1.0)
    assert late.fingerprint() == "hash-li-v1-d8"
