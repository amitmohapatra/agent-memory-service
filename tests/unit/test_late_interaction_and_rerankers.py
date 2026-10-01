"""The tensor paths of the ColBERT encoder and the cross-encoders, against fake sessions and
a small real tokenizer: what is fed, what is kept, what comes back. The real graphs are
compared with PyLate and with the offline reranker scores in the model suite."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from memory_service.adapters.models.late_interaction import ColbertGraph, HashLateInteraction
from memory_service.adapters.models.reranker import LexicalReranker, RerankerGraph
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


def _tokenizer() -> Tokenizer:
    tok = Tokenizer(WordLevel(VOCAB, unk_token="[PAD]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return tok


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


class _Logits:
    def __init__(self, inputs: tuple[str, ...], value: float | None = None) -> None:
        self._inputs = inputs
        self.value = value
        self.fed: list[dict] = []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self._inputs]

    def run(self, _outputs, feed):
        self.fed.append(feed)
        if self.value is not None:
            return [np.full((feed["input_ids"].shape[0], 1), self.value, dtype=np.float32)]
        # the logit is the number of real tokens in the pair
        return [feed["attention_mask"].sum(axis=1, keepdims=True).astype(np.float32)]


def test_a_reranker_feeds_the_pairs_it_declares_and_reads_the_first_logit() -> None:
    session = _Logits(("input_ids", "attention_mask", "token_type_ids"))
    graph = RerankerGraph(_tokenizer(), session, pad_id=0)
    scores = graph.score("hello", ["world", "hello world ."])
    assert scores == [5.0, 7.0]
    fed = session.fed[0]
    assert set(fed) == {"input_ids", "attention_mask", "token_type_ids"}
    assert fed["token_type_ids"].tolist()[0] == [0, 0, 0, 1, 1, 0, 0]
    assert graph.score("hello", []) == []


def test_a_reranker_refuses_unknown_inputs_and_non_finite_scores() -> None:
    with pytest.raises(DependencyUnavailable):
        RerankerGraph(_tokenizer(), _Logits(("input_ids", "pixel_values")), pad_id=0)
    graph = RerankerGraph(_tokenizer(), _Logits(("input_ids",), value=float("nan")), pad_id=0)
    with pytest.raises(ValueError, match="non-finite"):
        graph.score("hello", ["world"])


async def test_the_lexical_stand_in_scores_word_overlap() -> None:
    scores = await LexicalReranker().score("When did we go camping?", ["camping trip", "none"])
    assert scores[0] > scores[1] == 0.0
