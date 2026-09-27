"""NLI label order, pair alignment, bounds and local-only loading."""

from types import SimpleNamespace

import numpy as np
import pytest

from memory_service.adapters.models._runner import SerialRunner
from memory_service.adapters.models.onnx_nli import OnnxNLI, OnnxNLIScorer
from memory_service.config.constants import NLIModel
from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.unit


class Tokenizer:
    def encode_batch(self, pairs):
        self.pairs = pairs
        return [
            SimpleNamespace(ids=[1, 4, 2], attention_mask=[1, 1, 1], type_ids=[0, 0, 1]),
            SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 1]),
        ][: len(pairs)]


class Session:
    def __init__(self, logits=None, inputs=("input_ids", "attention_mask", "token_type_ids")):
        self.logits = np.asarray(logits if logits is not None else [[-1000, 1000, 0], [0, 0, 9]])
        self.inputs = inputs

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.inputs]

    def run(self, outputs, feed):
        self.feed = feed
        return [self.logits]


CONFIG = {
    "id2label": {"0": "neutral", "1": "contradiction", "2": "entailment"},
    "pad_token_id": 7,
}


def test_nli_uses_model_labels_real_pair_segments_and_dynamic_padding():
    tokenizer, session = Tokenizer(), Session()
    scorer = OnnxNLIScorer(tokenizer, session, CONFIG)
    pairs = [("a", "b"), ("c", "d")]
    scores = scorer.score(pairs)
    assert tokenizer.pairs == pairs
    assert scores[0].contradiction == 1
    assert scores[1].entailment > 0.99
    assert session.feed["input_ids"].tolist() == [[1, 4, 2], [1, 2, 7]]
    assert session.feed["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert session.feed["token_type_ids"].tolist() == [[0, 0, 1], [0, 1, 0]]
    assert all(value.dtype == np.int64 for value in session.feed.values())


@pytest.mark.parametrize("logits", [[[1, 2]], [[float("nan"), 0, 1]]])
def test_invalid_nli_outputs_fail_instead_of_becoming_verdicts(logits):
    scorer = OnnxNLIScorer(Tokenizer(), Session(logits=logits), CONFIG)
    with pytest.raises(ValueError, match="invalid logits"):
        scorer.score([("premise", "hypothesis")])


def test_empty_pairs_never_enter_session():
    session = Session()
    assert OnnxNLIScorer(Tokenizer(), session, CONFIG).score([]) == []
    assert not hasattr(session, "feed")


def test_unknown_inputs_and_labels_fail_at_load_time():
    with pytest.raises(ValueError, match="Unsupported"):
        OnnxNLIScorer(Tokenizer(), Session(inputs=("position_ids",)), CONFIG)
    with pytest.raises(KeyError):
        OnnxNLIScorer(Tokenizer(), Session(), {"id2label": {"0": "LABEL_0"}})


def test_missing_graph_never_falls_back_to_network_or_lexical(tmp_path):
    with pytest.raises(DependencyUnavailable, match="local graph"):
        OnnxNLI(NLIModel(model_path=str(tmp_path), runtime="onnx"))


async def test_trained_runtime_preserves_group_boundaries_and_batch_cap():
    from memory_service.ports.models import NLIScore

    class CountingScorer:
        def __init__(self):
            self.batches = []

        def score(self, pairs):
            self.batches.append(pairs)
            return [
                NLIScore(entailment=int(p == h), neutral=int(p != h), contradiction=0)
                for p, h in pairs
            ]

    model = OnnxNLI.__new__(OnnxNLI)
    model.spec = NLIModel(runtime="onnx", batch_size=2)
    model._scorer = CountingScorer()
    model._runner = SerialRunner("test-nli")
    try:
        result = await model.entail_groups([(["a"], "a"), ([], "x"), (["c", "b"], "b")])
        assert [len(scores) for scores in result] == [1, 0, 2]
        assert [score.entailment for scores in result for score in scores] == [1, 0, 1]
        assert [len(batch) for batch in model._scorer.batches] == [2, 1]
        model._scorer.batches.clear()
        assert await model.entail([], "x") == []
        assert await model.entail_groups([]) == []
        assert model._scorer.batches == []
    finally:
        model.close()
