"""NLI provider contract: index-aligned scores that sum to one, deterministic, and the
frozen multilingual head (mDeBERTa, ONNX) against real weights when ``BENCH_MODELS_DIR``
or ``./models`` holds them."""

from __future__ import annotations

import pytest

from memory_service.adapters.models.nli import LexicalNLI
from memory_service.config.constants import FROZEN_MODELS, NLIModel
from memory_service.ports.models import NLIProvider

pytestmark = pytest.mark.contract

PREMISE = "Adjusted EBITDA increased to EUR 98 million from EUR 81 million, despite lower revenue."
PAIRS = [
    ("EBITDA rose to EUR 98 million.", "entailment"),
    ("Adjusted EBITDA decreased to EUR 98 million.", "contradiction"),
    ("Adjusted EBITDA was EUR 150 million.", "contradiction"),
    ("The company opened a plant in Warsaw.", "neutral"),
]


async def _contract(nli: NLIProvider) -> None:
    assert isinstance(nli, NLIProvider)
    scores = await nli.entail([PREMISE, "The weather was mild in March."], PAIRS[0][0])
    assert len(scores) == 2
    for s in scores:
        assert abs(s.entailment + s.neutral + s.contradiction - 1.0) < 1e-4
    assert scores[0].entailment > scores[1].entailment
    assert await nli.entail([], "x") == []
    again = await nli.entail([PREMISE], PAIRS[0][0])
    assert again[0] == (await nli.entail([PREMISE], PAIRS[0][0]))[0]
    assert nli.fingerprint()


async def test_lexical_nli_contract() -> None:
    nli = LexicalNLI()
    await _contract(nli)
    assert nli.representative is False
    for hypothesis, label in PAIRS[1:]:
        score = (await nli.entail([PREMISE], hypothesis))[0]
        top = max(("entailment", "neutral", "contradiction"), key=lambda k: getattr(score, k))
        assert top == label, (hypothesis, score)


@pytest.mark.models
async def test_frozen_nli_real_weights_contract() -> None:
    """Runs only when the frozen graph is present (BENCH_MODELS_DIR or ./models)."""
    from tests.support_models import requires_onnxruntime, requires_weights

    requires_onnxruntime()
    weights = requires_weights(FROZEN_MODELS.nli.local_dir) / FROZEN_MODELS.nli.local_dir
    from memory_service.adapters.models.onnx_nli import OnnxNLI

    nli = OnnxNLI(NLIModel(model_path=str(weights), batch_size=2))
    await _contract(nli)
    assert nli.representative is True
    assert nli.fingerprint().startswith("nli-onnx-")
    assert nli.info.license == "MIT" and "mDeBERTa" in nli.info.name
    for hypothesis, label in PAIRS:
        score = (await nli.entail([PREMISE], hypothesis))[0]
        top = max(("entailment", "neutral", "contradiction"), key=lambda k: getattr(score, k))
        assert top == label, (hypothesis, score)
    # a Spanish premise supports an English claim: one model, every script
    spanish = "El EBITDA ajustado aumentó a 98 millones de euros pese a la caída de ingresos."
    cross = (await nli.entail([spanish], PAIRS[0][0]))[0]
    assert cross.entailment > cross.contradiction
    # batching keeps the order
    many = await nli.entail([PREMISE] * 5 + ["Unrelated sentence."], PAIRS[0][0])
    assert [round(s.entailment, 4) for s in many[:5]] == [round(many[0].entailment, 4)] * 5
    assert many[5].entailment < many[0].entailment
