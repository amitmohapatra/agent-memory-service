"""NLI provider contract: index-aligned scores that sum to one, deterministic, and the
DeBERTa adapter against real weights when ``MEMORY_MODELS_DIR`` holds them."""

from __future__ import annotations

import pytest

from memory_service.adapters.models.nli import LexicalNLI
from memory_service.config.settings import NLISettings
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
async def test_deberta_real_weights_contract() -> None:
    """Runs only when the DeBERTa weights are present (MEMORY_MODELS_DIR)."""
    from tests.support_models import requires_torch, requires_weights

    requires_torch()
    path = requires_weights("deberta-v3-base-mnli-fever-anli") / "deberta-v3-base-mnli-fever-anli"
    from memory_service.adapters.models.nli import TransformersNLI

    nli = TransformersNLI(NLISettings(model_path=str(path), batch_size=2))
    await _contract(nli)
    assert nli.representative is True
    assert nli.fingerprint() == "nli-deberta-v3-base-mnli-fever-anli"
    for hypothesis, label in PAIRS:
        score = (await nli.entail([PREMISE], hypothesis))[0]
        top = max(("entailment", "neutral", "contradiction"), key=lambda k: getattr(score, k))
        assert top == label, (hypothesis, score)
    # batching keeps the order
    many = await nli.entail([PREMISE] * 5 + ["Unrelated sentence."], PAIRS[0][0])
    assert [round(s.entailment, 4) for s in many[:5]] == [round(many[0].entailment, 4)] * 5
    assert many[5].entailment < many[0].entailment
