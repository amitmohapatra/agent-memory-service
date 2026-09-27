"""NLI score aggregation cannot credit missing classes or mix same/cross-language arms."""

import pytest
from benchmark.multilingual_nli import summarize

pytestmark = pytest.mark.unit


def test_language_mode_denominators_and_confusion_keep_failures():
    def row(mode, language, actual, predicted):
        return {
            "mode": mode,
            "language": language,
            "label": actual,
            "predicted": predicted,
            "latency_ms": 10.0,
        }

    summary = summarize(
        [
            row("same", "en", "entailment", "entailment"),
            row("same", "en", "contradiction", "neutral"),
            row("cross", "zh", "neutral", "entailment"),
        ]
    )
    assert summary["same/en"]["accuracy"] == 0.5
    assert summary["same/en"]["pairs"] == 2
    assert summary["cross/zh"]["accuracy"] == 0
    assert summary["same/en"]["confusion"]["contradiction"]["neutral"] == 1
    assert summary["cross/zh"]["confusion"]["neutral"]["entailment"] == 1
    assert summarize([]) == {}
