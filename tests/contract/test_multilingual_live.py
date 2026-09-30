# ruff: noqa: RUF001 - literal multilingual fixtures.
"""Opt-in: the real gateway reading messages the English rules cannot.

Runs only with ``MEMORY_TEST_LIVE_LLM=1`` and the gateway configured in the environment
(``MEMORY__MODELS__LLM__ENABLED=true`` plus a base URL, fast model and virtual key - the
local ``.env``): it spends real tokens. What it checks is what the scripted tests cannot,
that a real model follows the source-language rule and cites sentences, and that what it
returns survives the service's own checks.
"""

from __future__ import annotations

import os

import pytest

from memory_service.adapters.models.llm import BifrostLLM
from memory_service.config.settings import Settings
from memory_service.domain.language import detect_language
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.native import split_sentences
from memory_service.modules.memory.source_facts import extract_source_facts

pytestmark = [pytest.mark.contract, pytest.mark.bifrost]

MESSAGES = {
    "ja": "私は東京に住んでいます。毎朝コーヒーを飲むのが好きです。",
    "hi": "मैं दिल्ली में रहता हूँ। मुझे चाय बहुत पसंद है।",
    "de": "Ich wohne jetzt in Köln. Ich arbeite bei Siemens als Ingenieur.",
    "es": "Vivo en Madrid desde el año pasado. Me encanta el café con leche.",
    "zh": "我住在上海。我在阿里巴巴工作。",
}


@pytest.fixture(scope="module")
def assist() -> LLMAssist:
    if os.environ.get("MEMORY_TEST_LIVE_LLM") != "1":
        pytest.skip("live LLM tests are opt-in: MEMORY_TEST_LIVE_LLM=1")
    settings = Settings().models.llm
    if settings.enabled is not True or not settings.api_key:
        pytest.skip("gateway not configured (MEMORY__MODELS__LLM__*)")
    settings = settings.model_copy(
        update={"uses": ["contextual_extraction"], "timeout_seconds": 90}
    )
    return LLMAssist(BifrostLLM(settings), settings)


@pytest.mark.parametrize("lang", sorted(MESSAGES))
async def test_a_real_model_extracts_facts_in_the_source_language(assist, lang) -> None:
    sentences = split_sentences(MESSAGES[lang])
    facts = await extract_source_facts(assist, sentences, set(range(len(sentences))))
    if facts is None:
        pytest.skip("the gateway did not answer (rate limit, quota or outage)")
    assert facts, f"{lang}: the model returned nothing that passed the checks"
    for fact in facts:
        assert detect_language(fact.text) in (lang, "und"), fact.text
