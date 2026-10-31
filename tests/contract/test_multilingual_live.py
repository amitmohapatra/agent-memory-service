# ruff: noqa: RUF001 - literal multilingual fixtures.
"""Opt-in: a real model reading messages the English rules cannot.

Runs only with ``MEMORY_TEST_LIVE_LLM=1`` and an endpoint configured in the environment
(``LLM_BASE_URL``, and ``LLM_API_KEY``, ``LLM_GATEWAY``, ``LLM_AUTH`` as it needs them; the
model is chosen from its ``/models`` unless ``MEMORY_TEST_LIVE_LLM_MODEL`` names one): it spends real tokens. What it checks is
what the scripted tests cannot, that a real model follows the source-language rule and cites
sentences, and that what it returns survives the service's own checks.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from memory_service.adapters.models.llm import EndpointLLM
from memory_service.domain.language import detect_language
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.native import split_sentences
from memory_service.modules.memory.source_facts import extract_source_facts
from tests.conftest import live_llm_settings, live_llm_tuning

pytestmark = [pytest.mark.contract, pytest.mark.live_llm]

MESSAGES = {
    "ja": "私は東京に住んでいます。毎朝コーヒーを飲むのが好きです。",
    "hi": "मैं दिल्ली में रहता हूँ। मुझे चाय बहुत पसंद है।",
    "de": "Ich wohne jetzt in Köln. Ich arbeite bei Siemens als Ingenieur.",
    "es": "Vivo en Madrid desde el año pasado. Me encanta el café con leche.",
    "zh": "我住在上海。我在阿里巴巴工作。",
}


# One adapter per test, on the test's own loop. A module-scoped one shared its HTTP pool
# across the per-test event loops: every other case reused a connection bound to a closed
# loop, the assist swallowed the error as "no answer", and those cases skipped.
@pytest_asyncio.fixture(loop_scope="function")
async def assist() -> AsyncIterator[LLMAssist]:
    if os.environ.get("MEMORY_TEST_LIVE_LLM") != "1":
        pytest.skip("live LLM tests are opt-in: MEMORY_TEST_LIVE_LLM=1")
    settings = live_llm_settings()
    if settings.endpoint is None:
        pytest.skip("no endpoint configured (LLM_BASE_URL)")
    llm = EndpointLLM(settings, tuning=live_llm_tuning(timeout_seconds=90))
    try:
        yield LLMAssist(llm, settings)
    finally:
        await llm.close()


@pytest.mark.parametrize("lang", sorted(MESSAGES))
async def test_a_real_model_extracts_facts_in_the_source_language(assist, lang) -> None:
    sentences = split_sentences(MESSAGES[lang])
    facts = await extract_source_facts(assist, sentences, set(range(len(sentences))))
    if facts is None:
        pytest.skip("the endpoint did not answer (rate limit, quota or outage)")
    assert facts, f"{lang}: the model returned nothing that passed the checks"
    for fact in facts:
        assert detect_language(fact.text) in (lang, "und"), fact.text
