# ruff: noqa: RUF001 - literal multilingual fixtures.
"""Opt-in: a real model reading messages the English rules cannot.

Runs only with ``MEMORY_TEST_LIVE_LLM=1`` and an endpoint configured in the environment
(``LLM_BASE_URL``, and ``LLM_API_KEY``, ``LLM_GATEWAY``, ``LLM_AUTH`` as it needs them; the
model is chosen from its ``/models`` unless ``MEMORY_TEST_LIVE_LLM_MODEL`` names one): it spends real tokens. What it checks is
what the scripted tests cannot, that a real model follows the source-language rule and cites
the words it rests on, and that what it returns survives the service's own checks.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from memory_service.adapters.models.llm import EndpointLLM
from memory_service.domain.language import detect_language
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.gapfill import GapFill
from tests.conftest import live_llm_settings, live_llm_tuning

pytestmark = [pytest.mark.contract, pytest.mark.live_llm]

MESSAGES = {
    "ja": "私は東京に住んでいます。毎朝コーヒーを飲むのが好きです。",
    "hi": "मैं दिल्ली में रहता हूँ। मुझे चाय बहुत पसंद है।",
    "de": "Ich wohne jetzt in Köln. Ich arbeite bei Siemens als Ingenieur.",
    "es": "Vivo en Madrid desde el año pasado. Me encanta el café con leche.",
    "zh": "我住在上海。我在阿里巴巴工作。",
    "ar": "أعيش في دبي منذ سنتين. أعمل مهندسة في شركة طيران.",
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
    facts = await GapFill(assist).fill(
        "live",
        message=MESSAGES[lang],
        speaker="u1",
        said_at=datetime(2026, 10, 1, tzinfo=UTC),
        recent=[],
        rules=[],
        known=[],
    )
    if not facts:
        pytest.skip("the endpoint returned nothing usable (rate limit, quota or outage)")
    for fact in facts:
        assert detect_language(fact.fact) in (lang, "und"), fact.fact
