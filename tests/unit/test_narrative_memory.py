"""Contextual ingestion uses bounded source spans across non-benchmark domains."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.enums import MemoryType
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.narrative import extract_narrative_units
from memory_service.modules.memory.native import NativeMemoryIntelligence
from tests.support_llm import mocked_gateway
from tests.unit.test_memory_native import CTX, _obs

pytestmark = pytest.mark.unit

MESSAGE = (
    "Maya restored the payment queue after the outage. She did not restart the database. "
    "Omar shipped the invoice exporter on Tuesday. He postponed the dashboard until Friday."
)
UNITS = {"units": [{"start": 0, "end": 1}, {"start": 2, "end": 3}]}


async def test_multiple_topics_keep_antecedents_negation_and_source_identity():
    observation = _obs(MESSAGE)
    with mocked_gateway([UNITS]) as gateway:
        native = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        candidates = await native.extract(observation, CTX)
    units = [c for c in candidates if c.category == "narrative_unit"]
    assert len(units) == 2
    assert units[0].content == MESSAGE.split(" Omar", maxsplit=1)[0]
    assert units[1].content == "Omar" + MESSAGE.split(" Omar")[1]
    for unit in units:
        assert unit.subject == "user:u1" and unit.memory_type is MemoryType.OBSERVATION
        assert unit.evidence[0].source_id == observation.message_id
        assert unit.evidence[0].observed_at == observation.occurred_at
        assert unit.valid_from is None and unit.valid_to is None
        classified = await native.classify(unit, CTX)
        verbatim = next(c for c in candidates if c.category == "verbatim_turn")
        assert classified.visibility == (await native.classify(verbatim, CTX)).visibility
    assert gateway.route.call_count == 1
    assert gateway.prompts()[0]["model"] == "test/fast"
    messages = gateway.prompts()[0]["messages"]
    assert "ZERO-BASED" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["sentences"][0] == {
        "index": 0,
        "text": "Maya restored the payment queue after the outage.",
    }
    assert [s["index"] for s in payload["sentences"]] == list(range(4))
    assert any(c.category == "event" for c in candidates), "retain the native facts"


async def test_live_detached_pronoun_regression_preserves_raw_turn_and_other_topic():
    # GPT-4o-mini returned this valid JSON but semantically incomplete source span.
    output = {"units": [{"start": 1, "end": 1}, {"start": 2, "end": 3}]}
    observation = _obs(MESSAGE)
    with mocked_gateway([output]) as gateway:
        native = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        candidates = await native.extract(observation, CTX)
    units = [c.content for c in candidates if c.category == "narrative_unit"]
    assert units == [
        "Omar shipped the invoice exporter on Tuesday. He postponed the dashboard until Friday."
    ]
    assert any(c.content == MESSAGE and c.category == "verbatim_turn" for c in candidates)
    assert gateway.route.call_count == 1


@pytest.mark.parametrize("opening", ["She", "they", "HER", "“It", "(Those", "This"])
async def test_dependent_openings_are_not_stored_as_standalone_units(opening):
    assist = LLMAssist.disabled()
    assist.wants = lambda use: True
    assist.structured = AsyncMock(return_value={"units": [{"start": 1, "end": 1}]})
    assert (
        await extract_narrative_units(
            assist, ["A named participant introduced the project.", opening + " continued."], {0, 1}
        )
        == []
    )


@pytest.mark.parametrize("opening", ["I", "We", "The team", "Theodore", "Isabel"])
async def test_speaker_and_named_openings_are_not_pronoun_false_positives(opening):
    assist = LLMAssist.disabled()
    assist.wants = lambda use: True
    assist.structured = AsyncMock(return_value={"units": [{"start": 1, "end": 1}]})
    text = opening + " completed the survey."
    assert await extract_narrative_units(assist, ["A separate topic.", text], {0, 1}) == [text]


@pytest.mark.parametrize(
    "text",
    [
        "My timezone is CET.",
        "My timezone is CET. I prefer short answers.",
        "How do I reset it? Which server should I choose?",
        "Thanks! Good morning!",
        "Omar shipped the invoice exporter on Tuesday.",
    ],
)
async def test_simple_known_or_noise_messages_do_not_consult_model(text):
    with mocked_gateway([UNITS]) as gateway:
        assisted = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
        observation = _obs(text)
        assert await assisted.extract(observation, CTX) == await native.extract(observation, CTX)
    assert gateway.route.call_count == 0


@pytest.mark.parametrize("role", ["assistant", "agent", "system", "tool"])
async def test_agent_working_notes_never_use_narrative_enrichment(role):
    with mocked_gateway([UNITS]) as gateway:
        native = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        observation = _obs(MESSAGE).model_copy(update={"custom_metadata": {"role": role}})
        assert not any(
            c.category == "narrative_unit" for c in await native.extract(observation, CTX)
        )
    assert gateway.route.call_count == 0


@pytest.mark.parametrize("failed", [False, True])
async def test_failed_or_empty_contextual_attempt_does_not_fan_out(failed):
    with mocked_gateway([{"units": []}], failing=failed) as gateway:
        assisted = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(),
            assist=gateway.assist(
                uses=["contextual_extraction", "ambiguous_extraction", "ambiguous_worthiness"]
            ),
        )
        observation = _obs(MESSAGE)
        native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
        assert await assisted.extract(observation, CTX) == await native.extract(observation, CTX)
    assert gateway.route.call_count == 1


@pytest.mark.parametrize(
    "unit",
    [
        {"start": -1, "end": 1},
        {"start": 2, "end": 1},
        {"start": 0, "end": 4},
        {"start": True, "end": 1},
        {"start": "0", "end": 1},
        {"start": 0, "end": 1, "content": "Invented detail"},
        {"start": 0, "end": 1, "visibility": "TENANT"},
        {"start": 0, "end": 1, "subject": "user:another"},
        {"start": 0},
        None,
    ],
)
async def test_untrusted_output_cannot_introduce_text_scope_or_invalid_offsets(unit):
    assist = LLMAssist.disabled()
    assist.wants = lambda use: True
    assist.structured = AsyncMock(return_value={"units": [unit]})
    assert await extract_narrative_units(assist, ["A.", "B."], {0, 1}) == []


async def test_unit_limits_deduplication_and_eligible_coverage():
    assist = LLMAssist.disabled()
    assist.wants = lambda use: True
    sentences = ["A.", "B.", "C.", "D.", "E."]
    assist.structured = AsyncMock(
        return_value={
            "units": [
                {"start": 0, "end": 4},  # excessive span
                {"start": 4, "end": 4},  # no uncovered sentence
                {"start": 0, "end": 1},
                {"start": 0, "end": 1},
            ]
        }
    )
    assert await extract_narrative_units(assist, sentences, {0, 1}) == ["A. B."]
    assist.structured = AsyncMock(return_value={"units": [{"start": 0, "end": 1}] * 7})
    assert await extract_narrative_units(assist, sentences, {0, 1}) == []
    assist.structured = AsyncMock(return_value=UNITS)
    assert await extract_narrative_units(assist, ["a" * 1200, "b" * 1200], {0, 1}) == []


@pytest.mark.parametrize("sentences", [["text"] * 17, ["a" * 4000, "b" * 4000]])
async def test_oversized_input_is_not_truncated_or_sent(sentences):
    assist = LLMAssist.disabled()
    assist.wants = lambda use: True
    assist.structured = AsyncMock()
    assert await extract_narrative_units(assist, sentences, set(range(len(sentences)))) is None
    assist.structured.assert_not_awaited()


async def test_cancellation_is_not_swallowed():
    assist = LLMAssist.disabled()
    assist.wants = lambda use: True
    assist.structured = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await extract_narrative_units(assist, ["A.", "B."], {0, 1})


@pytest.mark.parametrize(
    "text",
    [
        "Lina catalogued the ceramics. She kept the broken pieces for conservation.",
        "Arjun planted the seedlings near the fence. He covered them before the frost.",
        "Zoë repaired the telescope mount. She postponed calibration until Monday.",
    ],
)
async def test_domains_and_names_are_not_benchmark_specific(text):
    with mocked_gateway([{"units": [{"start": 0, "end": 1}]}]) as gateway:
        native = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        candidates = await native.extract(_obs(text), CTX)
    assert gateway.route.call_count == 1
    # A selection identical to the full turn adds no representation; keep the raw record.
    assert [c.content for c in candidates if c.category == "verbatim_turn"] == [text]
    assert not any(c.category == "narrative_unit" for c in candidates)
    assert len({c.content for c in candidates}) == len(candidates)
