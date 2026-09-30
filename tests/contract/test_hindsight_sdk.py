"""SDK serialization, bounded failures and native ownership across the HTTP boundary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("hindsight_client")

from hindsight_client_api.models.dry_run_extraction_result import (  # noqa: E402
    DryRunExtractionResult,
)

from memory_service.adapters.models.hindsight import HindsightExtractor, _validated_texts
from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.config.settings import HindsightSettings
from memory_service.modules.llm.cost import llm_tokens_used, start_llm_accounting
from memory_service.modules.memory.native import NativeMemoryIntelligence
from tests.support_hindsight import preview_server
from tests.support_llm import mocked_gateway
from tests.unit.test_memory_native import CTX, _obs
from tests.unit.test_narrative_memory import MESSAGE

pytestmark = pytest.mark.contract

FACT = {
    "text": "Omar postponed the dashboard until Friday.",
    "fact_type": "world",
    "chunk_index": 0,
    "occurred_start": "2099-01-01",  # never accepted as local chronology
    "entities": ["tenant:other", "user:intruder"],
}


async def test_real_sdk_preview_keeps_local_identity_chronology_and_native_facts():
    obs = _obs(MESSAGE).model_copy(update={"occurred_at": datetime(2023, 5, 8, tzinfo=UTC)})
    start_llm_accounting()
    async with preview_server(facts=[FACT]) as server:
        with mocked_gateway(failing=True) as gateway:
            provider = NativeMemoryIntelligence(
                MemoryIntelligenceSettings(),
                assist=gateway.assist(uses=["contextual_extraction"]),
                contextual_extractor=server.extractor,
            )
            candidates = await provider.extract(obs, CTX)
        assert gateway.route.call_count == 0
        assert len(server.requests) == 1
        request = server.requests[0]
        assert request["path"].endswith("/extraction-preview/memories/dry-run-extract")
        assert request["body"]["content"] == MESSAGE
        assert request["body"]["timestamp"].startswith("2023-05-08")
        assert "agent_name" not in request["body"]
        fact = next(c for c in candidates if c.category == "contextual_fact")
        assert fact.subject == "user:u1" and fact.provider == "hindsight"
        assert fact.valid_from is None and fact.valid_to is None and fact.entities == []
        assert fact.evidence[0].source_id == obs.message_id
        assert fact.evidence[0].observed_at == obs.occurred_at
        raw = next(c for c in candidates if c.category == "verbatim_turn")
        assert (await provider.classify(fact, CTX)).visibility == (
            await provider.classify(raw, CTX)
        ).visibility
        assert any(c.category == "event" for c in candidates)
    assert llm_tokens_used() == 42


@pytest.mark.parametrize("status", [400, 429, 500, 503])
async def test_preview_failure_has_one_attempt_and_no_secondary_model_calls(status):
    async with preview_server(facts=[], status=status) as server:
        with mocked_gateway(failing=True) as gateway:
            provider = NativeMemoryIntelligence(
                MemoryIntelligenceSettings(),
                assist=gateway.assist(uses=["contextual_extraction"]),
                contextual_extractor=server.extractor,
            )
            candidates = await provider.extract(_obs(MESSAGE), CTX)
        assert len(server.requests) == 1 and gateway.route.call_count == 0
        assert any(c.category == "verbatim_turn" for c in candidates)
        assert not any(c.category == "contextual_fact" for c in candidates)


@pytest.mark.parametrize("case", ["disabled", "simple", "agent", "oversized", "skip"])
async def test_ineligible_messages_never_contact_preview(case):
    obs = _obs(MESSAGE)
    if case == "simple":
        obs = _obs("My timezone is Europe/Berlin.")
    elif case == "agent":
        obs = obs.model_copy(update={"custom_metadata": {"role": "assistant"}})
    elif case == "oversized":
        obs = _obs(MESSAGE * 100)
    elif case == "skip":
        obs = obs.model_copy(
            update={"hints": obs.hints.model_copy(update={"skip_extraction": True})}
        )
    async with preview_server(facts=[FACT]) as server:
        with mocked_gateway(failing=True) as gateway:
            provider = NativeMemoryIntelligence(
                MemoryIntelligenceSettings(),
                assist=gateway.assist(uses=[] if case == "disabled" else ["contextual_extraction"]),
                contextual_extractor=server.extractor,
            )
            await provider.extract(obs, CTX)
        assert server.requests == [] and gateway.route.call_count == 0


async def test_timeout_includes_queue_and_cancellation_propagates():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pending(**kwargs):
        entered.set()
        await release.wait()

    call = AsyncMock(side_effect=pending)
    client = SimpleNamespace(memory=SimpleNamespace(dry_run_extract_memories=call))
    adapter = HindsightExtractor(HindsightSettings(timeout_seconds=0.05), client=client)
    task = asyncio.create_task(adapter.extract(MESSAGE, timestamp=datetime.now(UTC)))
    await entered.wait()
    assert await adapter.extract(MESSAGE, timestamp=datetime.now(UTC)) == []
    assert await task == []
    # Only the first request reaches the server before both deadlines expire.
    assert call.await_count <= 2
    task = asyncio.create_task(adapter.extract(MESSAGE, timestamp=datetime.now(UTC)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "facts,chunks",
    [
        ([{**FACT, "chunk_index": -1}], [{"text": MESSAGE, "fact_count": 1}]),
        ([{**FACT, "chunk_index": None}], [{"text": MESSAGE, "fact_count": 1}]),
        ([FACT], [{"text": "another tenant's source", "fact_count": 1}]),
        ([FACT] * 7, [{"text": MESSAGE, "fact_count": 7}]),
        ([{**FACT, "text": "x" * 2001}], [{"text": MESSAGE, "fact_count": 1}]),
    ],
)
def test_unassociated_or_unbounded_output_is_discarded(facts, chunks):
    result = DryRunExtractionResult.model_validate({"facts": facts, "chunks": chunks})
    assert _validated_texts(result, MESSAGE) == []
