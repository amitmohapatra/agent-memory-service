"""SDK-enriched ingestion retains canonical isolation, replay and deletion semantics."""

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from tests.integration.test_memory import U1, U2, _memories, _observe
from tests.support_hindsight import preview_server
from tests.support_llm import mocked_gateway
from tests.unit.test_narrative_memory import MESSAGE

pytestmark = pytest.mark.integration


async def test_sdk_ingest_recall_replay_forget_and_cross_user_isolation(container, uow_factory):
    fact = "Omar postponed the dashboard until Friday."
    async with preview_server(
        facts=[{"text": fact, "fact_type": "world", "chunk_index": 0}]
    ) as server:
        with mocked_gateway(failing=True) as gateway:
            provider = container.services["memory_provider"]
            provider.assist = gateway.assist(uses=["contextual_extraction"])
            provider.contextual_extractor = server.extractor
            ack = await _observe(
                container,
                uow_factory,
                U1,
                MESSAGE,
                hints=ProcessingHints(visibility=Visibility.USER),
            )
            memories = await _memories(uow_factory, U1, container)
            generated = next(
                m for m in memories if m.system_metadata.get("provider") == "hindsight"
            )
            assert generated.content == fact
            assert generated.evidence[0].source_id == ack.observation_id
            assert generated.system_metadata["category"] == "contextual_fact"
            pipeline = container.services["observation_pipeline"]
            assert (
                await pipeline.run({"tenant_id": "acme", "observation_id": ack.observation_id})
                == []
            )
            engine = container.services["retrieval"]
            result = await engine.retrieve(U1, "Who postponed the dashboard?", kinds=("memory",))
            assert generated.memory_id in {c.record_id for c in result.candidates}
            from memory_service.modules.context.builder import candidate_to_item
            from tests.unit.test_bundle_rendering import _bundle

            candidate = next(c for c in result.candidates if c.record_id == generated.memory_id)
            item = candidate_to_item(candidate)
            assert item.attributes["category"] == "contextual_fact"
            assert item.attributes["provider"] == "hindsight"
            from memory_service.api.routers.v1.retrieval import RecallItem

            assert RecallItem.model_validate(item.model_dump()).attributes == item.attributes
            assert "model-extracted, unverified" in _bundle([item]).render()
            graph = container.services["graph"]
            from unittest.mock import AsyncMock, patch

            with patch.object(graph.provider, "enrich_memory", new_callable=AsyncMock) as enrich:
                assert await graph.enrich_memories("acme", [generated.memory_id]) == 0
                enrich.assert_not_awaited()
            async with uow_factory() as uow:
                pending = await uow.memories.reflection_pending(tenant_id="acme")
            assert generated.memory_id not in {m.memory_id for m in pending}
            for outsider in (U2, U1.model_copy(update={"tenant_id": "other"})):
                assert (
                    await engine.retrieve(outsider, f"show {generated.memory_id}")
                ).candidates == []
            async with uow_factory() as uow:
                await container.services["memory"].forget(uow, U1, generated.memory_id)
                await uow.commit()
            await container.tasks.drain()
            result = await engine.retrieve(U1, "Who postponed the dashboard?", kinds=("memory",))
            assert generated.memory_id not in {c.record_id for c in result.candidates}
        assert gateway.route.call_count == 0
        assert len(server.requests) == 1, "replay, recall and forgetting cannot call the model"
