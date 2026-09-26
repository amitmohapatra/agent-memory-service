"""Contextual extraction through durable ingestion, indexing, recall and forgetting."""

import pytest

from memory_service.domain.enums import MemoryType, Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.context.builder import candidate_to_item
from tests.integration.test_memory import U1, U2, _memories, _observe
from tests.support_llm import mocked_gateway
from tests.unit.test_narrative_memory import MESSAGE, UNITS

pytestmark = pytest.mark.integration


async def test_narrative_units_are_indexed_scoped_replay_safe_and_forgettable(
    container, uow_factory
):
    with mocked_gateway([UNITS]) as gateway:
        container.services["memory_provider"].assist = gateway.assist(
            uses=["contextual_extraction"]
        )
        ack = await _observe(
            container,
            uow_factory,
            U1,
            MESSAGE,
            hints=ProcessingHints(visibility=Visibility.USER),
        )
        memories = await _memories(uow_factory, U1, container)
        units = [m for m in memories if m.system_metadata.get("category") == "narrative_unit"]
        assert len(units) == 2
        dashboard = next(m for m in units if m.content.startswith("Omar"))
        assert any(m.system_metadata.get("category") == "verbatim_turn" for m in memories)
        assert all(m.memory_type is MemoryType.OBSERVATION for m in units)
        pipeline = container.services["observation_pipeline"]
        assert await pipeline.run({"tenant_id": "acme", "observation_id": ack.observation_id}) == []
        assert len(await _memories(uow_factory, U1, container)) == len(memories)
        engine = container.services["retrieval"]
        result = await engine.retrieve(U1, "Who postponed the dashboard?", kinds=("memory",))
        selected = next(c for c in result.candidates if c.record_id == dashboard.memory_id)
        assert candidate_to_item(selected).evidence == dashboard.evidence
        assert dashboard.evidence[0].source_id == ack.observation_id
        assert (await engine.retrieve(U2, f"show {dashboard.memory_id}")).candidates == []
        outsider = U1.model_copy(update={"tenant_id": "other"})
        assert (await engine.retrieve(outsider, f"show {dashboard.memory_id}")).candidates == []
        async with uow_factory() as uow:
            await container.services["memory"].forget(uow, U1, dashboard.memory_id)
            await uow.commit()
        await container.tasks.drain()
        after = await engine.retrieve(U1, f"show {dashboard.memory_id}")
        assert dashboard.memory_id not in {c.record_id for c in after.candidates}
        assert dashboard.memory_id not in {
            m.memory_id for m in await _memories(uow_factory, U1, container)
        }
    assert gateway.route.call_count == 1, (
        "replay and retrieval must not invoke the extraction model"
    )
