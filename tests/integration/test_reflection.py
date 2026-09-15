"""Reflection end to end against PostgreSQL: recent memories -> model insights -> a stored
memory with evidence pointing at its sources, indexed through ``memory.index``."""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, ObservationKind, Visibility
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory.reflection import ReflectionService
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


async def _observe(container, uow_factory, ctx, content):
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["memory"].submit_observation(
            uow, ctx, kind=ObservationKind.MESSAGE, content=content
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _memories(container, uow_factory, ctx):
    async with uow_factory() as uow:
        return await container.services["memory"].list_memories(uow, ctx)


async def test_reflection_stores_indexed_insight_with_source_evidence(
    container, uow_factory
) -> None:
    await _observe(container, uow_factory, U1, "I prefer concise answers with code samples.")
    await _observe(container, uow_factory, U1, "I prefer bullet points over long paragraphs.")
    sources = await _memories(container, uow_factory, U1)
    assert len(sources) == 2
    ids = sorted(m.memory_id for m in sources)
    assert "memory.reflect" not in container.tasks.handlers  # flag off in test settings
    reply = {
        "insights": [
            {
                "content": "User prefers terse, skimmable answers",
                "memory_type": "PREFERENCE",
                "source_memory_ids": ids,
            }
        ]
    }
    with mocked_gateway([reply]) as gw:
        service = ReflectionService(uow_factory, assist=gw.assist(uses=["reflection"]))
        created = await service.reflect_all()
        assert gw.route.call_count == 1 and len(created) == 1
        await container.tasks.drain()  # memory.index
        assert await service.reflect_all() == []  # nothing newer than the insight
        assert gw.route.call_count == 1
        assert await service.reflect("acme", "user:u1", sources) == []  # identical content
        assert gw.route.call_count == 2
    async with uow_factory() as uow:
        insight = await uow.memories.get("acme", created[0])
        keys = await uow.memories.visibility_keys("acme", created[0])
    assert insight is not None
    assert insight.content == "User prefers terse, skimmable answers"
    assert insight.memory_type is MemoryType.PREFERENCE and insight.confidence == 0.6
    assert insight.visibility is Visibility.PRIVATE and insight.owner_principal == "user:u1"
    assert keys == ["principal:acme/user:u1"]
    assert sorted(e.source_id for e in insight.evidence) == ids
    assert insight.system_metadata["category"] == "reflection"
    assert insight.system_metadata["source_memory_ids"] == ids
    mine = await _memories(container, uow_factory, U1)
    assert {m.memory_id for m in mine} == {*ids, created[0]}
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "terse skimmable answers", kinds=("memory",))
    assert any(c.record_id == created[0] for c in res.candidates)


async def test_reflection_without_model_answer_changes_nothing(container, uow_factory) -> None:
    await _observe(container, uow_factory, U1, "I prefer concise answers with code samples.")
    before = await _memories(container, uow_factory, U1)
    assert before
    with mocked_gateway(failing=True) as gw:
        service = ReflectionService(uow_factory, assist=gw.assist(uses=["reflection"]))
        assert await service.reflect_all() == []
    assert gw.route.call_count >= 1
    with mocked_gateway(['{"insights": [{"content": "x", "memory_type": "SEMANTIC"}]}']) as gw:
        service = ReflectionService(uow_factory, assist=gw.assist(uses=["summaries"]))
        assert await service.reflect_all() == []
    assert gw.route.call_count == 0
    assert await ReflectionService(uow_factory).reflect_all() == []
    after = await _memories(container, uow_factory, U1)
    assert [m.memory_id for m in after] == [m.memory_id for m in before]
