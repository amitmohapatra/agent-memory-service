"""Reflection end to end against PostgreSQL: recent memories -> model insights -> a stored
memory with evidence pointing at its sources, indexed through ``memory.index``."""

from __future__ import annotations

import pytest
from benchmark.common import submit_observation

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, ObservationKind
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory.consolidation import about_a_person
from memory_service.modules.memory.reflection import ReflectionService
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

#: an agent writing down facts about a project: no person's (a person's facts are
#: consolidated into beliefs, never also reflected on)
U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", agent_id="scout")
SCOUT = "agent:u1/scout"
ATLAS = ("The Atlas project uses Postgres 16.", "The Atlas project runs on Kubernetes.")


async def _observe(container, uow_factory, ctx, content, *, hints=None):
    register_handlers(container)
    async with uow_factory() as uow:
        await submit_observation(
            uow, ctx, kind=ObservationKind.MESSAGE, content=content, hints=hints
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _memories(container, uow_factory, ctx):
    async with uow_factory() as uow:
        return await container.services["memory"].list_memories(uow, ctx)


async def atlas_facts(container, uow_factory, ctx=U1, *, hints=None):
    """The agent's two facts about the Atlas project, each in its own observation."""
    for fact in ATLAS:
        await _observe(container, uow_factory, ctx, fact, hints=hints)
    return [
        m
        for m in await _memories(container, uow_factory, ctx)
        if m.content in ATLAS
        and not m.system_metadata.get("source_revisions")
        and not about_a_person(m)  # the turn itself is the user's: consolidation's
    ]


async def test_reflection_stores_indexed_insight_with_source_evidence(
    container, uow_factory
) -> None:
    sources = await atlas_facts(container, uow_factory)
    assert len(sources) == 2
    ids = sorted(m.memory_id for m in sources)
    assert "memory.reflect" not in container.tasks.handlers  # flag off in test settings
    reply = {
        "insights": [
            {
                "content": "Atlas runs Postgres 16 on Kubernetes",
                "memory_type": "SEMANTIC",
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
        assert await service.reflect("acme", SCOUT, sources) == []  # identical content
        assert gw.route.call_count == 2
    async with uow_factory() as uow:
        insight = await uow.memories.get("acme", created[0])
        keys = await uow.memories.visibility_keys("acme", created[0])
    assert insight is not None
    assert insight.content == "Atlas runs Postgres 16 on Kubernetes"
    assert insight.memory_type is MemoryType.SEMANTIC and insight.confidence == 0.6
    assert insight.visibility is sources[0].visibility and insight.owner_principal == SCOUT
    assert keys == sorted(sources[0].system_metadata["visibility_keys"])
    assert sorted(e.source_id for e in insight.evidence) == ids
    assert insight.system_metadata["category"] == "reflection"
    assert insight.system_metadata["source_memory_ids"] == ids
    mine = await _memories(container, uow_factory, U1)
    assert {*ids, created[0]} <= {m.memory_id for m in mine}
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "Atlas Postgres Kubernetes", kinds=("memory",))
    assert any(c.record_id == created[0] for c in res.candidates)


async def test_reflection_without_model_answer_changes_nothing(container, uow_factory) -> None:
    await atlas_facts(container, uow_factory)
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


async def test_a_persons_facts_are_acknowledged_without_a_reflection(container, uow_factory):
    """What a user says is consolidated into their beliefs: reflection does not also
    synthesise it (no model call), and acknowledges it so it is not rescanned."""
    user = U1.model_copy(update={"agent_id": None})
    await _observe(container, uow_factory, user, "I prefer concise answers with code samples.")
    await _observe(container, uow_factory, user, "I prefer bullet points over long paragraphs.")
    with mocked_gateway([]) as gw:
        service = ReflectionService(uow_factory, assist=gw.assist(uses=["reflection"]))
        assert await service.reflect_all() == []
        assert gw.route.call_count == 0
    async with uow_factory() as uow:
        assert await uow.memories.reflection_pending(tenant_id="acme") == []
