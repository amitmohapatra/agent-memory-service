"""Consolidation progress is durable, revision-aware and independent of source content."""

from datetime import UTC, datetime

import pytest

from memory_service.modules.memory.reflection import ReflectionService
from tests.integration.test_consolidation import sources
from tests.integration.test_memory import U1, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration


async def test_receipts_survive_service_restart_and_source_changes(container, uow_factory):
    memories = await sources(container, uow_factory)
    with mocked_gateway([{"insights": []}, {"insights": []}]) as gateway:
        for _ in range(2):
            service = ReflectionService(uow_factory, assist=gateway.assist(uses=["reflection"]))
            assert await service.reflect_all() == []
        assert gateway.route.call_count == 1
        async with uow_factory() as uow:
            current = await uow.memories.get("acme", memories[0].memory_id)
            current.importance = 0.7
            await uow.memories.update(current)
            await uow.commit()
        async with uow_factory() as uow:
            pending = await uow.memories.reflection_pending()
        assert memories[0].memory_id in {m.memory_id for m in pending}
        await service.reflect_all()
        assert gateway.route.call_count == 2
    async with uow_factory() as uow:
        # A late receipt for an earlier revision cannot undo newer progress.
        await uow.memories.mark_reflected(memories, at=datetime.now(UTC))
        await uow.commit()
    async with uow_factory() as uow:
        assert await uow.memories.reflection_pending() == []


async def test_related_history_can_include_original_turn_without_changing_landing(
    container, uow_factory
):
    await _observe(
        container, uow_factory, U1, "The supplier delivered the copper valves on Monday."
    )
    old = next(
        m
        for m in await _memories(uow_factory, U1, container)
        if m.system_metadata.get("category") == "verbatim_turn"
    )
    kwargs = {
        "scope_key": old.scope.key(),
        "subject": old.subject,
        "owner_principal": old.owner_principal,
        "visibility_keys": old.system_metadata["visibility_keys"],
        "include_derived": False,
    }
    async with uow_factory() as uow:
        landing = await uow.memories.related(U1.tenant_id, **kwargs)
        history = await uow.memories.related(U1.tenant_id, **kwargs, include_verbatim=True)
    assert old.memory_id not in {m.memory_id for m in landing}
    assert old.memory_id in {m.memory_id for m in history}


async def test_cross_thread_insight_is_recalled_and_hidden_after_source_deletion(
    container, uow_factory
):
    first = U1.model_copy(update={"thread_id": "planning"})
    second = U1.model_copy(update={"thread_id": "follow-up"})
    reader = U1.model_copy(update={"thread_id": "new-session"})
    await _observe(container, uow_factory, first, "I prefer concise answers.")
    original = await _memories(uow_factory, first, container)
    assert len(original) == 1
    with mocked_gateway([]) as gateway:
        service = ReflectionService(uow_factory, assist=gateway.assist(uses=["reflection"]))
        assert await service.reflect_all() == []
        assert gateway.route.call_count == 0  # one fact cannot justify an observation
    await _observe(container, uow_factory, second, "I prefer bullet points.")
    facts = await _memories(uow_factory, reader, container)
    assert len(facts) == 2
    reply = {
        "insights": [
            {
                "content": "User prefers concise answers and bullet points.",
                "memory_type": "PREFERENCE",
                "source_memory_ids": [m.memory_id for m in facts],
            }
        ]
    }
    with mocked_gateway([reply]) as gateway:
        restarted = ReflectionService(uow_factory, assist=gateway.assist(uses=["reflection"]))
        created = await restarted.reflect_all()
        assert len(created) == gateway.route.call_count == 1
        await container.tasks.drain()
        assert await restarted.reflect_all() == []
        assert gateway.route.call_count == 1
    engine = container.services["retrieval"]
    result = await engine.retrieve(reader, "concise answers bullet points", kinds=("memory",))
    assert created[0] in {c.record_id for c in result.candidates}
    other = reader.model_copy(update={"user_id": "another-user"})
    result = await engine.retrieve(other, "concise answers bullet points", kinds=("memory",))
    assert not result.candidates
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, first, original[0].memory_id)
        await uow.commit()
    # Do not drain index removal: canonical validation must hide a stale derived hit.
    result = await engine.retrieve(reader, "concise answers bullet points", kinds=("memory",))
    assert created[0] not in {c.record_id for c in result.candidates}
