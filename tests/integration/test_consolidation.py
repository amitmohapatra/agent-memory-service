"""Source lifecycle, audience isolation and replay on real PostgreSQL."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.enums import MemoryType, ObservationKind, TemporalStatus, Visibility
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.derived import EntitySummaryService
from memory_service.modules.memory.landing import LandingReflection
from memory_service.modules.memory.reflection import ReflectionService
from tests.integration.test_memory import U1, U2, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration


def enable(container):
    container.services["observation_pipeline"].landing = LandingReflection(
        MemoryIntelligenceSettings(consolidation_enabled=True)
    )


async def sources(container, uow_factory):
    await _observe(container, uow_factory, U1, "I prefer concise answers with code samples.")
    await _observe(container, uow_factory, U1, "I prefer bullet points over long paragraphs.")
    return [
        m
        for m in await _memories(uow_factory, U1, container)
        if not m.system_metadata.get("source_revisions")
    ]


async def test_landing_replay_reinforcement_and_source_forget(container, uow_factory):
    enable(container)
    base = await sources(container, uow_factory)
    before = await _memories(uow_factory, U1, container)
    derived = [m for m in before if m.system_metadata.get("source_revisions")]
    assert {m.memory_type for m in derived} == {MemoryType.BELIEF, MemoryType.ENTITY_SUMMARY}
    assert all(
        set(m.system_metadata["source_revisions"]) == {s.memory_id for s in base} for m in derived
    )
    landing = container.services["observation_pipeline"].landing
    async with uow_factory() as uow:
        assert await landing.on_landed(uow, U1, base[0], now=datetime.now(UTC)) == set()
        await uow.commit()
    await _observe(container, uow_factory, U1, base[0].content)
    current = await _memories(uow_factory, U1, container)
    assert len(current) == len(before)
    refreshed = [m for m in current if m.system_metadata.get("source_revisions")]
    assert {m.memory_id for m in refreshed}.isdisjoint(m.memory_id for m in derived)
    assert next(m for m in current if m.memory_id == base[0].memory_id).reinforcement_count == 2
    engine = container.services["retrieval"]
    assert (await engine.retrieve(U2, "concise answers", kinds=("memory",))).candidates == []
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, U1, base[0].memory_id)
        await uow.commit()
    # Deliberately leave stale vector entries in place: canonical validation must hide them.
    result = await engine.retrieve(U1, "concise bullet answers", kinds=("memory",))
    assert {m.memory_id for m in refreshed}.isdisjoint(c.record_id for c in result.candidates)
    async with uow_factory() as uow:
        assert await uow.memories.get_many("acme", [m.memory_id for m in refreshed]) == []
    async with container.database.engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text("SELECT payload FROM job_outbox WHERE task_name = 'memory.index'")
                )
            )
            .scalars()
            .all()
        )
    assert any({m.memory_id for m in refreshed} <= set(r["memory_ids"]) for r in rows)


async def test_landing_does_not_mix_private_and_shared_sources(container, uow_factory):
    enable(container)
    await _observe(
        container,
        uow_factory,
        U1,
        "I prefer tea over coffee.",
        hints=ProcessingHints(visibility=Visibility.USER),
    )
    await _observe(
        container,
        uow_factory,
        U1,
        "I prefer private medical reminders.",
        hints=ProcessingHints(visibility=Visibility.PRIVATE),
    )
    memories = await _memories(uow_factory, U1, container)
    assert not any(m.system_metadata.get("source_revisions") for m in memories)
    with mocked_gateway([]) as gw:
        reflection = ReflectionService(uow_factory, assist=gw.assist(uses=["reflection"]))
        assert await reflection.reflect("acme", "user:u1", memories) == []
        assert gw.route.call_count == 0


async def test_changed_source_rejected_and_derived_invalidated_recursively(container, uow_factory):
    base = await sources(container, uow_factory)
    service = EntitySummaryService()
    now = datetime.now(UTC)
    async with uow_factory() as uow:
        child, _ = await service.rebuild(
            uow, U1, scope=base[0].scope, subject="profile", facts=base, now=now
        )
        child = await uow.memories.get("acme", child.memory_id)
        grandchild, _ = await service.rebuild(
            uow, U1, scope=base[0].scope, subject="overview", facts=[child], now=now
        )
        await uow.commit()
    async with uow_factory() as uow:
        await uow.memories.set_status("acme", base[0].memory_id, TemporalStatus.SUPERSEDED, now=now)
        await uow.commit()
    async with uow_factory() as uow:
        assert await uow.memories.get_many("acme", [child.memory_id, grandchild.memory_id]) == []
        with pytest.raises(ValidationFailed):
            await service.rebuild(
                uow, U1, scope=base[0].scope, subject="stale snapshot", facts=base, now=now
            )


async def test_derived_expiry_and_rollback(container, uow_factory):
    base = await sources(container, uow_factory)
    now = datetime.now(UTC)
    async with uow_factory() as uow:
        base[0].system_metadata["expires_at"] = (now + timedelta(hours=1)).isoformat()
        await uow.memories.update(base[0])
        child, _ = await EntitySummaryService().rebuild(
            uow, U1, scope=base[0].scope, subject="profile", facts=base, now=now
        )
        await uow.commit()
    async with uow_factory() as uow:
        await uow.memories.forget("acme", base[0].memory_id)
        # No commit: source, dependent state and removal outbox must all roll back.
    async with uow_factory() as uow:
        assert await uow.memories.get("acme", child.memory_id) is not None
        await uow.memories.expire_due(now=now + timedelta(hours=2))
        await uow.commit()
    async with uow_factory() as uow:
        assert await uow.memories.get("acme", child.memory_id) is None


async def test_concurrent_admission_has_one_live_summary(container, uow_factory):
    enable(container)
    pipeline = container.services["observation_pipeline"]
    ids = []
    async with uow_factory() as uow:
        for content in ["I prefer concise answers.", "I prefer bullet points."]:
            ack = await container.services["memory"].submit_observation(
                uow, U1, kind=ObservationKind.MESSAGE, content=content
            )
            ids.append(ack.observation_id)
        await uow.commit()
    await asyncio.wait_for(
        asyncio.gather(*(pipeline.run({"tenant_id": "acme", "observation_id": i}) for i in ids)),
        timeout=10,
    )
    memories = await _memories(uow_factory, U1, container)
    summaries = [m for m in memories if m.memory_type is MemoryType.ENTITY_SUMMARY]
    assert len(summaries) == 1 and len(summaries[0].evidence) == 2


async def test_reflection_requires_two_sources_and_rejects_unseen_citations(container, uow_factory):
    base = await sources(container, uow_factory)
    reply = {
        "insights": [
            {
                "content": "User values concise formatting",
                "memory_type": "PREFERENCE",
                "source_memory_ids": [base[0].memory_id, "not-in-prompt"],
            }
        ]
    }
    with mocked_gateway([reply]) as gw:
        service = ReflectionService(uow_factory, assist=gw.assist(uses=["reflection"]))
        assert await service.reflect("acme", "user:u1", base[:1]) == []
        assert gw.route.call_count == 0
        assert await service.reflect("acme", "user:u1", base) == []
        assert gw.route.call_count == 1


async def test_retrieval_expands_summary_to_original_sources(container, uow_factory):
    enable(container)
    base = await sources(container, uow_factory)
    memories = await _memories(uow_factory, U1, container)
    summary = next(m for m in memories if m.memory_type is MemoryType.ENTITY_SUMMARY)
    engine = container.services["retrieval"]
    result = await engine.retrieve(U1, f"show {summary.memory_id}", kinds=("memory",))
    assert {m.memory_id for m in base} <= {c.record_id for c in result.candidates}
    assert result.diagnostics["derived_sources"] == 2
    assert all(
        c.expansion_edge == "DERIVED_SOURCE"
        for c in result.candidates
        if c.record_id in {m.memory_id for m in base}
    )


async def test_source_changed_during_model_call_discards_insight(container, uow_factory):
    base = await sources(container, uow_factory)

    class ConcurrentAssist:
        def wants(self, use):
            return True

        async def structured(self, *args, **kwargs):
            async with uow_factory() as uow:
                await uow.memories.forget("acme", base[0].memory_id)
                await uow.commit()
            return {
                "insights": [
                    {
                        "content": "User prefers skimmable answers",
                        "memory_type": "PREFERENCE",
                        "source_memory_ids": [m.memory_id for m in base],
                    }
                ]
            }

    service = ReflectionService(uow_factory, assist=ConcurrentAssist())
    assert await service.reflect("acme", "user:u1", base) == []
    assert not any(
        m.system_metadata.get("category") == "reflection"
        for m in await _memories(uow_factory, U1, container)
    )


async def test_run_only_reflection_does_not_grant_the_owner_extra_access(container, uow_factory):
    from memory_service.domain.ids import new_id

    originals = await sources(container, uow_factory)
    run_keys = ["run:acme/run-1"]
    run_sources = []
    async with uow_factory() as uow:
        for original in originals:
            memory = original.model_copy(
                deep=True, update={"memory_id": new_id("memory"), "visibility": Visibility.RUN}
            )
            await uow.memories.add(memory, visibility_keys=run_keys)
            run_sources.append(await uow.memories.get("acme", memory.memory_id))
        await uow.commit()
    reply = {
        "insights": [
            {
                "content": "The user explicitly prefers concise answers and bullet points.",
                "memory_type": "PREFERENCE",
                "source_memory_ids": [m.memory_id for m in run_sources],
            }
        ]
    }
    with mocked_gateway([reply]) as gw:
        service = ReflectionService(uow_factory, assist=gw.assist(uses=["reflection"]))
        created = await service.reflect("acme", "user:u1", run_sources)
    assert len(created) == 1
    async with uow_factory() as uow:
        insight = await uow.memories.get("acme", created[0])
        assert insight.visibility is Visibility.RUN
        assert await uow.memories.visibility_keys("acme", created[0]) == run_keys
    assert created[0] not in {m.memory_id for m in await _memories(uow_factory, U1, container)}
