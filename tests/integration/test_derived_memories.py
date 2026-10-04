"""Derived memories on real PostgreSQL: source lifecycle, audience isolation, expansion,
and the reflection that writes them."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.enums import MemoryType, TemporalStatus, Visibility
from memory_service.domain.errors import ValidationFailed
from memory_service.modules.memory.derived import _derived
from memory_service.modules.memory.reflection import ReflectionService
from tests.integration.test_memory import U1, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration


async def derive(uow, base, subject: str):
    """A derived memory over ``base``, written the way reflection writes one."""
    memory, keys = _derived(
        U1,
        memory_type=MemoryType.SEMANTIC,
        scope=base[0].scope,
        sources=base,
        content=f"What the sources say about {subject}.",
        subject=subject,
        predicate="insight:" + subject,
        confidence=0.6,
        importance=0.6,
        now=datetime.now(UTC),
        category="reflection",
        extra={"source_memory_ids": sorted(m.memory_id for m in base)},
    )
    memory.system_metadata["visibility_keys"] = keys
    await uow.memories.add(memory, visibility_keys=keys)
    return memory


async def sources(container, uow_factory):
    await _observe(container, uow_factory, U1, "I prefer concise answers with code samples.")
    await _observe(container, uow_factory, U1, "I prefer bullet points over long paragraphs.")
    return [
        m
        for m in await _memories(uow_factory, U1, container)
        if not m.system_metadata.get("source_revisions")
    ]


async def test_changed_source_rejected_and_derived_invalidated_recursively(container, uow_factory):
    base = await sources(container, uow_factory)
    now = datetime.now(UTC)
    async with uow_factory() as uow:
        child = await derive(uow, base, "profile")
        await uow.commit()
    async with uow_factory() as uow:
        child = await uow.memories.get("acme", child.memory_id)
        grandchild = await derive(uow, [child, base[1]], "overview")
        await uow.commit()
    async with uow_factory() as uow:
        await uow.memories.set_status("acme", base[0].memory_id, TemporalStatus.SUPERSEDED, now=now)
        await uow.commit()
    async with uow_factory() as uow:
        assert await uow.memories.get_many("acme", [child.memory_id, grandchild.memory_id]) == []
        with pytest.raises(ValidationFailed):
            await derive(uow, base, "stale snapshot")


async def test_derived_expiry_and_rollback(container, uow_factory):
    base = await sources(container, uow_factory)
    now = datetime.now(UTC)
    async with uow_factory() as uow:
        base[0].system_metadata["expires_at"] = (now + timedelta(hours=1)).isoformat()
        await uow.memories.update(base[0])
        child = await derive(uow, base, "profile")
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


async def test_retrieval_expands_a_derived_memory_to_its_sources(container, uow_factory):
    base = await sources(container, uow_factory)
    async with uow_factory() as uow:
        summary = await derive(uow, base, "profile")
        await uow.commit()
    engine = container.services["retrieval"]
    result = await engine.retrieve(U1, f"show {summary.memory_id}", kinds=("memory",))
    assert {m.memory_id for m in base} <= {c.record_id for c in result.candidates}
    # the exact hit leads; the hybrid search that always follows finds the sources too
    assert result.candidates[0].record_id == summary.memory_id
    # the expansion fetches whichever sources the search did not bring
    diagnostics: dict = {}
    expanded = await engine._expand_derived_sources(
        U1, result.candidates[:1], result.visibility, diagnostics
    )
    assert diagnostics["derived_sources"] == 2
    assert {c.record_id for c in expanded[1:]} == {m.memory_id for m in base}
    assert all(c.expansion_edge == "DERIVED_SOURCE" for c in expanded[1:])


async def test_source_changed_during_model_call_discards_insight(container, uow_factory):
    base = await sources(container, uow_factory)

    class ConcurrentAssist:
        def wants(self, use):
            return True

        @asynccontextmanager
        async def bound(self, identity):
            yield

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


async def test_forgetting_a_source_retracts_what_was_derived_from_it(container, uow_factory):
    """``DELETE /v1/memories/{id}`` and the ``memory_forget`` tool both forget through
    ``MemoryService.forget``. What was derived from the forgotten memory goes with it, however
    deep and whatever else it was derived from: a synthesis of A and B still says what A
    said. What was derived only from other memories stays. The dependents leave the index and
    the readers' caches in the same commit."""
    from memory_service.adapters.db.orm import MemoryRow
    from memory_service.modules.memory.revisions import memory_revision_keys

    forgotten, kept = await sources(container, uow_factory)
    async with uow_factory() as uow:
        both = await derive(uow, [forgotten, kept], "profile")
        await uow.commit()
    async with uow_factory() as uow:
        both = await uow.memories.get("acme", both.memory_id)
        deeper = await derive(uow, [both, kept], "overview")
        unrelated = await derive(uow, [kept], "formatting")
        await uow.commit()
    # a dependent anchored where the forgotten memory is not: only its own revisions can
    # invalidate a reader who cached it
    async with uow_factory() as uow:
        row = await uow.memories.s.get(MemoryRow, deeper.memory_id)
        row.thread_id, row.scope_level = "thr-insight", "THREAD"
        await uow.commit()
    async with uow_factory() as uow:
        deeper = (await uow.memories.get_many("acme", [deeper.memory_id]))[0]
        assert deeper.scope.thread_id == "thr-insight"
    watched = sorted(memory_revision_keys(deeper))
    async with uow_factory() as uow:
        before = await uow.revisions.get_many("acme", watched)

    async with uow_factory() as uow:
        assert await container.services["memory"].forget(uow, U1, forgotten.memory_id)
        await uow.commit()

    async with uow_factory() as uow:
        live = {
            m.memory_id
            for m in await uow.memories.get_many(
                "acme", [both.memory_id, deeper.memory_id, unrelated.memory_id, kept.memory_id]
            )
        }
        assert live == {unrelated.memory_id, kept.memory_id}
        for memory_id in (both.memory_id, deeper.memory_id):
            row = await uow.memories.s.get(MemoryRow, memory_id)
            assert row.temporal_status == TemporalStatus.RETRACTED.value
            assert row.indexed_at is None, "queued for removal from the search index"
        after = await uow.revisions.get_many("acme", watched)
    assert all(after[k] > before[k] for k in before), (before, after)
    # forgetting again is a no-op, not an error
    async with uow_factory() as uow:
        assert await container.services["memory"].forget(uow, U1, forgotten.memory_id) is None
