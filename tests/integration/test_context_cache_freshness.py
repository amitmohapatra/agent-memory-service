"""A cached bundle must not outlive the memory it failed to see.

The writing transaction bumps the revisions a bundle is keyed on — but it does so while
*enqueuing* the indexing job. A context request landing in that gap builds a bundle that
cannot see the new memory yet, and caches it under the revision that was supposed to mean
"this memory exists". Nothing moved the revision again, so the query that motivated the
write kept returning the old answer for the whole cache TTL.

Measured against a running service before the fix: a memory was written, then the same query
polled for two minutes, and never saw it — every response came back ``cache_hit=True`` with
nothing in it.
"""

from __future__ import annotations

import pytest
from benchmark.common import submit_observation

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.domain.ids import new_id
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.jobs.registry import TASK_MEMORY_INDEX
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]

QUERY = "what do we know about SKU-31?"


def _ctx() -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id="u-freshness",
        workspace_id="ws1",
        agent_id="freshness-agent",
        thread_id=new_id("thread"),
    )


async def _revisions(container, ctx) -> dict[str, int]:
    async with container.services["uow_factory"]() as uow:
        return await uow.revisions.get_many(
            ctx.tenant_id,
            [
                (RevisionKind.USER, ctx.user_id or ""),
                (RevisionKind.THREAD, ctx.thread_id or ""),
                (RevisionKind.AGENT, ctx.agent_id or ""),
            ],
        )


async def test_indexing_moves_the_revisions_again_so_a_bundle_cached_mid_write_is_dropped(
    container,
) -> None:
    ctx = _ctx()
    async with container.services["uow_factory"]() as uow:
        # the thread an observation names is ensured by the API router, which is the
        # only production caller that writes observations; a test that reaches past it
        # has to grant the thread itself or its THREAD-scoped memories are readable
        # by nobody, including their author
        await container.services["conversation"].create_thread(uow, ctx)
        await submit_observation(
            uow,
            ctx,
            kind=ObservationKind.EVENT,
            content="SKU-31 is discontinued as of September.",
        )
        await uow.commit()
    await container.tasks.drain()

    written = await _revisions(container, ctx)
    assert written, "writing a memory must move at least one revision"

    # Re-running just the indexing step must move them again. That second bump is the one
    # that happens when the memory can actually be found, and it is what drops any bundle
    # cached while indexing was still in flight.
    async with container.services["uow_factory"]() as uow:
        memories = await container.services["memory"].list_memories(uow, ctx)
    memory_ids = [m.memory_id for m in memories]
    assert memory_ids, "the observation produced no memory"

    await container.tasks.handlers[TASK_MEMORY_INDEX](
        {"tenant_id": ctx.tenant_id, "memory_ids": memory_ids}
    )
    indexed = await _revisions(container, ctx)
    assert any(indexed[key] > written.get(key, 0) for key in indexed), (
        "indexing must bump the revisions again, or a bundle cached during indexing stays "
        "addressed until its TTL expires"
    )


async def test_the_bundle_is_rebuilt_once_the_memory_is_indexed(container) -> None:
    ctx = _ctx()
    builder = container.services["context_builder"]

    await builder.build(ctx, QUERY)
    assert (await builder.build(ctx, QUERY)).cache_hit, "an unchanged scope serves from cache"

    async with container.services["uow_factory"]() as uow:
        # the thread an observation names is ensured by the API router, which is the
        # only production caller that writes observations; a test that reaches past it
        # has to grant the thread itself or its THREAD-scoped memories are readable
        # by nobody, including their author
        await container.services["conversation"].create_thread(uow, ctx)
        await submit_observation(
            uow,
            ctx,
            kind=ObservationKind.EVENT,
            content="SKU-31 is discontinued as of September.",
        )
        await uow.commit()
    await container.tasks.drain()

    assert not (await builder.build(ctx, QUERY)).cache_hit, (
        "a memory was indexed, so the bundle must be rebuilt rather than served from cache"
    )


async def test_the_cache_still_works_when_nothing_changed(container) -> None:
    """The fix must not amount to turning the cache off."""
    ctx = _ctx()
    builder = container.services["context_builder"]
    await builder.build(ctx, QUERY)
    for _ in range(3):
        assert (await builder.build(ctx, QUERY)).cache_hit


async def test_semantic_cache_uses_real_revisions_and_forget_invalidates_it(container) -> None:
    """Real database, ingestion and retrieval; hash vectors test plumbing, not accuracy."""
    ctx = _ctx()
    builder = container.services["context_builder"]
    factory = container.services["uow_factory"]
    memory = container.services["memory"]
    async with factory() as uow:
        await container.services["conversation"].create_thread(uow, ctx)
        await submit_observation(
            uow, ctx, kind=ObservationKind.EVENT, content="My timezone is Asia/Kolkata."
        )
        await uow.commit()
    await container.tasks.drain()
    first = await builder.build(ctx, "my timezone?")
    await builder.drain()
    assert first.memories
    second = await builder.build(ctx, "my timezone!")
    assert second.cache_hit and second.query == "my timezone!"
    assert second.bundle_id != first.bundle_id
    await builder.drain()
    async with factory() as uow:
        memories = await memory.list_memories(uow, ctx)
        for item in memories:
            await memory.forget(uow, ctx, item.memory_id)
        await uow.commit()
    await container.tasks.drain()
    third = await builder.build(ctx, "my timezone.")
    assert not third.cache_hit
    assert not third.memories


async def test_semantic_cache_cannot_cross_real_user_scope(container) -> None:
    ctx = _ctx()
    builder = container.services["context_builder"]
    await builder.build(ctx, "my timezone?")
    await builder.drain()
    outsider = ctx.model_copy(update={"user_id": "unrelated", "thread_id": None})
    bundle = await builder.build(outsider, "my timezone!")
    assert not bundle.cache_hit


@pytest.mark.parametrize("anchored", [False, True])
async def test_forgetting_tenant_memory_invalidates_other_readers_before_and_after_index(
    container, uow_factory, anchored
):
    from memory_service.domain.enums import MemoryType, Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, U2, _memories, _observe

    await _observe(
        container,
        uow_factory,
        U1,
        "My timezone is Asia/Kolkata.",
        hints=ProcessingHints(
            visibility=Visibility.TENANT,
            memory_type=MemoryType.USER if anchored else MemoryType.SEMANTIC,
        ),
    )
    builder = container.services["context_builder"]
    first = await builder.build(U2, "my timezone?")
    await builder.drain()
    assert first.memories
    assert await builder.cached(U2, first.bundle_id) is not None
    assert (await builder.build(U2, "my timezone!")).cache_hit
    await builder.drain()
    memories = await _memories(uow_factory, U1, container)
    async with uow_factory() as uow:
        for memory in memories:
            await container.services["memory"].forget(uow, U1, memory.memory_id)
        await uow.commit()
    # No worker/index/graph revision may mask the synchronous invalidation.
    assert await builder.cached(U2, first.bundle_id) is None
    mid_write = await builder.build(U2, "my timezone?")
    await builder.drain()
    assert not mid_write.cache_hit
    await container.tasks.drain()
    assert await builder.cached(U2, mid_write.bundle_id) is None
    final = await builder.build(U2, "my timezone!")
    assert not final.cache_hit and not final.memories


async def test_threadless_reader_of_shared_thread_loses_replay_immediately_on_forget(
    container, uow_factory
):
    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, U2, _memories, _observe

    author = U1.model_copy(update={"thread_id": new_id("thread")})
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, author)
        await uow.commit()
    await container.services["authz"].grant_thread(U2, author.thread_id, workspace_id="ws1")
    await _observe(
        container,
        uow_factory,
        author,
        "We decided to use PostgreSQL for the dashboard.",
        kind=ObservationKind.DECISION,
        hints=ProcessingHints(visibility=Visibility.THREAD),
    )
    builder = container.services["context_builder"]
    bundle = await builder.build(U2, "what did we decide about the dashboard?")
    await builder.drain()
    assert bundle.memories
    assert await builder.cached(U2, bundle.bundle_id) is not None
    memories = await _memories(uow_factory, author, container)
    async with uow_factory() as uow:
        for memory in memories:
            await container.services["memory"].forget(uow, author, memory.memory_id)
        await uow.commit()
    assert await builder.cached(U2, bundle.bundle_id) is None


async def test_private_memory_in_a_thread_invalidates_owners_threadless_replay(
    container, uow_factory
):
    from memory_service.domain.enums import MemoryType, Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, _memories, _observe

    author = U1.model_copy(update={"thread_id": new_id("thread")})
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, author)
        await uow.commit()
    builder = container.services["context_builder"]
    query = "what was said about the dashboard?"
    empty = await builder.build(U1, query)
    await builder.drain()
    await _observe(
        container,
        uow_factory,
        author,
        "The dashboard deployment moved to Friday.",
        hints=ProcessingHints(visibility=Visibility.PRIVATE, memory_type=MemoryType.OBSERVATION),
    )
    assert await builder.cached(U1, empty.bundle_id) is None
    bundle = await builder.build(U1, query)
    await builder.drain()
    assert bundle.memories and await builder.cached(U1, bundle.bundle_id) is not None
    memories = await _memories(uow_factory, author, container)
    assert memories and all(m.scope.user_id is None for m in memories)
    async with uow_factory() as uow:
        for memory in memories:
            await container.services["memory"].forget(uow, author, memory.memory_id)
        await uow.commit()
    assert await builder.cached(U1, bundle.bundle_id) is None
