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
        await container.services["memory"].submit_observation(
            uow, ctx, kind=ObservationKind.EVENT,
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
        await container.services["memory"].submit_observation(
            uow, ctx, kind=ObservationKind.EVENT,
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
