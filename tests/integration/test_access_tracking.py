"""The forgetting policy's access signal, end to end.

``forgetting.score()`` is
``importance * 0.5**(idle_days/half_life) * (1 - 0.5**(accesses + reinforcements))`` and reads
``access_count`` / ``last_accessed_at``. Migration 0006 added both columns for it.

Nothing incremented them. ``bump_access`` was implemented on the repository and declared on
the port and had **no call site anywhere in the codebase** — so ``access_count`` stayed 0 for
every memory, the access term was the constant 0.5 for all of them, and decay ran on
importance and recency alone. The half of the policy that keeps frequently-used memories
alive did nothing at all, while looking present in the code, in the port and in the migration.

A test that only asserted "the column exists" would have passed throughout.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
FACT = "Priya owns the rollback plan and pins the previous image digest in the release manifest."


async def _remember(container, uow_factory, text: str) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["memory"].submit_observation(
            uow, CTX, kind=ObservationKind.MESSAGE, content=text
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _all(uow_factory) -> list:
    from datetime import UTC, datetime, timedelta

    async with uow_factory() as uow:
        return await uow.memories.list_recent(
            since=datetime.now(UTC) - timedelta(hours=1), limit=100
        )


async def _counts(uow_factory) -> list[int]:
    return [m.access_count for m in await _all(uow_factory)]


async def test_serving_a_memory_counts_as_using_it(container, uow_factory) -> None:
    await _remember(container, uow_factory, FACT)
    before = await _counts(uow_factory)
    assert before, "nothing was remembered, so the rest of this proves nothing"
    assert all(c == 0 for c in before)

    builder = container.services["context_builder"]
    bundle = await builder.build(CTX, "who owns the rollback plan?")
    assert bundle.memories, "no memory reached the caller; access cannot be attributed"

    after = await _counts(uow_factory)
    assert sum(after) > sum(before), (
        "a memory was served to the caller and its access count did not move — the forgetting "
        "policy's access term is scoring against a counter nothing increments"
    )


async def test_a_memory_that_was_not_served_is_not_counted(container, uow_factory) -> None:
    """Retrieved-but-dropped must not count, or every query reinforces what nobody read."""
    await _remember(container, uow_factory, FACT)
    await _remember(container, uow_factory, "The Berlin office moved to a four-day week in May.")
    builder = container.services["context_builder"]
    bundle = await builder.build(CTX, "who owns the rollback plan?")
    served = {item.item_id for item in bundle.memories}

    for m in await _all(uow_factory):
        if m.memory_id not in served:
            assert m.access_count == 0, (
                f"{m.memory_id} never reached the caller but was counted as accessed"
            )
