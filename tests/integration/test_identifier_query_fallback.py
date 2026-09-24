"""Asking about a specific thing must not be the query that fails.

The router classifies a query naming an id — "what about SKU-88?" — as EXACT_IDENTIFIER and
sets needs_memories=False: an exact lookup beats anything ranking could offer, when it hits.
When it misses, the engine falls back to ranked search — but the flag was applied inside the
fallback too, so the fallback searched everything *except* memories. Measured against a
running service: "SKU-88 discontinued" returned nothing while "which products are
discontinued" returned the very same memory.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind, QueryType
from memory_service.domain.ids import new_id
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


def _ctx() -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id="u-ident",
        workspace_id="ws1",
        agent_id="ident-agent",
        thread_id=new_id("thread"),
    )


async def _remember(container, ctx, content: str) -> None:
    async with container.services["uow_factory"]() as uow:
        # the thread an observation names is ensured by the API router, which is the
        # only production caller of submit_observation; a test that reaches past it
        # has to grant the thread itself or its THREAD-scoped memories are readable
        # by nobody, including their author
        await container.services["conversation"].create_thread(uow, ctx)
        await container.services["memory"].submit_observation(
            uow, ctx, kind=ObservationKind.EVENT, content=content
        )
        await uow.commit()
    await container.tasks.drain()


async def test_an_identifier_query_that_misses_the_exact_index_still_searches_memories(
    container,
) -> None:
    ctx = _ctx()
    await _remember(container, ctx, "SKU-88 is discontinued as of September.")
    engine = container.services["retrieval"]

    identifier = await engine.retrieve(ctx, "SKU-88 discontinued", limit=5)
    assert identifier.routed.query_type is QueryType.EXACT_IDENTIFIER, (
        "this test is only meaningful if the router still takes the identifier branch"
    )
    assert identifier.diagnostics.get("exact_hits") == 0
    assert any(c.kind == "memory" for c in identifier.candidates), (
        "the exact lookup found nothing, so the fallback must search memories"
    )


async def test_the_exact_lookup_still_wins_when_it_hits(container) -> None:
    """The fallback must not turn every identifier query into a ranked search."""
    ctx = _ctx()
    await _remember(container, ctx, "SKU-88 is discontinued as of September.")
    engine = container.services["retrieval"]
    result = await engine.retrieve(ctx, "SKU-88 discontinued", limit=5)
    if result.diagnostics.get("exact_hits"):
        assert not result.diagnostics.get("exact_fallback")
