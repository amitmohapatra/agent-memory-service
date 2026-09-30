"""Asking about a specific thing must not be the query that fails.

The router classifies a query naming an id — "what about SKU-88?" — as EXACT_IDENTIFIER. The
exact lookup's hits lead; ranked search over memories and documents always runs behind them.
Two ways this failed: a lookup that missed searched everything *except* memories ("SKU-88
discontinued" returned nothing while "which products are discontinued" returned the very
same memory), and a lookup that hit ended the search, so a context for "update quote Q-1183
with the EMEA price" carried the memories naming Q-1183 and none of the pricing documents.
"""

from __future__ import annotations

import pytest
from benchmark.common import submit_observation

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
        # only production caller that writes observations; a test that reaches past it
        # has to grant the thread itself or its THREAD-scoped memories are readable
        # by nobody, including their author
        await container.services["conversation"].create_thread(uow, ctx)
        await submit_observation(uow, ctx, kind=ObservationKind.EVENT, content=content)
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


async def test_an_exact_hit_leads_and_the_ranked_search_still_runs(container) -> None:
    ctx = _ctx()
    await _remember(container, ctx, "SKU-88 is discontinued as of September.")
    await _remember(container, ctx, "Discontinued products are removed from the price list.")
    engine = container.services["retrieval"]
    result = await engine.retrieve(ctx, "SKU-88 discontinued", limit=5)
    texts = [c.text for c in result.candidates]
    assert any("SKU-88" in t for t in texts)
    assert any("price list" in t for t in texts), "what does not name the id still answers"
    if result.diagnostics.get("exact_hits"):
        assert "SKU-88" in texts[0], "the exact hit leads"
