"""A document becomes searchable when its index job runs, and that is what must move the
revisions its readers' bundles are keyed on (ADR 0031).

Ingestion bumps DOCUMENT, which no context lookup reads, and the tenant-wide GRAPH counter
moved only when graph enrichment ran - so with enrichment off, a bundle cached before the
index job kept answering without the document until its TTL. This module runs with graph
enrichment disabled on purpose: the indexer alone has to invalidate.
"""

from __future__ import annotations

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.revisions import RevisionKind
from tests.integration.conftest import requires_pg
from tests.integration.test_retrieval import OTHER_USER, OWNER, _ingest

pytestmark = [pytest.mark.integration, requires_pg]

QUERY = "why did Adjusted EBITDA increase despite lower revenue?"


@pytest.fixture
def container_overrides() -> dict:
    return {"graph_enrichment": "disabled"}


async def _served(builder, ctx, query: str) -> bool:
    found = await builder._lookup(ctx, query, builder.cfg.token_budget, None, None, True)
    return found.bundle is not None


async def test_an_indexed_document_drops_the_bundle_cached_before_it(container) -> None:
    assert "graph" not in container.services
    builder = container.services["context_builder"]
    empty = await builder.build(OWNER, QUERY)
    await builder.drain()
    assert not empty.knowledge and await _served(builder, OWNER, QUERY)

    await _ingest(container, container.services["uow_factory"])

    assert not await _served(builder, OWNER, QUERY), (
        "the document is searchable, so the bundle cached before it must not be served"
    )
    fresh = await builder.build(OWNER, QUERY)
    assert not fresh.cache_hit and fresh.knowledge


async def test_a_user_document_leaves_other_users_bundles_cached(container) -> None:
    """The index job's bump follows the document's audience: a USER document moves that
    user's counter, not the tenant's, so every other reader keeps its cache. (Accepting the
    upload grants the document, which moves the tenant's MEMBERSHIP; that happens before
    the bundles below are built, and the index pass is run again on its own.)"""
    builder = container.services["context_builder"]
    uow_factory = container.services["uow_factory"]
    document_id = await _ingest(container, uow_factory, visibility=Visibility.USER)
    for ctx in (OWNER, OTHER_USER):
        await builder.build(ctx, QUERY)
    await builder.drain()
    assert await _served(builder, OWNER, QUERY) and await _served(builder, OTHER_USER, QUERY)
    async with uow_factory() as uow:
        before = await uow.revisions.get_many("acme", [(RevisionKind.TENANT, "")])

    await container.services["indexer"].index_document("acme", document_id, force=True)

    async with uow_factory() as uow:
        after = await uow.revisions.get_many("acme", [(RevisionKind.TENANT, "")])
    assert after == before
    assert not await _served(builder, OWNER, QUERY)
    assert await _served(builder, OTHER_USER, QUERY)


async def test_deleting_a_document_from_the_index_moves_its_audience(container) -> None:
    builder = container.services["context_builder"]
    document_id = await _ingest(container, container.services["uow_factory"])
    assert (await builder.build(OWNER, QUERY)).knowledge
    await builder.drain()
    assert await _served(builder, OWNER, QUERY)

    await container.services["indexer"].delete_document("acme", document_id)

    assert not await _served(builder, OWNER, QUERY)
    assert not (await builder.build(OWNER, QUERY)).knowledge
