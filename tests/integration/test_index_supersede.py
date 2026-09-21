"""Re-indexing a document must not leave the previous generation behind.

Parsing assigns fresh chunk ids, so a re-parse writes a whole new generation of vectors.
Measured on a running service: one upload, retried by the job's own retry policy, left 72
vectors for 10 chunks — four generations, of which Postgres kept one. The orphans took top
ranks, occupied evidence seeds, and their node ids resolved to nothing, so the evidence
stage reported COMPLETE for a check it could not run.
"""

from __future__ import annotations

import pytest

from memory_service.modules.rag.indexer import KNOWLEDGE
from memory_service.ports.search import SearchFilter
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]

MARKDOWN = b"""# ACME Report

## 1. Definitions
**Adjusted EBITDA** means earnings before interest, taxes, depreciation and amortisation.

## 3. Results
Adjusted EBITDA increased to EUR 98 million from EUR 81 million.
"""


async def _indexed_ids(container, tenant_id: str, document_id: str) -> list[str]:
    indexer = container.services["indexer"]
    return await indexer.store.record_ids(
        indexer.collection(KNOWLEDGE),
        SearchFilter(tenant_id=tenant_id, must={"document_id": document_id}),
    )


async def test_reparsing_replaces_vectors_rather_than_adding_a_generation(container, tmp_path):
    from memory_service.domain.context import MemoryExecutionContext
    from memory_service.domain.ids import new_id
    from memory_service.modules.jobs.registry import register_handlers

    register_handlers(container)
    ctx = MemoryExecutionContext(
        tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id=new_id("thread")
    )
    async with container.services["uow_factory"]() as uow:
        handle = await container.services["ingestion"].accept_file(
            uow,
            ctx,
            filename="acme.md",
            media_type="text/markdown",
            data=MARKDOWN,
            title="ACME",
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()

    first = await _indexed_ids(container, "acme", handle.document_id)
    assert first, "the document indexed at all"

    # re-parse and re-index the same document, exactly as a job retry would
    await container.tasks.handlers["document.parse"](
        {"tenant_id": "acme", "document_id": handle.document_id}
    )
    await container.tasks.drain()

    second = await _indexed_ids(container, "acme", handle.document_id)
    async with container.services["uow_factory"]() as uow:
        chunks = await uow.documents.list_chunks("acme", handle.document_id)

    chunk_ids = {c.chunk_id for c in chunks}
    orphans = [r for r in second if r.startswith("chk_") and r not in chunk_ids]
    assert orphans == [], f"{len(orphans)} vectors survive whose chunk row is gone"
    assert len([r for r in second if r.startswith("chk_")]) == len(chunk_ids)
