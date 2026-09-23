"""Rebuild the search index from the canonical store.

PostgreSQL is the source of truth; Qdrant is a derived index. After an index loss, a
model change (new embedding fingerprint = new collections) or corruption, this rebuilds
every READY document's chunks and summaries and every CURRENT memory:

    uv run python -m memory_service.tools.reindex [--tenant acme] [--drop]

``--drop`` deletes the current collections first (full rebuild); without it the rebuild
upserts over what is there. Documents and memories that fail are reported, never skipped
silently. Also used by the failure-injection suite (``search_rebuild`` scenario).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from memory_service.__about__ import __version__
from memory_service.adapters.db.orm import DocumentRow, MemoryRow
from memory_service.application.container import Container, build_container
from memory_service.config.settings import Settings
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReindexReport:
    documents: int = 0
    chunks: int = 0
    memories: int = 0
    dropped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "memories": self.memories,
            "dropped": self.dropped,
            "failures": self.failures,
        }


async def rebuild_search_index(
    container: Container, *, tenant_id: str | None = None, drop: bool = False
) -> ReindexReport:
    indexer = container.services["indexer"]
    report = ReindexReport()
    if drop:
        # A collection is shared by every tenant, and the rebuild below is filtered by
        # tenant - so dropping the collection while rebuilding one tenant destroys every
        # other tenant's vectors and then declares success. The two flags together were a
        # documented instruction; now a tenant-scoped drop removes only that tenant's
        # points, and only a whole-store rebuild may drop a collection.
        from memory_service.ports.search import SearchFilter

        for base in (KNOWLEDGE, MEMORIES):
            name = indexer.collection(base)
            if tenant_id:
                removed = await container.search.delete_by_filter(
                    name, SearchFilter(tenant_id=tenant_id)
                )
                report.dropped.append(f"{name} (tenant {tenant_id}: {removed} points)")
            elif await container.search.drop_collection(name):
                report.dropped.append(name)
    await indexer.ensure_collections()
    async with container.database.session_factory() as session:
        docs = select(DocumentRow.tenant_id, DocumentRow.document_id).where(
            DocumentRow.status == "READY", DocumentRow.deleted_at.is_(None)
        )
        mems = select(MemoryRow.tenant_id, MemoryRow.memory_id).where(
            MemoryRow.temporal_status == "CURRENT", MemoryRow.deleted_at.is_(None)
        )
        if tenant_id:
            docs = docs.where(DocumentRow.tenant_id == tenant_id)
            mems = mems.where(MemoryRow.tenant_id == tenant_id)
        doc_rows = list((await session.execute(docs)).all())
        mem_rows = list((await session.execute(mems)).all())
    for tenant, document_id in doc_rows:
        try:
            report.chunks += await indexer.index_document(tenant, document_id, force=True)
            report.documents += 1
        except Exception as exc:
            report.failures.append(f"document {tenant}/{document_id}: {type(exc).__name__}: {exc}")
    by_tenant: dict[str, list[str]] = {}
    for tenant, memory_id in mem_rows:
        by_tenant.setdefault(tenant, []).append(memory_id)
    for tenant, ids in by_tenant.items():
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            try:
                report.memories += await indexer.index_memories(tenant, batch)
            except Exception as exc:
                report.failures.append(f"memories {tenant} [{start}:{start + len(batch)}]: {exc}")
    log.info("reindex.done", **report.as_dict())
    return report


async def _main(args: argparse.Namespace) -> int:
    container = await build_container(Settings(), __version__)
    try:
        report = await rebuild_search_index(container, tenant_id=args.tenant, drop=args.drop)
    finally:
        await container.close()
    sys.stdout.write(f"{report.as_dict()}\n")
    return 0 if report.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the search index from PostgreSQL")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--drop", action="store_true", help="delete collections first")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
