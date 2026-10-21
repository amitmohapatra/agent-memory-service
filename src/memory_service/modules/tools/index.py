"""Tool search: catalog entries embedded into the ``tools`` collection (name, description,
argument names and the arguments' schema descriptions: ``ToolDescriptor.index_text``),
searched with the same hybrid fusion as memories, per tenant and workspace.

Each record carries the ``text_hash`` of the text it was embedded from. A changed catalog
entry is re-indexed by its own ``tools.index`` job; a release that changes what a tool is
searched by changes the hash of entries nobody touched, and ``refresh`` (a periodic job)
re-indexes those, so an existing deployment picks the new text up by itself.

A tool record's audience is its catalog scope: ``tenant:<t>`` for a tenant-wide entry,
``workspace:<t>/<w>`` for a workspace's own; a search reads both, and a workspace's entry
shadows the tenant's of the same name.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Final

from memory_service.domain.ids import content_hash
from memory_service.domain.tools import ToolDescriptor
from memory_service.modules.rag.indexer import Indexer
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.observability.logging import get_logger
from memory_service.ports.search import CollectionSpec, SearchFilter, SearchRecord, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

TOOLS: Final = "tools"
#: Candidates one search asks the store for (before the caller's available set narrows them).
SEARCH_LIMIT: Final = 32
_RRF_K: Final = 1
#: Catalog entries one ``refresh`` page compares with their records.
REFRESH_BATCH: Final = 256
#: How long a process trusts a ``refresh`` it already ran: a new release is a new process, so
#: the first tick after a deploy compares every record; later ticks only catch a record an
#: older process wrote during a rolling deploy.
REFRESH_EVERY_SECONDS: Final = 6 * 3600


def tool_audience(tenant_id: str, workspace_id: str | None) -> list[str]:
    """The keys a catalog scope is searched by."""
    keys = [f"tenant:{tenant_id}"]
    if workspace_id:
        keys.append(f"workspace:{tenant_id}/{workspace_id}")
    return keys


class ToolIndex:
    def __init__(self, uow_factory: UnitOfWorkFactory, indexer: Indexer, store: SearchStore):
        self.uow_factory = uow_factory
        self.indexer = indexer
        self.store = store
        self._ensured: set[str] = set()
        #: collection -> when this process last compared every record with its entry
        self._refreshed: dict[str, float] = {}

    @property
    def collection(self) -> str:
        return self.indexer.collection(TOOLS)

    def forget_collections(self) -> None:
        """The collection was dropped outside this index (a test reset): ensure it again."""
        self._ensured.clear()
        self._refreshed.clear()

    async def ensure(self) -> None:
        if self.collection in self._ensured:
            return
        await self.store.ensure_collection(
            CollectionSpec(
                name=self.collection,
                dense=self.indexer.spaces.dimensions,
                sparse=True,
                sparse_idf=getattr(self.indexer.sparse, "server_side_idf", True),
                # a catalog is small and every tool search reads it
                on_disk_payload=False,
            )
        )
        self._ensured.add(self.collection)

    async def index(self, tenant_id: str, tool_ids: Sequence[str]) -> int:
        """Embed the named catalog entries (the ``tools.index`` job)."""
        async with self.uow_factory() as uow:
            entries = await uow.tools.catalog_by_ids(tenant_id, tool_ids)
        if not entries:
            return 0
        await self.ensure()
        await self.store.upsert(await self._records(entries))
        log.info("tools.indexed", tenant_id=tenant_id, count=len(entries))
        return len(entries)

    async def refresh(self, *, force: bool = False) -> int:
        """Re-embed every indexed entry whose record was embedded from other text than its
        ``index_text`` gives now (a release changed what tools are searched by, or a record
        predates ``text_hash``). Entries without a record are left to their own index job.
        Runs once per process per ``REFRESH_EVERY_SECONDS`` unless ``force``; returns how
        many records were re-indexed."""
        collection = self.collection
        started = time.monotonic()
        last = self._refreshed.get(collection)
        if not force and last is not None and started - last < REFRESH_EVERY_SECONDS:
            return 0
        await self.ensure()
        refreshed, after = 0, ""
        while True:
            async with self.uow_factory() as uow:
                entries = await uow.tools.catalog_page(after=after, limit=REFRESH_BATCH)
            if not entries:
                break
            after = entries[-1].tool_id
            indexed = {
                r.record_id: r.payload.get("text_hash")
                for r in await self.store.get(collection, [e.tool_id for e in entries])
            }
            stale = [
                e
                for e in entries
                if e.tool_id in indexed and indexed[e.tool_id] != content_hash(e.index_text())
            ]
            if stale:
                await self.store.upsert(await self._records(stale))
                refreshed += len(stale)
            if len(entries) < REFRESH_BATCH:
                break
        self._refreshed[collection] = started
        if refreshed:
            log.info("tools.refreshed", count=refreshed)
        return refreshed

    async def _records(self, entries: Sequence[ToolDescriptor]) -> list[SearchRecord]:
        texts = [e.index_text() for e in entries]
        dense = await self.indexer.embed_cached(texts, [content_hash(t) + ":tool1" for t in texts])
        sparse = self.indexer.sparse.encode_documents(texts)
        return [
            SearchRecord(
                record_id=entry.tool_id,
                collection=self.collection,
                tenant_id=entry.tenant_id,
                dense={space: vectors[i] for space, vectors in dense.items()},
                sparse=sparse[i],
                payload={
                    "kind": "tool",
                    "text": entry.name,
                    "text_hash": content_hash(texts[i]),
                    "visibility_keys": tool_audience(entry.tenant_id, entry.workspace_id)[-1:],
                },
            )
            for i, entry in enumerate(entries)
        ]

    async def search(
        self,
        tenant_id: str,
        workspace_id: str | None,
        task: str,
        *,
        vectors: QueryVectors | None = None,
        limit: int = SEARCH_LIMIT,
    ) -> list[tuple[str, float]]:
        """``(tool name, relevance 0..1)``, best first, one per name. ``vectors`` are the
        task's, when a retrieval already encoded it (the pushed context): no second encode."""
        await self.ensure()
        if vectors is not None and vectors.dense:
            dense, sparse = vectors.dense, vectors.sparse
        else:
            dense = await self.indexer.spaces.embed_query(task)
            sparse = self.indexer.sparse.encode_query(task)
        hits = await self.store.search_hybrid(
            self.collection,
            dense=dense,
            sparse=sparse,
            flt=SearchFilter(
                tenant_id=tenant_id,
                must={"kind": "tool"},
                must_any={"visibility_keys": tool_audience(tenant_id, workspace_id)},
            ),
            limit=limit,
            prefetch_limit=limit,
            rrf_k=_RRF_K,
        )
        best = max((h.score for h in hits), default=0.0) or 1.0
        out: dict[str, float] = {}
        for hit in hits:
            name = str(hit.payload.get("text", ""))
            if name and name not in out:
                out[name] = hit.score / best
        return list(out.items())
