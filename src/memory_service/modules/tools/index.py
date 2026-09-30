"""Tool search: catalog entries embedded into the ``tools`` collection (name, description and
argument names), searched with the same hybrid fusion as memories, per tenant and workspace.

A tool record's audience is its catalog scope: ``tenant:<t>`` for a tenant-wide entry,
``workspace:<t>/<w>`` for a workspace's own; a search reads both, and a workspace's entry
shadows the tenant's of the same name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from memory_service.domain.ids import content_hash
from memory_service.domain.tools import ToolDescriptor
from memory_service.modules.rag.indexer import Indexer
from memory_service.observability.logging import get_logger
from memory_service.ports.search import CollectionSpec, SearchFilter, SearchRecord, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

TOOLS: Final = "tools"
#: Candidates one search asks the store for (before the caller's available set narrows them).
SEARCH_LIMIT: Final = 32
_RRF_K: Final = 1


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

    @property
    def collection(self) -> str:
        return self.indexer.collection(TOOLS)

    def forget_collections(self) -> None:
        """The collection was dropped outside this index (a test reset): ensure it again."""
        self._ensured.clear()

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
                    "visibility_keys": tool_audience(entry.tenant_id, entry.workspace_id)[-1:],
                },
            )
            for i, entry in enumerate(entries)
        ]

    async def search(
        self, tenant_id: str, workspace_id: str | None, task: str, *, limit: int = SEARCH_LIMIT
    ) -> list[tuple[str, float]]:
        """``(tool name, relevance 0..1)``, best first, one per name."""
        await self.ensure()
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
