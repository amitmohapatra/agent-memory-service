"""Benchmark-gated retrieval strategies (M10). All default OFF; each is an *extra retriever*
whose hit list is fused with the baseline hybrid list by reciprocal-rank fusion, so turning
one on can add evidence but never removes the baseline's.

- ``pageindex``: tree search. Route the question to the best sections by their hierarchical
  summaries, then search only the chunks beneath those sections (a deterministic, LLM-free
  PageIndex): precise for document-local questions, cheap because the filter is store-side.
- ``raptor``: multi-level retrieval. Summary nodes (section/document) compete with chunks in
  the fused list, so a question answered by the gist of a section surfaces the summary and
  the chunks beneath it (RAPTOR without the LLM-built tree: our tree is the document's own
  hierarchy).
- ``graph_ppr``: personalised PageRank over the graph neighbourhood seeded at the query's
  entities; lives in :mod:`memory_service.modules.graph.retrieval`.
- ``late_interaction`` (ColBERT), ``splade``/``minicoil`` (learned sparse) and
  ``late_chunking`` are adapters in :mod:`memory_service.adapters.models` that need model
  weights; the harness records them as skipped when the weights are absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ContextGraphEdge, QueryType
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.rag.indexer import KNOWLEDGE, Indexer
from memory_service.modules.retrieval.router import RoutedQuery
from memory_service.ports.search import SearchFilter, SearchHit, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory


class ExtraRetriever(Protocol):
    name: str

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        visibility: VisibilitySpecification,
        document_ids: Sequence[str] | None,
    ) -> list[SearchHit]: ...


def _tag(hits: Sequence[SearchHit], retriever: str) -> list[SearchHit]:
    return [h.model_copy(update={"retriever": retriever}) for h in hits]


class PageIndexRetriever:
    name = "pageindex"

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        store: SearchStore,
        indexer: Indexer,
        *,
        sections: int = 3,
        limit: int = 20,
        prefetch: int = 30,
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.indexer = indexer
        self.sections = sections
        self.limit = limit
        self.prefetch = prefetch

    async def _search(
        self,
        query: str,
        flt: SearchFilter,
        *,
        limit: int,
    ) -> list[SearchHit]:
        return await self.store.search_hybrid(
            self.indexer.collection(KNOWLEDGE),
            dense=await self.indexer.embedding.embed_query(query),
            sparse=self.indexer.sparse.encode_query(query),
            flt=flt,
            limit=limit,
            prefetch_limit=self.prefetch,
        )

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        visibility: VisibilitySpecification,
        document_ids: Sequence[str] | None,
    ) -> list[SearchHit]:
        if routed.query_type in (QueryType.CONVERSATION_HISTORY, QueryType.USER_MEMORY):
            return []
        # 1. route: best sections by summary
        flt = visibility.search_filter(kind="summary")
        if document_ids:
            flt = flt.model_copy(
                update={"must_any": {**flt.must_any, "document_id": list(document_ids)}}
            )
        routes = await self._search(routed.query, flt, limit=self.sections)
        section_nodes = [str(h.payload.get("node_id")) for h in routes if h.payload.get("node_id")]
        if not section_nodes:
            return []
        # 2. descend: every node beneath the routed sections (bounded depth)
        async with self.uow_factory() as uow:
            frontier = list(section_nodes)
            scope_nodes = set(frontier)
            for _ in range(3):
                edges = await uow.documents.edges_from(
                    ctx.tenant_id, frontier, kinds=[ContextGraphEdge.CHILD]
                )
                frontier = [e.target_id for e in edges if e.target_id not in scope_nodes]
                if not frontier:
                    break
                scope_nodes.update(frontier)
        # 3. search only inside those nodes (store-side filter, visibility still applied)
        chunk_flt = visibility.search_filter(kind="chunk").model_copy(
            update={
                "must_any": {
                    **visibility.search_filter().must_any,
                    "node_id": sorted(scope_nodes)[:512],
                }
            }
        )
        hits = await self._search(routed.query, chunk_flt, limit=self.limit)
        return _tag(hits, self.name)


class RaptorRetriever:
    name = "raptor"

    def __init__(self, store: SearchStore, indexer: Indexer, *, limit: int = 8) -> None:
        self.store = store
        self.indexer = indexer
        self.limit = limit

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        visibility: VisibilitySpecification,
        document_ids: Sequence[str] | None,
    ) -> list[SearchHit]:
        if routed.query_type in (QueryType.CONVERSATION_HISTORY, QueryType.USER_MEMORY):
            return []
        flt = visibility.search_filter(kind="summary")
        if document_ids:
            flt = flt.model_copy(
                update={"must_any": {**flt.must_any, "document_id": list(document_ids)}}
            )
        hits = await self.store.search_hybrid(
            self.indexer.collection(KNOWLEDGE),
            dense=await self.indexer.embedding.embed_query(routed.query),
            sparse=self.indexer.sparse.encode_query(routed.query),
            flt=flt,
            limit=self.limit,
            prefetch_limit=self.limit * 2,
        )
        return _tag(hits, self.name)


class LateInteractionRetriever:
    """ColBERT-style multivector search (needs a late-interaction model + store support)."""

    name = "late_interaction"

    def __init__(self, store: SearchStore, indexer: Indexer, encoder, *, limit: int = 20) -> None:
        self.store = store
        self.indexer = indexer
        self.encoder = encoder
        self.limit = limit

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        visibility: VisibilitySpecification,
        document_ids: Sequence[str] | None,
    ) -> list[SearchHit]:
        search_late = getattr(self.store, "search_late", None)
        if search_late is None:
            return []
        flt = visibility.search_filter(kind="chunk")
        if document_ids:
            flt = flt.model_copy(
                update={"must_any": {**flt.must_any, "document_id": list(document_ids)}}
            )
        vectors = await self.encoder.embed_query_multi(routed.query)
        hits = await search_late(self.indexer.collection(KNOWLEDGE), vectors, flt, limit=self.limit)
        return _tag(hits, self.name)
