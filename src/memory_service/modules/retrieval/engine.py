"""RetrievalEngine: authorized scope -> exact -> route -> hybrid (BM25 + dense, RRF) ->
prune -> bounded CPU rerank -> (M9: expansion + evidence verification).

Every candidate comes out of the store already filtered by tenant + visibility keys; the
engine never sees another principal's data, so there is nothing to "filter in memory".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from memory_service.config.settings import RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType, Representation
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES, Indexer
from memory_service.modules.retrieval.router import QueryRouter, RoutedQuery
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.models import Reranker
from memory_service.ports.search import SearchHit, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)


@dataclass
class Candidate:
    record_id: str
    kind: str  # chunk | memory
    text: str
    score: float
    retrievers: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
    expanded_from: str | None = None
    expansion_edge: str | None = None

    @property
    def representation(self) -> Representation:
        return Representation.CHUNK if self.kind == "chunk" else Representation.MEMORY


@dataclass
class RetrievalResult:
    routed: RoutedQuery
    candidates: list[Candidate]
    visibility: VisibilitySpecification
    diagnostics: dict[str, Any] = field(default_factory=dict)


def rrf_fuse(
    lists: Sequence[Sequence[SearchHit]], *, k: int = 60
) -> list[tuple[str, float, list[str], dict[str, Any]]]:
    """Client-side reciprocal rank fusion (used when the store cannot fuse natively)."""
    scores: dict[str, float] = {}
    retrievers: dict[str, list[str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for hits in lists:
        for rank, hit in enumerate(hits):
            scores[hit.record_id] = scores.get(hit.record_id, 0.0) + 1.0 / (k + rank + 1)
            retrievers.setdefault(hit.record_id, []).append(hit.retriever)
            payloads.setdefault(hit.record_id, hit.payload)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(rid, s, retrievers[rid], payloads[rid]) for rid, s in ordered]


class RetrievalEngine:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        authz: AuthorizationService,
        store: SearchStore,
        indexer: Indexer,
        reranker: Reranker | None,
        *,
        settings: RetrievalSettings,
        rerank_k: int = 20,
        router: QueryRouter | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.store = store
        self.indexer = indexer
        self.reranker = reranker
        self.cfg = settings
        self.rerank_k = rerank_k
        self.router = router or QueryRouter()
        # pipeline stages appended by later milestones (graph M8, expansion/verification M9, M10)
        self.post_stages: dict[str, Any] = {}
        # exact lookups by id prefix (memories M7)
        self.exact_lookups: dict[str, Any] = {}

    async def retrieve(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        *,
        limit: int | None = None,
        kinds: Sequence[str] = ("chunk", "memory"),
        document_ids: Sequence[str] | None = None,
        visibility: VisibilitySpecification | None = None,
    ) -> RetrievalResult:
        limit = limit or self.cfg.final_k
        routed = self.router.route(query, has_thread=ctx.thread_id is not None)
        diagnostics: dict[str, Any] = {
            "query_type": routed.query_type.value,
            "signals": routed.signals,
        }
        with (
            span("retrieval", tenant_id=ctx.tenant_id, query_type=routed.query_type.value),
            stage_seconds.labels("retrieval").time(),
        ):
            if visibility is None:
                async with self.uow_factory() as uow:
                    visibility = await self.authz.visibility(ctx, revisions=uow.revisions)
            candidates: list[Candidate] = []
            # 1. exact identifiers (O(1)/O(log n) lookups, no ranking)
            if routed.identifiers and self.cfg.exact:
                candidates.extend(await self._exact(ctx, routed.identifiers, visibility))
                diagnostics["exact_hits"] = len(candidates)
            # 2. hybrid lexical + dense with native RRF inside the store
            if routed.query_type is not QueryType.EXACT_IDENTIFIER or not candidates:
                for kind in kinds:
                    if kind == "chunk" and not routed.needs_knowledge:
                        continue
                    if kind == "memory" and not routed.needs_memories:
                        continue
                    hits = await self._hybrid(
                        routed.query, visibility, kind=kind, document_ids=document_ids
                    )
                    for h in hits:
                        candidates.append(
                            Candidate(
                                record_id=h.record_id,
                                kind=kind,
                                text=str(h.payload.get("text", "")),
                                score=h.score,
                                retrievers=[h.retriever],
                                payload=h.payload,
                            )
                        )
                diagnostics["fused_candidates"] = len(candidates)
            # 3. prune to fused_k, keeping exact hits first
            candidates = _dedup(candidates)[: max(self.cfg.fused_k, limit)]
            # 4. bounded CPU rerank
            if self.cfg.rerank and self.reranker is not None and len(candidates) > 1:
                candidates = await self._rerank(routed.query, candidates, limit=limit)
                diagnostics["reranked"] = True
            else:
                candidates = candidates[:limit]
            # 5. strategy hooks (graph M8, expansion/verification M9)
            for name, stage in self.post_stages.items():
                candidates = await stage(ctx, routed, candidates, visibility, diagnostics)
                diagnostics.setdefault("stages", []).append(name)
        return RetrievalResult(
            routed=routed, candidates=candidates, visibility=visibility, diagnostics=diagnostics
        )

    async def _exact(
        self,
        ctx: MemoryExecutionContext,
        identifiers: Sequence[str],
        visibility: VisibilitySpecification,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        chunk_ids = [i for i in identifiers if i.startswith("chk_")]
        if chunk_ids:
            async with self.uow_factory() as uow:
                chunks = await uow.documents.get_chunks(ctx.tenant_id, chunk_ids)
                for c in chunks:
                    keys = await uow.documents.visibility_keys(ctx.tenant_id, c.document_id)
                    if visibility.allows(c.tenant_id, keys):
                        out.append(
                            Candidate(
                                record_id=c.chunk_id,
                                kind="chunk",
                                text=c.text,
                                score=1.0,
                                retrievers=["exact"],
                                payload={
                                    "document_id": c.document_id,
                                    "page": c.page,
                                    "section_path": c.section_path,
                                    "node_id": c.node_id,
                                },
                            )
                        )
        for prefix, lookup in self.exact_lookups.items():
            matching = [i for i in identifiers if i.startswith(prefix)]
            if matching:
                out.extend(await lookup(ctx, matching, visibility))
        return out

    async def _hybrid(
        self,
        query: str,
        visibility: VisibilitySpecification,
        *,
        kind: str,
        document_ids: Sequence[str] | None,
    ) -> list[SearchHit]:
        collection = self.indexer.collection(KNOWLEDGE if kind == "chunk" else MEMORIES)
        flt = visibility.search_filter(kind=kind)
        if document_ids:
            flt = flt.model_copy(
                update={"must_any": {**flt.must_any, "document_id": list(document_ids)}}
            )
        dense = await self.indexer.embedding.embed_query(query) if self.cfg.dense else None
        sparse = self.indexer.sparse.encode_query(query) if self.cfg.bm25 else None
        if self.cfg.fusion == "rrf":
            return await self.store.search_hybrid(
                collection,
                dense=dense,
                sparse=sparse,
                flt=flt,
                limit=self.cfg.fused_k,
                prefetch_limit=self.cfg.prefetch_k,
            )
        lists = []
        if dense is not None:
            lists.append(
                await self.store.search_dense(collection, dense, flt, limit=self.cfg.prefetch_k)
            )
        if sparse is not None:
            lists.append(
                await self.store.search_sparse(collection, sparse, flt, limit=self.cfg.prefetch_k)
            )
        return [
            SearchHit(record_id=rid, score=s, retriever="fusion", payload=p)
            for rid, s, _, p in rrf_fuse(lists, k=self.cfg.rrf_k)
        ][: self.cfg.fused_k]

    async def _rerank(
        self, query: str, candidates: list[Candidate], *, limit: int
    ) -> list[Candidate]:
        head = candidates[: self.rerank_k]
        tail = candidates[self.rerank_k :]
        with span("rerank", k=len(head)), stage_seconds.labels("rerank").time():
            results = await self.reranker.rerank(query, [c.text for c in head], top_k=len(head))  # type: ignore[union-attr]
        reranked: list[Candidate] = []
        for r in results:
            c = head[r.index]
            c.rerank_score = r.score
            reranked.append(c)
        return (reranked + tail)[:limit]


def _dedup(candidates: list[Candidate]) -> list[Candidate]:
    seen: dict[str, Candidate] = {}
    for c in candidates:
        if c.record_id in seen:
            existing = seen[c.record_id]
            existing.retrievers = sorted(set(existing.retrievers) | set(c.retrievers))
            existing.score = max(existing.score, c.score)
        else:
            seen[c.record_id] = c
    return list(seen.values())
