"""RetrievalEngine: authorized scope -> exact -> route -> hybrid (BM25 + dense, RRF) ->
prune -> bounded CPU rerank -> (M9: expansion + evidence verification).

Every candidate comes out of the store already filtered by tenant + visibility keys; the
engine never sees another principal's data, so there is nothing to "filter in memory".
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from memory_service.config.settings import RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType, Representation
from memory_service.domain.ids import content_hash
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES, Indexer
from memory_service.modules.retrieval.router import QueryRouter, RoutedQuery
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.models import Reranker
from memory_service.ports.search import SearchHit, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

_QUERY_TYPES = {t.value: t for t in QueryType}
_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": list(_QUERY_TYPES)},
        "terms": {"type": "array", "items": {"type": "string"}},
        "identifiers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query_type", "terms", "identifiers"],
    "additionalProperties": False,
}
_EXPANSION_SYSTEM = (
    "You classify a search query against a memory and document store and propose search "
    "terms. Query types: EXACT_IDENTIFIER (lookup of an explicit id), CONVERSATION_HISTORY "
    "(what was said earlier in this chat), USER_MEMORY (the user's own preferences or facts), "
    "DECISION (why something was decided), DOCUMENT_LOCAL (a specific place in a document), "
    "DOCUMENT_MULTI_HOP (needs several passages combined), ENTITY_RELATION (who or what "
    "relates to whom), TEMPORAL (time-bound or how something changed), GLOBAL_SUMMARY "
    "(overview of a whole document), GENERAL_SEMANTIC (anything else). Return JSON only: "
    '{"query_type": ..., "terms": up to 6 short synonyms or closely related terms not already '
    'in the query, "identifiers": explicit ids mentioned in the query (usually empty)}.'
)
_MAX_TERMS = 6
_MAX_IDENTIFIERS = 4
UNUSED_MAX = 8


@dataclass
class QueryExpansion:
    query_type: QueryType | None
    terms: list[str]
    identifiers: list[str]


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
        return {
            "chunk": Representation.CHUNK,
            "memory": Representation.MEMORY,
            "fact": Representation.RELATION,
            "summary": Representation.SUMMARY,
        }.get(self.kind, Representation.CHUNK)


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
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.store = store
        self.indexer = indexer
        self.reranker = reranker
        self.cfg = settings
        self.rerank_k = rerank_k
        self.router = router or QueryRouter()
        self.assist = assist or LLMAssist.disabled()
        # pipeline stages appended by later milestones (graph M8, expansion/verification M9)
        self.post_stages: dict[str, Any] = {}
        # extra retrievers (M10 strategies): their hit lists are RRF-fused with the hybrid list
        self.retrievers: dict[str, Any] = {}
        # exact lookups by id prefix (graph facts M8)
        self.exact_lookups: dict[str, Any] = {}
        self._ensured: set[str] = set()

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
        # Bound the query before anything expensive touches it. See
        # RetrievalSettings.max_query_chars: the cost of a query is paid again for every
        # cross-encoder pair, and the embedding models truncate at 512 tokens anyway.
        if len(query) > self.cfg.max_query_chars:
            log.warning(
                "retrieval.query_truncated",
                original_chars=len(query),
                kept_chars=self.cfg.max_query_chars,
            )
            query = query[: self.cfg.max_query_chars]
        has_thread = ctx.thread_id is not None
        routed = self.router.route(query, has_thread=has_thread)
        search_text = routed.query
        diagnostics: dict[str, Any] = {}
        if (
            self.assist.wants("query_expansion")
            and routed.query_type is QueryType.GENERAL_SEMANTIC
            and not any(routed.signals.values())
        ):
            expansion = await self._expand_query(routed.query)
            if expansion is not None:
                if expansion.query_type is not None:
                    routed = self.router.routed(
                        routed.query,
                        expansion.query_type,
                        identifiers=[*routed.identifiers, *expansion.identifiers],
                        signals=routed.signals,
                        has_thread=has_thread,
                    )
                if expansion.terms:
                    search_text = f"{routed.query} {' '.join(expansion.terms)}"
                diagnostics["query_expansion"] = expansion.terms
        diagnostics["query_type"] = routed.query_type.value
        diagnostics["signals"] = routed.signals
        with (
            span("retrieval", tenant_id=ctx.tenant_id, query_type=routed.query_type.value),
            stage_seconds.labels("retrieval").time(),
        ):
            if visibility is None:
                async with self.uow_factory() as uow:
                    visibility = await self.authz.visibility(ctx, revisions=uow.revisions)
            if self.indexer.fingerprint not in self._ensured:
                # a fresh deployment answers "nothing yet" before anything was indexed,
                # instead of failing on a missing collection
                await self.indexer.ensure_collections()
                self._ensured.add(self.indexer.fingerprint)
            candidates: list[Candidate] = []
            # 1. exact identifiers (O(1)/O(log n) lookups, no ranking)
            if routed.identifiers and self.cfg.exact:
                candidates.extend(await self._exact(ctx, routed.identifiers, visibility))
                diagnostics["exact_hits"] = len(candidates)
                if not candidates:
                    diagnostics["exact_fallback"] = True
            # 2. hybrid lexical + dense with native RRF inside the store
            #
            # An identifier lookup that found nothing has to fall back to ranked search, and
            # that fallback has to include memories. The router sets needs_memories=False for
            # EXACT_IDENTIFIER — reasonable when the lookup succeeds, since an exact hit beats
            # anything ranking could offer — but it was applied to the fallback as well. So
            # "what about SKU-88?" searched everything except memories and returned nothing,
            # while the vaguer "which products are discontinued" found the very same memory.
            # Asking about a specific thing is the most natural question there is; it must not
            # be the one that fails.
            exact_lookup_found_nothing = (
                routed.query_type is QueryType.EXACT_IDENTIFIER and not candidates
            )
            if routed.query_type is not QueryType.EXACT_IDENTIFIER or not candidates:
                wanted = list(kinds)
                if routed.needs_summaries and "summary" not in wanted:
                    wanted.append("summary")
                per_kind: list[list[Candidate]] = []
                encoded = await self._encode(search_text)
                for kind in wanted:
                    if kind == "chunk" and not routed.needs_knowledge:
                        continue
                    if kind == "memory" and not (
                        routed.needs_memories or exact_lookup_found_nothing
                    ):
                        continue
                    hits = await self._hybrid(
                        search_text,
                        visibility,
                        kind=kind,
                        document_ids=document_ids,
                        encoded=encoded,
                    )
                    retrievers_of: dict[str, list[str]] = {h.record_id: [h.retriever] for h in hits}
                    if kind == "chunk" and self.retrievers:
                        lists: list[Sequence[SearchHit]] = [hits]
                        for name, extra in self.retrievers.items():
                            extra_hits = await extra(ctx, routed, visibility, document_ids)
                            diagnostics.setdefault("strategies", {})[name] = len(extra_hits)
                            lists.append(extra_hits)
                        fused = rrf_fuse(lists, k=self.cfg.rrf_k)[: self.cfg.fused_k]
                        hits = [
                            SearchHit(record_id=rid, score=s, retriever="fusion", payload=p)
                            for rid, s, _, p in fused
                        ]
                        retrievers_of = {rid: names for rid, _, names, _ in fused}
                    per_kind.append(
                        [
                            Candidate(
                                record_id=h.record_id,
                                kind=str(h.payload.get("kind") or kind),
                                text=str(h.payload.get("text", "")),
                                score=h.score,
                                retrievers=retrievers_of.get(h.record_id, [h.retriever]),
                                payload=h.payload,
                            )
                            for h in hits
                        ]
                    )
                # interleave the per-kind lists by rank so a long document result list can
                # never crowd out the memories (or summaries) before the reranker sees them
                for group in itertools.zip_longest(*per_kind):
                    candidates.extend(c for c in group if c is not None)
                diagnostics["fused_candidates"] = len(candidates)
            # 3. prune to fused_k, keeping exact hits first; collapse exact-duplicate texts
            #    (copies of the same document) so they cannot crowd out other evidence
            before = len(candidates)
            candidates = _dedup(candidates)[: max(self.cfg.fused_k, limit)]
            if before != len(candidates):
                diagnostics["duplicates_collapsed"] = before - len(candidates)
            # 4. bounded CPU rerank
            pool = candidates
            if self.cfg.rerank and self.reranker is not None and len(candidates) > 1:
                candidates = await self._rerank(routed.query, candidates, limit=limit)
                diagnostics["reranked"] = True
            else:
                candidates = candidates[:limit]
            kept = {c.record_id for c in candidates}
            unused = [c for c in pool if c.record_id not in kept][:UNUSED_MAX]
            if unused:
                # retrieved but ranked out: the grounding cascade scans these for contradictions
                diagnostics["unused"] = [
                    {"record_id": c.record_id, "kind": c.kind, "text": c.text} for c in unused
                ]
            # 5. strategy hooks (graph M8, expansion/verification M9)
            for name, stage in self.post_stages.items():
                candidates = await stage(ctx, routed, candidates, visibility, diagnostics)
                diagnostics.setdefault("stages", []).append(name)
            if self.post_stages:
                candidates = _cap_evidence(candidates, limit)
        return RetrievalResult(
            routed=routed, candidates=candidates, visibility=visibility, diagnostics=diagnostics
        )

    async def _expand_query(self, query: str) -> QueryExpansion | None:
        """Model-assisted routing + lexical expansion when no rule fired. The original query
        is kept for reranking and verification; the terms only widen the hybrid search."""
        out = await self.assist.structured(
            "query_expansion",
            system=_EXPANSION_SYSTEM,
            user=f"Query: {query[:500]}",
            schema=_EXPANSION_SCHEMA,
            max_tokens=200,
        )
        if out is None:
            return None
        lowered = query.lower()
        terms: list[str] = []
        for raw in out.get("terms", []):
            term = " ".join(str(raw).split())[:48]
            if term and term.lower() not in lowered and term.lower() not in map(str.lower, terms):
                terms.append(term)
        identifiers = [i for i in (str(x).strip()[:64] for x in out.get("identifiers", [])) if i][
            :_MAX_IDENTIFIERS
        ]
        qt = _QUERY_TYPES.get(str(out.get("query_type")))
        if qt is QueryType.EXACT_IDENTIFIER and not identifiers:
            qt = None
        if qt is QueryType.GENERAL_SEMANTIC:
            qt = None
        return QueryExpansion(query_type=qt, terms=terms[:_MAX_TERMS], identifiers=identifiers)

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
        summary_nodes = [i[4:] for i in identifiers if i.startswith("sum_")]
        if summary_nodes:
            async with self.uow_factory() as uow:
                summaries = await uow.documents.node_summaries(ctx.tenant_id, summary_nodes)
                nodes = await uow.documents.get_nodes(ctx.tenant_id, list(summaries))
                for n in nodes:
                    keys = await uow.documents.visibility_keys(ctx.tenant_id, n.document_id)
                    if visibility.allows(n.tenant_id, keys):
                        out.append(
                            Candidate(
                                record_id=f"sum_{n.node_id}",
                                kind="summary",
                                text=summaries[n.node_id],
                                score=1.0,
                                retrievers=["exact"],
                                payload={
                                    "document_id": n.document_id,
                                    "node_id": n.node_id,
                                    "page": n.page_start,
                                    "section_path": n.section_path,
                                },
                            )
                        )
        memory_ids = [i for i in identifiers if i.startswith("mem_")]
        if memory_ids:
            async with self.uow_factory() as uow:
                for m in await uow.memories.get_many(ctx.tenant_id, memory_ids):
                    keys = m.system_metadata.get("visibility_keys", [])
                    if visibility.allows(m.tenant_id, keys):
                        out.append(
                            Candidate(
                                record_id=m.memory_id,
                                kind="memory",
                                text=m.content,
                                score=1.0,
                                retrievers=["exact"],
                                payload={
                                    "memory_type": m.memory_type.value,
                                    "temporal_status": m.temporal.status.value,
                                    "subject": m.subject,
                                    "predicate": m.predicate,
                                    "object": m.object,
                                    "observed_at": m.temporal.observed_at.isoformat(),
                                },
                            )
                        )
        for prefix, lookup in self.exact_lookups.items():
            matching = [i for i in identifiers if i.startswith(prefix)]
            if matching:
                out.extend(await lookup(ctx, matching, visibility))
        return out

    async def _encode(self, query: str) -> tuple[list[float] | None, Any]:
        """Encode the query once for every kind that will be searched.

        This used to live inside ``_hybrid``, which is called once per kind — so a query
        wanting both "memory" and "chunk" paid the same embedding forward pass twice. That
        pass is the dominant cost of a search: measured on this box, dense off is p50
        58.6 ms and dense on is p50 532.9 ms over an identical corpus, and the encoder alone
        is 178 ms mean. Hoisting it out is a pure refactor with no behavioural change.
        """
        dense = await self.indexer.embedding.embed_query(query) if self.cfg.dense else None
        sparse = self.indexer.sparse.encode_query(query) if self.cfg.bm25 else None
        return dense, sparse

    async def _hybrid(
        self,
        query: str,
        visibility: VisibilitySpecification,
        *,
        kind: str,
        document_ids: Sequence[str] | None,
        encoded: tuple[list[float] | None, Any] | None = None,
    ) -> list[SearchHit]:
        collection = self.indexer.collection(MEMORIES if kind == "memory" else KNOWLEDGE)
        flt = visibility.search_filter(kind=kind)
        if kind == "memory":
            flt = flt.model_copy(update={"must": {**flt.must, "current": True}})
        if document_ids:
            flt = flt.model_copy(
                update={"must_any": {**flt.must_any, "document_id": list(document_ids)}}
            )
        dense, sparse = encoded if encoded is not None else await self._encode(query)
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


def _cap_evidence(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Keep at most ``limit`` *ranked* evidence items (chunks/memories) after post-stages.
    Expansions, escalated companions, facts and summaries ride along uncounted: they are
    bounded by their own budgets and exist precisely to complete the ranked evidence."""
    out: list[Candidate] = []
    evidence = 0
    for c in candidates:
        if c.kind in ("chunk", "memory") and c.expansion_edge is None:
            if evidence >= limit:
                continue
            evidence += 1
        out.append(c)
    return out


def _dedup(candidates: list[Candidate]) -> list[Candidate]:
    """Merge repeated record ids and collapse identical texts onto the first occurrence.

    Collapsing used to apply to chunks only — ``if c.kind == "chunk"`` — because memories
    carry no ``text_hash`` in their search payload, so there was nothing to group them by.
    The effect was that identical memories never collapsed at all. Measured on a live
    bundle: eleven memory items with **two** distinct texts, six copies of one sentence and
    five of another, crowding out every other piece of evidence. Hashing the text here
    instead of trusting a payload field fixes it for data already indexed, with no reindex.

    Candidates arrive ranked, so the first occurrence is the best-scoring one; the rest are
    recorded as ``duplicates`` rather than discarded silently. Every twin already passed the
    store's visibility filter, so collapsing cannot widen what this caller may see.
    """
    seen: dict[str, Candidate] = {}
    by_hash: dict[str, Candidate] = {}
    for c in candidates:
        if c.record_id in seen:
            existing = seen[c.record_id]
            existing.retrievers = sorted(set(existing.retrievers) | set(c.retrievers))
            existing.score = max(existing.score, c.score)
            continue
        h = c.payload.get("text_hash") or (content_hash(c.text) if c.text else None)
        if h:
            twin = by_hash.get(h)
            if twin is not None:
                twin.payload.setdefault("duplicates", []).append(c.record_id)
                twin.score = max(twin.score, c.score)
                twin.retrievers = sorted(set(twin.retrievers) | set(c.retrievers))
                continue
            by_hash[h] = c
        seen[c.record_id] = c
    return list(seen.values())
