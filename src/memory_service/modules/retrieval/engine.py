"""RetrievalEngine: authorized scope -> exact -> route -> hybrid (BM25 + dense, RRF) ->
prune -> (M9: expansion + evidence verification).

Every candidate comes out of the store already filtered by tenant + visibility keys; the
engine never sees another principal's data, so there is nothing to "filter in memory".
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from memory_service.config.constants import RetrievalSettings, derived_k
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType, Representation
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.ids import content_hash
from memory_service.domain.learning import standing_factor
from memory_service.domain.memory import CanonicalMemory, unverified_representation
from memory_service.domain.script import Script, detect_script
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.grounding.lexical import content_tokens
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES, Indexer
from memory_service.modules.retrieval.memory_queries import plan_memory_queries
from memory_service.modules.retrieval.router import QueryRouter, RoutedQuery
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.timings import Timings
from memory_service.observability.tracing import span
from memory_service.ports.search import (
    AnchoredPrefetch,
    Retriever,
    SearchHit,
    SearchStore,
    SparseVector,
    VectorName,
)
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
    query_embedding: list[float] | None = None


def is_derived(candidate: Candidate) -> bool:
    """Includes pre-migration index records without the explicit derived flag."""
    return candidate.kind == "memory" and (
        bool(candidate.payload.get("derived"))
        or candidate.payload.get("memory_type") in {"BELIEF", "ENTITY_SUMMARY"}
        or any(
            ref.get("source_type") == "memory" for ref in candidate.payload.get("source_refs", [])
        )
    )


def memory_candidate(memory: CanonicalMemory, *, retriever: str, score: float) -> Candidate:
    """Project canonical evidence identically for exact and graph retrieval."""
    return Candidate(
        record_id=memory.memory_id,
        kind="memory",
        text=memory.content,
        score=score,
        retrievers=[retriever],
        payload={
            "memory_type": memory.memory_type.value,
            "category": memory.system_metadata.get("category"),
            "provider": memory.system_metadata.get("provider"),
            "derived": bool(memory.system_metadata.get("source_revisions")),
            "source_observed_to": memory.system_metadata.get("source_observed_to"),
            "temporal_status": memory.temporal.status.value,
            "subject": memory.subject,
            "predicate": memory.predicate,
            "object": memory.object,
            "observed_at": memory.temporal.observed_at.isoformat(),
            "valid_from": memory.temporal.valid_from.isoformat()
            if memory.temporal.valid_from
            else None,
            "valid_to": memory.temporal.valid_to.isoformat() if memory.temporal.valid_to else None,
            "source_refs": [
                ref.model_dump(mode="json", exclude_none=True) for ref in memory.evidence
            ],
        },
    )


def rrf_fuse(
    lists: Sequence[Sequence[SearchHit]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float, list[str], dict[str, Any]]]:
    """Client-side reciprocal rank fusion (used when the store cannot fuse natively).

    ``weights`` scales each list's contribution, one per list, the way the store's weighted
    RRF does; absent, every list weighs 1.0.
    """
    scores: dict[str, float] = {}
    retrievers: dict[str, list[str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for position, hits in enumerate(lists):
        weight = 1.0 if weights is None else float(weights[position])
        for rank, hit in enumerate(hits):
            scores[hit.record_id] = scores.get(hit.record_id, 0.0) + weight / (k + rank + 1)
            retrievers.setdefault(hit.record_id, []).append(str(hit.retriever))
            payloads.setdefault(hit.record_id, hit.payload)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(rid, s, retrievers[rid], payloads[rid]) for rid, s in ordered]


@dataclass(frozen=True)
class QueryVectors:
    """What one query was encoded into: a vector per dense space its script is searched in,
    the sparse vector, and the script that decided the spaces."""

    dense: dict[VectorName, list[float]]
    sparse: SparseVector | None
    script: Script


#: entity anchors read off a query; a question names a handful of things at most
MAX_QUERY_ENTITIES = 6


def query_entities(query: str) -> list[str]:
    """The entities a query names, in the canonical form the index stores them in.

    A sentence-initial "What" is capitalised like a name; a candidate with no content token
    is a function word and anchors nothing.
    """
    out: list[str] = []
    for name in extract_entities(query, max_entities=MAX_QUERY_ENTITIES):
        canonical = canonical_entity(name)
        if canonical and canonical not in out and content_tokens(canonical):
            out.append(canonical)
    return out


class RetrievalEngine:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        authz: AuthorizationService,
        store: SearchStore,
        indexer: Indexer,
        *,
        settings: RetrievalSettings,
        router: QueryRouter | None = None,
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.store = store
        self.indexer = indexer
        self.cfg = settings
        self.router = router or QueryRouter(semantic_graph=settings.semantic_graph)
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
        query_embedding: tuple[str, list[float]] | None = None,
    ) -> RetrievalResult:
        explicit_limit = limit is not None
        limit = limit or self.cfg.final_k
        selected_documents = frozenset(document_ids or ())
        selected_kinds = frozenset(kinds)
        # Bound the query before anything expensive touches it. See
        # RetrievalSettings.max_query_chars: the embedding models truncate at 512 tokens
        # anyway, so the text past it was only ever paid for.
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
            timings = Timings()
            diagnostics["timings_ms"] = timings.ms
            # The encoder is the floor of every query (178 ms mean on this box). Started
            # here, the visibility lookup, the collection check, the exact lookups and the
            # graph traversal all run under it instead of after it; what is awaited later
            # is only whatever is left of it.
            reused = (
                {self.indexer.spaces.primary_space.name: query_embedding[1]}
                if query_embedding and query_embedding[0] == search_text
                else None
            )
            encode_task = asyncio.ensure_future(self._encode(search_text, known=reused))
            encoded: QueryVectors | None = None
            prefetched: dict[str, asyncio.Future[Any]] = {}
            try:
                if visibility is None:
                    with timings.stage("visibility"):
                        async with self.uow_factory() as uow:
                            visibility = await self.authz.visibility(ctx, revisions=uow.revisions)
                if self.indexer.fingerprint not in self._ensured:
                    # a fresh deployment answers "nothing yet" before anything was indexed,
                    # instead of failing on a missing collection
                    await self.indexer.ensure_collections()
                    self._ensured.add(self.indexer.fingerprint)
                # a post-stage that depends only on the route and the scope (the graph
                # traversal) starts now and is awaited when its turn comes
                for name, stage in self.post_stages.items():
                    starter = getattr(stage, "prefetch", None)
                    if starter is not None and (task := starter(ctx, routed, visibility)):
                        prefetched[name] = task
                candidates: list[Candidate] = []
                # 1. exact identifiers (O(1)/O(log n) lookups, no ranking)
                if routed.identifiers and self.cfg.exact:
                    with timings.stage("exact"):
                        candidates.extend(
                            _within_selection(
                                await self._exact(ctx, routed.identifiers, visibility),
                                selected_documents,
                                selected_kinds,
                            )
                        )
                    diagnostics["exact_hits"] = len(candidates)
                    if not candidates:
                        diagnostics["exact_fallback"] = True
                # 2. hybrid lexical + dense with native RRF inside the store
                #
                # An identifier lookup that found nothing has to fall back to ranked search,
                # and that fallback has to include memories. The router sets
                # needs_memories=False for EXACT_IDENTIFIER — reasonable when the lookup
                # succeeds, since an exact hit beats anything ranking could offer — but it
                # was applied to the fallback as well. So "what about SKU-88?" searched
                # everything except memories and returned nothing, while the vaguer "which
                # products are discontinued" found the very same memory. Asking about a
                # specific thing is the most natural question there is; it must not be the
                # one that fails.
                exact_lookup_found_nothing = (
                    routed.query_type is QueryType.EXACT_IDENTIFIER and not candidates
                )
                if routed.query_type is not QueryType.EXACT_IDENTIFIER or not candidates:
                    wanted = list(kinds)
                    if routed.needs_summaries and "chunk" in wanted and "summary" not in wanted:
                        wanted.append("summary")
                    wanted = [
                        kind
                        for kind in wanted
                        if (kind != "chunk" or routed.needs_knowledge)
                        and (
                            kind != "memory" or routed.needs_memories or exact_lookup_found_nothing
                        )
                    ]
                    with timings.stage("encode"):
                        encoded = await encode_task
                    diagnostics["query_script"] = encoded.script.value
                    # one store round trip per kind, concurrently; `wanted` order is kept so
                    # the interleave below is what it was when they ran one after another
                    with timings.stage("search"):
                        per_kind = await asyncio.gather(
                            *(
                                self._search_kind(
                                    ctx,
                                    routed,
                                    search_text,
                                    visibility,
                                    kind=kind,
                                    document_ids=document_ids,
                                    encoded=encoded,
                                    diagnostics=diagnostics,
                                )
                                for kind in wanted
                            )
                        )
                    # interleave the per-kind lists by rank so a long document result list
                    # can never crowd out the memories (or summaries) before the cut
                    for group in itertools.zip_longest(*per_kind):
                        candidates.extend(c for c in group if c is not None)
                    diagnostics["fused_candidates"] = len(candidates)
                # 3. prune to fused_k, keeping exact hits first; collapse exact-duplicate
                #    texts (copies of the same document) so they cannot crowd out other
                #    evidence
                candidates = _within_selection(candidates, selected_documents, selected_kinds)
                if (
                    self.cfg.memory_entity_search
                    and routed.query_type is not QueryType.EXACT_IDENTIFIER
                    and not selected_documents
                    and not explicit_limit
                    and routed.signals.get("multi_hop")
                    and candidates
                    and all(c.kind == "memory" for c in candidates)
                ):
                    with timings.stage("memory_entity_search"):
                        candidates = await self._entity_search(
                            routed.query, candidates, visibility, diagnostics
                        )
                pool_limit = max(self.cfg.fused_k, limit)
                if (
                    not explicit_limit
                    and not selected_documents
                    and self.cfg.memory_recall_k > limit
                    and candidates
                    and all(c.kind == "memory" for c in candidates)
                ):
                    limit = self.cfg.memory_recall_k
                    pool_limit = max(pool_limit, derived_k(limit))
                    diagnostics["memory_recall_k"] = limit
                before = len(candidates)
                candidates = _dedup(candidates)[:pool_limit]
                candidates = await self._validate_derived(ctx, candidates, visibility)
                checked_derived = {c.record_id for c in candidates if is_derived(c)}
                if before != len(candidates):
                    diagnostics["duplicates_collapsed"] = before - len(candidates)
                # list(), not an alias. ``unused`` below is everything in the pool that the
                # cut dropped, and it feeds EvidenceReport.unused, which the grounding cascade
                # scans for contradictions. Today the only rebinding between here and that
                # computation is the ``candidates[:limit]`` in the else-branch below, so an
                # alias happens to work; delete or move that one line and ``kept`` becomes the
                # whole pool, ``unused`` goes silently empty, and the cascade stops seeing
                # contradictions with every test still green.
                pool = list(candidates)
                candidates = diverse_head(
                    candidates,
                    limit=limit,
                    per_document=self.cfg.max_chunks_per_document
                    if len(selected_documents) != 1
                    else 0,
                )
                kept = {c.record_id for c in candidates}
                unused = [c for c in pool if c.record_id not in kept][:UNUSED_MAX]
                if unused:
                    # retrieved but ranked out: the grounding cascade scans these for
                    # contradictions
                    diagnostics["unused"] = [
                        {"record_id": c.record_id, "kind": c.kind, "text": c.text}
                        for c in unused
                        if not unverified_representation(c.payload)
                    ]
                # 5. strategy hooks (graph M8, expansion/verification M9)
                for name, stage in self.post_stages.items():
                    extra = {"prefetched": prefetched.pop(name)} if name in prefetched else {}
                    with timings.stage(name):
                        candidates = await stage(
                            ctx, routed, candidates, visibility, diagnostics, **extra
                        )
                    # Source selectors constrain graph facts and companions before any
                    # later stage consumes them, just as they constrain ranked search.
                    candidates = _within_selection(candidates, selected_documents, selected_kinds)
                    diagnostics.setdefault("stages", []).append(name)
                if self.post_stages:
                    candidates = _cap_evidence(candidates, limit)
                    candidates = await self._validate_derived(
                        ctx, candidates, visibility, checked=checked_derived
                    )
                candidates = await self._expand_derived_sources(
                    ctx, candidates, visibility, diagnostics
                )
                candidates = _within_selection(candidates, selected_documents, selected_kinds)
            finally:
                # an exact hit never needs the encoding; a stage that raised never consumed
                # its prefetch - neither may outlive the request or log as never retrieved
                for task in (encode_task, *prefetched.values()):
                    _discard(task)
        return RetrievalResult(
            routed=routed,
            candidates=candidates,
            visibility=visibility,
            diagnostics=diagnostics,
            # the primary space's vector, for the semantic cache to remember the query by
            query_embedding=(
                encoded.dense.get(self.indexer.spaces.primary_space.name) if encoded else None
            ),
        )

    async def _expand_derived_sources(
        self,
        ctx: MemoryExecutionContext,
        candidates: list[Candidate],
        visibility: VisibilitySpecification,
        diagnostics: dict[str, Any],
    ) -> list[Candidate]:
        """Bounded one-hop evidence fetch; no model, graph walk or per-source query."""
        present = {c.record_id for c in candidates}
        wanted: dict[str, Candidate] = {}
        for candidate in candidates:
            if not is_derived(candidate):
                continue
            for ref in candidate.payload.get("source_refs", []):
                source_id = ref.get("source_id")
                if (
                    ref.get("source_type") == "memory"
                    and source_id not in present
                    and source_id not in wanted
                    and len(wanted) < self.cfg.derived_source_k
                ):
                    wanted[source_id] = candidate
        if not wanted:
            return candidates
        async with self.uow_factory() as uow:
            sources = await uow.memories.get_many(ctx.tenant_id, list(wanted))
        additions = []
        for memory in sources:
            if (
                memory.temporal.status.value != "CURRENT"
                or memory.system_metadata.get("source_revisions")
                or not visibility.allows(
                    memory.tenant_id, memory.system_metadata.get("visibility_keys", [])
                )
            ):
                continue
            parent = wanted[memory.memory_id]
            additions.append(
                replace(
                    memory_candidate(memory, retriever="derived_source", score=parent.score),
                    expanded_from=parent.record_id,
                    expansion_edge="DERIVED_SOURCE",
                )
            )
        diagnostics["derived_sources"] = len(additions)
        return [*candidates, *additions]

    async def _validate_derived(
        self,
        ctx: MemoryExecutionContext,
        candidates: list[Candidate],
        visibility: VisibilitySpecification,
        *,
        checked: set[str] | None = None,
    ) -> list[Candidate]:
        checked = checked or set()
        ids = {c.record_id for c in candidates if is_derived(c) and c.record_id not in checked}
        if not ids:
            return candidates
        async with self.uow_factory() as uow:
            current = {
                m.memory_id: m
                for m in await uow.memories.get_many(ctx.tenant_id, sorted(ids))
                if m.temporal.status.value == "CURRENT"
                and visibility.allows(m.tenant_id, m.system_metadata.get("visibility_keys", []))
            }
        return [
            replace(c, text=current[c.record_id].content, payload={**c.payload, "derived": True})
            if c.record_id in current
            else c
            for c in candidates
            if c.record_id not in ids or c.record_id in current
        ]

    async def _search_kind(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        search_text: str,
        visibility: VisibilitySpecification,
        *,
        kind: str,
        document_ids: Sequence[str] | None,
        encoded: QueryVectors,
        diagnostics: dict[str, Any],
    ) -> list[Candidate]:
        """Ranked candidates of one kind: the store's hybrid search, fused with any extra
        chunk retrievers. One of these runs per wanted kind, concurrently."""
        hits = await self._hybrid(
            search_text, visibility, kind=kind, document_ids=document_ids, encoded=encoded
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
                SearchHit(record_id=rid, score=s, retriever=Retriever.FUSION, payload=p)
                for rid, s, _, p in fused
            ]
            retrievers_of = {rid: names for rid, _, names, _ in fused}
        candidates = [
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
        return by_standing(candidates) if kind == "memory" else candidates

    async def _entity_search(
        self,
        query: str,
        candidates: list[Candidate],
        visibility: VisibilitySpecification,
        diagnostics: dict[str, Any],
    ) -> list[Candidate]:
        plan = plan_memory_queries(query, candidates)
        if plan is None:
            return candidates
        detail: dict[str, Any] = {"subjects": list(plan.subjects), "topic": plan.topic}
        diagnostics["memory_entity_search"] = detail
        tasks: list[asyncio.Task[list[SearchHit]]] = []
        try:
            async with asyncio.timeout(self.cfg.memory_entity_search_timeout_ms / 1000):
                encoded = await self._encode(plan.topic)
                tasks = [
                    asyncio.create_task(
                        self._hybrid(
                            plan.topic,
                            visibility,
                            kind="memory",
                            document_ids=None,
                            encoded=encoded,
                            subject=subject,
                        )
                    )
                    for subject in plan.subjects
                ]
                extra = await asyncio.gather(*tasks)
        except (TimeoutError, DependencyUnavailable) as exc:
            detail["fallback"] = type(exc).__name__
            return candidates
        finally:
            for task in tasks:
                _discard(task)
        original = {c.record_id: c for c in candidates}
        base = [
            SearchHit(
                record_id=c.record_id,
                score=c.score,
                retriever=Retriever.FUSION,
                payload=c.payload,
            )
            for c in candidates
        ]
        # The original question gets twice the weight of each actor view. A topic view
        # supplements the original intent, including facts spoken by somebody else.
        fused = rrf_fuse([base, base, *extra], k=self.cfg.rrf_k)
        detail["new_candidates"] = sum(rid not in original for rid, *_ in fused)
        return [
            replace(original[rid], score=score)
            if rid in original
            else Candidate(
                record_id=rid,
                kind="memory",
                text=str(payload.get("text", "")),
                score=score,
                retrievers=["entity_topic"],
                payload=payload,
            )
            for rid, score, _, payload in fused
        ]

    async def _expand_query(self, query: str) -> QueryExpansion | None:
        """Model-assisted routing + lexical expansion when no rule fired. The original query
        is kept for ranking and verification; the terms only widen the hybrid search."""
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
                        out.append(memory_candidate(m, retriever="exact", score=1.0))
        for prefix, lookup in self.exact_lookups.items():
            matching = [i for i in identifiers if i.startswith(prefix)]
            if matching:
                out.extend(await lookup(ctx, matching, visibility))
        return out

    async def _encode(
        self, query: str, *, known: dict[VectorName, list[float]] | None = None
    ) -> QueryVectors:
        """Encode the query once for every kind that will be searched.

        This used to live inside ``_hybrid``, which is called once per kind — so a query
        wanting both "memory" and "chunk" paid the same embedding forward pass twice. That
        pass is the dominant cost of a search: measured on this box, dense off is p50
        58.6 ms and dense on is p50 532.9 ms over an identical corpus, and the encoder alone
        is 178 ms mean. Hoisting it out is a pure refactor with no behavioural change.

        The query's script decides which dense spaces are encoded and searched: the English
        specialist only sees Latin-script text, so a Cyrillic or Thai question pays one
        encode, not two. The spaces it does need are encoded concurrently.
        """
        script = detect_script(query)
        dense = (
            await self.indexer.spaces.embed_query(query, script=script, known=known)
            if self.cfg.dense
            else {}
        )
        sparse = self.indexer.sparse.encode_query(query) if self.cfg.bm25 else None
        return QueryVectors(dense=dense, sparse=sparse, script=script)

    def _anchors(self, query: str, encoded: QueryVectors, *, kind: str) -> list[AnchoredPrefetch]:
        """The entity prefetch: memories sharing an entity with the query, as their own RRF
        list, searched with the vector the query already has for the primary space."""
        primary = self.indexer.spaces.primary_space.name
        if not (self.cfg.entity_prefetch and kind == "memory" and primary in encoded.dense):
            return []
        entities = query_entities(query)
        return (
            [AnchoredPrefetch(vector=primary, must_any={"entities": entities})] if entities else []
        )

    async def _hybrid(
        self,
        query: str,
        visibility: VisibilitySpecification,
        *,
        kind: str,
        document_ids: Sequence[str] | None,
        encoded: QueryVectors | None = None,
        subject: str | None = None,
    ) -> list[SearchHit]:
        collection = self.indexer.collection(MEMORIES if kind == "memory" else KNOWLEDGE)
        flt = visibility.search_filter(kind=kind)
        if kind == "memory":
            flt = flt.model_copy(update={"must": {**flt.must, "current": True}})
            if subject is not None:
                flt = flt.model_copy(update={"must": {**flt.must, "subject": subject}})
        if document_ids:
            flt = flt.model_copy(
                update={"must_any": {**flt.must_any, "document_id": list(document_ids)}}
            )
        vectors = encoded if encoded is not None else await self._encode(query)
        memory_depth = derived_k(self.cfg.memory_recall_k) if kind == "memory" else 0
        return await self.store.search_hybrid(
            collection,
            dense=vectors.dense,
            sparse=vectors.sparse,
            flt=flt,
            limit=max(self.cfg.fused_k, memory_depth),
            prefetch_limit=max(self.cfg.prefetch_k, memory_depth),
            rrf_k=self.cfg.hybrid_rrf_k,
            weights=self.cfg.hybrid_weights,
            anchors=self._anchors(query, vectors, kind=kind),
        )


def by_standing(candidates: list[Candidate]) -> list[Candidate]:
    """Memories re-scored by their standing - confidence and reinforcement, which feedback and
    restatement move - within a bounded factor, and re-sorted (stable: ties keep the fused
    order). A memory's standing reorders near-ties; it never outweighs relevance."""
    for c in candidates:
        c.score *= standing_factor(c.payload.get("confidence"), c.payload.get("reinforcement"))
    return sorted(candidates, key=lambda c: -c.score)


def diverse_head(candidates: list[Candidate], *, limit: int, per_document: int) -> list[Candidate]:
    """O(n) soft document cap, stable within each tier, with no loss of result capacity.

    Diversity is for primary document discovery. Explicit identifiers and companions
    retain their positions; a single selected document bypasses this at the call site.
    """
    if per_document <= 0:
        return candidates[:limit]
    counts: dict[str, int] = {}
    preferred: list[Candidate] = []
    overflow: list[Candidate] = []
    for candidate in candidates:
        document = candidate.payload.get("document_id")
        if (
            candidate.kind != "chunk"
            or not document
            or candidate.expansion_edge
            or "exact" in candidate.retrievers
        ):
            preferred.append(candidate)
        elif counts.get(document, 0) < per_document:
            counts[document] = counts.get(document, 0) + 1
            preferred.append(candidate)
        else:
            overflow.append(candidate)
        if len(preferred) >= limit:
            return preferred
    return preferred + overflow[: limit - len(preferred)]


def _within_selection(
    candidates: list[Candidate], documents: frozenset[str], kinds: frozenset[str]
) -> list[Candidate]:
    """Apply the same source contract to every retrieval path in linear time.

    Summaries belong to documents. Relations accompany their selected source; a
    relation without source lineage is eligible only for an unrestricted source query.
    """
    unrestricted = {"chunk", "memory"} <= kinds
    out: list[Candidate] = []
    for candidate in candidates:
        payload = candidate.payload
        if documents and payload.get("document_id") not in documents:
            continue
        allowed = candidate.kind in kinds
        if candidate.kind == "summary":
            allowed = allowed or "chunk" in kinds
        elif candidate.kind == "fact":
            allowed = (
                allowed
                or unrestricted
                or ("chunk" in kinds and bool(payload.get("document_id")))
                or ("memory" in kinds and bool(payload.get("memory_id")))
            )
        if allowed:
            out.append(candidate)
    return out


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


def _discard(task: asyncio.Future[Any]) -> None:
    """Drop a task the request no longer needs: cancel it if it is still running, and if it
    already failed, take the exception so asyncio does not log it as never retrieved."""
    if not task.done():
        task.cancel()
    elif not task.cancelled():
        task.exception()


def _dedup_context(candidate: Candidate) -> tuple:
    """Equal words need equal attribution before memory evidence can be collapsed."""
    if candidate.kind != "memory":
        return (candidate.kind,)
    payload = candidate.payload
    refs = payload.get("source_refs") or []
    sources = tuple(
        sorted(
            (ref.get("source_type", ""), ref.get("source_id", ""))
            for ref in refs
            if ref.get("source_type") and ref.get("source_id")
        )
    )
    # Legacy/ephemeral hits without provenance cannot establish the same event.
    if not sources or len(sources) != len(refs):
        return (candidate.kind, candidate.record_id)
    subject = payload.get("subject")
    if not subject or (isinstance(subject, str) and subject.startswith(("thread:", "workspace:"))):
        # Native event/task extraction uses scope identifiers when no actor is parsed.
        # They do not name a different speaker from the same original observation.
        subject = payload.get("owner_principal")
        if not subject:
            return (candidate.kind, candidate.record_id)
    return (
        candidate.kind,
        subject,
        payload.get("owner_principal"),
        payload.get("observed_at"),
        sources,
    )


def _dedup(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Collapse representations of the same evidence, preserving speaker and event.

    Other text collapses within its representation kind. Memories compare within a provenance
    group: equal words from different people, dates or source turns remain distinct.
    Normalization is linear in input text; containment costs O(sum(group_size**2)),
    bounded by the retrieval pool, rather than comparing every memory with every other.
    """
    out: list[Candidate] = []
    aliases: dict[tuple[str, str], Candidate] = {}
    by_hash: dict[tuple, Candidate] = {}
    groups: dict[tuple, list[tuple[str, Candidate]]] = {}
    with stage_seconds.labels("retrieval.dedup").time():
        for candidate in candidates:
            identity = (candidate.kind, candidate.record_id)
            twin = aliases.get(identity)
            context = _dedup_context(candidate)
            digest = candidate.payload.get("text_hash") or (
                content_hash(candidate.text) if candidate.text else None
            )
            hash_key = (context, digest)
            if twin is None and digest:
                twin = by_hash.get(hash_key)
            norm = _normalised(candidate.text) if COLLAPSE_SUBSUMED else ""
            if twin is None and COLLAPSE_SUBSUMED:
                twin = _subsumed_by(norm, candidate.record_id, groups.get(context, ()))
            if twin is None:
                twin = candidate
                out.append(candidate)
                groups.setdefault(context, []).append((norm, candidate))
            else:
                if twin.record_id != candidate.record_id and identity not in aliases:
                    twin.payload.setdefault("duplicates", []).append(candidate.record_id)
                twin.score = max(twin.score, candidate.score)
                twin.retrievers = sorted(set(twin.retrievers) | set(candidate.retrievers))
            aliases[identity] = twin
            if digest:
                by_hash[hash_key] = twin
    return out


#: Collapse a candidate whose text is wholly contained in one already kept. Enabled in the
#: shipped baseline; changes must be evaluated against that baseline.
#:
#: Identical-text dedup above collapses exact twins. It cannot see the shape this service
#: actually produces: since a turn is kept verbatim as well as extracted, one sentence yields
#: BOTH "Melanie prefers tea" and the turn "I prefer tea, and my name is Amit..." - different
#: text, different hash, two slots, one fact. A hundred-memory bundle can therefore carry
#: closer to fifty distinct things, and the distractor ratio behind every position argument is
#: twice as bad as the slot count suggests.
#:
#: That is not what the lost-in-the-middle work models - it assumes N distinct documents with
#: one gold among them - so the remedy is not a better position, it is fewer copies of the
#: same evidence competing for the positions there are.
#: Collapse a candidate that another already contains. Multi-hop needs evidence from two or
#: more DIFFERENT turns, and a relevance-only top-10 can be ten phrasings of one of them - on
#: the full set only 51.2% of multi_hop gold evidence reaches the head against 71.4% for
#: temporal. Dropping a subsumed candidate frees a slot for the hop that is missing.
COLLAPSE_SUBSUMED = True

#: Below this, containment is coincidence rather than subsumption ("tea" inside anything).
SUBSUMPTION_MIN_CHARS = 25


def _subsumed_by(
    text: str, record_id: str, kept: Sequence[tuple[str, Candidate]]
) -> Candidate | None:
    """An already-kept candidate whose text contains this one's, or None.

    Containment only, not similarity: the longer text carries everything the shorter one says,
    so dropping the shorter loses no information from the bundle. The reverse - a kept short
    fact and a longer arrival that subsumes it - is deliberately left alone, because replacing
    an accepted candidate would reorder a ranking the caller is entitled to.

    Takes ``text`` already normalised, and ``kept`` as ``(normalised body, candidate)`` pairs,
    because the caller is walking the same kept list for every candidate and normalising a
    body once per comparison rather than once per body is the whole cost of this scan.
    """
    if len(text) < SUBSUMPTION_MIN_CHARS:
        return None
    for body, other in kept:
        if other.record_id == record_id:
            continue
        if len(body) > len(text) and text in body:
            return other
    return None


def _normalised(text: str) -> str:
    return " ".join((text or "").lower().split())
