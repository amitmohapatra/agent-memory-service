"""Thin ContextBuilder: retrieval result + recent conversation -> bounded ContextBundle.

write / select / compress / isolate: select ranked evidence within a token budget, compress
the conversation window to its most recent turns (a rolling summary is attached by M9 when
available), isolate by scope (already enforced upstream). Bundles are cached by
scope fingerprint + revision fingerprint + query hash + retrieval config fingerprint.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal

import orjson

from memory_service.config.constants import ContextSettings, RetrievalSettings
from memory_service.config.settings import LLMUse
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EarlierChatLine,
    EvidenceReport,
    ProfileBlockView,
    ScoreKind,
    UnusedEvidence,
    WindowMessage,
    earlier_chats_section,
)
from memory_service.domain.conversation import Message
from memory_service.domain.enums import (
    EvidenceStatus,
    MessageKind,
    MessageRole,
    QueryType,
    WindowFormat,
)
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.ids import stable_key
from memory_service.domain.memory import unverified_representation
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.context.coverage import unanswered
from memory_service.modules.context.evidence import NO_SHARED_TERM
from memory_service.modules.context.handles import BundleRecords, record_of
from memory_service.modules.context.sections import (
    ContextSections,
    Pinned,
    ToolsRequest,
    within_budget,
)
from memory_service.modules.context.semantic_cache import SemanticBundleCache, semantic_key
from memory_service.modules.context.standing import standing_blocks
from memory_service.modules.context.views import full_view, prompt_view
from memory_service.modules.context.window import replay_window
from memory_service.modules.conversation.service import ConversationService
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import model_calls_allowed
from memory_service.modules.memory.ephemeral import EphemeralMemory
from memory_service.modules.retrieval import slots
from memory_service.modules.retrieval.engine import (
    UNUSED_MAX,
    Candidate,
    QueryVectors,
    RetrievalEngine,
    RetrievalResult,
)
from memory_service.modules.retrieval.periods import months_dated
from memory_service.modules.retrieval.router import (
    QueryRouter,
    RoutedQuery,
    continues,
    points_to_earlier_chat,
)
from memory_service.modules.tools.hints import asks_change, ruled
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import evidence_status_total, stage_seconds
from memory_service.observability.timings import Timings
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)


#: Standing rules and preferences that apply to the request, as profile sections (ADR 0041).
STANDING = True
#: A short question asked in a thread (at most this many words) whose own search finds nothing
#: close (its best similarity under :data:`FOLLOW_UP_FLOOR`) is searched again together with
#: the thread's previous questions, newest first (ADR 0044): "who runs that?" names nothing the search can find, the
#: question before it does. The concatenation baseline of conversational search (TREC CAsT),
#: gated on a weak first retrieval as corrective retrieval does (CRAG), so a short question on
#: a new topic, which finds its own evidence, keeps its ranking and its size. Measured on
#: pronoun follow-ups and topic changes (``benchmark/follow_ups.py``).
FOLLOW_UP_WORDS: Final = 8
#: The best dense similarity under which a short question's own search counts as weak. On the
#: frozen English encoder the benchmark's topic changes found their evidence at 0.49 to 0.77,
#: and the follow-ups whose evidence ranked past third, or was missed, peaked at 0.40 or less.
FOLLOW_UP_FLOOR: Final = 0.45
#: The most of the previous questions the search text takes.
FOLLOW_UP_CHARS: Final = 500
#: How many of the thread's newest messages are looked through for its previous question.
FOLLOW_UP_MESSAGES: Final = 6
#: What the assistant answered in the user's earlier conversations, for a question
#: (``Searcher.earlier_chats``, ADR 0044).
EarlierChats = Callable[[MemoryExecutionContext, str], Awaitable[list[EarlierChatLine]]]


def _cacheable(bundle: ContextBundle) -> bool:
    return not any(
        item.attributes.get("derived") or item.item_id.startswith("wm_") for item in bundle.memories
    )


#: Earlier values (ADR 0038) a bundle carries at most, in its history section: enough for "what
#: was it before", never enough to crowd the current evidence's budget.
HISTORY_MAX = 4

#: The model uses a cached bundle's content depends on. A deployment that expands queries
#: assembles a different bundle from the same query, so it must not read another's cache.
CACHED_MODEL_USES: tuple[LLMUse, ...] = ("query_expansion",)
#: What a ``format=prompt`` response carries of the bundle.

ContextFormat = Literal["prompt", "full"]


#: Fusion scores are rank aggregates, not probabilities: they are mapped into a low band
#: beneath an exact identifier hit. Order is preserved; the number does not claim a
#: confidence nobody measured.
_FUSION_CEILING = 0.05


def _relevance(c: Candidate) -> tuple[float, ScoreKind, float]:
    """The raw ranking number, what produced it, and a comparable 0..1 relevance.

    The raw number has two incompatible scales - an RRF fusion score (~0.001..0.25) and a
    hardcoded 1.0 for exact identifier hits - so clients compare ``relevance`` instead: the
    dense similarity to the question when the context read it (``score_similarity``), else
    the fusion score mapped into a low band.
    """
    if "exact" in (c.retrievers or []):
        return float(c.score), "exact", 1.0
    if c.similarity is not None:
        return float(c.score), "fusion", min(max(c.similarity, 0.0), 1.0)
    # RRF scores are sums of 1/(k+rank): bounded and monotone in rank, but not a probability
    return float(c.score), "fusion", min(float(c.score), 1.0) * _FUSION_CEILING


#: The evidence a retrieved candidate is, by its kind (summary and message name themselves).
_EVIDENCE_SOURCE = {
    "chunk": EvidenceSource.DOCUMENT_CHUNK,
    "memory": EvidenceSource.MEMORY,
    "fact": EvidenceSource.GRAPH_FACT,
}


def candidate_to_item(c: Candidate) -> ContextItem:
    p = c.payload
    citation = {
        "chunk": f"chunk_id:{c.record_id}",
        "memory": f"memory_id:{c.record_id}",
        "fact": f"relation_id:{c.record_id}",
        "summary": f"summary_id:{c.record_id}",
    }.get(c.kind, f"{c.kind}:{c.record_id}")
    evidence = [
        EvidenceRef(
            source_type=_EVIDENCE_SOURCE.get(c.kind) or EvidenceSource(c.kind),
            source_id=c.record_id,
            document_id=p.get("document_id"),
            chunk_id=c.record_id if c.kind == "chunk" else p.get("chunk_id"),
            node_id=p.get("node_id"),
            page=p.get("page"),
            observed_at=datetime.now(UTC),
        )
    ]
    # Preserve the raw-source lineage, while keeping the memory's own citation stable.
    # Older indexes and working-memory candidates have no source_refs; their record
    # pointer remains valid until the index is rebuilt.
    if c.kind == "memory" and (source_refs := p.get("source_refs")):
        evidence = [EvidenceRef.model_validate(ref) for ref in source_refs]
    # _relevance existed, was correct, and was called by nothing but its own test, so every
    # item on the wire carried relevance=0.0 and score_kind="fusion" regardless of what
    # produced it — the incomparable-scale problem its docstring describes was still live.
    raw, kind, relevance = _relevance(c)
    return ContextItem(
        item_id=c.record_id,
        representation=c.representation,
        text=c.text,
        score=raw,
        score_kind=kind,
        relevance=relevance,
        retrievers=c.retrievers,
        evidence=evidence,
        citation=citation,
        document_id=p.get("document_id"),
        page=p.get("page"),
        section_path=p.get("section_path"),
        expanded_from=c.expanded_from,
        expansion_edge=c.expansion_edge,
        token_estimate=estimate_tokens(c.text),
        attributes={
            k: v
            for k, v in p.items()
            if k
            in (
                "predicate",
                "subject",
                "object",
                "attributes",
                "contradicts",
                # typed connections written by the background pass (supersedes / contradicts /
                # relates): a caller that reads one memory can see what it was linked to
                # a later statement of the same user replaced it (ADR 0038)
                "earlier_value",
                "connections",
                "contributors",
                "memory_type",
                "category",
                "provider",
                "derived",
                "source_observed_to",
                "visibility",
                "owner_principal",
                "confidence",
                "status",
                # when it was observed, so a renderer can order memories in time and show
                # the date beside the fact - the memories section prints the date, the
                # weekday and the speaker once per line from this field and the subject,
                # instead of a rank-ordered list with no time in it
                "observed_at",
                # the relative dates the text names, already resolved against observed_at
                "dated_mentions",
            )
            and v not in (None, [], {})
        },
    )


@dataclass(frozen=True)
class _Lookup:
    """What one database read and one cache read say about a query, before any work."""

    bundle_id: str
    #: where the bundle record is cached (the in-process reader's and the semantic cache's)
    cache_key: str
    #: where the API's prompt form is cached
    prompt_key: str
    #: where the API's full form is cached
    full_key: str
    revision_fp: str
    #: the revisions an authorization scope depends on, in that cache's key format
    authz_fp: str
    #: the cached AuthorizedScope bytes, or None when that key missed
    scope: bytes | None
    #: the cached answer in the requested form: present means it is already made
    bundle: bytes | None
    semantic_key: str | None = None
    semantic: bytes | None = None


class _StandingRead:
    """The read of the standing rules and preferences that apply to a request
    (``standing.standing_blocks``), started when the retrieval has the route and the query's
    vectors (``RetrievalEngine.retrieve(on_encoded=...)``): it runs under the ranking."""

    def __init__(self, engine: Any, query: str, visibility: Any) -> None:
        self.engine, self.query, self.visibility = engine, query, visibility
        self.task: asyncio.Future[list[ProfileBlockView]] | None = None

    def start(self, routed: RoutedQuery, vectors: QueryVectors) -> None:
        if STANDING and routed.needs_memories and self.task is None:
            self.task = asyncio.ensure_future(
                standing_blocks(self.engine, self.query, self.visibility, vectors)
            )


class _Encoded:
    """The query's vectors, handed over the moment the retrieval has them
    (``RetrievalEngine.retrieve(on_encoded=...)``): they start the standing rules' read, and
    a learned procedure the query's words miss is matched by them (``ContextSections``)."""

    def __init__(self, standing: _StandingRead) -> None:
        self.standing = standing
        self.vectors: asyncio.Future[QueryVectors | None] = (
            asyncio.get_running_loop().create_future()
        )

    def __call__(self, routed: RoutedQuery, vectors: QueryVectors) -> None:
        self.standing.start(routed, vectors)
        self.settle(vectors)

    def settle(self, vectors: QueryVectors | None) -> None:
        """Hand ``vectors`` over unless they were: a retrieval that reported none still
        lets the procedures read finish, on the result's own."""
        if not self.vectors.done():
            self.vectors.set_result(vectors)


class ContextBuilder:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        engine: RetrievalEngine,
        conversation: ConversationService,
        cache: CacheProvider | None,
        *,
        settings: ContextSettings,
        retrieval: RetrievalSettings,
        cache_ttl_seconds: int = 300,
        working: EphemeralMemory | None = None,
        assist: LLMAssist | None = None,
        sections: ContextSections | None = None,
        records: BundleRecords | None = None,
        earlier_chats: EarlierChats | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.earlier_chats = earlier_chats
        self.sections = sections
        self.engine = engine
        self.conversation = conversation
        self.cache = cache
        self.working = working
        self.cfg = settings
        self.retrieval_cfg = retrieval
        self.cache_ttl = cache_ttl_seconds
        self.semantic_cache = (
            SemanticBundleCache(cache, ttl_seconds=cache_ttl_seconds) if cache else None
        )
        self.records = records or BundleRecords(cache)
        self._cache_router = QueryRouter()
        self.assist = assist or LLMAssist.disabled()
        #: cache writes and access flushes still in flight; awaited by drain()
        self._pending: set[asyncio.Task[None]] = set()
        #: tenant -> the memory ids served since the last flush. A set, because a memory
        #: served twice inside one window is bumped once (see _buffer_access).
        self._access: dict[str, set[str]] = {}
        self._buffered = 0
        self._flusher: asyncio.Task[None] | None = None
        self._flush_now = asyncio.Event()
        self._flush_lock = asyncio.Lock()
        # Nothing in it changes for the life of the process - the tuning is frozen and the
        # embedding fingerprint is fixed at construction (model use is per request) - and
        # it was recomputed per request: two model_dump_json() calls and a hash, before the
        # cache key it feeds could even be formed.
        self._fingerprint = self._config_fingerprint()

    def _config_fingerprint(self) -> str:
        parts = [
            "source-lineage-v2:rag-evidence-v7:multilingual-packing-v1:run-record-window-v1",
            self.retrieval_cfg.model_dump_json(),
            self.cfg.model_dump_json(),
            self.engine.indexer.fingerprint,
        ]
        return stable_key(*parts)

    async def _lookup(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        budget: int,
        document_ids: Sequence[str] | None,
        tools: ToolsRequest | None,
        window: WindowFormat | None,
        output: ContextFormat | Literal["record"] = "record",
        *,
        conversation_budget: int | None = None,
    ) -> _Lookup:
        """Everything the cache decision needs: one database round trip, one cache read.

        It was four. The revision fingerprint came from one unit of work, the caller's
        visibility from a second read inside the engine that opened its own session and asked
        the same table again, and then two sequential cache GETs - the authorization scope,
        then the bundle - each a full round trip to Dragonfly before the next could start.
        None of them depends on the others: the bundle key is a function of the revisions,
        the scope key is a function of a subset of the same revisions, and both keys exist
        before either value is needed. So: read the revisions once, derive both keys, and ask
        for both values in one MGET.

        Resolving the scope is left to the miss path. On a hit the caller is served from the
        bundle and the authorization scope is never needed, so the listing the resolver does
        when the scope cache is cold was work for an answer nobody read.
        """
        revisions = await self._revisions(ctx)
        revision_fp = self._revision_fingerprint(revisions)
        authz_fp = AuthorizationService.revision_fingerprint(ctx, revisions)
        namespace = stable_key(
            "source-backed-consolidation-v1",
            ctx.tenant_id,
            ctx.scope_fingerprint(),
            revision_fp,
            self._fingerprint,
            str(model_calls_allowed()),
            # per request, not per process: which uses are active depends on the bound
            # identity's key and policy, and a bundle built with model assistance must not
            # be served to a caller without it (or the reverse)
            self.assist.cache_fingerprint(CACHED_MODEL_USES),
            str(budget),
            ",".join(document_ids or []),
            tools.fingerprint() if tools else "",
            WINDOW_KEYS[window],
            # a window cut to the caller's own budget; none named keeps the key it always had
            *(
                [f"conversation-budget={conversation_budget}"]
                if conversation_budget is not None and window is not None
                else []
            ),
        )
        bundle_id = stable_key(namespace, query)
        cache_key = self._cache_key(ctx, bundle_id)
        prompt_key = f"ctxp:{cache_key}"
        full_key = f"ctxf:{cache_key}"
        # the namespace binds the tools by their digest (``ToolsRequest.fingerprint``: the
        # sorted names, k, hints), so a request with tools reuses only a bundle built for the
        # same toolbox, whose hints and skills were found for the same words
        semantic = self._semantic_key(namespace, ctx, query, tools)
        scope_key = AuthorizationService.scope_cache_key(ctx, authz_fp)
        answer = {"prompt": prompt_key, "full": full_key}.get(output, cache_key)
        keys = [scope_key, answer, semantic]
        values: list[bytes | None] = [None] * len(keys)
        if self.cache is not None:
            with contextlib.suppress(CacheUnavailable):
                wanted = [k for k in keys if k]
                found = iter(await self.cache.mget(wanted))
                values = [next(found) if k else None for k in keys]
        scope_raw, bundle_raw, semantic_raw = values
        return _Lookup(
            bundle_id=bundle_id,
            cache_key=cache_key,
            prompt_key=prompt_key,
            full_key=full_key,
            revision_fp=revision_fp,
            authz_fp=authz_fp,
            scope=scope_raw,
            bundle=bundle_raw,
            semantic_key=semantic,
            semantic=semantic_raw,
        )

    async def _revisions(self, ctx: MemoryExecutionContext) -> dict[str, int]:
        """One shared dependency set for automatic lookup and explicit bundle replay."""
        async with self.uow_factory() as uow:
            # Every revision a bundle's content depends on. AGENT and GRAPH were missing, so
            # an agent-scoped memory or a graph enrichment bumped a counter nobody read and
            # the bundle stayed cached regardless.
            keys = [
                (RevisionKind.TENANT, ""),
                (RevisionKind.USER, ctx.user_id or ""),
                (RevisionKind.THREAD, ctx.thread_id or ""),
                (RevisionKind.AGENT, ctx.agent_id or ""),
                (RevisionKind.GRAPH, ""),
                # the authorization scope's own keys, read in the same round trip: it
                # depends on grants, not on any of the content revisions above
                (RevisionKind.MEMBERSHIP, ""),
                (RevisionKind.MEMBERSHIP, ctx.user_id or ""),
                (RevisionKind.MEMBERSHIP, ctx.agent_id or ""),
            ]
            return await uow.revisions.get_many(ctx.tenant_id, keys)

    @staticmethod
    def _revision_fingerprint(revisions: dict[str, int]) -> str:
        return stable_key(*(f"{k}={v}" for k, v in sorted(revisions.items())))

    def _semantic_key(
        self,
        namespace: str,
        ctx: MemoryExecutionContext,
        query: str,
        tools: ToolsRequest | None = None,
    ) -> str | None:
        if self.semantic_cache is None or not self.retrieval_cfg.dense:
            return None
        if self.assist.wants("query_expansion"):
            return None
        route = self._cache_router.route(query, has_thread=ctx.thread_id is not None)
        if route.query_type not in {QueryType.USER_MEMORY, QueryType.GENERAL_SEMANTIC}:
            return None
        if tools:
            # the hints' next step depends on whether the task is a question (a tool that
            # changes something never leads one), and the guarded words drop the question
            # mark: a question's hints are no hints for a request to act, nor the reverse
            namespace = stable_key(namespace, f"asks_change={asks_change(query)}")
        return semantic_key(namespace, query)

    async def build(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        *,
        token_budget: int | None = None,
        document_ids: Sequence[str] | None = None,
        tools: ToolsRequest | None = None,
        window: bool = True,
        window_format: WindowFormat = WindowFormat.RENDERED,
        conversation_token_budget: int | None = None,
    ) -> ContextBundle:
        """The bundle itself, for in-process readers (the profile job, the benchmarks)."""
        budget = token_budget or self.cfg.token_budget
        form = window_format if window else None
        with (
            span("context.build", tenant_id=ctx.tenant_id),
            stage_seconds.labels("context.build").time(),
        ):
            timings = Timings()
            with timings.stage("scope"):
                found = await self._lookup(
                    ctx,
                    query,
                    budget,
                    document_ids,
                    tools,
                    form,
                    conversation_budget=conversation_token_budget,
                )
            if found.bundle is not None:
                return ContextBundle.model_validate_json(found.bundle)
            bundle = await self._fresh(
                ctx,
                query,
                budget,
                document_ids,
                tools,
                form,
                found,
                timings,
                conversation_token_budget or self.cfg.conversation_token_budget,
            )
        evidence_status_total.labels(bundle.evidence.status.value).inc()
        self._after_build(ctx, bundle, found)
        return bundle

    async def build_api(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        *,
        token_budget: int | None = None,
        document_ids: Sequence[str] | None = None,
        tools: ToolsRequest | None = None,
        window: bool = True,
        window_format: WindowFormat = WindowFormat.RENDERED,
        output: ContextFormat = "prompt",
        debug: bool = False,
        conversation_token_budget: int | None = None,
    ) -> bytes:
        """The context as the API sends it, serialised once.

        Both forms (``modules/context/views.py``) are cached as the bytes a hit answers with,
        so a hit in either is the stored bytes untouched.
        ``debug`` asks for the build's diagnostics, which only a build has: it is never
        served from the cache. Callers that need the ContextBundle itself use ``build``.
        ``conversation_token_budget`` sizes the window (``None``: the service's
        ``conversation_token_budget``).
        """
        budget = token_budget or self.cfg.token_budget
        form = window_format if window else None
        with (
            span("context.build", tenant_id=ctx.tenant_id),
            stage_seconds.labels("context.build").time(),
        ):
            timings = Timings()
            with timings.stage("scope"):
                found = await self._lookup(
                    ctx,
                    query,
                    budget,
                    document_ids,
                    tools,
                    form,
                    output,
                    conversation_budget=conversation_token_budget,
                )
            if found.bundle is not None and not debug:
                self._track(self.records.given(ctx, found.bundle_id))
                return found.bundle
            bundle = await self._fresh(
                ctx,
                query,
                budget,
                document_ids,
                tools,
                form,
                found,
                timings,
                conversation_token_budget or self.cfg.conversation_token_budget,
            )
        evidence_status_total.labels(bundle.evidence.status.value).inc()
        self._after_build(ctx, bundle, found)
        body = prompt_view(bundle) if output == "prompt" else full_view(bundle)
        if debug:
            body["diagnostics"] = bundle.diagnostics
        return orjson.dumps(body)

    async def _fresh(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        budget: int,
        document_ids: Sequence[str] | None,
        tools: ToolsRequest | None,
        window: WindowFormat | None,
        found: _Lookup,
        timings: Timings,
        conversation_budget: int,
    ) -> ContextBundle:
        """Retrieve, window (in the form asked for; ``None``: none - within
        ``conversation_budget``) and assemble - the cache-miss half of a build."""
        cached, embedding = await self._semantic_lookup(found, query)
        if cached is not None:
            return cached
        with timings.stage("visibility"):
            visibility = await self.engine.authz.visibility(
                ctx, revision_fingerprint=found.authz_fp, cached_scope=found.scope
            )
        tokens_before = self.assist.tokens_used()
        # the standing rules and preferences need only the route and the query's vectors:
        # read under the ranking, from the moment the vectors exist
        standing = _StandingRead(self.engine, query, visibility)
        encoded = _Encoded(standing)
        # the pinned sections and the thread's recent messages are independent of retrieval:
        # they are read under it
        sections = (
            asyncio.ensure_future(
                self.sections.gather(
                    ctx,
                    query,
                    visibility,
                    procedures=tools is not None and tools.any,
                    vectors=encoded.vectors,
                )
            )
            if self.sections is not None
            else None
        )
        recent = asyncio.ensure_future(
            self._recent_messages(ctx, window) if window is not None else _nothing()
        )
        previous = asyncio.ensure_future(self._previous_question(ctx, query))
        earlier = asyncio.ensure_future(self._earlier_chats(ctx, query))
        try:
            with timings.stage("retrieve"):
                result = await self.engine.retrieve(
                    ctx,
                    query,
                    document_ids=document_ids,
                    visibility=visibility,
                    query_embedding=(query, embedding) if embedding is not None else None,
                    on_encoded=encoded,
                )
            encoded.settle(result.query_vectors)
            with timings.stage("sections"):
                pinned = await sections if sections is not None else Pinned()
                conversation = self._window(
                    ctx,
                    await recent,
                    after=pinned.covered_to,
                    form=window or WindowFormat.RENDERED,
                    token_budget=conversation_budget,
                )
                asked_before = await previous
                said_before = await earlier
        except BaseException:
            for side in (sections, recent, previous, earlier, standing.task):
                if side is not None:
                    side.cancel()
            raise
        with timings.stage("similarity"):
            await self._similarity_and_standing(query, visibility, result, pinned, standing.task)
        result = await self._followed_up(
            ctx, query, document_ids, visibility, result, asked_before, timings
        )
        await self._add_extras(ctx, query, tools, visibility, result, pinned, timings)
        # the engine's own stage split, plus what happened around it
        result.diagnostics.setdefault("timings_ms", {}).update(timings.ms)
        pinned, pinned_tokens = within_budget(pinned, budget)
        said_before, said_tokens = _within(said_before, budget - pinned_tokens)
        bundle = self._assemble(
            query, result, conversation, budget - pinned_tokens - said_tokens, found.revision_fp
        )
        self._remember_semantic(found, result, bundle)
        spent = self.assist.tokens_used() - tokens_before
        return bundle.model_copy(
            update={
                "bundle_id": found.bundle_id,
                "evidence": bundle.evidence.model_copy(update={"llm_tokens": spent}),
                "token_budget": budget,
                "token_estimate": bundle.token_estimate + pinned_tokens + said_tokens,
                "earlier_chats": said_before,
                "profile": pinned.profile,
                "thread_summary": pinned.thread_summary,
                "procedures": pinned.procedures,
                "tools": pinned.tools,
            }
        )

    async def _similarity_and_standing(
        self,
        query: str,
        visibility: Any,
        result: RetrievalResult,
        pinned: Pinned,
        started: asyncio.Future[list[ProfileBlockView]] | None = None,
    ) -> None:
        """Score the candidates' similarity and, beside it, read the user's standing rules
        and preferences that apply (``standing``) into the profile: one round trip, the
        query's own vectors. ``started``: that read, already under way (``_fresh`` starts it
        as soon as the retrieval has the query's vectors)."""
        standing = started or (
            asyncio.ensure_future(
                standing_blocks(self.engine, query, visibility, result.query_vectors)
            )
            if STANDING and result.routed.needs_memories
            else None
        )
        try:
            await self.engine.score_similarity(result)
        except BaseException:
            if standing is not None:
                standing.cancel()
            raise
        if standing is not None:
            pinned.profile = [*await standing, *pinned.profile]

    async def _add_extras(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        tools: ToolsRequest | None,
        visibility: Any,
        result: RetrievalResult,
        pinned: Pinned,
        timings: Timings,
    ) -> None:
        """Working memory ahead of the ranked memories, the tool hints (which read the
        memories in hand and the profile), and the learned skills' steps a standing rule of
        the user forbids for this task, with or without hints (``ruled``)."""
        if self.working is not None and result.routed.needs_memories:
            for i, item in enumerate(await self.working.recall(ctx)):
                result.candidates.insert(
                    i,
                    Candidate(
                        record_id=f"wm_{i}",
                        kind="memory",
                        text=str(item.get("content", "")),
                        score=1.0,
                        retrievers=["working"],
                        payload={"memory_type": item.get("memory_type", "WORKING")},
                    ),
                )
        if tools is not None and tools.hinted and self.sections is not None:
            with timings.stage("tools"):
                pinned.tools = await self.sections.tools(
                    ctx,
                    query,
                    tools,
                    visibility,
                    memories=[
                        candidate_to_item(c) for c in result.candidates if c.kind == "memory"
                    ],
                    profile=pinned.profile,
                    vectors=result.query_vectors,
                )
        pinned.procedures = ruled(pinned.procedures, query, pinned.profile, pinned.tools)

    async def _semantic_lookup(
        self, found: _Lookup, query: str
    ) -> tuple[ContextBundle | None, list[float] | None]:
        """Return a reusable bundle or an encoding the retrieval miss can consume."""
        if found.semantic is None or self.semantic_cache is None:
            return None, None
        embedding = await self.engine.indexer.spaces.embed_primary_query(query)
        raw = await self.semantic_cache.lookup(found.semantic, embedding)
        if raw is None:
            return None, embedding
        try:
            cached = ContextBundle.model_validate_json(raw)
        except ValueError:
            return None, embedding
        return cached.model_copy(
            update={"query": query, "bundle_id": found.bundle_id, "cache_hit": True}
        ), embedding

    def _remember_semantic(
        self, found: _Lookup, result: RetrievalResult, bundle: ContextBundle
    ) -> None:
        """Keep only a vector and reference; the existing bundle cache owns the payload."""
        if (
            self.semantic_cache is not None
            and found.semantic_key is not None
            and result.query_embedding is not None
            and _cacheable(bundle)
        ):
            self._track(
                self.semantic_cache.store(
                    found.semantic_key, found.cache_key, result.query_embedding
                )
            )

    def _after_build(
        self,
        ctx: MemoryExecutionContext,
        bundle: ContextBundle,
        found: _Lookup,
    ) -> None:
        """The bookkeeping a built bundle leaves behind, none of it on the request path: the
        served-access counts, the cached bundle, and the record of its handles and evidence
        that later calls refer back to (``modules.context.handles``)."""
        self._buffer_access(ctx.tenant_id, [i.item_id for i in bundle.memories if i.item_id])
        if self.cache is None:
            return
        if _cacheable(bundle):
            self._track(self._store(self.cache, found, bundle))
        self._track(self._remember(ctx, bundle))

    async def _remember(self, ctx: MemoryExecutionContext, bundle: ContextBundle) -> None:
        await self.records.store(ctx, record_of(bundle))
        await self.records.given(ctx, bundle.bundle_id)

    async def _store(
        self,
        cache: CacheProvider,
        found: _Lookup,
        bundle: ContextBundle,
    ) -> None:
        """Write the bundle record and both API forms to the cache after the caller has been
        answered.

        A bundle is 30-80 KB. Serialising it and pushing it to Dragonfly took that long off
        the front of every cache-miss response for the benefit of the *next* caller, who is
        not waiting on anything. ``cache_hit`` is stored true because a value read back from
        here is, by construction, a hit - that is what lets ``build_api`` return the stored
        bytes untouched. The diagnostics are not stored: a response carries them only from
        a build.
        """
        try:
            record = bundle.model_copy(update={"cache_hit": True, "diagnostics": {}})
            # one pipelined round trip for the three forms, not three sequential SETs
            await cache.mset(
                {
                    found.cache_key: record.model_dump_json(
                        exclude={"revision_fingerprint"}
                    ).encode(),
                    found.prompt_key: orjson.dumps(prompt_view(bundle)),
                    found.full_key: orjson.dumps(full_view(bundle)),
                },
                ttl_seconds=self.cache_ttl,
            )
        except CacheUnavailable:
            return
        except Exception as exc:  # nothing awaits this task; a failure here is silent
            log.warning("context.bundle_cache_write_failed", error_message=str(exc))

    def _track(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Flush the buffered access counts and wait for the writes started by earlier builds.

        Called by tests and at shutdown. It wakes the flush timer rather than cancelling it:
        a timer caught mid-flush holds a pooled connection, and aborting that connection
        costs the pool more than waiting out one statement.
        """
        self._flush_now.set()
        flusher = self._flusher
        if flusher is not None:
            await asyncio.gather(flusher, return_exceptions=True)
        await self._flush_access()
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def close(self) -> None:
        await self.drain()

    def _buffer_access(self, tenant_id: str, ids: Sequence[str]) -> None:
        """Count a memory as used when it actually reaches the caller.

        Off the request path, and now coalesced. Each bump is an UPDATE over up to
        ``memories_max`` rows followed by a COMMIT - a WAL flush - on the same connection
        pool the reads check out of; one per served bundle is ~20 of those a second at the
        20 rps target, for a counter read once a night by the forgetting pass. Ids are
        buffered per tenant and flushed every ``access_flush_seconds`` or at
        ``access_flush_max_ids``, whichever comes first, so a window costs one statement per
        tenant. A memory served twice inside one window counts once: the forgetting score's
        access term is ``1 - 0.5**accesses``, which saturates, so the distinction is below
        anything the policy can act on.

        The forgetting policy scores
        ``importance * 0.5**(idle/half_life) * (1 - 0.5**(accesses + reinforcements))`` and
        reads ``access_count`` and ``last_accessed_at``. Migration 0006 added those columns for
        it. Nothing ever incremented them: ``bump_access`` existed on the repository and on the
        port and had no call site anywhere, so ``access_count`` stayed 0 for every memory and
        the access term collapsed to the constant 0.5. Decay was running on importance and
        recency alone, and the "frequently used memories survive" half of the policy silently
        did nothing - while looking, in the code and in the migration, entirely present.

        Served, not merely retrieved: candidates that were ranked but dropped by the token
        budget never reached anyone and must not count as use, or every query would reinforce
        memories nobody read.
        """
        if not ids:
            return
        self._access.setdefault(tenant_id, set()).update(ids)
        self._buffered += len(ids)
        if self._buffered >= self.cfg.access_flush_max_ids:
            # reset before the task is spawned, not inside it: every build between the spawn
            # and the flush would otherwise see a full buffer and spawn another one
            self._buffered = 0
            self._track(self._flush_access())
            return
        if self._flusher is None or self._flusher.done():
            self._flush_now.clear()
            self._flusher = asyncio.ensure_future(self._flush_after())

    async def _flush_after(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._flush_now.wait(), self.cfg.access_flush_seconds)
        await self._flush_access()

    async def _flush_access(self) -> None:
        """One bulk bump per tenant for everything buffered since the last flush."""
        async with self._flush_lock:
            pending, self._access, self._buffered = self._access, {}, 0
            for tenant_id, served in pending.items():
                try:
                    async with self.uow_factory() as uow:
                        await uow.memories.bump_access(
                            tenant_id, sorted(served), at=datetime.now(UTC)
                        )
                        await uow.commit()
                except Exception as exc:  # usage accounting must never fail a read
                    log.warning(
                        "memory.access_bump_failed",
                        error_message=str(exc),
                        count=len(served),
                    )

    @staticmethod
    def _cache_key(ctx: MemoryExecutionContext, bundle_id: str) -> str:
        """Tenant, the caller's security scope, and the bundle. All three, always."""
        return f"ctx:{ctx.tenant_id}:{ctx.scope_fingerprint()}:{bundle_id}"

    async def _recent_messages(
        self, ctx: MemoryExecutionContext, form: WindowFormat
    ) -> list[Message]:
        """The thread's most recent messages, read while retrieval runs. A caller that keeps
        no history of its own (``window``) gets them whatever the question: the turn before
        is what a follow-up means, and the route cannot know what it refers to. The messages
        form reads the tool results of the calls in it too, and counts
        ``conversation_max_messages`` as turns: a turn is replayed whole or not at all."""
        if not ctx.thread_id:
            return []
        limit = self.cfg.conversation_max_messages
        async with self.uow_factory() as uow:
            if form is WindowFormat.MESSAGES:
                return await self.conversation.window_messages(
                    uow,
                    ctx,
                    ctx.thread_id,
                    turns=limit,
                    limit=self.cfg.conversation_read_max_messages,
                )
            thread = await uow.threads.get(ctx.tenant_id, ctx.thread_id)
            if thread is None:
                return []
            return list(await self.conversation.list_messages(uow, ctx, ctx.thread_id, limit=limit))

    async def _followed_up(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        document_ids: Sequence[str] | None,
        visibility: Any,
        result: RetrievalResult,
        asked_before: str | None,
        timings: Timings,
    ) -> RetrievalResult:
        """``result``, or - when the question's own search found nothing close (under
        :data:`FOLLOW_UP_FLOOR`) and it follows ``asked_before`` up - the search again with
        that question in its search text, its candidates scored the same way."""
        if asked_before is None or _best_similarity(result) >= FOLLOW_UP_FLOOR:
            return result
        with timings.stage("follow_up"):
            again = await self.engine.retrieve(
                ctx, query, document_ids=document_ids, visibility=visibility, follow_up=asked_before
            )
            await self.engine.score_similarity(again)
        return again

    async def _earlier_chats(
        self, ctx: MemoryExecutionContext, query: str
    ) -> list[EarlierChatLine]:
        """What the assistant answered in the user's earlier conversations, when ``query``
        points to one (ADR 0044), or carries on from something (:func:`router.continues`) as
        the first question of its conversation, which has nothing of its own it could carry
        on from; nothing otherwise, at no cost. Read while retrieval runs."""
        if self.earlier_chats is None or not ctx.user_id:
            return []
        if points_to_earlier_chat(query) or (
            continues(query) and not await self._asked_before(ctx, query)
        ):
            return await self.earlier_chats(ctx, query)
        return []

    async def _asked_before(self, ctx: MemoryExecutionContext, query: str) -> bool:
        """Whether this conversation holds a question of the user's besides ``query``."""
        if not ctx.thread_id:
            return False
        async with self.uow_factory() as uow:
            if await uow.threads.get(ctx.tenant_id, ctx.thread_id) is None:
                return False
            newest = await self.conversation.list_messages(
                uow, ctx, ctx.thread_id, limit=FOLLOW_UP_MESSAGES
            )
        asked = query.strip()
        return any(
            m.role is MessageRole.USER and m.content.strip() not in ("", asked) for m in newest
        )

    async def _previous_question(self, ctx: MemoryExecutionContext, query: str) -> str | None:
        """The thread's previous questions, when ``query`` is short enough to follow them up
        (:data:`FOLLOW_UP_WORDS`): the user's messages among its newest
        :data:`FOLLOW_UP_MESSAGES` that are not ``query`` itself (a caller may append the
        question before it asks for its context), newest first, within
        :data:`FOLLOW_UP_CHARS`. "Which of the two?" points at more than the question just
        before it. ``None`` outside a thread, for a longer question, or when the thread has
        none. Read while the question's own search runs."""
        if not ctx.thread_id or len(query.split()) > FOLLOW_UP_WORDS:
            return None
        async with self.uow_factory() as uow:
            if await uow.threads.get(ctx.tenant_id, ctx.thread_id) is None:
                return None
            newest = await self.conversation.list_messages(
                uow, ctx, ctx.thread_id, limit=FOLLOW_UP_MESSAGES
            )
        asked = query.strip()
        before = [
            text
            for message in reversed(newest)
            if message.role is MessageRole.USER
            and (text := message.content.strip()) != asked
            and text
        ]
        return " ".join(before)[:FOLLOW_UP_CHARS] or None

    def _window(
        self,
        ctx: MemoryExecutionContext,
        messages: Sequence[Message],
        *,
        after: int,
        form: WindowFormat,
        token_budget: int,
    ) -> ConversationWindow:
        """The messages after the thread's durable summary (which covers everything up to
        ``after``) that fit ``token_budget``, in the form asked for."""
        if not ctx.thread_id or not messages:
            return ConversationWindow(thread_id=ctx.thread_id, form=form)
        after_summary = [m for m in messages if m.sequence > after]
        if form is WindowFormat.MESSAGES:
            return replay_window(
                ctx.thread_id, after_summary, token_budget, self.cfg.conversation_max_messages
            )
        return render_window(ctx.thread_id, after_summary, token_budget)

    def _below_floor(self, c: Candidate, slot: slots.Slot | None = None) -> bool:
        """A ranked item whose similarity to the question is under the floor is not packed:
        a fusion score always fills the budget, relevant or not. Exact hits, expansion and
        graph companions and items with no similarity (graph facts, working memory) are
        judged by what brought them in, not by this; so is a memory that is a list answering
        the question (``slots.answers_as_list``): a long list sits far from its question in
        the encoder's space, however plainly it answers it."""
        return (
            c.similarity is not None
            and c.similarity < self.engine.relevance_floor
            and c.expanded_from is None
            and "exact" not in (c.retrievers or [])
            and not (
                c.kind == "memory" and slot is not None and slots.answers_as_list(slot, c.text)
            )
        )

    def _coverage(
        self,
        report: EvidenceReport,
        query: str,
        shown: Sequence[ContextItem],
        retrieved: Sequence[str] = (),
    ) -> EvidenceReport:
        """INSUFFICIENT when the evidence shown does not hold the value the question asks for
        (``coverage.unanswered``, ADR 0040); each item dated by when it was said and the days
        its relative dates name, as the bundle prints them."""
        if (
            not self.retrieval_cfg.abstain_when_insufficient
            or not self.retrieval_cfg.coverage_check
            or report.status is EvidenceStatus.INSUFFICIENT
        ):
            return report
        gap = unanswered(
            query,
            [i.text for i in shown],
            [months_dated(i.attributes) for i in shown],
            retrieved=retrieved,
        )
        if not gap:
            return report
        return report.model_copy(
            update={"status": EvidenceStatus.INSUFFICIENT, "notes": [*report.notes, gap]}
        )

    def _related_evidence(
        self,
        report: EvidenceReport,
        memories: Sequence[ContextItem],
        knowledge: Sequence[ContextItem],
    ) -> EvidenceReport:
        """A memory-only bundle that shares no word with the question but carries a current
        memory the encoder puts close to it is evidence, not nothing (ADR 0040)."""
        floor = self.retrieval_cfg.related_evidence_relevance
        if (
            floor <= 0
            or report.status is not EvidenceStatus.INSUFFICIENT
            or knowledge
            or not memories
            or NO_SHARED_TERM not in report.notes
        ):
            return report
        best = max(m.relevance for m in memories)
        if best < floor:
            return report
        return report.model_copy(
            update={
                "status": EvidenceStatus.COMPLETE,
                "notes": [
                    *(n for n in report.notes if n != NO_SHARED_TERM),
                    f"no word in common with the question; a memory is related at {best:.2f}",
                ],
            }
        )

    def _assemble(
        self,
        query: str,
        result: RetrievalResult,
        window: ConversationWindow,
        budget: int,
        revision_fp: str,
    ) -> ContextBundle:
        # A rendered window is in the rendered text, all of it. The messages form is replayed by
        # the caller as its own conversation: it takes the conversation's usual share of the
        # budget and no more, so a larger window the caller asked for (or a newest turn over
        # its budget, given whole) never crowds the memories out.
        drawn = (
            window.token_estimate
            if window.form is WindowFormat.RENDERED
            else min(window.token_estimate, self.cfg.conversation_token_budget)
        )
        remaining = budget - drawn
        memories: list[ContextItem] = []
        primary_memories = 0
        knowledge: list[ContextItem] = []
        graph_facts: list[ContextItem] = []
        summaries: list[ContextItem] = []
        history: list[ContextItem] = []
        # a seed chunk travels with the companions its required evidence groups point at
        # (definition, footnote, cross-reference): either the whole unit fits the budget or
        # the seed is left out, so the bundle never carries a claim without its companion
        targets = result.diagnostics.get("evidence_targets") or {}
        seed_groups = result.diagnostics.get("evidence_seed_groups") or {}
        included_ids: set[str] = set()
        skipped_seeds: list[str] = []
        top_seed_skipped: list[str] = []  # groups of the best-ranked seed, if it was left out
        seen_seed = False
        items = [(c, candidate_to_item(c)) for c in result.candidates]
        by_node: dict[str, list[tuple[Any, ContextItem]]] = {}
        for c, item in items:
            node = c.payload.get("node_id")
            if node and c.kind in ("chunk", "summary"):
                by_node.setdefault(str(node), []).append((c, item))

        def companions_for(node: str) -> list[tuple[Any, ContextItem]]:
            out: list[tuple[Any, ContextItem]] = []
            for group in seed_groups.get(node, []):
                accepted = set(targets.get(group, []))
                if any(
                    str(cc.payload.get("node_id")) in accepted
                    for cc, _ in items
                    if cc.record_id in included_ids
                ):
                    continue  # already satisfied by something in the bundle
                for target in targets.get(group, []):
                    found = [
                        (cc, it)
                        for cc, it in by_node.get(target, [])
                        if cc.record_id not in included_ids
                    ]
                    if found:
                        out.append(found[0])
                        break
            return out

        below_floor = 0
        slot = slots.question_slot(query)
        for c, item in items:
            if c.record_id in included_ids:
                continue
            if self._below_floor(c, slot):
                below_floor += 1
                continue
            if c.kind == "chunk" and len(knowledge) < self.cfg.knowledge_max:
                node = str(c.payload.get("node_id") or "")
                unit = [(c, item)] + (companions_for(node) if node in seed_groups else [])
                cost = sum(it.token_estimate for _, it in unit)
                if cost > remaining:
                    if len(unit) > 1:
                        skipped_seeds.append(c.record_id)
                        if not seen_seed:
                            top_seed_skipped.extend(seed_groups.get(node, []))
                    seen_seed = seen_seed or node in seed_groups
                    continue
                seen_seed = seen_seed or node in seed_groups
                for cc, it in unit:
                    if cc.record_id in included_ids:
                        continue
                    if cc.kind == "summary":
                        summaries.append(it)
                    else:
                        knowledge.append(it)
                    included_ids.add(cc.record_id)
                remaining -= cost
                continue
            if item.token_estimate > remaining:
                continue
            if (
                c.kind == "memory"
                and c.payload.get("earlier_value")
                and not result.routed.signals.get("history")
            ):
                # an earlier value goes to the history section, never among current evidence
                # - unless the question asks what was (``router._HISTORY``): then it answers
                if len(history) >= HISTORY_MAX:
                    continue
                history.append(item)
            elif c.kind == "memory" and (
                primary_memories < self.cfg.memories_max
                or c.expansion_edge
                in {"GRAPH_EVIDENCE", "DERIVED_SOURCE", "REPLACEMENT", "SECOND_HOP"}
            ):
                memories.append(item)
                if c.expansion_edge not in {
                    "GRAPH_EVIDENCE",
                    "DERIVED_SOURCE",
                    "REPLACEMENT",
                    "SECOND_HOP",
                }:
                    primary_memories += 1
            elif c.kind == "fact" and len(graph_facts) < self.cfg.graph_facts_max:
                graph_facts.append(item)
            elif c.kind == "summary" and len(summaries) < self.cfg.summaries_max:
                summaries.append(item)
            else:
                continue
            included_ids.add(c.record_id)
            remaining -= item.token_estimate
        if below_floor:
            result.diagnostics["below_relevance_floor"] = below_floor
        evidence = result.diagnostics.get("evidence")
        report = (
            EvidenceReport.model_validate(evidence)
            if isinstance(evidence, dict)
            else EvidenceReport(
                status=EvidenceStatus.COMPLETE
                if (knowledge or memories or graph_facts)
                else EvidenceStatus.INSUFFICIENT
            )
        )
        # the report must describe what is IN the bundle: companions that were retrieved but
        # dropped by the token budget count as missing
        included_nodes = {
            ev.node_id for item in (*knowledge, *summaries) for ev in item.evidence if ev.node_id
        }
        if report.required_groups and report.status is not EvidenceStatus.INSUFFICIENT:
            # groups required only by seeds that were left out (with their companions) no
            # longer apply to what is in the bundle
            seeds_in = {
                str(c.payload.get("node_id"))
                for c in result.candidates
                if c.record_id in included_ids and c.kind == "chunk"
            }
            applicable = {
                g for node, names in seed_groups.items() if node in seeds_in for g in names
            } or set(report.required_groups)
            report = report.model_copy(
                update={
                    "required_groups": [g for g in report.required_groups if g in applicable],
                    "satisfied_groups": [g for g in report.satisfied_groups if g in applicable],
                    "missing_groups": [g for g in report.missing_groups if g in applicable],
                    "notes": [
                        *report.notes,
                        *(
                            [
                                f"{len(skipped_seeds)} lower-ranked passage(s) left out: their "
                                "companion evidence did not fit the token budget"
                            ]
                            if skipped_seeds
                            else []
                        ),
                    ],
                }
            )
            targets_by_group = dict(
                zip(
                    report.required_groups,
                    [list(targets.get(g, [])) for g in report.required_groups],
                    strict=False,
                )
            )
            dropped = [
                g for g, ids in targets_by_group.items() if ids and not (set(ids) & included_nodes)
            ]
            if dropped:
                missing = sorted(set(report.missing_groups) | set(dropped))
                report = report.model_copy(
                    update={
                        "status": EvidenceStatus.INCOMPLETE,
                        "missing_groups": missing,
                        "satisfied_groups": [
                            g for g in report.satisfied_groups if g not in missing
                        ],
                        "notes": [*report.notes, "companion evidence exceeded the token budget"],
                    }
                )
        report = self._related_evidence(report, memories, knowledge)
        # current evidence only: an earlier value answers nothing about now
        report = self._coverage(
            report,
            query,
            (*memories, *knowledge, *graph_facts, *summaries),
            retrieved=[c.text for c in result.candidates],
        )
        if not (knowledge or memories or graph_facts or summaries or history):
            report = report.model_copy(update={"status": EvidenceStatus.INSUFFICIENT})
        # what the retriever found but this bundle does not carry (ranked out, over budget):
        # the grounding cascade checks answers against it for contradictions
        unused = [
            UnusedEvidence(item_id=c.record_id, kind=c.kind, text=c.text)
            for c, _ in items
            if c.record_id not in included_ids
            and c.kind in ("chunk", "memory", "summary")
            and not unverified_representation(c.payload)
        ]
        unused.extend(
            UnusedEvidence(item_id=str(u["record_id"]), kind=str(u["kind"]), text=str(u["text"]))
            for u in result.diagnostics.get("unused") or []
        )
        report = report.model_copy(update={"unused": unused[:UNUSED_MAX]})
        diagnostics = {
            k: v
            for k, v in result.diagnostics.items()
            if k not in ("evidence", "evidence_targets", "evidence_seed_groups", "unused")
        }
        return ContextBundle(
            query=query,
            query_type=result.routed.query_type,
            conversation=window,
            memories=memories,
            knowledge=knowledge,
            graph_facts=graph_facts,
            summaries=summaries,
            history=history,
            evidence=report,
            token_budget=budget,
            token_estimate=budget - remaining,
            revision_fingerprint=revision_fp,
            diagnostics=diagnostics,
        )


def render_window(
    thread_id: str, messages: Sequence[Message], token_budget: int
) -> ConversationWindow:
    """Most recent visible messages that fit the budget, oldest first."""
    chosen: list[Message] = []
    used = 0
    for m in reversed(messages):
        if m.kind is not MessageKind.VISIBLE:
            continue
        t = estimate_tokens(m.content) + 4
        if used + t > token_budget and chosen:
            break
        chosen.append(m)
        used += t
    chosen.reverse()
    rendered = "\n".join(f"{m.role.value}: {m.content}" for m in chosen)
    return ConversationWindow(
        thread_id=thread_id,
        message_ids=[m.message_id for m in chosen],
        messages=[
            WindowMessage(message_id=m.message_id, role=m.role.value, text=m.content)
            for m in chosen
        ],
        rendered=rendered,
        token_estimate=used,
    )


def _within(lines: list[EarlierChatLine], budget: int) -> tuple[list[EarlierChatLine], int]:
    """The best of the earlier conversations' lines (``lines`` is best first) whose section
    fits half of ``budget``, and the tokens that section takes."""
    kept: list[EarlierChatLine] = []
    for line in lines:
        if _section_tokens([*kept, line]) <= budget // 2:
            kept.append(line)
    return kept, _section_tokens(kept)


def _section_tokens(lines: list[EarlierChatLine]) -> int:
    """The tokens the earlier conversations' section takes: none without a line."""
    return sum(estimate_tokens(part) for part in earlier_chats_section(lines))


def _best_similarity(result: RetrievalResult) -> float:
    """The closest a ranked candidate came to the question (0: none was scored)."""
    return max((c.similarity for c in result.candidates if c.similarity is not None), default=0.0)


async def _nothing() -> list[Message]:
    return []


#: What each window (``None``: none) adds to a bundle's cache key; the rendered form's is the
#: key it always had.
WINDOW_KEYS: Final[Mapping[WindowFormat | None, str]] = {
    None: "no-window",
    WindowFormat.RENDERED: "window",
    WindowFormat.MESSAGES: "window:messages",
}
