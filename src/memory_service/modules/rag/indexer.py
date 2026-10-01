"""Indexer: canonical chunks/memories -> search records (dense spaces + BM25 sparse + payload).

The search index is rebuildable: ``rebuild_document`` re-indexes from PostgreSQL. Collection
names embed the fingerprint of every dense space, of the sparse encoder and of the
late-interaction encoder, so a model change creates a new vector space instead of mixing
spaces. A memory is indexed under two keys (ADR 0025): its own text, and its text read with
the turn it answers. Payload carries only security/filter fields and short display
text; every record is tagged with the Unicode script of its text.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.conversation import Thread
from memory_service.domain.documents import Chunk, Document, DocumentNode
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.script import detect_script
from memory_service.modules.context.summaries import abstractive_summaries, build_summaries
from memory_service.modules.conversation.summary import (
    SUMMARY_MAX_CHARS,
    SUMMARY_SOURCE_MESSAGES,
    digest,
)
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import document_identity
from memory_service.modules.memory.connections import payload_edges
from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.models import EmbeddingProvider, LateInteractionEncoder, SparseEncoder
from memory_service.ports.search import (
    CollectionSpec,
    SearchFilter,
    SearchRecord,
    SearchStore,
    VectorName,
)
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

KNOWLEDGE = "knowledge"
MEMORIES = "memories"
#: The open ends of a memory's valid and knowledge time, written as instants so a range
#: filter can read them: a store range filter never matches a field that is absent.
TIME_MIN = datetime(1900, 1, 1, tzinfo=UTC)
TIME_MAX = datetime(9999, 12, 31, tzinfo=UTC)


def memory_index_text(m: CanonicalMemory) -> str:
    """What gets embedded and BM25-indexed for a memory: the fact *with its context*.

    Documents already do this — a chunk is indexed as ``contextual_text``, not ``text`` —
    and memories were the one collection that skipped it: ``"semantic: <content>"``, no date,
    no subject. The date and the subject are both on the object and both written to the
    payload, so the system was handed the attribution and threw it away before indexing.
    On dated conversational memory every question is "who did what, when"; a temporal
    query ("when did Caroline ...") cannot lexically match a fact whose index text has no
    date, and a dense query is pulled toward the wrong person when the subject is absent.

    This is the memory's own key, and what its rerankers read. The turn it answers is the
    second key (``memory_context_text``).
    """
    return memory_key_text(
        observed=m.temporal.observed_at.date().isoformat(),
        subject=m.subject,
        memory_type=m.memory_type.value,
        content=m.content,
    )


def memory_key_text(*, observed: str, subject: str | None, memory_type: str, content: str) -> str:
    """``memory_index_text`` from its parts, which a search hit's payload also carries: the
    rerankers read a candidate exactly as it was indexed, without a database round trip."""
    name = (subject or "").split(":", 1)[-1].strip()
    who = f" {name}:" if name and not name.startswith(("thr_", "run_")) else ""
    return f"[{observed[:10]}]{who} {memory_type.lower()}: {content}"


def memory_context_text(m: CanonicalMemory) -> str:
    """The memory's second key: a verbatim turn read after the message it answers, which a
    reply rarely restates (``MemoryIntelligenceSettings.index_preceding_turn``). Indexing a
    turn *only* this way lifted LoCoMo recall@10 0.711 -> 0.736 and cost multi-hop and
    temporal questions the turn's own words; both keys keep both (ADR 0025). Every other
    memory's second key is its first."""
    text = memory_index_text(m)
    prior = m.system_metadata.get("preceding_turn") or {}
    if m.system_metadata.get("category") == "verbatim_turn" and (said := prior.get("text")):
        speaker = str(prior.get("speaker") or "").split(":", 1)[-1].strip()
        return f"{speaker}: {said}\n{text}" if speaker else f"{said}\n{text}"
    return text


EPISODE_PREFIX = "epi_"


def episode_id(thread_id: str) -> str:
    """A thread's one episode record: re-indexing a newer summary replaces it."""
    return f"{EPISODE_PREFIX}{thread_id}"


def episode_index_text(
    summary: str, title: str | None, observed_from: datetime, observed_to: datetime
) -> str:
    """The episode's dates and title lead its text, for the reason ``memory_index_text``
    dates a memory: a "last month" question has to be able to match it lexically."""
    start, end = observed_from.date().isoformat(), observed_to.date().isoformat()
    when = start if start == end else f"{start} to {end}"
    name = f" {' '.join(title.split())[:200]}:" if title and title.strip() else ""
    return f"[{when}] conversation{name} {summary.strip()}"


#: Memory states the search index holds: the live ones, and the ones a later memory replaced
#: (history for point-in-time search). Everything else is out of the index.
INDEXED_STATUSES = frozenset({"CURRENT", "SUPERSEDED"})


def memory_time_payload(m: CanonicalMemory) -> dict[str, str]:
    """A memory's valid time and the end of its knowledge time, as filterable instants.

    An unknown start of validity is open (the fact holds as far back as anything asks),
    as is an unknown end. Knowledge ends when the memory was superseded: both supersede
    paths stamp ``updated_at`` at that moment (a later edit of a superseded row would move
    it later, which only ever widens what ``known_at`` returns)."""
    t = m.temporal
    superseded = t.status.value == "SUPERSEDED"
    return {
        "valid_from": (t.valid_from or TIME_MIN).isoformat(),
        "valid_to": (t.valid_to or TIME_MAX).isoformat(),
        "known_to": (m.updated_at if superseded else TIME_MAX).isoformat(),
    }


class Indexer:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        store: SearchStore,
        spaces: DenseSpaces,
        sparse: SparseEncoder,
        cache: CacheProvider | None = None,
        *,
        late: LateInteractionEncoder | None = None,
        batch_size: int = 32,
        embedding_cache_ttl: int = 7 * 24 * 3600,
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.spaces = spaces
        self.sparse = sparse
        self.late = late
        self.cache = cache
        self.batch_size = batch_size
        self.embedding_cache_ttl = embedding_cache_ttl
        self.assist = assist or LLMAssist.disabled()

    @property
    def embedding(self) -> EmbeddingProvider:
        """The one encoder that stands for a text where a single vector is wanted."""
        return self.spaces.primary

    #: The memories' key layout: ``mk2`` is two keys per memory (ADR 0025). A layout change
    #: is a new collection, like a model change.
    MEMORY_KEYS = "mk2"

    @property
    def fingerprint(self) -> str:
        """Every encoder's fingerprint and the key layout. The late-interaction part is
        hashed: the whole is stored in a 200-character column and is part of a collection
        name, and the readable dense part already takes three quarters of that."""
        late = ""
        if self.late is not None:
            late = "|li-" + hashlib.sha256(self.late.fingerprint().encode()).hexdigest()[:12]
        return f"{self.spaces.fingerprint()}|{self.sparse.fingerprint()}{late}|{self.MEMORY_KEYS}"

    #: Qdrant accepts letters, digits, hyphen and underscore in a collection name. Anything
    #: else has to be folded, or a provider whose fingerprint contains a path or a colon
    #: takes down every write with a 422 that names the collection rather than the cause.
    _SAFE = re.compile(r"[^a-z0-9_-]+")

    def collection(self, base: str) -> str:
        """Collection name bound to the embedding space (dense + sparse fingerprints)."""
        space = self.fingerprint.replace("|", "__")
        return self._SAFE.sub("_", f"{base}_{space}".lower())

    async def ensure_collections(self) -> None:
        sparse_idf = getattr(self.sparse, "server_side_idf", True)
        dims = self.spaces.dimensions
        for base in (KNOWLEDGE, MEMORIES):
            memories = base == MEMORIES
            await self.store.ensure_collection(
                CollectionSpec(
                    name=self.collection(base),
                    # the memories carry each space twice: their own key and the context key
                    dense={**dims, **{n.context: w for n, w in dims.items()}} if memories else dims,
                    sparse=True,
                    sparse_idf=sparse_idf,
                    sparse_context=memories,
                    late=self.late.dimension if self.late is not None else None,
                    # Memories are small and every query reads them: holding their payloads
                    # in RAM costs a few MB and saves a page-cache read per returned hit on
                    # the remote store. Knowledge chunks are the bulk of the index and stay
                    # on disk.
                    on_disk_payload=base != MEMORIES,
                )
            )

    async def embed_cached(
        self, texts: Sequence[str], hashes: Sequence[str]
    ) -> dict[VectorName, list[list[float]]]:
        """Every space's vectors for ``texts``, the spaces encoded concurrently, each behind
        a content-hash cache keyed by that space's fingerprint."""
        vectors = await asyncio.gather(
            *(self._embed_space(space, texts, hashes) for space in self.spaces.spaces)
        )
        return dict(zip(self.spaces.names, vectors, strict=True))

    async def embed_late(self, texts: Sequence[str]) -> list[list[list[float]]] | None:
        """The late-interaction token vectors of ``texts``; ``None`` without that encoder.
        Not cached: a memory's are ~5 KB and a chunk's ~130 KB, too large to keep in the
        cache beside every dense vector, and they are written to the store once."""
        if self.late is None or not texts:
            return None
        with stage_seconds.labels("index.late").time():
            return await self.late.embed_documents(list(texts))

    async def _embed_space(
        self, space: DenseSpace, texts: Sequence[str], hashes: Sequence[str]
    ) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        keys = [f"emb:{space.encoder.fingerprint()}:{h}" for h in hashes]
        if self.cache is not None:
            try:
                cached = await self.cache.mget(keys)
            except CacheUnavailable:
                cached = [None] * len(keys)
            for i, raw in enumerate(cached):
                if raw is not None:
                    out[i] = _decode(raw)
        missing = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = await space.encoder.embed_documents([texts[i] for i in batch])
            for i, vec in zip(batch, vectors, strict=True):
                out[i] = vec
        if self.cache is not None and missing:
            with contextlib.suppress(CacheUnavailable):
                await self.cache.mset(
                    {keys[i]: _encode(out[i] or []) for i in missing},
                    ttl_seconds=self.embedding_cache_ttl,
                )
        return [v or [] for v in out]

    async def index_document(self, tenant_id: str, document_id: str, *, force: bool = False) -> int:
        await self.ensure_collections()
        async with self.uow_factory() as uow:
            chunks = await uow.documents.list_chunks(
                tenant_id, document_id, unindexed_only=not force
            )
            document = await uow.documents.get(tenant_id, document_id)
            keys = await uow.documents.visibility_keys(tenant_id, document_id)
            all_chunks = (
                chunks if force else await uow.documents.list_chunks(tenant_id, document_id)
            )
            nodes = await uow.documents.list_nodes(tenant_id, document_id)
        if not chunks or document is None:
            return 0
        async with self.assist.bound(document_identity(document)):
            with (
                span("index.document", tenant_id=tenant_id),
                stage_seconds.labels("index.document").time(),
            ):
                n = await self._index_chunks(
                    chunks,
                    visibility_keys=keys,
                    document_title=document.title,
                    thread_id=document.thread_id,
                )
                # hierarchical summaries (M9): one per section/subsection/document, indexed as
                # kind="summary" records so GLOBAL_SUMMARY questions can find them
                extractive = build_summaries(nodes, all_chunks, title=document.title)
                summaries = extractive
                if self.assist.wants("summaries"):
                    summaries = await abstractive_summaries(
                        self.assist, nodes, all_chunks, summaries, title=document.title
                    )
                await self._index_summaries(
                    summaries,
                    nodes=nodes,
                    tenant_id=tenant_id,
                    document=document,
                    visibility_keys=keys,
                    generated_ids={
                        nid for nid, value in summaries.items() if value != extractive[nid]
                    },
                )
                async with self.uow_factory() as uow:
                    # SQL summaries feed parent expansion, which has no generated-provenance
                    # column. Keep that evidence extractive; generated search representations
                    # above carry their provider label and cannot prove their own claims.
                    await uow.documents.set_node_summaries(tenant_id, extractive)
                    await uow.documents.mark_chunks_indexed(
                        [c.chunk_id for c in chunks],
                        fingerprint=self.fingerprint,
                        indexed_at=datetime.now(UTC),
                    )
                    await uow.commit()
                stale = await self._purge_superseded(
                    tenant_id,
                    document_id,
                    keep={c.chunk_id for c in all_chunks} | {f"sum_{nid}" for nid in summaries},
                )
        log.info(
            "index.document_done",
            tenant_id=tenant_id,
            document_id=document_id,
            chunks=n,
            summaries=len(summaries),
            superseded=stale,
        )
        return n

    async def _purge_superseded(self, tenant_id: str, document_id: str, *, keep: set[str]) -> int:
        """Drop index entries for chunks this document no longer has.

        Parsing assigns fresh chunk and node ids, so a re-parse writes a whole new generation
        of vectors and leaves the previous one behind. Measured on a running service: one
        upload, retried three times by the job's own retry policy, left **72 vectors for 10
        chunks** — four generations, of which Postgres kept one. The orphans are not inert:
        they take top ranks, occupy evidence seeds, and their node ids resolve to nothing, so
        the evidence stage silently reported COMPLETE for a check it could not run.

        The index must mirror the document's chunks, so anything not in ``keep`` goes. The
        summary records share the document filter and are rewritten on every pass, so they are
        kept by id rather than by generation.
        """
        collection = self.collection(KNOWLEDGE)
        flt = SearchFilter(tenant_id=tenant_id, must={"document_id": document_id})
        try:
            present = await self.store.record_ids(collection, flt)
        except Exception as exc:  # a purge failure must not fail the indexing that preceded it
            log.warning("index.purge_failed", document_id=document_id, error=type(exc).__name__)
            return 0
        stale = [r for r in present if r.startswith(("chk_", "sum_")) and r not in keep]
        if stale:
            await self.store.delete(collection, stale)
        return len(stale)

    async def _index_summaries(
        self,
        summaries: dict[str, str],
        *,
        nodes: Sequence[DocumentNode],
        tenant_id: str,
        document: Document,
        visibility_keys: Sequence[str],
        generated_ids: set[str],
    ) -> None:
        if not summaries:
            return
        by_id = {n.node_id: n for n in nodes}
        ids = list(summaries)
        texts = [summaries[i] for i in ids]
        dense = await self.embed_cached(texts, [content_hash(t) + ":sum" for t in texts])
        sparse = self.sparse.encode_documents(texts)
        late = await self.embed_late(texts)
        records = [
            SearchRecord(
                record_id=f"sum_{nid}",
                collection=self.collection(KNOWLEDGE),
                tenant_id=tenant_id,
                dense=_vectors_at(dense, i),
                sparse=sparse[i],
                late=late[i] if late else None,
                payload={
                    "kind": "summary",
                    "provider": "llm" if nid in generated_ids else "native",
                    "visibility_keys": list(visibility_keys),
                    "document_id": document.document_id,
                    "document_title": document.title,
                    "node_id": nid,
                    "thread_id": document.thread_id,
                    "page": by_id[nid].page_start if nid in by_id else None,
                    "section_path": by_id[nid].section_path if nid in by_id else "",
                    "representation": by_id[nid].representation.value
                    if nid in by_id
                    else "SUMMARY",
                    "script": detect_script(texts[i]).value,
                    "text": texts[i][:2000],
                    "text_hash": content_hash(texts[i]),
                },
            )
            for i, nid in enumerate(ids)
        ]
        await self.store.upsert(records)

    async def _index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        visibility_keys: Sequence[str],
        document_title: str,
        thread_id: str | None,
    ) -> int:
        texts = [c.contextual_text for c in chunks]
        dense = await self._dense_for_chunks(chunks, texts)
        sparse = self.sparse.encode_documents(texts)
        late = await self.embed_late(texts)
        records = [
            SearchRecord(
                record_id=c.chunk_id,
                collection=self.collection(KNOWLEDGE),
                tenant_id=c.tenant_id,
                dense=_vectors_at(dense, i),
                sparse=sparse[i],
                late=late[i] if late else None,
                payload={
                    "kind": "chunk",
                    "visibility_keys": list(visibility_keys),
                    "document_id": c.document_id,
                    "document_version_id": c.document_version_id,
                    "document_title": document_title,
                    "node_id": c.node_id,
                    "thread_id": thread_id,
                    "page": c.page,
                    "section_path": c.section_path,
                    "script": detect_script(c.text).value,
                    "text": c.text[:2000],
                    "text_hash": c.text_hash,
                    "token_estimate": c.token_estimate,
                },
            )
            for i, c in enumerate(chunks)
        ]
        await self.store.upsert(records)
        return len(records)

    async def index_memories(self, tenant_id: str, memory_ids: Sequence[str]) -> int:
        """Upsert CURRENT memories into the memories collection, and SUPERSEDED ones as
        history (``current: false``); remove every other state (expired, contradicted,
        retracted, forgotten). Every ordinary search filters ``current``, so history is
        read only by a search that asks for a point in time (``as_of``: what was true then;
        ``known_at``: what had been learned by then)."""
        await self.ensure_collections()
        async with self.uow_factory() as uow:
            memories = await uow.memories.get_many(tenant_id, memory_ids)
        found = {m.memory_id for m in memories}
        live = [m for m in memories if m.temporal.status.value in INDEXED_STATUSES]
        gone = [
            m.memory_id for m in memories if m.temporal.status.value not in INDEXED_STATUSES
        ] + [i for i in memory_ids if i not in found]
        collection = self.collection(MEMORIES)
        if gone:
            await self.store.delete(collection, gone)
        n = 0
        if live:
            with (
                span("index.memories", tenant_id=tenant_id),
                stage_seconds.labels("index.memories").time(),
            ):
                texts = [memory_index_text(m) for m in live]
                contexts = [memory_context_text(m) for m in live]
                # The cache key names the string that was embedded: ":mem3" is the memory's
                # own key, the hash of the context text is the context key's (":mem2" was
                # the context text alone, under the memory's hash).
                dense = await self.embed_cached(texts, [m.normalized_hash + ":mem3" for m in live])
                # a memory whose second key is its first is not encoded twice
                differs = [
                    i for i, (t, c) in enumerate(zip(texts, contexts, strict=True)) if t != c
                ]
                dense_ctx = await self.embed_cached(
                    [contexts[i] for i in differs],
                    [content_hash(contexts[i]) + ":ctx3" for i in differs],
                )
                sparse = self.sparse.encode_documents(texts)
                sparse_ctx = self.sparse.encode_documents([contexts[i] for i in differs])
                where = {i: n for n, i in enumerate(differs)}
                late = await self.embed_late(texts)
                records = [
                    SearchRecord(
                        record_id=m.memory_id,
                        collection=collection,
                        tenant_id=m.tenant_id,
                        dense={
                            **_vectors_at(dense, i),
                            **{
                                name.context: vector
                                for name, vector in (
                                    _vectors_at(dense_ctx, where[i])
                                    if i in where
                                    else _vectors_at(dense, i)
                                ).items()
                            },
                        },
                        sparse=sparse[i],
                        sparse_context=sparse_ctx[where[i]] if i in where else sparse[i],
                        late=late[i] if late else None,
                        payload={
                            "kind": "memory",
                            "visibility_keys": list(m.system_metadata.get("visibility_keys", [])),
                            "memory_type": m.memory_type.value,
                            "category": m.system_metadata.get("category"),
                            "provider": m.system_metadata.get("provider"),
                            "derived": bool(m.system_metadata.get("source_revisions")),
                            "source_observed_to": m.system_metadata.get("source_observed_to"),
                            # lifetime, temporal_status, representation and thread_id are
                            # not written: no reader reads them and the payload projection
                            # (ports.search.PAYLOAD_FIELDS) would not return them if one
                            # did. They were roughly a tenth of every point's payload,
                            # which is resident memory as soon as the memories collection
                            # keeps its payload in RAM. `current` stays: it is a filter.
                            "current": m.temporal.status.value == "CURRENT",
                            **memory_time_payload(m),
                            "subject": m.subject,
                            "predicate": m.predicate,
                            "object": (m.object or "")[:300],
                            "owner_principal": m.owner_principal,
                            "contributors": list(m.system_metadata.get("contributors", [])),
                            "contradicts": list(m.temporal.contradicts),
                            # Only when there are any: an empty list on every memory point is
                            # payload that is resident memory as soon as the collection keeps
                            # its payload in RAM, for a field most memories never have.
                            **({"connections": edges} if (edges := payload_edges(m)) else {}),
                            "confidence": m.confidence,
                            "reinforcement": m.reinforcement_count,
                            "observed_at": m.temporal.observed_at.isoformat(),
                            "script": detect_script(m.content).value,
                            # relative dates the text names, resolved against observed_at
                            "dated_mentions": list(m.system_metadata.get("dated_mentions", [])),
                            "source_refs": [
                                ref.model_dump(mode="json", exclude_none=True) for ref in m.evidence
                            ],
                            "text": m.content[:2000],
                            # the conversation's previous turn, which retrieval reads to
                            # score a turn with its neighbours (learned_fusion's lift)
                            **(
                                {"preceding_source_id": prior_source}
                                if (
                                    prior_source := (
                                        m.system_metadata.get("preceding_turn") or {}
                                    ).get("source_id")
                                )
                                else {}
                            ),
                            # so identical memories group in the store's payload rather than
                            # being rehashed on every retrieval (see engine._dedup)
                            "text_hash": content_hash(m.content),
                        },
                    )
                    for i, m in enumerate(live)
                ]
                await self.store.upsert(records)
                n = len(records)
        async with self.uow_factory() as uow:
            await uow.memories.mark_indexed(
                [m.memory_id for m in memories],
                fingerprint=self.fingerprint,
                indexed_at=datetime.now(UTC),
            )
            await uow.commit()
        log.info("index.memories_done", tenant_id=tenant_id, upserted=n, removed=len(gone))
        return n

    async def index_episode(self, tenant_id: str, thread_id: str) -> bool:
        """Upsert a thread's one searchable episode (its latest summary, then a digest of
        the messages after it), or remove the episode when the thread is gone or has no
        visible messages. True when an episode is indexed.

        An episode is what lets a later conversation find an earlier one ("what did we
        decide last month?"); the extracted memories alone lose the thread's narrative.

        Its audience is the thread's owning user (``user:`` key: that user and the agents
        acting for them, in any thread), because reaching across threads is the point and
        the owner can already read every message the summary is folded from. A thread with
        no owning user keeps the thread's own audience, readable only inside it. Nothing
        reads episodes unless a search names the ``episode`` kind.
        """
        await self.ensure_collections()
        collection = self.collection(MEMORIES)
        record_id = episode_id(thread_id)
        source = await self._episode_source(tenant_id, thread_id)
        if source is None:
            await self.store.delete(collection, [record_id])
            return False
        thread, body, observed_from, observed_to = source
        text = episode_index_text(body, thread.title, observed_from, observed_to)
        keys = (
            [f"user:{tenant_id}/{thread.owner_user_id}"]
            if thread.owner_user_id
            else [f"thread:{tenant_id}/{thread_id}"]
        )
        with span("index.episode", tenant_id=tenant_id):
            dense = await self.embed_cached([text], [f"{content_hash(text)}:epi1"])
            sparse = self.sparse.encode_documents([text])
            late = await self.embed_late([text])
            await self.store.upsert(
                [
                    SearchRecord(
                        record_id=record_id,
                        collection=collection,
                        tenant_id=tenant_id,
                        # an episode is one text: its context key is its own key
                        dense={
                            **_vectors_at(dense, 0),
                            **{name.context: v for name, v in _vectors_at(dense, 0).items()},
                        },
                        sparse=sparse[0],
                        sparse_context=sparse[0],
                        late=late[0] if late else None,
                        payload={
                            "kind": "episode",
                            "visibility_keys": keys,
                            "thread_id": thread_id,
                            "current": True,
                            "observed_at": observed_to.isoformat(),
                            "observed_from": observed_from.isoformat(),
                            "script": detect_script(body).value,
                            "text": text[:4000],
                            "text_hash": content_hash(text),
                        },
                    )
                ]
            )
        log.info("index.episode_done", tenant_id=tenant_id, chars=len(text))
        return True

    async def _episode_source(
        self, tenant_id: str, thread_id: str
    ) -> tuple[Thread, str, datetime, datetime] | None:
        """The thread, its episode body and the span of its messages; ``None`` when the
        thread is gone or has nothing visible to recall."""
        async with self.uow_factory() as uow:
            thread = await uow.threads.get(tenant_id, thread_id)
            if thread is None:
                return None
            summary = await uow.summaries.latest(tenant_id, thread_id)
            covered = summary.covers_to_sequence if summary else 0
            head = await uow.messages.list_after(tenant_id, thread_id, after_sequence=0, limit=1)
            # what the summary does not cover yet, as the extractive digest (no model): a
            # thread shorter than one summary period is still an episode
            tail = await uow.messages.list_after(
                tenant_id, thread_id, after_sequence=covered, limit=SUMMARY_SOURCE_MESSAGES
            )
            # with everything summarised, the episode ends at the last message it covers
            end = tail[-1:] or await uow.messages.list_after(
                tenant_id, thread_id, after_sequence=max(covered - 1, 0), limit=1
            )
        lines = digest(tail)
        while lines and sum(len(line) + 1 for line in lines) > SUMMARY_MAX_CHARS:
            lines.pop(0)
        body = "\n".join(part for part in ((summary.text if summary else ""), *lines) if part)
        if not body.strip():
            return None
        observed_to = end[0].occurred_at if end else datetime.now(UTC)
        observed_from = head[0].occurred_at if head else observed_to
        return thread, body, observed_from, observed_to

    async def _dense_for_chunks(
        self, chunks: Sequence[Chunk], texts: list[str]
    ) -> dict[VectorName, list[list[float]]]:
        """Per-chunk embeddings.

        This used to branch into late chunking when the provider exposed ``embed_spans``. That
        path is gone with the flag: it needs token-level output, which a served embedding
        endpoint cannot give — it returns one pooled vector per input — so it was mutually
        exclusive with running the models as their own tier.
        """
        return await self.embed_cached(texts, [c.text_hash + ":ctx" for c in chunks])

    async def rebuild_document(self, tenant_id: str, document_id: str) -> int:
        return await self.index_document(tenant_id, document_id, force=True)

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        await self.store.delete_by_filter(
            self.collection(KNOWLEDGE),
            SearchFilter(tenant_id=tenant_id, must={"document_id": document_id}),
        )


def _vectors_at(dense: dict[VectorName, list[list[float]]], i: int) -> dict[VectorName, Any]:
    return {space: vectors[i] for space, vectors in dense.items()}


def _encode(vec: list[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def _decode(raw: bytes) -> list[float]:
    import struct

    n = len(raw) // 4
    return list(struct.unpack(f"<{n}f", raw))
