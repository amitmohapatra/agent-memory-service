"""Indexer: canonical chunks/memories -> search records (dense spaces + BM25 sparse + payload).

The search index is rebuildable: ``rebuild_document`` re-indexes from PostgreSQL. Collection
names embed the fingerprint of every dense space so a model change creates a new vector
space instead of mixing spaces. Payload carries only security/filter fields and short display
text; every record is tagged with the Unicode script of its text.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.documents import Chunk, Document, DocumentNode
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.script import detect_script
from memory_service.modules.context.summaries import abstractive_summaries, build_summaries
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import document_identity
from memory_service.modules.memory.connections import payload_edges
from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.models import EmbeddingProvider, SparseEncoder
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


def memory_index_text(m: CanonicalMemory) -> str:
    """What gets embedded and BM25-indexed for a memory: the fact *with its context*.

    Documents already do this — a chunk is indexed as ``contextual_text``, not ``text`` —
    and memories were the one collection that skipped it: ``"semantic: <content>"``, no date,
    no subject. The date and the subject are both on the object and both written to the
    payload, so the system was handed the attribution and threw it away before indexing.
    On dated conversational memory every question is "who did what, when"; a temporal
    query ("when did Caroline ...") cannot lexically match a fact whose index text has no
    date, and a dense query is pulled toward the wrong person when the subject is absent.
    """
    when = m.temporal.observed_at.date().isoformat()
    subject = (m.subject or "").split(":", 1)[-1].strip()
    who = f" {subject}:" if subject and not subject.startswith(("thr_", "run_")) else ""
    return f"[{when}]{who} {m.memory_type.value.lower()}: {m.content}"


class Indexer:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        store: SearchStore,
        spaces: DenseSpaces,
        sparse: SparseEncoder,
        cache: CacheProvider | None = None,
        *,
        batch_size: int = 32,
        embedding_cache_ttl: int = 7 * 24 * 3600,
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.spaces = spaces
        self.sparse = sparse
        self.cache = cache
        self.batch_size = batch_size
        self.embedding_cache_ttl = embedding_cache_ttl
        self.assist = assist or LLMAssist.disabled()

    @property
    def embedding(self) -> EmbeddingProvider:
        """The one encoder that stands for a text where a single vector is wanted."""
        return self.spaces.primary

    @property
    def fingerprint(self) -> str:
        return f"{self.spaces.fingerprint()}|{self.sparse.fingerprint()}"

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
        for base in (KNOWLEDGE, MEMORIES):
            await self.store.ensure_collection(
                CollectionSpec(
                    name=self.collection(base),
                    dense=self.spaces.dimensions,
                    sparse=True,
                    sparse_idf=sparse_idf,
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
        records = [
            SearchRecord(
                record_id=f"sum_{nid}",
                collection=self.collection(KNOWLEDGE),
                tenant_id=tenant_id,
                dense=_vectors_at(dense, i),
                sparse=sparse[i],
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
        records = [
            SearchRecord(
                record_id=c.chunk_id,
                collection=self.collection(KNOWLEDGE),
                tenant_id=c.tenant_id,
                dense=_vectors_at(dense, i),
                sparse=sparse[i],
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
        """Upsert CURRENT memories into the memories collection; remove every other state
        (superseded, expired, retracted, forgotten) so only live intelligence is searchable.
        Superseded memories stay in PostgreSQL for temporal/audit queries."""
        await self.ensure_collections()
        async with self.uow_factory() as uow:
            memories = await uow.memories.get_many(tenant_id, memory_ids)
        found = {m.memory_id for m in memories}
        live = [m for m in memories if m.temporal.status.value == "CURRENT"]
        gone = [m.memory_id for m in memories if m.temporal.status.value != "CURRENT"] + [
            i for i in memory_ids if i not in found
        ]
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
                # ":mem2": the text changed shape, so vectors cached under the old key
                # would be embeddings of a different string.
                dense = await self.embed_cached(texts, [m.normalized_hash + ":mem2" for m in live])
                sparse = self.sparse.encode_documents(texts)
                records = [
                    SearchRecord(
                        record_id=m.memory_id,
                        collection=collection,
                        tenant_id=m.tenant_id,
                        dense=_vectors_at(dense, i),
                        sparse=sparse[i],
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
                            "current": True,
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
