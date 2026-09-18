"""Indexer: canonical chunks/memories -> search records (dense + BM25 sparse + payload).

The search index is rebuildable: ``rebuild_document`` re-indexes from PostgreSQL. Collection
names embed the embedding fingerprint so a model change creates a new vector space instead
of mixing spaces. Payload carries only security/filter fields and short display text.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.documents import Chunk, Document, DocumentNode
from memory_service.domain.ids import content_hash
from memory_service.modules.context.summaries import abstractive_summaries, build_summaries
from memory_service.modules.llm.assist import LLMAssist
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.models import EmbeddingProvider, SparseEncoder
from memory_service.ports.search import CollectionSpec, SearchFilter, SearchRecord, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

KNOWLEDGE = "knowledge"
MEMORIES = "memories"


class Indexer:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        store: SearchStore,
        embedding: EmbeddingProvider,
        sparse: SparseEncoder,
        cache: CacheProvider | None = None,
        *,
        batch_size: int = 32,
        embedding_cache_ttl: int = 7 * 24 * 3600,
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.embedding = embedding
        self.sparse = sparse
        self.cache = cache
        self.batch_size = batch_size
        self.embedding_cache_ttl = embedding_cache_ttl
        self.assist = assist or LLMAssist.disabled()
        # M10 (benchmark-gated): ColBERT multivectors are added to chunk records when set
        self.late_interaction: Any = None

    @property
    def fingerprint(self) -> str:
        late = f"|{self.late_interaction.fingerprint()}" if self.late_interaction else ""
        return f"{self.embedding.fingerprint()}|{self.sparse.fingerprint()}{late}"

    def collection(self, base: str) -> str:
        """Collection name bound to the embedding space (dense + sparse + late fingerprints)."""
        space = self.fingerprint.replace("|", "__")
        return f"{base}_{space}".replace("/", "_").replace(".", "_").lower()

    async def ensure_collections(self) -> None:
        sparse_idf = getattr(self.sparse, "server_side_idf", True)
        late_dim = self.late_interaction.dimension if self.late_interaction else None
        for base in (KNOWLEDGE, MEMORIES):
            await self.store.ensure_collection(
                CollectionSpec(
                    name=self.collection(base),
                    dense_dim=self.embedding.dimension,
                    sparse=True,
                    sparse_idf=sparse_idf,
                    late_interaction_dim=late_dim if base == KNOWLEDGE else None,
                )
            )

    async def embed_cached(self, texts: Sequence[str], hashes: Sequence[str]) -> list[list[float]]:
        """Embed with a content-hash cache keyed by the embedding fingerprint."""
        out: list[list[float] | None] = [None] * len(texts)
        keys = [f"emb:{self.embedding.fingerprint()}:{h}" for h in hashes]
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
            vectors = await self.embedding.embed_documents([texts[i] for i in batch])
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
            summaries = build_summaries(nodes, all_chunks, title=document.title)
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
            )
            async with self.uow_factory() as uow:
                await uow.documents.set_node_summaries(tenant_id, summaries)
                await uow.documents.mark_chunks_indexed(
                    [c.chunk_id for c in chunks],
                    fingerprint=self.fingerprint,
                    indexed_at=datetime.now(UTC),
                )
                await uow.commit()
            stale = await self._purge_superseded(
                tenant_id, document_id, keep={c.chunk_id for c in all_chunks}
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
        stale = [r for r in present if r.startswith("chk_") and r not in keep]
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
    ) -> None:
        if not summaries:
            return
        by_id = {n.node_id: n for n in nodes}
        ids = list(summaries)
        texts = [summaries[i] for i in ids]
        dense = await self.embed_cached(texts, [content_hash(t) + ":sum" for t in texts])
        sparse = self.sparse.encode_documents(texts)
        # every point in a multivector collection needs the vector (local mode requires it)
        late = (
            await self.late_interaction.embed_documents_multi(texts)
            if self.late_interaction is not None
            else None
        )
        records = [
            SearchRecord(
                record_id=f"sum_{nid}",
                collection=self.collection(KNOWLEDGE),
                tenant_id=tenant_id,
                dense=dense[i],
                sparse=sparse[i],
                late_interaction=late[i] if late is not None else None,
                payload={
                    "kind": "summary",
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
        late = (
            await self.late_interaction.embed_documents_multi(texts)
            if self.late_interaction is not None
            else None
        )
        records = [
            SearchRecord(
                record_id=c.chunk_id,
                collection=self.collection(KNOWLEDGE),
                tenant_id=c.tenant_id,
                dense=dense[i],
                sparse=sparse[i],
                late_interaction=late[i] if late is not None else None,
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
                    "text": c.text[:2000],
                    "text_hash": c.text_hash,
                    "entities": c.entities[:12],
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
                texts = [f"{m.memory_type.value.lower()}: {m.content}" for m in live]
                dense = await self.embed_cached(texts, [m.normalized_hash + ":mem" for m in live])
                sparse = self.sparse.encode_documents(texts)
                records = [
                    SearchRecord(
                        record_id=m.memory_id,
                        collection=collection,
                        tenant_id=m.tenant_id,
                        dense=dense[i],
                        sparse=sparse[i],
                        payload={
                            "kind": "memory",
                            "visibility_keys": list(m.system_metadata.get("visibility_keys", [])),
                            "memory_type": m.memory_type.value,
                            "lifetime": m.lifetime.value,
                            "temporal_status": m.temporal.status.value,
                            "current": True,
                            "subject": m.subject,
                            "predicate": m.predicate,
                            "object": (m.object or "")[:300],
                            "owner_principal": m.owner_principal,
                            "contributors": list(m.system_metadata.get("contributors", [])),
                            "contradicts": list(m.temporal.contradicts),
                            "importance": m.importance,
                            "confidence": m.confidence,
                            "observed_at": m.temporal.observed_at.isoformat(),
                            "thread_id": m.scope.thread_id,
                            "text": m.content[:2000],
                            "representation": "MEMORY",
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
    ) -> list[list[float]]:
        """Per-chunk embeddings, or late chunking (M10) when the provider supports spans:
        the document's chunks are embedded as one sequence and pooled per chunk span."""
        embed_spans = getattr(self.embedding, "embed_spans", None)
        if embed_spans is None or len(chunks) < 2:
            return await self.embed_cached(texts, [c.text_hash + ":ctx" for c in chunks])
        document = ""
        spans: list[tuple[int, int]] = []
        for c in chunks:
            start = len(document)
            document += c.text + "\n\n"
            spans.append((start, start + len(c.text)))
        return await embed_spans(document, spans, texts)

    async def rebuild_document(self, tenant_id: str, document_id: str) -> int:
        return await self.index_document(tenant_id, document_id, force=True)

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        from memory_service.ports.search import SearchFilter

        await self.store.delete_by_filter(
            self.collection(KNOWLEDGE),
            SearchFilter(tenant_id=tenant_id, must={"document_id": document_id}),
        )


def _encode(vec: list[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def _decode(raw: bytes) -> list[float]:
    import struct

    n = len(raw) // 4
    return list(struct.unpack(f"<{n}f", raw))
