"""Indexer: canonical chunks/memories -> search records (dense + BM25 sparse + payload).

The search index is rebuildable: ``rebuild_document`` re-indexes from PostgreSQL. Collection
names embed the embedding fingerprint so a model change creates a new vector space instead
of mixing spaces. Payload carries only security/filter fields and short display text.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from datetime import UTC, datetime

from memory_service.domain.documents import Chunk
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.models import EmbeddingProvider, SparseEncoder
from memory_service.ports.search import CollectionSpec, SearchRecord, SearchStore
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
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.embedding = embedding
        self.sparse = sparse
        self.cache = cache
        self.batch_size = batch_size
        self.embedding_cache_ttl = embedding_cache_ttl

    @property
    def fingerprint(self) -> str:
        return f"{self.embedding.fingerprint()}|{self.sparse.fingerprint()}"

    def collection(self, base: str) -> str:
        """Collection name bound to the embedding space."""
        return f"{base}_{self.embedding.fingerprint()}".replace("/", "_").replace(".", "_").lower()

    async def ensure_collections(self) -> None:
        for base in (KNOWLEDGE, MEMORIES):
            await self.store.ensure_collection(
                CollectionSpec(
                    name=self.collection(base), dense_dim=self.embedding.dimension, sparse=True
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
            async with self.uow_factory() as uow:
                await uow.documents.mark_chunks_indexed(
                    [c.chunk_id for c in chunks],
                    fingerprint=self.fingerprint,
                    indexed_at=datetime.now(UTC),
                )
                await uow.commit()
        log.info("index.document_done", tenant_id=tenant_id, document_id=document_id, chunks=n)
        return n

    async def _index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        visibility_keys: Sequence[str],
        document_title: str,
        thread_id: str | None,
    ) -> int:
        texts = [c.contextual_text for c in chunks]
        dense = await self.embed_cached(texts, [c.text_hash + ":ctx" for c in chunks])
        sparse = self.sparse.encode_documents(texts)
        records = [
            SearchRecord(
                record_id=c.chunk_id,
                collection=self.collection(KNOWLEDGE),
                tenant_id=c.tenant_id,
                dense=dense[i],
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
                    "text": c.text[:2000],
                    "entities": c.entities[:12],
                    "token_estimate": c.token_estimate,
                },
            )
            for i, c in enumerate(chunks)
        ]
        await self.store.upsert(records)
        return len(records)

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
