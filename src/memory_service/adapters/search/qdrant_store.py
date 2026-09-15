"""Qdrant SearchStore (Apache-2.0).

Collections carry a named dense vector (``dense``) and a named sparse vector (``bm25``) with
Qdrant's server-side IDF modifier, so BM25 scoring happens in the store. Hybrid search uses
``query_points`` with two prefetches fused by native RRF. Every query carries a tenant
filter and a ``visibility_keys`` MatchAny filter that Qdrant applies before ranking.

``qdrant_local_path`` (e.g. ``:memory:``) runs the same client API in-process for tests.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from memory_service.config.settings import SearchSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.models import ProviderInfo
from memory_service.ports.search import (
    CollectionSpec,
    Retriever,
    SearchFilter,
    SearchHit,
    SearchRecord,
    SparseVector,
)

DENSE = "dense"
SPARSE = "bm25"


def point_id(record_id: str) -> str:
    """Qdrant ids must be ints or UUIDs; derive a stable UUID5 from our ULID-based ids."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


def _filter(flt: SearchFilter) -> models.Filter:
    must: list[Any] = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=flt.tenant_id))
    ]
    for key, value in flt.must.items():
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    for key, values in flt.must_any.items():
        must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    must_not: list[Any] = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in flt.must_not.items()
    ]
    if must_not:
        return models.Filter(must=must, must_not=must_not)
    return models.Filter(must=must)


class QdrantSearchStore:
    info = ProviderInfo(
        name="qdrant",
        license="Apache-2.0",
        origin="qdrant/qdrant",
        locality="local",
        data_residency="deployment",
    )

    def __init__(self, settings: SearchSettings) -> None:
        self.settings = settings
        if settings.qdrant_local_path:
            self._client = (
                AsyncQdrantClient(location=settings.qdrant_local_path)
                if settings.qdrant_local_path == ":memory:"
                else AsyncQdrantClient(path=settings.qdrant_local_path)
            )
            self._local = True
        else:
            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
                timeout=int(settings.timeout_seconds),
            )
            self._local = False
        self._known: set[str] = set()

    def _name(self, collection: str) -> str:
        return f"{self.settings.collection_prefix}_{collection}"

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        name = self._name(spec.name)
        if name in self._known:
            return
        try:
            exists = await self._client.collection_exists(name)
            if not exists:
                vectors: dict[str, Any] = {}
                if spec.dense_dim:
                    vectors[DENSE] = models.VectorParams(
                        size=spec.dense_dim, distance=models.Distance.COSINE, on_disk=spec.on_disk
                    )
                sparse = (
                    {
                        SPARSE: models.SparseVectorParams(
                            modifier=models.Modifier.IDF if spec.sparse_idf else None,
                            index=models.SparseIndexParams(on_disk=spec.on_disk),
                        )
                    }
                    if spec.sparse
                    else None
                )
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=vectors,
                    sparse_vectors_config=sparse,
                    on_disk_payload=self.settings.on_disk_payload and not self._local,
                )
                if not self._local:  # local mode has no payload indexes
                    for field in ("tenant_id", "visibility_keys", "kind", "document_id"):
                        await self._client.create_payload_index(
                            name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                        )
        except Exception as exc:
            raise DependencyUnavailable(
                f"qdrant ensure_collection failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._known.add(name)

    async def upsert(self, records: Sequence[SearchRecord]) -> None:
        if not records:
            return
        by_collection: dict[str, list[models.PointStruct]] = {}
        for r in records:
            vector: dict[str, Any] = {}
            if r.dense is not None:
                vector[DENSE] = list(r.dense)
            if r.sparse is not None:
                vector[SPARSE] = models.SparseVector(
                    indices=r.sparse.indices, values=r.sparse.values
                )
            payload = {**r.payload, "record_id": r.record_id, "tenant_id": r.tenant_id}
            by_collection.setdefault(self._name(r.collection), []).append(
                models.PointStruct(id=point_id(r.record_id), vector=vector, payload=payload)
            )
        with span("search.upsert"), stage_seconds.labels("search.upsert").time():
            for name, points in by_collection.items():
                try:
                    await self._client.upsert(collection_name=name, points=points, wait=True)
                except Exception as exc:
                    raise DependencyUnavailable(
                        f"qdrant upsert failed: {type(exc).__name__}"
                    ) from exc

    async def delete(self, collection: str, record_ids: Sequence[str]) -> None:
        if not record_ids:
            return
        await self._client.delete(
            collection_name=self._name(collection),
            points_selector=models.PointIdsList(points=[point_id(r) for r in record_ids]),
            wait=True,
        )

    async def delete_by_filter(self, collection: str, flt: SearchFilter) -> int:
        name = self._name(collection)
        before = await self.count(collection, flt)
        await self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(filter=_filter(flt)),
            wait=True,
        )
        return before

    def _hit(self, p: Any, retriever: Retriever) -> SearchHit:
        payload = dict(p.payload or {})
        return SearchHit(
            record_id=str(payload.get("record_id", p.id)),
            score=float(p.score),
            retriever=retriever,
            payload=payload,
        )

    async def search_dense(
        self, collection: str, vector: Sequence[float], flt: SearchFilter, *, limit: int
    ) -> list[SearchHit]:
        with span("search.dense"), stage_seconds.labels("retrieval.dense").time():
            res = await self._client.query_points(
                collection_name=self._name(collection),
                query=list(vector),
                using=DENSE,
                query_filter=_filter(flt),
                limit=limit,
                with_payload=True,
            )
        return [self._hit(p, "dense") for p in res.points]

    async def search_sparse(
        self, collection: str, vector: SparseVector, flt: SearchFilter, *, limit: int
    ) -> list[SearchHit]:
        if not vector.indices:
            return []
        with span("search.sparse"), stage_seconds.labels("retrieval.bm25").time():
            res = await self._client.query_points(
                collection_name=self._name(collection),
                query=models.SparseVector(indices=vector.indices, values=vector.values),
                using=SPARSE,
                query_filter=_filter(flt),
                limit=limit,
                with_payload=True,
            )
        return [self._hit(p, "bm25") for p in res.points]

    async def search_hybrid(
        self,
        collection: str,
        *,
        dense: Sequence[float] | None,
        sparse: SparseVector | None,
        flt: SearchFilter,
        limit: int,
        prefetch_limit: int,
    ) -> list[SearchHit]:
        prefetch: list[models.Prefetch] = []
        qf = _filter(flt)
        if dense is not None:
            prefetch.append(
                models.Prefetch(query=list(dense), using=DENSE, limit=prefetch_limit, filter=qf)
            )
        if sparse is not None and sparse.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=prefetch_limit,
                    filter=qf,
                )
            )
        if not prefetch:
            return []
        if len(prefetch) == 1:
            single = prefetch[0]
            res = await self._client.query_points(
                collection_name=self._name(collection),
                query=single.query,
                using=single.using,
                query_filter=qf,
                limit=limit,
                with_payload=True,
            )
            retriever: Retriever = "dense" if single.using == DENSE else "bm25"
            return [self._hit(p, retriever) for p in res.points]
        with span("search.hybrid"), stage_seconds.labels("retrieval.hybrid").time():
            try:
                res = await self._client.query_points(
                    collection_name=self._name(collection),
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=qf,
                    limit=limit,
                    with_payload=True,
                )
            except Exception as exc:
                raise DependencyUnavailable(
                    f"qdrant hybrid query failed: {type(exc).__name__}: {exc}"
                ) from exc
        return [self._hit(p, "fusion") for p in res.points]

    async def get(self, collection: str, record_ids: Sequence[str]) -> list[SearchRecord]:
        if not record_ids:
            return []
        points = await self._client.retrieve(
            collection_name=self._name(collection),
            ids=[point_id(r) for r in record_ids],
            with_payload=True,
            with_vectors=False,
        )
        return [
            SearchRecord(
                record_id=str(p.payload.get("record_id")),
                collection=collection,
                tenant_id=str(p.payload.get("tenant_id")),
                payload=dict(p.payload or {}),
            )
            for p in points
            if p.payload
        ]

    async def count(self, collection: str, flt: SearchFilter) -> int:
        res = await self._client.count(
            collection_name=self._name(collection), count_filter=_filter(flt), exact=True
        )
        return int(res.count)

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
