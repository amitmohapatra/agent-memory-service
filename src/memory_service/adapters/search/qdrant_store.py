"""Qdrant SearchStore (Apache-2.0).

Collections carry one named dense vector per space (``dense_en``, ``dense_ml``) and a named
sparse vector (``bm25``) with Qdrant's server-side IDF modifier, so BM25 scoring happens in
the store. Hybrid search uses ``query_points`` with one prefetch per arm fused by native RRF,
weighted when the fitted weights say so. Every query carries a tenant filter and a
``visibility_keys`` MatchAny filter that Qdrant applies before ranking.

A server is addressed over gRPC (``prefer_grpc``): the query path sends vectors and receives
payloads on every request, and protobuf costs the event loop far less than REST JSON does.
``local_path`` (e.g. ``:memory:``) runs the same client API in-process for tests; it is a
``build_container`` override, never a setting.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx
from prometheus_client import Counter
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from memory_service.config.constants import SEARCH
from memory_service.config.settings import SearchSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import REGISTRY, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.models import ProviderInfo
from memory_service.ports.search import (
    PAYLOAD_FIELDS,
    AnchoredPrefetch,
    CollectionSpec,
    Retriever,
    SearchFilter,
    SearchHit,
    SearchRecord,
    SparseVector,
    VectorName,
)

log = get_logger(__name__)

#: Only the fields a reader actually uses come back from a query (see ``PAYLOAD_FIELDS``).
_PAYLOAD = models.PayloadSelectorInclude(include=list(PAYLOAD_FIELDS))

search_read_retries_total = Counter(
    "memory_search_read_retries_total",
    "Idempotent search reads retried after a connection-level failure",
    ["operation"],
    registry=REGISTRY,
)

#: A connection that died between two requests is not a failure of the query: the same read,
#: issued again on a fresh connection, answers. A judged run hit "Server disconnected without
#: sending a response" four times in 304 queries through Docker's host gateway, each one
#: costing a whole question. One retry, only for reads (they are idempotent; an upsert or a
#: delete is not), after a short pause so a restarting server is not hammered.
_RETRY_PAUSE_SECONDS = (0.05, 0.10)
_CONNECTION_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)


def _is_connection_error(exc: BaseException) -> bool:
    """A broken connection, however this client happens to be wrapping it.

    The REST transport raises httpx errors, usually wrapped in ``ResponseHandlingException``;
    the gRPC transport raises ``AioRpcError`` with status UNAVAILABLE. The gRPC case is read
    through ``.code()`` rather than by importing ``grpc`` for one enum, which also keeps the
    unit test's fake client honest: anything that answers ``code().name == "UNAVAILABLE"``
    is treated the way the real one is.
    """
    if isinstance(exc, ResponseHandlingException):
        return _is_connection_error(exc.source)
    if isinstance(exc, _CONNECTION_ERRORS):
        return True
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            return getattr(code(), "name", "") == "UNAVAILABLE"
        except Exception:
            return False
    return False


async def _read[T](operation: str, call: Callable[[], Awaitable[T]]) -> T:
    try:
        return await call()
    except Exception as exc:
        if not _is_connection_error(exc):
            raise
        search_read_retries_total.labels(operation).inc()
        await asyncio.sleep(random.uniform(*_RETRY_PAUSE_SECONDS))  # noqa: S311 - jitter
        return await call()


#: Payload fields the search filters use; each one is indexed (see _ensure_payload_indexes).
#: ``script`` is the record's Unicode script; ``entities`` anchors the entity prefetch.
_PAYLOAD_INDEXES = {
    "tenant_id": models.PayloadSchemaType.KEYWORD,
    "visibility_keys": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "current": models.PayloadSchemaType.BOOL,
    "script": models.PayloadSchemaType.KEYWORD,
    "entities": models.PayloadSchemaType.KEYWORD,
}


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


def _weight(weights: Mapping[VectorName, float] | None, name: VectorName) -> float:
    return 1.0 if weights is None else float(weights.get(name, 1.0))


def _fusion(rrf_k: int, weights: Sequence[float]) -> models.FusionQuery | models.RrfQuery:
    """The wire form of RRF for our one-based rank convention.

    The historical default (``k=1``, equal arms) keeps the historical wire query: explicit
    RRF has equal scores but may pick different cutoff ties on the server. Any other
    constant, or a fitted weight, is spelled out - Qdrant's ``k`` is zero-based, so ours is
    translated by one, and a weight of 1.0 on every arm is not sent at all.
    """
    weighted = any(weight != 1.0 for weight in weights)
    if rrf_k == 1 and not weighted:
        return models.FusionQuery(fusion=models.Fusion.RRF)
    return models.RrfQuery(rrf=models.Rrf(k=rrf_k + 1, weights=list(weights) if weighted else None))


#: One fusion arm: the prefetch to send, the vector name it was built from, and its RRF
#: weight. The name is carried rather than read back from ``Prefetch.using``, which is an
#: optional string on the wire model - a label read out of it is a label that can be None.
type _Arm = tuple[models.Prefetch, VectorName, float]


def _anchor_filter(flt: SearchFilter, anchor: AnchoredPrefetch) -> models.Filter:
    """The query filter with the anchor's condition added.

    An anchor may only ADD. The base filter's ``must_any`` is what confines the query to the
    caller's audience (``visibility_keys``), and a dict merge lets a colliding key replace it
    instead of narrowing it: an anchor keyed ``visibility_keys`` would widen the arm to
    whatever it listed, inside the store where that boundary is meant to be unconditional.
    A collision is refused rather than silently resolved.
    """
    if collides := sorted(anchor.must_any.keys() & flt.must_any.keys()):
        raise ValueError(
            f"anchored prefetch would replace the query filter on {collides}: "
            "an anchor may only add a condition"
        )
    return _filter(flt.model_copy(update={"must_any": {**flt.must_any, **anchor.must_any}}))


def _arms(
    *,
    dense: Mapping[VectorName, Sequence[float]],
    sparse: SparseVector | None,
    flt: SearchFilter,
    qf: models.Filter,
    prefetch_limit: int,
    weights: Mapping[VectorName, float] | None,
    anchors: Sequence[AnchoredPrefetch],
) -> list[_Arm]:
    """Every arm of one hybrid query, in fusion order: the dense spaces, their anchored
    companions, then BM25. An anchored arm is never weighted: it is the same vector as its
    space, so a weight there would count that space twice."""
    arms: list[_Arm] = [
        (
            models.Prefetch(query=list(vector), using=space.value, limit=prefetch_limit, filter=qf),
            space,
            _weight(weights, space),
        )
        for space, vector in dense.items()
    ]
    for anchor in anchors:
        if anchor.vector not in dense:
            raise ValueError(f"anchored prefetch on {anchor.vector} without its query vector")
        arms.append(
            (
                models.Prefetch(
                    query=list(dense[anchor.vector]),
                    using=anchor.vector.value,
                    limit=prefetch_limit,
                    filter=_anchor_filter(flt, anchor),
                ),
                anchor.vector,
                1.0,
            )
        )
    if sparse is not None and sparse.indices:
        arms.append(
            (
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=VectorName.BM25.value,
                    limit=prefetch_limit,
                    filter=qf,
                ),
                VectorName.BM25,
                _weight(weights, VectorName.BM25),
            )
        )
    return arms


class QdrantSearchStore:
    info = ProviderInfo(
        name="qdrant",
        license="Apache-2.0",
        origin="qdrant/qdrant",
        locality="local",
        data_residency="deployment",
    )

    def __init__(self, settings: SearchSettings, *, local_path: str | None = None) -> None:
        self.settings = settings
        if local_path:
            self._client = (
                AsyncQdrantClient(location=local_path)
                if local_path == ":memory:"
                else AsyncQdrantClient(path=local_path)
            )
            self._local = True
        else:
            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
                timeout=int(SEARCH.timeout_seconds),
                # The REST API stays reachable on 6333 (collection management, the dashboard);
                # every call this client makes goes over gRPC on 6334.
                prefer_grpc=True,
                grpc_port=settings.qdrant_grpc_port,
            )
            self._local = False
        self._known: set[str] = set()

    def _name(self, collection: str) -> str:
        return f"{SEARCH.collection_prefix}_{collection}"

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        name = self._name(spec.name)
        if name in self._known:
            return
        try:
            exists = await self._client.collection_exists(name)
            if not exists:
                vectors = {
                    space.value: models.VectorParams(
                        size=width, distance=models.Distance.COSINE, on_disk=spec.on_disk
                    )
                    for space, width in spec.dense.items()
                }
                sparse = (
                    {
                        VectorName.BM25.value: models.SparseVectorParams(
                            modifier=models.Modifier.IDF if spec.sparse_idf else None,
                            index=models.SparseIndexParams(on_disk=spec.on_disk),
                        )
                    }
                    if spec.sparse
                    else None
                )
                try:
                    await self._client.create_collection(
                        collection_name=name,
                        vectors_config=vectors,
                        sparse_vectors_config=sparse,
                        on_disk_payload=spec.on_disk_payload
                        and SEARCH.on_disk_payload
                        and not self._local,
                    )
                except Exception:
                    # Check-then-create, and three API workers start cold against the same
                    # Qdrant at the same time: two of them see "does not exist" and both
                    # create it, and the loser used to answer DependencyUnavailable - a
                    # worker that never became ready because another one did its work. The
                    # collection existing is the outcome this method wanted; only a failure
                    # that left it absent is a failure.
                    if not await self._client.collection_exists(name):
                        raise
            if not self._local:  # local mode has neither payload indexes nor this setting
                await self._ensure_payload_indexes(name)
                await self._reconcile_payload_storage(name, spec)
        except Exception as exc:
            raise DependencyUnavailable(
                f"qdrant ensure_collection failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._known.add(name)

    async def _reconcile_payload_storage(self, name: str, spec: CollectionSpec) -> None:
        """Keep an existing collection's payload storage in agreement with the spec.

        ``on_disk_payload`` was only ever passed to ``create_collection``, so a collection
        that already existed kept whatever it was born with: the memories collection was
        running with its payload on disk while the code had been asking for it in memory,
        which is a page-cache read per returned hit at depth 100 - precisely the case a
        remote Qdrant under memory pressure makes expensive. Changing it does not need a
        reindex; the collection is updated in place.
        """
        wanted = spec.on_disk_payload and SEARCH.on_disk_payload
        try:
            info = await self._client.get_collection(name)
            if bool(info.config.params.on_disk_payload) == wanted:
                return
            await self._client.update_collection(
                collection_name=name,
                collection_params=models.CollectionParamsDiff(on_disk_payload=wanted),
            )
        except Exception as exc:  # a collection that works is worth more than this setting
            log.warning(
                "qdrant.payload_storage_not_reconciled",
                collection=name,
                wanted_on_disk=wanted,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def _ensure_payload_indexes(self, name: str) -> None:
        """Every field a filter touches, indexed - on a collection just created and on one
        that already existed. ``current`` was never indexed although every memories query
        filters on it, so Qdrant read payloads to apply it; and a field added here after a
        collection was created was never indexed on that collection at all."""
        info = await self._client.get_collection(name)
        have = set((info.payload_schema or {}).keys())
        for field, schema in _PAYLOAD_INDEXES.items():
            if field not in have:
                await self._client.create_payload_index(name, field_name=field, field_schema=schema)

    async def upsert(self, records: Sequence[SearchRecord]) -> None:
        if not records:
            return
        by_collection: dict[str, list[models.PointStruct]] = {}
        for r in records:
            vector: dict[str, Any] = {
                space.value: list(values) for space, values in r.dense.items()
            }
            if r.sparse is not None:
                vector[VectorName.BM25.value] = models.SparseVector(
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

    async def record_ids(self, collection: str, flt: SearchFilter) -> list[str]:
        name, out, offset = self._name(collection), [], None
        while True:
            points, offset = await _read(
                "scroll",
                lambda offset=offset: self._client.scroll(
                    collection_name=name,
                    scroll_filter=_filter(flt),
                    limit=512,
                    offset=offset,
                    with_payload=["record_id"],
                    with_vectors=False,
                ),
            )
            out.extend(str((p.payload or {}).get("record_id", p.id)) for p in points)
            if offset is None:
                return out

    def _hit(self, p: Any, retriever: Retriever) -> SearchHit:
        payload = dict(p.payload or {})
        return SearchHit(
            record_id=str(payload.get("record_id", p.id)),
            score=float(p.score),
            retriever=retriever,
            payload=payload,
        )

    async def search_dense(
        self,
        collection: str,
        name: VectorName,
        vector: Sequence[float],
        flt: SearchFilter,
        *,
        limit: int,
    ) -> list[SearchHit]:
        retriever = Retriever.for_vector(name)
        with span("search.dense"), stage_seconds.labels(f"retrieval.{name.value}").time():
            res = await _read(
                retriever.value,
                lambda: self._client.query_points(
                    collection_name=self._name(collection),
                    query=list(vector),
                    using=name.value,
                    query_filter=_filter(flt),
                    limit=limit,
                    with_payload=_PAYLOAD,
                ),
            )
        return [self._hit(p, retriever) for p in res.points]

    async def search_sparse(
        self, collection: str, vector: SparseVector, flt: SearchFilter, *, limit: int
    ) -> list[SearchHit]:
        if not vector.indices:
            return []
        with span("search.sparse"), stage_seconds.labels("retrieval.bm25").time():
            res = await _read(
                Retriever.BM25.value,
                lambda: self._client.query_points(
                    collection_name=self._name(collection),
                    query=models.SparseVector(indices=vector.indices, values=vector.values),
                    using=VectorName.BM25.value,
                    query_filter=_filter(flt),
                    limit=limit,
                    with_payload=_PAYLOAD,
                ),
            )
        return [self._hit(p, Retriever.BM25) for p in res.points]

    async def search_hybrid(
        self,
        collection: str,
        *,
        dense: Mapping[VectorName, Sequence[float]],
        sparse: SparseVector | None,
        flt: SearchFilter,
        limit: int,
        prefetch_limit: int,
        rrf_k: int = 1,
        weights: Mapping[VectorName, float] | None = None,
        anchors: Sequence[AnchoredPrefetch] = (),
    ) -> list[SearchHit]:
        if rrf_k < 0:
            raise ValueError("rrf_k must be nonnegative")
        qf = _filter(flt)
        arms = _arms(
            dense=dense,
            sparse=sparse,
            flt=flt,
            qf=qf,
            prefetch_limit=prefetch_limit,
            weights=weights,
            anchors=anchors,
        )
        if not arms:
            return []
        if len(arms) == 1:
            single, name, _ = arms[0]
            # one arm is not a hybrid query, and counting its retries as one makes
            # memory_search_read_retries_total{operation="hybrid"} a number about two
            # different reads
            retriever = Retriever.for_vector(name)
            res = await _read(
                retriever.value,
                lambda: self._client.query_points(
                    collection_name=self._name(collection),
                    query=single.query,
                    using=single.using,
                    query_filter=single.filter,
                    limit=limit,
                    with_payload=_PAYLOAD,
                ),
            )
            return [self._hit(p, retriever) for p in res.points]
        with span("search.hybrid"), stage_seconds.labels("retrieval.hybrid").time():
            try:
                res = await _read(
                    "hybrid",
                    lambda: self._client.query_points(
                        collection_name=self._name(collection),
                        prefetch=[arm for arm, _, _ in arms],
                        query=_fusion(rrf_k, [weight for _, _, weight in arms]),
                        query_filter=qf,
                        limit=limit,
                        with_payload=_PAYLOAD,
                    ),
                )
            except Exception as exc:
                raise DependencyUnavailable(
                    f"qdrant hybrid query failed: {type(exc).__name__}: {exc}"
                ) from exc
        # Native RRF leaves equal-score ordering unspecified. Resolve ties before the
        # engine deduplicates/cuts the pool, or identical queries can pack different
        # evidence. This stabilizes the returned pool without another RPC or wider search;
        # it cannot stabilize membership when the server cuts through a tie at its limit.
        hits = [self._hit(p, Retriever.FUSION) for p in res.points]
        hits.sort(key=lambda hit: (-hit.score, hit.record_id))
        return hits

    async def get(self, collection: str, record_ids: Sequence[str]) -> list[SearchRecord]:
        if not record_ids:
            return []
        points = await _read(
            "retrieve",
            lambda: self._client.retrieve(
                collection_name=self._name(collection),
                ids=[point_id(r) for r in record_ids],
                with_payload=_PAYLOAD,
                with_vectors=False,
            ),
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
        res = await _read(
            "count",
            lambda: self._client.count(
                collection_name=self._name(collection), count_filter=_filter(flt), exact=True
            ),
        )
        return int(res.count)

    async def drop_collection(self, collection: str) -> bool:
        name = self._name(collection)
        self._known.discard(name)
        if not await self._client.collection_exists(name):
            return False
        await self._client.delete_collection(name)
        return True

    async def list_collections(self) -> list[str]:
        prefix = f"{SEARCH.collection_prefix}_"
        try:
            listing = await self._client.get_collections()
        except Exception as exc:
            raise DependencyUnavailable(
                f"qdrant get_collections failed: {type(exc).__name__}: {exc}"
            ) from exc
        return [c.name[len(prefix) :] for c in listing.collections if c.name.startswith(prefix)]

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
