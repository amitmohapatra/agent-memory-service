"""Qdrant SearchStore (Apache-2.0).

Collections carry one named dense vector per space (``dense_en``, ``dense_ml``) and a named
sparse vector (``bm25``) with Qdrant's server-side IDF modifier, so BM25 scoring happens in
the store; the memories collection adds each of those a second time for the contextual key
(``*_ctx``), and both collections carry the late-interaction token vectors (``colbert``,
MaxSim, half precision, on disk, no HNSW graph: the arm only ever rescores the union of the
other arms' candidates, so the graph would be built and never walked). Hybrid search uses
``query_points`` with one prefetch per arm fused by native RRF, weighted when the fitted
weights say so; the memories' learned fusion reads every arm unfused through one
``query_batch_points``. Every query carries a tenant filter and a
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
    SPARSE_NAMES,
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

#: The other way a read comes back with nothing: the client unwraps a response it never got
#: and raises from inside its own code, ``'NoneType' object has no attribute 'result'`` (newer
#: paths assert instead, "Search returned None"). It is not a bad query -- the same call
#: answers before it and after it -- so it belongs with the connection failures rather than
#: with programming errors. Two 45- and 80-minute benchmark runs died on it under load, each
#: losing the whole run to one read, which is what a retry is for.
_EMPTY_RESPONSE_SIGNS = ("object has no attribute 'result'", "returned None")


def _unavailable(operation: str, exc: BaseException) -> DependencyUnavailable:
    """The 503 a failed call becomes. The client's message goes to the log, never into the
    problem's ``detail``: it can quote the server's URL, collection names and payloads."""
    log.warning(
        "qdrant.failed", operation=operation, error_type=type(exc).__name__, error=str(exc)[:500]
    )
    return DependencyUnavailable(f"qdrant {operation} failed: {type(exc).__name__}")


def _is_empty_response(exc: BaseException) -> bool:
    """Whether the client raised because it had no response to unwrap (see the note above)."""
    if not isinstance(exc, AttributeError | AssertionError):
        return False
    return any(sign in str(exc) for sign in _EMPTY_RESPONSE_SIGNS)


def _is_transient_read_error(exc: BaseException) -> bool:
    """Whether this read can be reissued: the connection died, or no response arrived."""
    return _is_connection_error(exc) or _is_empty_response(exc)


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
        if not _is_transient_read_error(exc):
            raise
        search_read_retries_total.labels(operation).inc()
        await asyncio.sleep(random.uniform(*_RETRY_PAUSE_SECONDS))  # noqa: S311 - jitter
        return await call()


#: Payload fields the search filters use; each one is indexed (see _ensure_payload_indexes).
#: ``script`` is the record's Unicode script.
#: ``tenant_id`` is the tenant key: every search filters on exactly one value of it, and
#: ``is_tenant`` tells Qdrant to lay segments out per tenant so that filter selects a
#: tenant's storage instead of scanning everyone's (multitenancy, Qdrant >= 1.11).
_TENANT_INDEX = models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True)

_PAYLOAD_INDEXES: dict[str, Any] = {
    "tenant_id": _TENANT_INDEX,
    "visibility_keys": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "current": models.PayloadSchemaType.BOOL,
    "script": models.PayloadSchemaType.KEYWORD,
    "observed_at": models.PayloadSchemaType.DATETIME,
    # a memory's valid time and the end of its knowledge time (``as_of`` / ``known_at``)
    "valid_from": models.PayloadSchemaType.DATETIME,
    "valid_to": models.PayloadSchemaType.DATETIME,
    "known_to": models.PayloadSchemaType.DATETIME,
}


def _is_tenant_index(info: Any) -> bool:
    """Whether an existing payload index (``PayloadIndexInfo``) is the tenant index."""
    params = getattr(info, "params", None)
    return bool(getattr(params, "is_tenant", False))


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
    for key, (start, end) in flt.within.items():
        must.append(models.FieldCondition(key=key, range=models.DatetimeRange(gte=start, lte=end)))
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


def _arms(
    *,
    dense: Mapping[VectorName, Sequence[float]],
    sparse: SparseVector | None,
    flt: SearchFilter,
    qf: models.Filter,
    prefetch_limit: int,
    weights: Mapping[VectorName, float] | None,
    late: Sequence[Sequence[float]] | None = None,
) -> list[_Arm]:
    """Every arm of one hybrid query, in fusion order: the dense spaces, BM25, then the
    late-interaction arm over the union of the others."""
    arms: list[_Arm] = [
        (
            models.Prefetch(query=list(vector), using=space.value, limit=prefetch_limit, filter=qf),
            space,
            _weight(weights, space),
        )
        for space, vector in dense.items()
    ]
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
    if late and arms:
        arms.append(
            (
                _late_prefetch(late, [arm for arm, _, _ in arms], qf, prefetch_limit),
                VectorName.COLBERT,
                _weight(weights, VectorName.COLBERT),
            )
        )
    return arms


def _late_prefetch(
    late: Sequence[Sequence[float]],
    inner: list[models.Prefetch],
    qf: models.Filter,
    limit: int,
    using: VectorName = VectorName.COLBERT,
) -> models.Prefetch:
    """The late-interaction arm: MaxSim over the union of the other arms' candidates.

    A rescoring of that union rather than a search of its own. The token vectors carry no
    HNSW graph (``m=0``), so a search of its own would be a scan of the tenant's every point
    at MaxSim cost; over the union it is bounded by the arms' depth whatever the tenant's
    size, and on LoCoMo the union already holds nearly every turn the full scan ranks first.
    """
    return models.Prefetch(
        prefetch=inner,
        query=[list(row) for row in late],
        using=using.value,
        limit=limit,
        filter=qf,
    )


def _collection_vectors(
    spec: CollectionSpec,
) -> tuple[dict[str, models.VectorParams], dict[str, models.SparseVectorParams] | None]:
    """The named dense, late-interaction and sparse vectors a collection is created with."""
    vectors = {
        space.value: models.VectorParams(
            size=width, distance=models.Distance.COSINE, on_disk=spec.on_disk
        )
        for space, width in spec.dense.items()
    }
    # the contextual keys' collection carries the late vectors of both keys (ADR 0026)
    late_names = [VectorName.COLBERT] if spec.late else []
    if spec.late and spec.sparse_context:
        late_names.append(VectorName.COLBERT_CTX)
    for name in late_names:
        vectors[name.value] = models.VectorParams(
            size=spec.late or 0,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            # rescoring only (see _late_prefetch): no graph to build
            hnsw_config=models.HnswConfigDiff(m=0),
            # Half precision moves no ranking measured here and halves the largest thing a
            # point holds (a 512-token chunk is 512 vectors).
            datatype=models.Datatype.FLOAT16,
            on_disk=True,
        )
    sparse_names = [VectorName.BM25] if spec.sparse else []
    if spec.sparse and spec.sparse_context:
        sparse_names.append(VectorName.BM25_CTX)
    sparse = {
        name.value: models.SparseVectorParams(
            modifier=models.Modifier.IDF if spec.sparse_idf else None,
            index=models.SparseIndexParams(on_disk=spec.on_disk),
        )
        for name in sparse_names
    }
    return vectors, sparse or None


def _arm_prefetches(
    dense: Mapping[VectorName, Sequence[float]],
    sparse: Mapping[VectorName, SparseVector],
    late: Sequence[Sequence[float]] | None,
    qf: models.Filter,
    limit: int,
) -> list[tuple[VectorName, models.Prefetch]]:
    """One prefetch per arm ``search_arms`` reads: each dense space, each non-empty sparse
    space, then the late-interaction arm over the union of the others."""
    arms: list[tuple[VectorName, models.Prefetch]] = [
        (name, models.Prefetch(query=list(vector), using=name.value, limit=limit, filter=qf))
        for name, vector in dense.items()
    ]
    for name, vector in sparse.items():
        if name not in SPARSE_NAMES:
            raise ValueError(f"{name} is not a sparse space")
        if vector.indices:
            query = models.SparseVector(indices=vector.indices, values=vector.values)
            arms.append(
                (name, models.Prefetch(query=query, using=name.value, limit=limit, filter=qf))
            )
    if late and arms:
        inner = [prefetch for _, prefetch in arms]
        arms.append((VectorName.COLBERT, _late_prefetch(late, inner, qf, limit)))
        # a search over the contextual keys reads the context key's late vectors too
        if any(name.value.endswith("_ctx") for name, _ in arms):
            arms.append(
                (
                    VectorName.COLBERT_CTX,
                    _late_prefetch(late, inner, qf, limit, VectorName.COLBERT_CTX),
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
                # Not a request for inference: this service sends vectors only, never a
                # ``models.Document``. Off, the client walks every request looking for one -
                # every float of every query vector, ~22 ms of CPU under the GIL per memory
                # search with seven arms and a 48x64 late-interaction query.
                cloud_inference=True,
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
                vectors, sparse = _collection_vectors(spec)
                try:
                    await self._client.create_collection(
                        collection_name=name,
                        vectors_config=vectors,
                        sparse_vectors_config=sparse,
                        on_disk_payload=spec.on_disk_payload
                        and SEARCH.on_disk_payload
                        and not self._local,
                        # the cluster's layout (SearchSettings); local mode has none
                        shard_number=None if self._local else self.settings.shard_number,
                        replication_factor=None
                        if self._local
                        else self.settings.replication_factor,
                        write_consistency_factor=None
                        if self._local
                        else self.settings.write_consistency_factor,
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
            raise _unavailable("ensure_collection", exc) from exc
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
        try:
            info = await self._client.get_collection(name)
            # in RAM only while it is small enough to be worth it (``payload_in_ram_max_points``)
            grown = (info.points_count or 0) > SEARCH.payload_in_ram_max_points
            wanted = (spec.on_disk_payload or grown) and SEARCH.on_disk_payload
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
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def _ensure_payload_indexes(self, name: str) -> None:
        """Every field a filter touches, indexed - on a collection just created and on one
        that already existed. ``current`` was never indexed although every memories query
        filters on it, so Qdrant read payloads to apply it; and a field added here after a
        collection was created was never indexed on that collection at all."""
        info = await self._client.get_collection(name)
        have = info.payload_schema or {}
        for field, schema in _PAYLOAD_INDEXES.items():
            if field not in have:
                await self._client.create_payload_index(name, field_name=field, field_schema=schema)
        if not _is_tenant_index(have.get("tenant_id")):
            # Created before ``is_tenant``: index it again with the flag. Qdrant then groups
            # new segments by tenant, so a tenant-filtered search reads that tenant's points
            # instead of filtering every segment; points already stored are regrouped as the
            # optimizer rewrites their segments, or at once by a rebuild (reindex --drop).
            await self._client.create_payload_index(
                name, field_name="tenant_id", field_schema=_TENANT_INDEX
            )

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
            if r.sparse_context is not None:
                vector[VectorName.BM25_CTX.value] = models.SparseVector(
                    indices=r.sparse_context.indices, values=r.sparse_context.values
                )
            if r.late:
                vector[VectorName.COLBERT.value] = [list(row) for row in r.late]
            if r.late_context:
                vector[VectorName.COLBERT_CTX.value] = [list(row) for row in r.late_context]
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

    async def similarity(
        self,
        collection: str,
        name: VectorName,
        vector: Sequence[float],
        record_ids: Sequence[str],
    ) -> dict[str, float]:
        if not record_ids:
            return {}
        ids = list(dict.fromkeys(record_ids))
        with span("search.similarity"), stage_seconds.labels("retrieval.similarity").time():
            res = await _read(
                "similarity",
                lambda: self._client.query_points(
                    collection_name=self._name(collection),
                    query=list(vector),
                    using=name.value,
                    query_filter=models.Filter(
                        must=[models.HasIdCondition(has_id=[point_id(r) for r in ids])]
                    ),
                    limit=len(ids),
                    with_payload=models.PayloadSelectorInclude(include=["record_id"]),
                ),
            )
        return {str((p.payload or {}).get("record_id", p.id)): float(p.score) for p in res.points}

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
        late: Sequence[Sequence[float]] | None = None,
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
            late=late,
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
                if _is_empty_response(exc):
                    log.warning("qdrant.failed", operation="hybrid query", error=str(exc)[:500])
                    raise DependencyUnavailable(
                        "qdrant hybrid query failed: the search server sent no response body "
                        "(retried once already)"
                    ) from exc
                raise _unavailable("hybrid query", exc) from exc
        # Native RRF leaves equal-score ordering unspecified. Resolve ties before the
        # engine deduplicates/cuts the pool, or identical queries can pack different
        # evidence. This stabilizes the returned pool without another RPC or wider search;
        # it cannot stabilize membership when the server cuts through a tie at its limit.
        hits = [self._hit(p, Retriever.FUSION) for p in res.points]
        hits.sort(key=lambda hit: (-hit.score, hit.record_id))
        return hits

    async def search_arms(
        self,
        collection: str,
        *,
        dense: Mapping[VectorName, Sequence[float]],
        sparse: Mapping[VectorName, SparseVector],
        late: Sequence[Sequence[float]] | None,
        flt: SearchFilter,
        limit: int,
    ) -> dict[VectorName, list[SearchHit]]:
        qf = _filter(flt)
        arms = _arm_prefetches(dense, sparse, late, qf, limit)
        if not arms:
            return {}
        requests = [
            models.QueryRequest(
                prefetch=prefetch.prefetch,
                query=prefetch.query,
                using=prefetch.using,
                filter=qf,
                limit=limit,
                # ids and scores only: see _payloads
                with_payload=False,
            )
            for _, prefetch in arms
        ]
        with span("search.arms"), stage_seconds.labels("retrieval.arms").time():
            try:
                responses = await _read(
                    "arms",
                    lambda: self._client.query_batch_points(
                        collection_name=self._name(collection), requests=requests
                    ),
                )
            except Exception as exc:
                raise _unavailable("arms query", exc) from exc
            payloads = await self._payloads(collection, responses)
        out: dict[VectorName, list[SearchHit]] = {}
        for (name, _), response in zip(arms, responses, strict=True):
            retriever = Retriever.for_vector(name)
            hits = [
                SearchHit(
                    record_id=str(payload.get("record_id", p.id)),
                    score=float(p.score),
                    retriever=retriever,
                    payload=payload,
                )
                for p in response.points
                if (payload := payloads.get(str(p.id))) is not None
            ]
            hits.sort(key=lambda hit: (-hit.score, hit.record_id))  # see search_hybrid
            out[name] = hits
        return out

    async def _payloads(self, collection: str, responses: Sequence[Any]) -> dict[str, dict]:
        """Every point's payload, read once.

        The arms overlap: ~800 hits a query are ~350 points. Turning a payload from protobuf
        into a dict is client CPU under the GIL, and read with every hit it was most of a
        memory search's Python time and what capped one worker's throughput.
        """
        ids = list(dict.fromkeys(str(p.id) for response in responses for p in response.points))
        try:
            points = await _read(
                "retrieve",
                lambda: self._client.retrieve(
                    collection_name=self._name(collection),
                    ids=ids,
                    with_payload=_PAYLOAD,
                    with_vectors=False,
                ),
            )
        except Exception as exc:
            raise _unavailable("arms payload read", exc) from exc
        return {str(p.id): dict(p.payload or {}) for p in points}

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
            raise _unavailable("get_collections", exc) from exc
        return [c.name[len(prefix) :] for c in listing.collections if c.name.startswith(prefix)]

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
