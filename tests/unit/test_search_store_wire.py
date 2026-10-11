"""What the search store puts on the wire, and what it does when the wire breaks.

Three properties are pinned here, all of them invisible from the outside until they cost a
question or a millisecond: the payload projection (only the keys a reader uses come back),
the one retry on a connection-level failure for reads and never for writes, and gRPC for a
server while local mode stays in-process.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

# The subject of this file is what the adapter puts on the wire, so it reads the wire types
# the adapter uses. Nothing else in tests/unit may import the SDK.
from qdrant_client import models  # noqa: TID251
from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: TID251

from memory_service.adapters.search.qdrant_store import (
    QdrantSearchStore,
    search_read_retries_total,
)
from memory_service.config.settings import SearchSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.search import (
    PAYLOAD_FIELDS,
    CollectionSpec,
    SearchFilter,
    SparseVector,
    VectorName,
)

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2] / "src" / "memory_service"

#: Every file that reads a payload off a search hit. The projection is derived from these and
#: from nothing else, so a new reader either adds its key here or gets None in production.
READERS = (
    "modules/retrieval/memory_ranking.py",
    "modules/retrieval/engine.py",
    "modules/context/builder.py",
    "modules/context/evidence.py",
    "modules/context/expansion.py",
    "modules/graph/retrieval.py",
    "modules/rag/indexer.py",
    "modules/retrieval/search.py",
)

#: Keys nobody reads through ``payload[...]``: the store's own identity fields, which it needs
#: to rebuild a record, and the graph stage's vocabulary, which reaches the builder's attribute
#: filter from in-process payloads rather than from Qdrant. Harmless to ask for, and cheaper
#: than a second projection per collection.
_NOT_FROM_A_READ = {"record_id", "tenant_id"}

# Relations are assembled by the graph stage, not indexed as Qdrant records.
# Their source-memory lineage is read only when filtering fact candidates.
_GRAPH_ONLY_FIELDS = {"memory_id"}
#: Written onto a hit by ranking itself and read back in the same request (``_dedup`` records
#: the twins it collapsed, ``in_hand`` reads them): never stored, so never projected.
_IN_PROCESS_FIELDS = {"duplicates"}


class _PayloadKeys(ast.NodeVisitor):
    """Collect every payload key a module reads.

    Three shapes appear in the code: ``hit.payload.get("k")``, ``c.payload["k"]``, and the
    builder's ``p = c.payload`` followed by ``p.get("k")`` and a ``k in (...)`` filter over
    ``p.items()``. All three are payload reads; a grep for the first two alone would miss the
    attribute allowlist that decides what the API returns.
    """

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.aliases: set[str] = set()

    def _is_payload(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "payload":
            return True
        return isinstance(node, ast.Name) and node.id in self.aliases

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast API
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "payload"
            and isinstance(node.targets[0], ast.Name)
        ):
            self.aliases.add(node.targets[0].id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and self._is_payload(func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.keys.add(node.args[0].value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802 - ast API
        if (
            self._is_payload(node.value)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self.keys.add(node.slice.value)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802 - ast API
        for gen in node.generators:
            iterable = gen.iter
            if (
                isinstance(iterable, ast.Call)
                and isinstance(iterable.func, ast.Attribute)
                and iterable.func.attr == "items"
                and self._is_payload(iterable.func.value)
            ):
                for condition in gen.ifs:
                    for const in ast.walk(condition):
                        if isinstance(const, ast.Constant) and isinstance(const.value, str):
                            self.keys.add(const.value)
        self.generic_visit(node)


def _keys_read_in_src() -> set[str]:
    found: set[str] = set()
    for name in READERS:
        visitor = _PayloadKeys()
        # two passes: an alias may be assigned after the first read in file order
        tree = ast.parse((SRC / name).read_text())
        visitor.visit(tree)
        visitor.visit(tree)
        found |= visitor.keys
    return found


def test_the_projection_is_exactly_what_the_readers_read() -> None:
    """The include list is a duplicate of knowledge that lives in six other files, so it is
    re-derived here rather than trusted. A key added to a reader and not here comes back as
    None from Qdrant while every test that uses a fake store keeps passing."""
    read = _keys_read_in_src() - _GRAPH_ONLY_FIELDS - _IN_PROCESS_FIELDS
    projected = set(PAYLOAD_FIELDS)
    assert read - projected == set(), (
        f"payload keys read in src but not requested from the store: {sorted(read - projected)}"
    )
    assert projected - read == _NOT_FROM_A_READ, (
        f"requested from the store but read nowhere: {sorted(projected - read - _NOT_FROM_A_READ)}"
    )


def test_graph_only_payload_fields_have_no_search_index_producer() -> None:
    """Keep the projection exception narrow if relation storage changes later."""
    indexer = ast.parse((SRC / "modules/rag/indexer.py").read_text())
    graph = ast.parse((SRC / "modules/graph/retrieval.py").read_text())

    def written_keys(tree: ast.AST) -> set[str]:
        return {
            key.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

    assert not (_GRAPH_ONLY_FIELDS & written_keys(indexer))
    assert written_keys(graph) >= _GRAPH_ONLY_FIELDS
    assert not (_GRAPH_ONLY_FIELDS & set(PAYLOAD_FIELDS))


def test_the_indexer_still_writes_the_keys_the_projection_asks_for() -> None:
    """A projection is only as good as the payload behind it: if the indexer stops writing a
    key the include list is silently asking for nothing."""
    written = set(
        re.findall(r'^\s+"([a-z_]+)":', (SRC / "modules/rag/indexer.py").read_text(), re.M)
    )
    for key in ("kind", "text", "text_hash", "node_id", "document_id", "subject", "observed_at"):
        assert key in written, key
        assert key in PAYLOAD_FIELDS, key


# --- a client that answers, fails, or dies ---------------------------------------------------


@dataclass
class _Point:
    id: str = "p1"
    score: float = 1.0
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.payload = self.payload if self.payload is not None else {"record_id": "r1"}


@dataclass
class _Result:
    points: list[_Point]


class _FakeGrpcError(Exception):
    """Shaped like grpc.aio.AioRpcError: a status with a name, read through code()."""

    def __init__(self, name: str = "UNAVAILABLE") -> None:
        self._name = name

    def code(self) -> Any:
        return type("Status", (), {"name": self._name})()


class FakeClient:
    """Records every call and fails the first ``fail_times`` of them."""

    def __init__(self, error: Exception | None = None, fail_times: int = 0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error = error
        self.fail_times = fail_times

    def _maybe_fail(self) -> None:
        if self.fail_times > 0 and self.error is not None:
            self.fail_times -= 1
            raise self.error

    async def query_points(self, **kwargs: Any) -> _Result:
        self.calls.append(("query_points", kwargs))
        self._maybe_fail()
        return _Result([_Point()])

    async def retrieve(self, **kwargs: Any) -> list[_Point]:
        self.calls.append(("retrieve", kwargs))
        self._maybe_fail()
        return [_Point(payload={"record_id": "r1", "tenant_id": "acme"})]

    async def count(self, **kwargs: Any) -> Any:
        self.calls.append(("count", kwargs))
        self._maybe_fail()
        return type("Count", (), {"count": 3})()

    async def scroll(self, **kwargs: Any) -> tuple[list[_Point], None]:
        self.calls.append(("scroll", kwargs))
        self._maybe_fail()
        return [_Point()], None

    async def upsert(self, **kwargs: Any) -> None:
        self.calls.append(("upsert", kwargs))
        self._maybe_fail()

    async def delete(self, **kwargs: Any) -> None:
        self.calls.append(("delete", kwargs))
        self._maybe_fail()


def _store(client: FakeClient) -> QdrantSearchStore:
    store = QdrantSearchStore(SearchSettings())
    store._client = client  # type: ignore[assignment]
    return store


def _flt() -> SearchFilter:
    return SearchFilter(tenant_id="acme")


def _retries(operation: str | None = None) -> float:
    return sum(
        sample.value
        for metric in search_read_retries_total.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
        and (operation is None or sample.labels.get("operation") == operation)
    )


async def test_every_read_projects_the_payload() -> None:
    client = FakeClient()
    store = _store(client)
    await store.search_dense("c", VectorName.DENSE_ML, [0.1] * 4, _flt(), limit=5)
    await store.search_sparse("c", SparseVector(indices=[1], values=[1.0]), _flt(), limit=5)
    await store.search_hybrid(
        "c",
        dense={VectorName.DENSE_ML: [0.1] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        flt=_flt(),
        limit=5,
        prefetch_limit=8,
    )
    await store.get("c", ["r1"])
    selectors = [kwargs["with_payload"] for _, kwargs in client.calls]
    assert selectors, "no read reached the client"
    for selector in selectors:
        assert isinstance(selector, models.PayloadSelectorInclude)
        assert selector.include == list(PAYLOAD_FIELDS)


@pytest.mark.parametrize("order", [("b", "a", "c"), ("c", "a", "b"), ("a", "b", "c")])
async def test_native_fusion_ties_have_stable_order_without_changing_scores(order) -> None:
    class TiedClient(FakeClient):
        async def query_points(self, **kwargs: Any) -> _Result:
            self.calls.append(("query_points", kwargs))
            return _Result(
                [
                    _Point(score=0.5 if name != "c" else 0.75, payload={"record_id": name})
                    for name in order
                ]
            )

    client = TiedClient()
    hits = await _store(client).search_hybrid(
        "c",
        dense={VectorName.DENSE_ML: [0.1] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        flt=_flt(),
        limit=3,
        prefetch_limit=8,
    )
    assert [(hit.record_id, hit.score) for hit in hits] == [("c", 0.75), ("a", 0.5), ("b", 0.5)]
    assert len(client.calls) == 1
    assert client.calls[0][1]["limit"] == 3


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("Server disconnected without sending a response"),
        httpx.ReadError("read error"),
        ResponseHandlingException(httpx.RemoteProtocolError("Server disconnected")),
        _FakeGrpcError(),
        AttributeError("'NoneType' object has no attribute 'result'"),
        AssertionError("Search returned None"),
    ],
    ids=[
        "connect",
        "remote_protocol",
        "read",
        "wrapped",
        "grpc_unavailable",
        "empty_response",
        "asserted_none",
    ],
)
async def test_a_read_survives_one_connection_failure(error: Exception) -> None:
    """The failure that cost four of 304 judged questions: the connection is gone, the query
    is not. One retry, and the counter says it happened."""
    before = _retries()
    client = FakeClient(error=error, fail_times=1)
    hits = await _store(client).search_dense("c", VectorName.DENSE_ML, [0.1] * 4, _flt(), limit=5)
    assert [h.record_id for h in hits] == ["r1"]
    assert len(client.calls) == 2, "the read was not retried exactly once"
    assert _retries() == before + 1


async def test_a_hybrid_read_with_no_response_says_so_instead_of_leaking_an_attribute_error() -> (
    None
):
    """A read whose response never arrived used to surface as ``AttributeError: 'NoneType'
    object has no attribute 'result'`` wrapped in DependencyUnavailable, which reads like a
    bug in this adapter and says nothing about what happened. Two long benchmark runs died
    on it; both would have survived the retry, and whoever read the log would have known why."""
    client = FakeClient(error=AttributeError("'NoneType' object has no attribute 'result'"))
    client.fail_times = 99
    with pytest.raises(DependencyUnavailable, match="sent no response body"):
        await _store(client).search_hybrid(
            "c",
            dense={VectorName.DENSE_EN: [0.1] * 4, VectorName.DENSE_ML: [0.2] * 4},
            sparse=None,
            flt=_flt(),
            limit=5,
            prefetch_limit=10,
        )
    assert len(client.calls) == 2, "the hybrid read was retried once before giving up"


async def test_a_programming_error_in_a_read_is_not_mistaken_for_a_lost_response() -> None:
    """The retry keys on the client having no response, not on AttributeError as a class:
    a genuine attribute bug must still surface on the first call."""
    client = FakeClient(error=AttributeError("'Store' object has no attribute 'collection'"))
    client.fail_times = 99
    with pytest.raises(AttributeError, match="no attribute 'collection'"):
        await _store(client).search_dense("c", VectorName.DENSE_ML, [0.1] * 4, _flt(), limit=5)
    assert len(client.calls) == 1, "not retried"


async def test_a_second_failure_is_not_retried_again() -> None:
    client = FakeClient(error=httpx.ConnectError("down"), fail_times=5)
    with pytest.raises(httpx.ConnectError):
        await _store(client).search_dense("c", VectorName.DENSE_ML, [0.1] * 4, _flt(), limit=5)
    assert len(client.calls) == 2, "one retry, not a loop"


@pytest.mark.parametrize("operation", ["retrieve", "count", "scroll"])
async def test_the_other_reads_are_retried_too(operation: str) -> None:
    """The retry lives in _read, so every read gets it - but only search_dense was ever
    exercised, and scroll is the one that carries an offset through a default argument."""
    client = FakeClient(error=httpx.ConnectError("down"), fail_times=1)
    store = _store(client)
    if operation == "retrieve":
        assert [r.record_id for r in await store.get("c", ["r1"])] == ["r1"]
    elif operation == "count":
        assert await store.count("c", _flt()) == 3
    else:
        assert await store.record_ids("c", _flt()) == ["r1"]
    assert [name for name, _ in client.calls] == [operation, operation]


async def test_a_retried_scroll_page_asks_for_the_offset_it_was_on() -> None:
    """``lambda offset=offset`` binds the page this attempt is fetching. Bound wrong, the
    retry restarts from the first page and record_ids returns the first page twice."""

    class _Pages(FakeClient):
        async def scroll(self, **kwargs: Any) -> tuple[list[_Point], Any]:
            self.calls.append(("scroll", kwargs))
            self._maybe_fail()
            first = kwargs["offset"] is None
            return [_Point(id="p1" if first else "p2", payload={"record_id": "r2"})], (
                "page-2" if first else None
            )

    client = _Pages(error=httpx.ConnectError("down"), fail_times=1)
    # fails the first page, retries it, then asks for page two: three calls, two pages
    assert await _store(client).record_ids("c", _flt()) == ["r2", "r2"]
    assert [kwargs["offset"] for _, kwargs in client.calls] == [None, None, "page-2"]


async def test_a_one_armed_hybrid_is_counted_as_the_read_it_is() -> None:
    """A query with only a dense prefetch is a dense read; fusing needs two arms. Counted as
    "hybrid", the retry counter mixes three different reads under one operation."""
    before = (_retries("dense_ml"), _retries("hybrid"))
    client = FakeClient(error=httpx.ConnectError("down"), fail_times=1)
    await _store(client).search_hybrid(
        "c",
        dense={VectorName.DENSE_ML: [0.1] * 4},
        sparse=None,
        flt=_flt(),
        limit=5,
        prefetch_limit=8,
    )
    assert (_retries("dense_ml"), _retries("hybrid")) == (before[0] + 1, before[1])


async def test_an_ordinary_error_is_not_retried() -> None:
    """Retrying a query that the server answered with an error is a second wasted round trip
    and, for a hybrid query, a second DependencyUnavailable a beat later."""
    client = FakeClient(error=ValueError("bad vector dimension"), fail_times=1)
    with pytest.raises(ValueError):
        await _store(client).search_dense("c", VectorName.DENSE_ML, [0.1] * 4, _flt(), limit=5)
    assert len(client.calls) == 1


async def test_a_write_is_never_retried() -> None:
    """An upsert that may or may not have landed is not idempotent from here: the client saw
    a broken connection, not a rejection."""
    from memory_service.domain.errors import DependencyUnavailable
    from memory_service.ports.search import SearchRecord

    client = FakeClient(error=httpx.RemoteProtocolError("Server disconnected"), fail_times=1)
    store = _store(client)
    record = SearchRecord(
        record_id="r1", collection="c", tenant_id="acme", dense={VectorName.DENSE_ML: [0.1] * 4}
    )
    with pytest.raises(DependencyUnavailable):
        await store.upsert([record])
    assert len(client.calls) == 1


async def test_the_hybrid_failure_is_still_a_dependency_error() -> None:
    client = FakeClient(error=httpx.ConnectError("down"), fail_times=5)
    from memory_service.domain.errors import DependencyUnavailable

    with pytest.raises(DependencyUnavailable):
        await _store(client).search_hybrid(
            "c",
            dense={VectorName.DENSE_ML: [0.1] * 4},
            sparse=SparseVector(indices=[1], values=[1.0]),
            flt=_flt(),
            limit=5,
            prefetch_limit=8,
        )


async def test_the_hybrid_failure_keeps_the_clients_message_out_of_the_problem() -> None:
    """The client's text can name the server's URL and collections; the problem detail a
    caller reads says which call failed and how, and the message goes to the log only."""
    client = FakeClient(
        error=RuntimeError("http://qdrant.internal:6333/collections/tm_memories refused"),
        fail_times=5,
    )
    with pytest.raises(DependencyUnavailable) as raised:
        await _store(client).search_hybrid(
            "c",
            dense={VectorName.DENSE_ML: [0.1] * 4},
            sparse=SparseVector(indices=[1], values=[1.0]),
            flt=_flt(),
            limit=5,
            prefetch_limit=8,
        )
    assert raised.value.message == "qdrant hybrid query failed: RuntimeError"


# --- transport --------------------------------------------------------------------------------


def test_a_server_is_addressed_over_grpc_and_local_mode_is_not() -> None:
    server = QdrantSearchStore(SearchSettings())
    inner = server._client._client
    assert getattr(inner, "_prefer_grpc", False) is True
    assert getattr(inner, "_grpc_port", None) == SearchSettings().qdrant_grpc_port
    local = QdrantSearchStore(SearchSettings(), local_path=":memory:")
    assert local._local is True
    assert not hasattr(local._client._client, "_prefer_grpc")


async def test_the_memories_collection_keeps_its_payload_in_memory() -> None:
    """Small enough to hold in RAM, read by every query; the knowledge collection is neither."""
    from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES

    specs: dict[str, bool] = {}

    class _Recorder:
        async def ensure_collection(self, spec: Any) -> None:
            specs[spec.name] = spec.on_disk_payload

    from memory_service.adapters.models.embeddings import HashEmbedding
    from memory_service.adapters.models.sparse import Bm25SparseEncoder
    from memory_service.modules.rag.indexer import Indexer
    from memory_service.modules.rag.spaces import DenseSpaces

    indexer = Indexer(
        uow_factory=None,  # type: ignore[arg-type]
        store=_Recorder(),  # type: ignore[arg-type]
        spaces=DenseSpaces.single(HashEmbedding(dimension=8)),
        sparse=Bm25SparseEncoder(),
    )
    await indexer.ensure_collections()
    assert specs[indexer.collection(MEMORIES)] is False
    assert specs[indexer.collection(KNOWLEDGE)] is True


# --- two workers starting against one empty Qdrant --------------------------------------


class _RacingClient:
    """Answers "no such collection", then has one by the time the create is attempted."""

    def __init__(self, exists_after_create: bool = True) -> None:
        self.exists = False
        self.exists_after_create = exists_after_create
        self.indexed: list[str] = []

    async def collection_exists(self, name: str) -> bool:
        return self.exists

    async def create_collection(self, **kwargs: Any) -> None:
        self.exists = self.exists_after_create
        raise ValueError(f"Collection `{kwargs['collection_name']}` already exists!")

    async def get_collection(self, name: str) -> Any:
        return type("Info", (), {"payload_schema": {}})()

    async def create_payload_index(self, name: str, *, field_name: str, **kwargs: Any) -> None:
        self.indexed.append(field_name)


async def test_a_collection_another_worker_created_first_is_not_a_failure() -> None:
    """Three API workers start cold against one Qdrant and all three see an empty store, so
    two of them create the same collection. The one that loses used to answer
    DependencyUnavailable and never become ready, over the outcome it had asked for."""
    client = _RacingClient()
    store = _store(client)  # type: ignore[arg-type]
    await store.ensure_collection(
        CollectionSpec(name="memories", dense={VectorName.DENSE_EN: 4, VectorName.DENSE_ML: 4})
    )
    assert client.indexed, "the loser still has to ensure the payload indexes exist"


async def test_a_create_that_leaves_no_collection_still_fails() -> None:
    client = _RacingClient(exists_after_create=False)
    with pytest.raises(DependencyUnavailable):
        await _store(client).ensure_collection(  # type: ignore[arg-type]
            CollectionSpec(name="memories", dense={VectorName.DENSE_EN: 4, VectorName.DENSE_ML: 4})
        )


# --- scale-out: layout, tenant index, payload storage (ADR 0031) ------------------------


class _LayoutClient:
    """An empty store that records what a create and the index pass ask for."""

    def __init__(self, *, points: int = 0, schema: dict[str, Any] | None = None) -> None:
        self.created: dict[str, Any] = {}
        self.indexes: dict[str, Any] = {}
        self.updates: list[Any] = []
        self.exists = schema is not None
        self.schema = schema or {}
        self.points = points

    async def collection_exists(self, name: str) -> bool:
        return self.exists

    async def create_collection(self, **kwargs: Any) -> None:
        self.created = kwargs
        self.exists = True

    async def get_collection(self, name: str) -> Any:
        params = type("Params", (), {"on_disk_payload": False})()
        config = type("Config", (), {"params": params})()
        return type(
            "Info",
            (),
            {"payload_schema": self.schema, "points_count": self.points, "config": config},
        )()

    async def create_payload_index(self, name: str, *, field_name: str, field_schema: Any) -> None:
        self.indexes[field_name] = field_schema

    async def update_collection(self, **kwargs: Any) -> None:
        self.updates.append(kwargs["collection_params"])


async def test_a_new_collection_takes_the_clusters_layout_and_a_tenant_index() -> None:
    client = _LayoutClient()
    store = QdrantSearchStore(
        SearchSettings(shard_number=6, replication_factor=2, write_consistency_factor=2)
    )
    store._client = client  # type: ignore[assignment]
    await store.ensure_collection(
        CollectionSpec(name="memories", dense={VectorName.DENSE_ML: 4}, on_disk_payload=False)
    )
    assert (
        client.created["shard_number"],
        client.created["replication_factor"],
        client.created["write_consistency_factor"],
    ) == (6, 2, 2)
    tenant = client.indexes["tenant_id"]
    assert isinstance(tenant, models.KeywordIndexParams) and tenant.is_tenant is True
    assert client.updates == [], "a small memories collection keeps its payload in RAM"


async def test_an_old_tenant_index_is_rebuilt_with_the_flag() -> None:
    old = type("IndexInfo", (), {"params": None})()
    schema = dict.fromkeys(("tenant_id", "visibility_keys", "kind", "document_id"), old)
    schema |= dict.fromkeys(
        ("current", "script", "observed_at", "valid_from", "valid_to", "known_to"), old
    )
    client = _LayoutClient(schema=schema)
    store = _store(client)  # type: ignore[arg-type]
    await store.ensure_collection(CollectionSpec(name="knowledge", dense={VectorName.DENSE_ML: 4}))
    assert list(client.indexes) == ["tenant_id"]
    assert client.indexes["tenant_id"].is_tenant is True


async def test_a_grown_memories_collection_moves_its_payload_to_disk() -> None:
    from memory_service.config.constants import SEARCH

    client = _LayoutClient(points=SEARCH.payload_in_ram_max_points + 1, schema={})
    store = _store(client)  # type: ignore[arg-type]
    await store.ensure_collection(
        CollectionSpec(name="memories", dense={VectorName.DENSE_ML: 4}, on_disk_payload=False)
    )
    assert [diff.on_disk_payload for diff in client.updates] == [True]


def test_write_consistency_cannot_exceed_the_replicas() -> None:
    with pytest.raises(ValueError, match="write_consistency_factor"):
        SearchSettings(replication_factor=1, write_consistency_factor=2)
