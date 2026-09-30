"""What the store sends Qdrant for a collection with two dense spaces: one named vector per
space at creation, every vector a record carries at upsert, the script and entity indexes,
and the arm a one-armed query is counted as."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from qdrant_client import models  # noqa: TID251 - adapter wire contract

from memory_service.adapters.search.qdrant_store import QdrantSearchStore
from memory_service.config.settings import SearchSettings
from memory_service.ports.search import (
    CollectionSpec,
    SearchFilter,
    SearchRecord,
    SparseVector,
    VectorName,
)

pytestmark = pytest.mark.unit


@dataclass
class _Info:
    payload_schema: dict[str, Any] = field(default_factory=dict)
    config: Any = None

    def __post_init__(self) -> None:
        params = type("Params", (), {"on_disk_payload": True})()
        self.config = type("Config", (), {"params": params})()


class _Client:
    """Records the collection management and search calls of a server-mode store."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exists = False

    async def collection_exists(self, name: str) -> bool:
        return self.exists

    async def create_collection(self, **kwargs: Any) -> None:
        self.calls.append(("create_collection", kwargs))
        self.exists = True

    async def get_collection(self, name: str) -> _Info:
        return _Info()

    async def create_payload_index(self, name: str, *, field_name: str, field_schema: Any) -> None:
        self.calls.append(("create_payload_index", {"field": field_name, "schema": field_schema}))

    async def update_collection(self, **kwargs: Any) -> None:
        self.calls.append(("update_collection", kwargs))

    async def upsert(self, **kwargs: Any) -> None:
        self.calls.append(("upsert", kwargs))

    async def query_points(self, **kwargs: Any) -> Any:
        self.calls.append(("query_points", kwargs))
        return type("Result", (), {"points": []})()


def _store() -> tuple[QdrantSearchStore, _Client]:
    store = QdrantSearchStore(SearchSettings())
    client = _Client()
    store._client = client  # type: ignore[assignment]
    return store, client


async def test_a_collection_gets_one_named_vector_per_space_and_the_filter_indexes() -> None:
    store, client = _store()
    await store.ensure_collection(
        CollectionSpec(name="c", dense={VectorName.DENSE_EN: 384, VectorName.DENSE_ML: 384})
    )
    created = next(kwargs for name, kwargs in client.calls if name == "create_collection")
    assert set(created["vectors_config"]) == {"dense_en", "dense_ml"}
    assert all(
        v.size == 384 and v.distance == models.Distance.COSINE
        for v in created["vectors_config"].values()
    )
    assert set(created["sparse_vectors_config"]) == {"bm25"}
    indexed = {kwargs["field"] for name, kwargs in client.calls if name == "create_payload_index"}
    assert {"script", "tenant_id", "visibility_keys", "kind", "current"} <= indexed
    assert "entities" not in indexed, "nothing filters on entities since the prefetch went"


async def test_an_upsert_writes_every_vector_the_record_carries() -> None:
    store, client = _store()
    await store.upsert(
        [
            SearchRecord(
                record_id="r1",
                collection="c",
                tenant_id="acme",
                dense={VectorName.DENSE_EN: [0.1] * 4, VectorName.DENSE_ML: [0.2] * 4},
                sparse=SparseVector(indices=[1], values=[1.0]),
                payload={"script": "latin"},
            )
        ]
    )
    point = client.calls[-1][1]["points"][0]
    assert set(point.vector) == {"dense_en", "dense_ml", "bm25"}
    assert point.payload["script"] == "latin" and point.payload["tenant_id"] == "acme"


@pytest.mark.parametrize(
    ("dense", "sparse", "expected"),
    [
        ({VectorName.DENSE_EN: [0.1] * 4}, None, "dense_en"),
        ({VectorName.DENSE_ML: [0.1] * 4}, None, "dense_ml"),
        ({}, SparseVector(indices=[1], values=[1.0]), "bm25"),
    ],
)
async def test_a_one_armed_query_is_labelled_with_its_arm(dense, sparse, expected) -> None:
    store, client = _store()
    await store.search_hybrid(
        "c",
        dense=dense,
        sparse=sparse,
        flt=SearchFilter(tenant_id="acme"),
        limit=5,
        prefetch_limit=8,
    )
    kwargs = client.calls[-1][1]
    assert kwargs["using"] == expected and "prefetch" not in kwargs


async def test_the_dense_arms_are_prefetched_in_mapping_order_then_bm25() -> None:
    store, client = _store()
    await store.search_hybrid(
        "c",
        dense={VectorName.DENSE_ML: [0.2] * 4, VectorName.DENSE_EN: [0.1] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        flt=SearchFilter(tenant_id="acme"),
        limit=5,
        prefetch_limit=8,
    )
    kwargs = client.calls[-1][1]
    assert [p.using for p in kwargs["prefetch"]] == ["dense_ml", "dense_en", "bm25"]
    assert all(p.limit == 8 for p in kwargs["prefetch"])


async def test_no_arms_is_an_empty_answer_without_a_round_trip() -> None:
    store, client = _store()
    assert (
        await store.search_hybrid(
            "c", dense={}, sparse=None, flt=SearchFilter(tenant_id="a"), limit=5, prefetch_limit=8
        )
        == []
    )
    assert client.calls == []
