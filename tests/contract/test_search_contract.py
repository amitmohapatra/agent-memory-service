"""One contract, every SearchStore adapter.

``search.provider`` chooses between an embedded Qdrant (``:memory:``) and a Qdrant server.
They are the same adapter against different backends, and retrieval correctness depends on
both behaving identically: tenant isolation is enforced by the filter, deletes really remove,
and hybrid fusion returns something ranked rather than one retriever's list.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from memory_service.config.settings import SearchSettings
from memory_service.ports.search import (
    CollectionSpec,
    SearchFilter,
    SearchRecord,
    SparseVector,
)

pytestmark = pytest.mark.contract

DIM = 8
ADAPTERS = ("embedded", "server")


def _build(name: str):
    from memory_service.adapters.search.qdrant_store import QdrantSearchStore

    settings = SearchSettings()
    if name == "embedded":
        settings = settings.model_copy(update={"qdrant_local_path": ":memory:"})
    return QdrantSearchStore(settings)


@pytest_asyncio.fixture(params=ADAPTERS, loop_scope="function")
async def store(request: pytest.FixtureRequest):
    adapter = _build(request.param)
    if not await adapter.ping():
        pytest.skip(f"{request.param} qdrant not reachable — start the dev stack")
    collection = f"contract_{uuid.uuid4().hex[:10]}"
    await adapter.ensure_collection(CollectionSpec(name=collection, dense_dim=DIM, sparse=True))
    adapter.contract_collection = collection  # type: ignore[attr-defined]
    try:
        yield adapter
    finally:
        await adapter.drop_collection(collection)
        await adapter.close()


def _dense(seed: float) -> list[float]:
    return [seed] * DIM


def _record(store, rid: str, tenant: str, seed: float, **payload) -> SearchRecord:
    return SearchRecord(
        record_id=rid,
        collection=store.contract_collection,
        tenant_id=tenant,
        dense=_dense(seed),
        sparse=SparseVector(indices=[1, 2], values=[0.5, 0.5]),
        payload={"tenant_id": tenant, **payload},
    )


async def test_an_upserted_record_is_retrievable_by_id(store) -> None:
    rid = uuid.uuid4().hex
    await store.upsert([_record(store, rid, "acme", 0.1, kind="memory")])
    got = await store.get(store.contract_collection, [rid])
    assert [r.record_id for r in got] == [rid]
    assert got[0].payload["kind"] == "memory"


async def test_dense_search_ranks_the_nearer_vector_first(store) -> None:
    near, far = uuid.uuid4().hex, uuid.uuid4().hex
    await store.upsert([_record(store, near, "acme", 0.9), _record(store, far, "acme", -0.9)])
    hits = await store.search_dense(
        store.contract_collection, _dense(0.9), SearchFilter(tenant_id="acme"), limit=2
    )
    assert [h.record_id for h in hits][:1] == [near], "ranking must reflect distance"


async def test_a_filter_cannot_see_another_tenants_records(store) -> None:
    """The isolation that every other guarantee rests on."""
    mine, theirs = uuid.uuid4().hex, uuid.uuid4().hex
    await store.upsert([_record(store, mine, "acme", 0.5), _record(store, theirs, "globex", 0.5)])
    hits = await store.search_dense(
        store.contract_collection, _dense(0.5), SearchFilter(tenant_id="acme"), limit=10
    )
    assert theirs not in {h.record_id for h in hits}
    assert await store.count(store.contract_collection, SearchFilter(tenant_id="acme")) == 1


async def test_hybrid_returns_a_fused_ranking(store) -> None:
    ids = [uuid.uuid4().hex for _ in range(3)]
    await store.upsert([_record(store, rid, "acme", 0.1 * i) for i, rid in enumerate(ids)])
    hits = await store.search_hybrid(
        store.contract_collection,
        dense=_dense(0.2),
        sparse=SparseVector(indices=[1, 2], values=[0.5, 0.5]),
        flt=SearchFilter(tenant_id="acme"),
        limit=3,
        prefetch_limit=10,
    )
    assert hits, "hybrid must return something when both retrievers can match"
    assert len({h.record_id for h in hits}) == len(hits), "fusion must not duplicate a record"
    assert all(h.score is not None for h in hits)


async def test_delete_removes_the_record_from_results(store) -> None:
    rid = uuid.uuid4().hex
    await store.upsert([_record(store, rid, "acme", 0.3)])
    await store.delete(store.contract_collection, [rid])
    assert await store.get(store.contract_collection, [rid]) == []


async def test_delete_by_filter_removes_exactly_the_matching_records(store) -> None:
    """How a superseded document generation is purged; over-deleting loses live data."""
    doomed, kept = uuid.uuid4().hex, uuid.uuid4().hex
    await store.upsert(
        [
            _record(store, doomed, "acme", 0.4, document_id="doc-1"),
            _record(store, kept, "acme", 0.4, document_id="doc-2"),
        ]
    )
    removed = await store.delete_by_filter(
        store.contract_collection, SearchFilter(tenant_id="acme", must={"document_id": "doc-1"})
    )
    assert removed >= 1
    assert [r.record_id for r in await store.get(store.contract_collection, [doomed, kept])] == [
        kept
    ]


async def test_record_ids_lists_what_the_filter_matches(store) -> None:
    ids = sorted(uuid.uuid4().hex for _ in range(3))
    await store.upsert([_record(store, rid, "acme", 0.6, document_id="doc-9") for rid in ids])
    found = await store.record_ids(
        store.contract_collection, SearchFilter(tenant_id="acme", must={"document_id": "doc-9"})
    )
    assert sorted(found) == ids
