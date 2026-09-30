"""The native fusion constant uses the same rank convention as client-side fusion, and a
fitted weight reaches the wire aligned with the arm it names."""

import pytest
from qdrant_client import models  # noqa: TID251 - adapter wire contract

from memory_service.modules.rag.indexer import KNOWLEDGE
from memory_service.modules.retrieval.engine import rrf_fuse
from memory_service.ports.search import AnchoredPrefetch, SparseVector, VectorName
from tests.unit import test_llm_retrieval as base
from tests.unit.test_search_store_wire import FakeClient, _flt, _store

parts = base.parts


@pytest.mark.parametrize("constant", [0, 1, 60])
async def test_native_scores_match_client_fusion(parts, constant):
    indexer, embedding, store = parts
    collection = indexer.collection(KNOWLEDGE)
    dense = await embedding.embed_query(base.QUERY)
    sparse = indexer.sparse.encode_query(base.QUERY)
    flt = base.VISIBILITY.search_filter(kind="chunk")
    lists = [
        await store.search_dense(collection, VectorName.DENSE_ML, dense, flt, limit=10),
        await store.search_sparse(collection, sparse, flt, limit=10),
    ]
    expected = rrf_fuse(lists, k=constant)
    hits = await store.search_hybrid(
        collection,
        dense={VectorName.DENSE_ML: dense},
        sparse=sparse,
        flt=flt,
        limit=10,
        prefetch_limit=10,
        rrf_k=constant,
    )
    assert {h.record_id: h.score for h in hits} == pytest.approx(
        {rid: score for rid, score, _, _ in expected}
    )


@pytest.mark.parametrize("constant", [1, 60])
async def test_wire_constant_is_translated_and_preserves_authorization(constant):
    client = FakeClient()
    await _store(client).search_hybrid(
        "c",
        dense={VectorName.DENSE_ML: [0.1] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        flt=_flt(),
        limit=5,
        prefetch_limit=10,
        rrf_k=constant,
    )
    kwargs = client.calls[-1][1]
    if constant == 1:
        assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    else:
        assert isinstance(kwargs["query"], models.RrfQuery)
        assert kwargs["query"].rrf.k == constant + 1
        assert kwargs["query"].rrf.weights is None
    assert all(p.filter == kwargs["query_filter"] for p in kwargs["prefetch"])
    assert kwargs["limit"] == 5


async def test_weights_reach_the_wire_in_arm_order_and_equal_weights_do_not():
    """A fitted weight is spelled out beside the arm it names, in prefetch order; a weight
    of 1.0 everywhere leaves the historical wire query untouched."""
    client = FakeClient()
    store = _store(client)
    call = {
        "dense": {VectorName.DENSE_EN: [0.1] * 4, VectorName.DENSE_ML: [0.2] * 4},
        "sparse": SparseVector(indices=[1], values=[1.0]),
        "flt": _flt(),
        "limit": 5,
        "prefetch_limit": 10,
    }
    await store.search_hybrid("c", **call, weights={VectorName.BM25: 1.0})
    assert client.calls[-1][1]["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    await store.search_hybrid("c", **call, weights={VectorName.DENSE_ML: 1.5, VectorName.BM25: 0.5})
    kwargs = client.calls[-1][1]
    assert [p.using for p in kwargs["prefetch"]] == ["dense_en", "dense_ml", "bm25"]
    assert kwargs["query"].rrf.k == 2 and kwargs["query"].rrf.weights == [1.0, 1.5, 0.5]


async def test_weighted_client_fusion_scales_each_list():
    from memory_service.ports.search import SearchHit

    def hits(name: str, *ids: str) -> list[SearchHit]:
        return [
            SearchHit(record_id=i, score=1.0, retriever=name, payload={})  # type: ignore[arg-type]
            for i in ids
        ]

    fused = rrf_fuse([hits("dense_ml", "a"), hits("bm25", "b")], k=1, weights=[2.0, 1.0])
    assert [(rid, score) for rid, score, _, _ in fused] == [("a", 1.0), ("b", 0.5)]


async def test_an_anchored_prefetch_narrows_the_filter_and_keeps_the_tenant():
    client = FakeClient()
    await _store(client).search_hybrid(
        "c",
        dense={VectorName.DENSE_ML: [0.2] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        flt=_flt(),
        limit=5,
        prefetch_limit=10,
        anchors=[AnchoredPrefetch(vector=VectorName.DENSE_ML, must_any={"entities": ["caroline"]})],
    )
    kwargs = client.calls[-1][1]
    assert [p.using for p in kwargs["prefetch"]] == ["dense_ml", "dense_ml", "bm25"]
    anchored = kwargs["prefetch"][1].filter
    conditions = {c.key: c for c in anchored.must}
    assert conditions["tenant_id"].match.value == "acme"
    assert conditions["entities"].match.any == ["caroline"]
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)


async def test_an_anchor_without_its_query_vector_is_refused():
    with pytest.raises(ValueError, match="anchored prefetch"):
        await _store(FakeClient()).search_hybrid(
            "c",
            dense={VectorName.DENSE_ML: [0.2] * 4},
            sparse=None,
            flt=_flt(),
            limit=5,
            prefetch_limit=10,
            anchors=[AnchoredPrefetch(vector=VectorName.DENSE_EN, must_any={"entities": ["x"]})],
        )
