"""The native fusion constant uses the same rank convention as client-side fusion."""

import pytest
from qdrant_client import models  # noqa: TID251 - adapter wire contract

from memory_service.modules.rag.indexer import KNOWLEDGE
from memory_service.modules.retrieval.engine import rrf_fuse
from memory_service.ports.search import SparseVector
from tests.unit import test_llm_retrieval as base
from tests.unit.test_search_store_wire import FakeClient, _flt, _store

parts = base.parts


@pytest.mark.parametrize("constant", [0, 1, 60])
async def test_native_scores_match_client_fusion(parts, constant):
    indexer, embedding, _, store = parts
    collection = indexer.collection(KNOWLEDGE)
    dense = await embedding.embed_query(base.QUERY)
    sparse = indexer.sparse.encode_query(base.QUERY)
    flt = base.VISIBILITY.search_filter(kind="chunk")
    lists = [
        await store.search_dense(collection, dense, flt, limit=10),
        await store.search_sparse(collection, sparse, flt, limit=10),
    ]
    expected = rrf_fuse(lists, k=constant)
    hits = await store.search_hybrid(
        collection,
        dense=dense,
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
        dense=[0.1] * 4,
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
    assert all(p.filter == kwargs["query_filter"] for p in kwargs["prefetch"])
    assert kwargs["limit"] == 5
