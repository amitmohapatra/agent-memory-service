"""``query_expansion`` is accepted by a tenant's policy and ignored: retrieval never asks the model
about a query. Measured on LoCoMo, the call missed its 250 ms deadline on all but one of ~1,280
eligible questions per run, and given 5 s to answer it left the judged answers where they were
(10 gained, 10 lost of 496) for +2.0 s p50 per question."""

from __future__ import annotations

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.adapters.search.qdrant_store import QdrantSearchStore
from memory_service.config.constants import RetrievalSettings
from memory_service.config.settings import SearchSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.rag.indexer import KNOWLEDGE, Indexer
from memory_service.modules.rag.spaces import DenseSpaces
from memory_service.modules.retrieval.engine import RetrievalEngine
from memory_service.ports.search import SearchRecord, VectorName
from tests.support_llm import mocked_gateway

CTX = MemoryExecutionContext(tenant_id="t", user_id="u1")
VISIBILITY = VisibilitySpecification(tenant_id="t", keys=frozenset({"tenant:t"}))
QUERY = "staff cuts at the factory"
TEXTS = {
    "chk_restructuring": "The restructuring programme reduced headcount at the Lyon plant.",
    "chk_ebitda": "Adjusted EBITDA increased to EUR 98 million despite lower revenue.",
}


class _SpyEmbedding(HashEmbedding):
    def __init__(self) -> None:
        super().__init__(dimension=32)
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


class _NoUoW:
    def __call__(self):  # pragma: no cover - visibility is passed in, ids never match
        raise AssertionError("no unit of work expected")


@pytest.fixture
async def parts() -> tuple[Indexer, _SpyEmbedding, QdrantSearchStore]:
    store = QdrantSearchStore(SearchSettings(), local_path=":memory:")
    embedding = _SpyEmbedding()
    sparse = Bm25SparseEncoder()
    indexer = Indexer(_NoUoW(), store, DenseSpaces.single(embedding), sparse, None)  # type: ignore[arg-type]
    await indexer.ensure_collections()
    dense = await embedding.embed_documents(list(TEXTS.values()))
    sv = sparse.encode_documents(list(TEXTS.values()))
    await store.upsert(
        [
            SearchRecord(
                record_id=rid,
                collection=indexer.collection(KNOWLEDGE),
                tenant_id="t",
                dense={VectorName.DENSE_ML: dense[i]},
                sparse=sv[i],
                payload={
                    "kind": "chunk",
                    "visibility_keys": ["tenant:t"],
                    "document_id": "doc_1",
                    "text": text,
                },
            )
            for i, (rid, text) in enumerate(TEXTS.items())
        ]
    )
    embedding.queries.clear()
    return indexer, embedding, store


def _engine(parts, assist=None) -> RetrievalEngine:
    indexer, _, store = parts
    return RetrievalEngine(
        _NoUoW(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        store,
        indexer,
        settings=RetrievalSettings(graph=False),
        assist=assist,
    )


async def test_a_policy_that_allows_query_expansion_changes_nothing_in_retrieval(parts) -> None:
    """The same candidates and the same encoded query with the use allowed as without it, and
    no model call: the written query is searched as written."""
    _, embedding, _ = parts
    with mocked_gateway(
        [{"query_type": "TEMPORAL", "terms": ["headcount"], "identifiers": []}]
    ) as gw:
        allowed = _engine(parts, gw.assist(uses=["query_expansion"]))
        res = await allowed.retrieve(CTX, QUERY, visibility=VISIBILITY)
        assert gw.prompts() == [], "no model call for a query"
    plain = await _engine(parts).retrieve(CTX, QUERY, visibility=VISIBILITY)
    assert [c.record_id for c in res.candidates] == [c.record_id for c in plain.candidates]
    assert res.diagnostics["query_type"] == plain.diagnostics["query_type"]
    assert "query_expansion" not in res.diagnostics
    assert embedding.queries == [QUERY, QUERY], "each search encodes the written query once"
