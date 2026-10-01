"""``query_expansion``: the model is consulted only when no routing rule fired; its terms widen
the hybrid search while the routed query stays the original; any failure leaves routing as is."""

from __future__ import annotations

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.adapters.search.qdrant_store import QdrantSearchStore
from memory_service.config.constants import RetrievalSettings
from memory_service.config.settings import SearchSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
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


async def test_expansion_reroutes_and_widens_search_but_keeps_the_original_query(parts) -> None:
    _, embedding, _ = parts
    reply = {
        "query_type": "ENTITY_RELATION",
        "terms": ["headcount", "restructuring", "plant", "factory"],
        "identifiers": [],
    }
    with mocked_gateway([reply]) as gw:
        engine = _engine(parts, gw.assist(uses=["query_expansion"]))
        res = await engine.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
        assert gw.route.call_count == 1
        assert gw.prompts()[0]["model"] == "gemini/gemini-3.8-flash"
        assert gw.prompts()[0]["messages"][1]["content"] == f"Query: {QUERY}"
    assert res.routed.query_type is QueryType.ENTITY_RELATION and res.routed.needs_graph
    assert res.routed.query == QUERY
    assert res.diagnostics["query_type"] == "ENTITY_RELATION"
    assert res.diagnostics["query_expansion"] == ["headcount", "restructuring", "plant"]
    # the written query is encoded while the model is asked (in case it misses its deadline);
    # the terms it returned change the search text, so that one is encoded and searched
    assert embedding.queries == [QUERY, f"{QUERY} headcount restructuring plant"]
    assert {c.record_id for c in res.candidates} >= {"chk_restructuring"}


async def test_gateway_failure_keeps_native_routing(parts) -> None:
    _, embedding, _ = parts
    with mocked_gateway(failing=True) as gw:
        engine = _engine(parts, gw.assist(uses=["query_expansion"]))
        res = await engine.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
        assert gw.route.call_count >= 1
    assert res.routed.query_type is QueryType.GENERAL_SEMANTIC
    assert "query_expansion" not in res.diagnostics
    assert embedding.queries == [QUERY]
    assert res.candidates


def _without_timings(diagnostics: dict) -> dict:
    return {k: v for k, v in diagnostics.items() if k != "timings_ms"}


async def test_flag_off_or_rule_fired_never_calls_the_model(parts) -> None:
    _, embedding, _ = parts
    with mocked_gateway(['{"query_type": "DECISION", "terms": ["x"], "identifiers": []}']) as gw:
        off = _engine(parts, gw.assist(uses=["grounding_judge"]))
        res = await off.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
        assert res.routed.query_type is QueryType.GENERAL_SEMANTIC
        on = _engine(parts, gw.assist(uses=["query_expansion"]))
        decided = await on.retrieve(
            CTX, "why did we decide to close the plant?", kinds=("chunk",), visibility=VISIBILITY
        )
        assert decided.routed.query_type is QueryType.DECISION
        assert gw.route.call_count == 0
    native = _engine(parts)
    plain = await native.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
    # the stage timings are the one per-request value; everything else must be identical
    assert _without_timings(plain.diagnostics) == _without_timings(res.diagnostics)
    assert "query_expansion" not in plain.diagnostics
    assert plain.diagnostics["query_type"] == "GENERAL_SEMANTIC"
    assert plain.diagnostics["signals"] == dict.fromkeys(plain.routed.signals, False)
    assert embedding.queries == [QUERY, "why did we decide to close the plant?", QUERY]


async def test_bounds_terms_capped_and_query_truncated_and_exact_needs_ids(parts) -> None:
    _, embedding, _ = parts
    reply = {
        "query_type": "EXACT_IDENTIFIER",
        "terms": [f"term{i}" for i in range(12)] + ["  term0 ", ""],
        "identifiers": [],
    }
    long_query = "plant " * 400
    with mocked_gateway([reply]) as gw:
        engine = _engine(parts, gw.assist(uses=["query_expansion"]))
        res = await engine.retrieve(CTX, long_query, kinds=("chunk",), visibility=VISIBILITY)
        sent = gw.prompts()[0]["messages"][1]["content"]
    assert len(sent) <= len("Query: ") + 500
    assert res.routed.query_type is QueryType.GENERAL_SEMANTIC
    assert res.diagnostics["query_expansion"] == [f"term{i}" for i in range(6)]
    assert embedding.queries[-1].endswith(" term0 term1 term2 term3 term4 term5")


async def test_a_slow_expansion_is_abandoned_at_its_deadline(parts) -> None:
    """The model call is on the read path; past ``query_expansion_timeout_ms`` the query is
    searched as written, with the encoding that ran while the model was asked."""
    import asyncio
    import time

    _, embedding, _ = parts
    with mocked_gateway([{"query_type": None, "terms": ["late"], "identifiers": []}]) as gw:
        engine = _engine(parts, gw.assist(uses=["query_expansion"]))
        engine.cfg = engine.cfg.model_copy(update={"query_expansion_timeout_ms": 50})

        async def slow(query: str):
            await asyncio.sleep(5)

        engine._expand_query = slow  # type: ignore[method-assign]
        started = time.perf_counter()
        res = await engine.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
        elapsed = time.perf_counter() - started
    assert elapsed < 2, "the search waited for the model past its deadline"
    assert res.diagnostics["query_expansion_timeout"] is True
    assert "query_expansion" not in res.diagnostics
    assert embedding.queries == [QUERY], "the written query is encoded once, not twice"
    assert res.candidates
