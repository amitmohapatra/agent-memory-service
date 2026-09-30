"""The engine encodes what the query's script calls for and says which script it saw."""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import asyncio
import time

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.adapters.search.qdrant_store import QdrantSearchStore
from memory_service.config.constants import RetrievalSettings
from memory_service.config.settings import SearchSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.script import Script
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES, Indexer
from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
from memory_service.modules.retrieval.engine import RetrievalEngine
from memory_service.ports.search import SearchRecord, VectorName

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="t", user_id="u1")
VISIBILITY = VisibilitySpecification(tenant_id="t", keys=frozenset({"tenant:t"}))
TEXTS = {
    "chk_berlin": "The Berlin office opened in May.",
    "chk_moscow": "Офис в Москве открылся в мае.",
}


class _Spy(HashEmbedding):
    def __init__(self, dimension: int, *, delay: float = 0.0) -> None:
        super().__init__(dimension=dimension)
        self.queries: list[str] = []
        self.delay = delay

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return await super().embed_query(text)

    def fingerprint(self) -> str:
        return f"spy-d{self.dimension}"


class _NoUoW:
    def __call__(self):  # pragma: no cover - visibility is passed in
        raise AssertionError("no unit of work expected")


class _RecordingStore(QdrantSearchStore):
    def __init__(self) -> None:
        super().__init__(SearchSettings(), local_path=":memory:")
        self.hybrid_calls: list[dict] = []

    async def search_hybrid(self, collection, **kwargs):  # type: ignore[override]
        self.hybrid_calls.append(kwargs)
        return await super().search_hybrid(collection, **kwargs)


async def _parts(delay: float = 0.0, **settings):
    store = _RecordingStore()
    english, multilingual = _Spy(8, delay=delay), _Spy(16, delay=delay)
    spaces = DenseSpaces(
        [
            DenseSpace(VectorName.DENSE_EN, english, query_scripts=frozenset({Script.LATIN})),
            DenseSpace(VectorName.DENSE_ML, multilingual),
        ]
    )
    indexer = Indexer(_NoUoW(), store, spaces, Bm25SparseEncoder(), None)  # type: ignore[arg-type]
    await indexer.ensure_collections()
    texts = list(TEXTS.values())
    dense = await indexer.embed_cached(texts, [f"h{n}" for n in range(len(texts))])
    sparse = indexer.sparse.encode_documents(texts)
    for kind, base in (("chunk", KNOWLEDGE), ("memory", MEMORIES)):
        await store.upsert(
            [
                SearchRecord(
                    record_id=f"{rid}_{kind}",
                    collection=indexer.collection(base),
                    tenant_id="t",
                    dense={space: vectors[i] for space, vectors in dense.items()},
                    sparse=sparse[i],
                    payload={
                        "kind": kind,
                        "visibility_keys": ["tenant:t"],
                        "current": True,
                        "text": text,
                        "entities": ["berlin"] if "Berlin" in text else ["москва"],
                    },
                )
                for i, (rid, text) in enumerate(TEXTS.items())
            ]
        )
    english.queries.clear()
    multilingual.queries.clear()
    engine = RetrievalEngine(
        _NoUoW(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        store,
        indexer,
        settings=RetrievalSettings(graph=False, **settings),
    )
    return engine, english, multilingual, store


async def test_a_latin_query_searches_both_spaces_and_says_so() -> None:
    engine, english, multilingual, store = await _parts()
    result = await engine.retrieve(CTX, "Berlin office", kinds=("chunk",), visibility=VISIBILITY)
    assert english.queries == ["Berlin office"] and multilingual.queries == ["Berlin office"]
    assert result.diagnostics["query_script"] == "latin"
    assert set(store.hybrid_calls[-1]["dense"]) == {VectorName.DENSE_EN, VectorName.DENSE_ML}
    assert {c.record_id for c in result.candidates} >= {"chk_berlin_chunk"}


async def test_a_cyrillic_query_never_pays_the_english_encode() -> None:
    engine, english, multilingual, store = await _parts()
    result = await engine.retrieve(CTX, "Офис в Москве", kinds=("chunk",), visibility=VISIBILITY)
    assert english.queries == [] and multilingual.queries == ["Офис в Москве"]
    assert result.diagnostics["query_script"] == "cyrillic"
    assert set(store.hybrid_calls[-1]["dense"]) == {VectorName.DENSE_ML}
    assert {c.record_id for c in result.candidates} >= {"chk_moscow_chunk"}


#: One encoder's delay in the concurrency test. Running both in sequence costs two of these,
#: running them together costs one, so the bound sits between the two -- at 0.05 s the old
#: bound of 0.12 s was *above* the 0.10 s serial floor, which made the test both weak (serial
#: encoders could pass it) and flaky (a loaded host spends the 0.02 s margin on scheduling).
ENCODE_DELAY = 0.4


async def test_the_two_encoders_run_at_the_same_time() -> None:
    engine, _, _, _ = await _parts(delay=ENCODE_DELAY)
    started = time.perf_counter()
    await engine.retrieve(CTX, "Berlin office", kinds=("chunk",), visibility=VISIBILITY)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.6 * ENCODE_DELAY, (
        f"the encoders ran one after the other: {elapsed:.3f}s for two "
        f"{ENCODE_DELAY}s encodes, and running them together costs one"
    )


async def test_the_shipped_fusion_is_unweighted() -> None:
    engine, _, _, store = await _parts()
    await engine.retrieve(CTX, "What opened in Berlin?", kinds=("memory",), visibility=VISIBILITY)
    assert store.hybrid_calls[-1]["weights"] is None
