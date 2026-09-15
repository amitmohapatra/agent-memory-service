"""Retrieval-level isolation gate (M6): unauthorized retrieval through the search store = 0.

Every visibility variant of every tenant is indexed into ONE shared Qdrant collection with
identical text (so ranking cannot hide a leak), then every reader configuration runs the full
RetrievalEngine pipeline. The set of returned records must be a subset of what the independent
oracle allows — for dense, sparse and hybrid retrieval, with and without a document filter.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.adapters.models.rerankers import LexicalReranker
from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.adapters.search.qdrant_store import QdrantSearchStore
from memory_service.config.settings import RetrievalSettings, SearchSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.modules.rag.indexer import KNOWLEDGE, Indexer
from memory_service.modules.retrieval.engine import RetrievalEngine
from memory_service.ports.search import SearchRecord
from tests.security.test_isolation import TENANTS, _keys, _oracle, _spec

pytestmark = pytest.mark.security

TEXT = "Adjusted EBITDA increased to EUR 98 million despite lower revenue"

OBJECTS: list[dict[str, Any]] = [
    {
        "tenant": tenant,
        "visibility": vis.value,
        "owner": owner,
        "user": user,
        "group": group,
        "thread": thread,
        "workspace": ws,
        "work": "w1",
        "agent_group": ag,
    }
    for tenant, vis, owner, user, group, thread, ws, ag in itertools.product(
        TENANTS,
        Visibility,
        ["user:u1", "agent:research"],
        ["u1", "u2"],
        ["legal", "finance"],
        ["thr1", "thr2"],
        ["ws1", "ws2"],
        ["crew", "other"],
    )
]

READERS: list[dict[str, Any]] = [
    {
        "tenant": tenant,
        "user": user,
        "is_agent": is_agent,
        "agent": "research",
        "groups": groups,
        "threads": threads,
        "workspaces": workspaces,
        "works": ["w1"],
        "agent_groups": agent_groups,
    }
    for tenant, user, is_agent, groups, threads, workspaces, agent_groups in itertools.product(
        TENANTS,
        ["u1", "u2"],
        [False, True],
        [[], ["legal"]],
        [[], ["thr1"]],
        [[], ["ws1"], ["ws1", "ws2"]],
        [[], ["crew"]],
    )
]


class _NoUoW:
    """The engine only touches the UoW for exact lookups / scope; both are bypassed here."""

    def __call__(self):  # pragma: no cover - must never be reached
        raise AssertionError("retrieval must not open a unit of work for hybrid search")


@pytest.fixture(scope="module")
async def engine_and_ids() -> tuple[RetrievalEngine, dict[str, dict[str, Any]]]:
    store = QdrantSearchStore(SearchSettings(provider="memory", qdrant_local_path=":memory:"))
    embedding = HashEmbedding(dimension=32)
    sparse = Bm25SparseEncoder()
    indexer = Indexer(_NoUoW(), store, embedding, sparse, None)  # type: ignore[arg-type]
    collection = indexer.collection(KNOWLEDGE)
    await indexer.ensure_collections()
    dense = (await embedding.embed_documents([TEXT]))[0]
    sv = sparse.encode_documents([TEXT])[0]
    records = []
    by_id: dict[str, dict[str, Any]] = {}
    for n, obj in enumerate(OBJECTS):
        rid = f"chk_{n:05d}"
        by_id[rid] = obj
        records.append(
            SearchRecord(
                record_id=rid,
                collection=collection,
                tenant_id=obj["tenant"],
                dense=dense,
                sparse=sv,
                payload={
                    "kind": "chunk",
                    "visibility_keys": _keys(obj),
                    "document_id": f"doc_{obj['tenant']}_{n % 3}",
                    "text": TEXT,
                    "page": 11,
                    "section_path": "3.2",
                },
            )
        )
    await store.upsert(records)
    cfg = RetrievalSettings(prefetch_k=len(records), fused_k=len(records), final_k=len(records))
    engine = RetrievalEngine(
        _NoUoW(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        store,
        indexer,
        LexicalReranker(),
        settings=cfg,
        rerank_k=5,
    )
    return engine, by_id


async def _leaks(engine: RetrievalEngine, by_id, reader: dict, **kw) -> list[str]:
    spec = _spec(reader)
    ctx = MemoryExecutionContext(
        tenant_id=reader["tenant"],
        user_id=reader["user"],
        agent_id=reader["agent"] if reader["is_agent"] else None,
    )
    result = await engine.retrieve(ctx, TEXT, limit=len(by_id), visibility=spec, **kw)
    return [c.record_id for c in result.candidates if not _oracle(reader, by_id[c.record_id])]


async def test_every_reader_gets_only_authorized_records(engine_and_ids) -> None:
    engine, by_id = engine_and_ids
    total_returned = 0
    for reader in READERS:
        leaked = await _leaks(engine, by_id, reader)
        assert leaked == [], f"reader={reader} leaked={leaked[:5]}"
        spec = _spec(reader)
        allowed = {rid for rid, obj in by_id.items() if spec.allows(obj["tenant"], _keys(obj))}
        ctx = MemoryExecutionContext(tenant_id=reader["tenant"], user_id=reader["user"])
        got = {
            c.record_id
            for c in (
                await engine.retrieve(ctx, TEXT, limit=len(by_id), visibility=spec)
            ).candidates
        }
        # completeness: the store returns everything the oracle allows (no over-filtering)
        assert got == allowed
        total_returned += len(got)
    assert total_returned > 0


async def test_isolation_holds_per_retriever_and_with_document_filter(engine_and_ids) -> None:
    engine, by_id = engine_and_ids
    reader = {
        "tenant": "acme",
        "user": "u2",
        "is_agent": False,
        "agent": "research",
        "groups": [],
        "threads": ["thr1"],
        "workspaces": ["ws2"],
        "works": [],
        "agent_groups": [],
    }
    for dense, bm25, fusion in ((True, False, "rrf"), (False, True, "rrf"), (True, True, "none")):
        engine.cfg = engine.cfg.model_copy(update={"dense": dense, "bm25": bm25, "fusion": fusion})
        assert await _leaks(engine, by_id, reader) == []
        assert (
            await _leaks(engine, by_id, reader, document_ids=["doc_acme_0", "doc_globex_0"]) == []
        )
    engine.cfg = engine.cfg.model_copy(update={"dense": True, "bm25": True, "fusion": "rrf"})


async def test_empty_visibility_returns_nothing(engine_and_ids) -> None:
    engine, _ = engine_and_ids
    from memory_service.modules.authz.visibility import VisibilitySpecification

    spec = VisibilitySpecification(tenant_id="acme", keys=frozenset())
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="nobody")
    assert (await engine.retrieve(ctx, TEXT, limit=50, visibility=spec)).candidates == []
