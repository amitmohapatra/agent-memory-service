"""Baseline retrieval: index job -> Qdrant (local mode) hybrid search -> engine -> bundle.

Uses the deterministic hash embedding + BM25 sparse encoder + lexical reranker, so the
assertions are about *plumbing and isolation* (store-side filtering, fusion, exact lookups,
budgets, caching), not about semantic quality. Quality is measured in tests/eval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import EvidenceStatus, MessageRole, QueryType, Visibility
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.rag.indexer import KNOWLEDGE
from memory_service.ports.search import SearchFilter

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
OWNER = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
OTHER_USER = MemoryExecutionContext(tenant_id="acme", user_id="u2", workspace_id="ws1")
OTHER_TENANT = MemoryExecutionContext(tenant_id="globex", user_id="u1", workspace_id="ws1")


async def _ingest(
    container,
    uow_factory,
    ctx=OWNER,
    *,
    thread_id: str | None = None,
    visibility: Visibility | None = None,
    salt: str = "",
) -> str:
    register_handlers(container)
    ingestion = container.services["ingestion"]
    scoped = ctx.model_copy(update={"thread_id": thread_id}) if thread_id else ctx
    async with uow_factory() as uow:
        ack = await ingestion.accept_file(
            uow,
            scoped,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes() + salt.encode(),  # content-hash dedup is per tenant
            title="ACME FY26",
            visibility=visibility,
        )
        await uow.commit()
    # parse job, then the chained index job
    await container.tasks.drain()
    await container.tasks.drain()
    return ack.document_id


async def test_index_job_writes_hybrid_records(container, uow_factory) -> None:
    doc_id = await _ingest(container, uow_factory)
    indexer = container.services["indexer"]
    store = container.search
    collection = indexer.collection(KNOWLEDGE)
    assert collection.startswith("knowledge_hash-v1")
    async with uow_factory() as uow:
        chunks = await uow.documents.list_chunks("acme", doc_id)
        pending = await uow.documents.list_chunks("acme", doc_id, unindexed_only=True)
    assert chunks and not pending, "every chunk is marked indexed after the job"
    assert all(c.index_fingerprint == indexer.fingerprint for c in chunks)
    assert await store.count(collection, SearchFilter(tenant_id="acme")) == len(chunks)
    # dense, sparse and hybrid all find the EBITDA chunk with a tenant-only filter
    flt = SearchFilter(tenant_id="acme")
    q = "Adjusted EBITDA increased despite lower revenue"
    dense = await store.search_dense(
        collection, await indexer.embedding.embed_query(q), flt, limit=5
    )
    sparse = await store.search_sparse(collection, indexer.sparse.encode_query(q), flt, limit=5)
    hybrid = await store.search_hybrid(
        collection,
        dense=await indexer.embedding.embed_query(q),
        sparse=indexer.sparse.encode_query(q),
        flt=flt,
        limit=5,
        prefetch_limit=20,
    )
    assert {h.retriever for h in dense} == {"dense"} and {h.retriever for h in sparse} == {"bm25"}
    assert {h.retriever for h in hybrid} == {"fusion"}
    assert any("increased to EUR 98" in h.payload["text"] for h in sparse)
    assert any("increased to EUR 98" in h.payload["text"] for h in hybrid)
    # payload carries only what retrieval needs: security keys + display fields
    p = hybrid[0].payload
    assert set(p) >= {"tenant_id", "visibility_keys", "document_id", "page", "section_path", "text"}
    assert len(p["text"]) <= 2000
    # re-index is idempotent (upsert by deterministic point id)
    assert await indexer.rebuild_document("acme", doc_id) == len(chunks)
    assert await store.count(collection, flt) == len(chunks)
    # delete removes only that document's points
    await indexer.delete_document("acme", doc_id)
    assert await store.count(collection, flt) == 0


async def test_store_side_visibility_filtering(container, uow_factory) -> None:
    """The store itself never returns a point outside the caller's tenant + visibility keys."""
    engine = container.services["retrieval"]
    authz = container.services["authz"]
    q = "restructuring savings"
    # 1. default upload by a user (no thread) -> USER visibility: only u1 sees it
    doc_id = await _ingest(container, uow_factory)
    owner = await engine.retrieve(OWNER, q, limit=5)
    assert owner.candidates and owner.candidates[0].kind == "chunk"
    assert all(c.payload["document_id"] == doc_id for c in owner.candidates)
    assert (await engine.retrieve(OTHER_USER, q, limit=5)).candidates == []
    # 2. WORKSPACE visibility -> members of ws1 see it, non-members don't
    async with uow_factory() as uow:  # grants bump the user revision -> cached scope invalid
        for user, ws in (("u1", "ws1"), ("u2", "ws1"), ("u3", "ws2")):
            await authz.grant_membership("acme", user, workspaces=[ws], revisions=uow.revisions)
        await uow.commit()
    shared_id = await _ingest(
        container, uow_factory, visibility=Visibility.WORKSPACE, salt="\n\nShared copy.\n"
    )
    peer = await engine.retrieve(OTHER_USER, q, limit=10)
    assert peer.candidates and {c.payload["document_id"] for c in peer.candidates} == {shared_id}
    stranger = OTHER_USER.model_copy(update={"user_id": "u3", "workspace_id": "ws2"})
    assert (await engine.retrieve(stranger, q, limit=10)).candidates == []
    # the owner now sees both documents; the shared one never leaks the private one
    both = await engine.retrieve(OWNER, q, limit=10)
    assert {c.payload["document_id"] for c in both.candidates} == {doc_id, shared_id}
    # 3. different tenant -> nothing, even though the collection is shared
    assert (await engine.retrieve(OTHER_TENANT, q, limit=5)).candidates == []
    # a caller with no visibility keys at all gets nothing (filter is must_any, never empty=all)
    async with uow_factory() as uow:
        vis = await authz.visibility(OTHER_TENANT, revisions=uow.revisions)
    assert not vis.allows("acme", ["ws:acme/ws1"])
    empty = await container.search.search_hybrid(
        engine.indexer.collection(KNOWLEDGE),
        dense=await engine.indexer.embedding.embed_query(q),
        sparse=engine.indexer.sparse.encode_query(q),
        flt=SearchFilter(tenant_id="acme", must_any={"visibility_keys": []}),
        limit=5,
        prefetch_limit=20,
    )
    assert empty == []


async def test_thread_private_document_is_not_visible_to_other_users(
    container, uow_factory
) -> None:
    from memory_service.domain.ids import new_id

    thread_id = new_id("thread")
    ctx = OWNER.model_copy(
        update={
            "thread_id": thread_id,
            "session_id": new_id("session"),
            "turn_id": new_id("turn"),
            "workspace_id": None,
        }
    )
    async with uow_factory() as uow:  # first message creates the thread and grants the owner
        await container.services["conversation"].append_message(
            uow, ctx, role=MessageRole.USER, content="attaching the report"
        )
        await uow.commit()
    await _ingest(container, uow_factory, ctx, thread_id=thread_id)
    engine = container.services["retrieval"]
    q = "Legacy Services revenue"
    mine = await engine.retrieve(ctx, q, limit=5)
    assert mine.candidates
    other = OTHER_USER.model_copy(update={"thread_id": thread_id, "workspace_id": None})
    assert (await engine.retrieve(other, q, limit=5)).candidates == []
    # agent acting for the owner inherits the owner's access
    agent = ctx.model_copy(update={"agent_id": "planner", "agent_run_id": new_id("agent_run")})
    assert (await engine.retrieve(agent, q, limit=5)).candidates


async def test_engine_pipeline_exact_rerank_and_kinds(container, uow_factory) -> None:
    doc_id = await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    async with uow_factory() as uow:
        chunks = await uow.documents.list_chunks("acme", doc_id)
    target = next(c for c in chunks if "increased to EUR 98" in c.text)
    # exact identifier -> O(1) lookup, no hybrid search
    exact = await engine.retrieve(OWNER, f"show {target.chunk_id}")
    assert exact.routed.query_type is QueryType.EXACT_IDENTIFIER
    assert [c.record_id for c in exact.candidates] == [target.chunk_id]
    assert (
        exact.candidates[0].retrievers == ["exact"] and "fused_candidates" not in exact.diagnostics
    )
    # exact lookup respects visibility
    assert (await engine.retrieve(OTHER_TENANT, f"show {target.chunk_id}")).candidates == []
    # multi-hop question: hybrid + rerank, bounded to limit
    res = await engine.retrieve(OWNER, "why did Adjusted EBITDA increase despite lower revenue?")
    assert res.routed.query_type is QueryType.DOCUMENT_MULTI_HOP
    assert res.diagnostics["reranked"] is True and res.diagnostics["fused_candidates"] >= 5
    assert len(res.candidates) <= container.settings.retrieval.final_k
    assert all(c.rerank_score is not None for c in res.candidates[: engine.rerank_k])
    assert res.candidates[0].record_id == target.chunk_id
    # limit is honoured; document_ids restricts; kinds=memory returns nothing yet (M7)
    assert len((await engine.retrieve(OWNER, "revenue", limit=2)).candidates) == 2
    assert (await engine.retrieve(OWNER, "revenue", document_ids=["doc_nope"])).candidates == []
    assert (await engine.retrieve(OWNER, "revenue", kinds=("memory",))).candidates == []


async def test_context_builder_budget_and_cache(container, uow_factory) -> None:
    from memory_service.domain.ids import new_id

    thread_id = new_id("thread")
    ctx = OWNER.model_copy(
        update={
            "thread_id": thread_id,
            "session_id": new_id("session"),
            "turn_id": new_id("turn"),
        }
    )
    await _ingest(container, uow_factory)
    conversation = container.services["conversation"]
    async with uow_factory() as uow:
        for text in ("Let's review the FY26 numbers.", "Focus on EBITDA please."):
            await conversation.append_message(uow, ctx, role=MessageRole.USER, content=text)
        await uow.commit()
    builder = container.services["context_builder"]
    bundle = await builder.build(ctx, "why did Adjusted EBITDA increase despite lower revenue?")
    assert bundle.cache_hit is False and bundle.query_type is QueryType.DOCUMENT_MULTI_HOP
    assert bundle.knowledge and bundle.evidence.status is EvidenceStatus.COMPLETE
    assert "Focus on EBITDA" in bundle.conversation.rendered
    assert len(bundle.conversation.message_ids) == 2
    assert bundle.token_estimate <= bundle.token_budget
    first = bundle.knowledge[0]
    assert first.citation == f"chunk_id:{first.item_id}" and first.evidence[0].page == 11
    rendered = bundle.render()
    assert "increased to EUR 98" in rendered and "Recent conversation" in rendered
    # second call: served from cache
    again = await builder.build(ctx, "why did Adjusted EBITDA increase despite lower revenue?")
    assert again.cache_hit is True and again.knowledge[0].item_id == first.item_id
    # a new message bumps the thread revision -> cache miss
    async with uow_factory() as uow:
        await conversation.append_message(
            uow, ctx, role=MessageRole.USER, content="And the footnote?"
        )
        await uow.commit()
    third = await builder.build(ctx, "why did Adjusted EBITDA increase despite lower revenue?")
    assert third.cache_hit is False and len(third.conversation.message_ids) == 3
    # tight budget: nothing exceeds it
    small = await builder.build(ctx, "restructuring programme headcount", token_budget=260)
    assert small.token_estimate <= 260
