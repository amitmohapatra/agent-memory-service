"""M10 strategies on the fixture corpus: PageIndex tree routing, RAPTOR-style summary
fusion, personalised PageRank in the graph stage, and ColBERT-style multivectors in Qdrant
(with a deterministic fake encoder — the store path is real, the model is not)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
from memory_service.modules.graph.retrieval import personalized_pagerank
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.retrieval.strategies import (
    LateInteractionRetriever,
    PageIndexRetriever,
    RaptorRetriever,
)

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
Q = "Why did Adjusted EBITDA increase despite lower revenue?"


class FakeLateInteraction:
    """Token-level hash vectors: identical tokens give identical vectors, so MaxSim rewards
    shared vocabulary. Exercises the multivector collection/upsert/query path for real."""

    dimension = 16

    def _tok(self, token: str) -> list[float]:
        h = hashlib.blake2b(token.lower().encode(), digest_size=self.dimension).digest()
        v = [(b - 128) / 128 for b in h]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed_documents_multi(self, texts: Sequence[str]) -> list[list[list[float]]]:
        return [[self._tok(t) for t in text.split()[:64]] for text in texts]

    async def embed_query_multi(self, text: str) -> list[list[float]]:
        return [self._tok(t) for t in text.split()[:32]]

    def fingerprint(self) -> str:
        return "colbert-fake-d16"


async def _ingest(container, uow_factory, ctx=U1, *, salt=""):
    register_handlers(container)
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            ctx,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes() + salt.encode(),
            title="ACME FY26",
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack.document_id


async def test_pageindex_and_raptor_fuse_without_losing_the_baseline(
    container, uow_factory
) -> None:
    await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    indexer = container.services["indexer"]
    base = await engine.retrieve(U1, Q)
    base_ids = {c.record_id for c in base.candidates if c.kind == "chunk"}
    engine.retrievers["pageindex"] = PageIndexRetriever(
        container.services["uow_factory"], container.search, indexer
    )
    engine.retrievers["raptor"] = RaptorRetriever(container.search, indexer)
    try:
        res = await engine.retrieve(U1, Q)
    finally:
        engine.retrievers.clear()
    assert res.diagnostics["strategies"]["pageindex"] > 0
    assert res.diagnostics["strategies"]["raptor"] > 0
    ids = {c.record_id for c in res.candidates if c.kind == "chunk"}
    assert base_ids <= ids | {c.record_id for c in res.candidates}
    # fused candidates carry every retriever that found them
    top = next(c for c in res.candidates if c.kind == "chunk" and c.expansion_edge is None)
    assert "pageindex" in top.retrievers or "fusion" in top.retrievers
    assert any(c.kind == "summary" and "raptor" in c.retrievers for c in res.candidates)
    # PageIndex alone: routed sections contain the evidence pages
    pi = PageIndexRetriever(container.services["uow_factory"], container.search, indexer)
    async with uow_factory() as uow:
        vis = await container.services["authz"].visibility(U1, revisions=uow.revisions)
    hits = await pi(U1, res.routed, vis, None)
    assert hits and all(h.retriever == "pageindex" for h in hits)
    assert {h.payload.get("page") for h in hits} & {11, 14, 20}
    # conversation-history questions are not routed through the tree
    assert (
        await pi(U1, engine.router.route("what did I say earlier in this thread?"), vis, None) == []
    )


async def test_graph_ppr_ranks_by_centrality(container, uow_factory) -> None:
    await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    stage = engine.post_stages["graph"]
    stage.ppr = True
    try:
        res = await engine.retrieve(U1, Q)
    finally:
        stage.ppr = False
    assert res.diagnostics["graph"]["ppr"] is True
    facts = [c for c in res.candidates if c.kind == "fact"]
    assert facts and res.diagnostics["evidence"]["status"] == "COMPLETE"
    # the pure function: seeds get the mass, unreachable nodes get none
    from datetime import UTC, datetime

    from memory_service.domain.evidence import EvidenceRef
    from memory_service.ports.intelligence import Relation

    now = datetime.now(UTC)

    def rel(s, o, w=0.5):
        return Relation(
            relation_id=f"{s}-{o}",
            tenant_id="acme",
            subject_id=s,
            predicate="x",
            object_id=o,
            observed_at=now,
            confidence=w,
            evidence=[EvidenceRef(source_type="memory", source_id="m", observed_at=now)],
        )

    scores = personalized_pagerank(
        [rel("a", "b"), rel("b", "c"), rel("c", "d"), rel("x", "y")], {"a"}
    )
    ranked = sorted(("a", "b", "c", "d"), key=lambda n: -scores[n])
    assert set(ranked[:2]) == {"a", "b"} and ranked[-1] == "d"
    assert scores["x"] == scores["y"] == 0.0 and abs(sum(scores.values()) - 1.0) < 1e-6
    assert personalized_pagerank([], {"a"}) == {}


async def test_late_interaction_multivectors_in_qdrant(container, uow_factory) -> None:
    indexer = container.services["indexer"]
    indexer.late_interaction = FakeLateInteraction()
    try:
        assert "colbert-fake-d16" in indexer.fingerprint
        doc_id = await _ingest(container, uow_factory)
        engine = container.services["retrieval"]
        retriever = LateInteractionRetriever(container.search, indexer, indexer.late_interaction)
        async with uow_factory() as uow:
            vis = await container.services["authz"].visibility(U1, revisions=uow.revisions)
        hits = await retriever(U1, engine.router.route(Q), vis, None)
        assert hits and all(h.retriever == "late_interaction" for h in hits)
        assert hits[0].payload["document_id"] == doc_id
        assert any("EBITDA" in h.payload["text"] for h in hits[:3])
        # visibility still applies inside the multivector query
        stranger = MemoryExecutionContext(tenant_id="acme", user_id="u2")
        async with uow_factory() as uow:
            vis2 = await container.services["authz"].visibility(stranger, revisions=uow.revisions)
        assert await retriever(stranger, engine.router.route(Q), vis2, None) == []
        # fused into the engine as an extra retriever
        engine.retrievers["late_interaction"] = retriever
        try:
            res = await engine.retrieve(U1, Q)
        finally:
            engine.retrievers.clear()
        assert res.diagnostics["strategies"]["late_interaction"] > 0
        assert res.routed.query_type is QueryType.DOCUMENT_MULTI_HOP
        assert res.diagnostics["evidence"]["status"] == "COMPLETE"
    finally:
        indexer.late_interaction = None
