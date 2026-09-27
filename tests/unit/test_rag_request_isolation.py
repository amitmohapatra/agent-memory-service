"""RAG companion requirements belong to one request and one evidence source."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from memory_service.config.constants import RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, ContextEdge
from memory_service.domain.enums import ContextGraphEdge, QueryType
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.context.evidence import VerificationStage
from memory_service.modules.context.expansion import ExpansionStage
from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.router import QueryRouter

pytestmark = pytest.mark.unit
CTX = MemoryExecutionContext(tenant_id="t", user_id="u")
VIS = VisibilitySpecification(tenant_id="t", keys=frozenset({"tenant:t"}))
CFG = RetrievalSettings(escalation_max_rounds=0, abstain_when_insufficient=False)
ROUTE = QueryRouter().routed(
    "Explain the revenue footnotes",
    QueryType.DOCUMENT_LOCAL,
    identifiers=[],
    signals={},
    has_thread=False,
)


def seed(name: str, *, kind: str = "chunk") -> Candidate:
    return Candidate(
        record_id=f"chk_{name}",
        kind=kind,
        text="Revenue rose.",
        score=0.9,
        payload={"node_id": f"node_{name}", "document_id": f"doc_{name}"},
    )


def edge(name: str, kind: ContextGraphEdge = ContextGraphEdge.FOOTNOTE) -> ContextEdge:
    return ContextEdge(
        tenant_id="t",
        document_id=f"doc_{name}",
        source_id=f"node_{name}",
        target_id=f"target_{name}",
        edge=kind,
        label="1",
    )


def factory(repo):
    @asynccontextmanager
    async def uow():
        yield SimpleNamespace(documents=repo)

    return uow


class Documents:
    async def edges_from(self, tenant, nodes, *, kinds):
        return [edge(n.removeprefix("node_")) for n in nodes if n.startswith("node_")]


async def test_memory_only_request_cannot_inherit_previous_document_requirements():
    stage = VerificationStage(factory(Documents()), settings=CFG)
    first, second = {}, {}
    await stage(CTX, ROUTE, [seed("a")], VIS, first)
    await stage(CTX, ROUTE, [seed("memory", kind="memory")], VIS, second)
    assert first["evidence_seed_groups"]
    assert second["evidence_seed_groups"] == {}
    assert second["evidence_targets"] == {}


async def test_concurrent_requests_do_not_exchange_companion_metadata():
    a_waiting, release_a = asyncio.Event(), asyncio.Event()

    class Interleaved(Documents):
        async def edges_from(self, tenant, nodes, *, kinds):
            if nodes == ["target_a"]:
                a_waiting.set()
                await release_a.wait()
            return await super().edges_from(tenant, nodes, kinds=kinds)

    stage = VerificationStage(factory(Interleaved()), settings=CFG)
    first, second = {}, {}
    task = asyncio.create_task(stage(CTX, ROUTE, [seed("a")], VIS, first))
    try:
        await asyncio.wait_for(a_waiting.wait(), 1)
        await stage(CTX, ROUTE, [seed("b")], VIS, second)
    finally:
        release_a.set()
        await task
    assert set(first["evidence_seed_groups"]) == {"node_a"}
    assert set(second["evidence_seed_groups"]) == {"node_b"}
    assert set(first["evidence_targets"]).isdisjoint(second["evidence_targets"])


async def test_same_footnote_label_in_two_documents_requires_both_sources():
    stage = VerificationStage(factory(Documents()), settings=CFG)
    companion = Candidate(
        record_id="footnote_a",
        kind="chunk",
        text="Revenue excludes taxes.",
        score=0.5,
        payload={"node_id": "target_a", "document_id": "doc_a"},
        expanded_from="chk_a",
        expansion_edge="FOOTNOTE",
    )
    diagnostics = {}
    await stage(CTX, ROUTE, [seed("a"), seed("b"), companion], VIS, diagnostics)
    report = diagnostics["evidence"]
    assert len(report["required_groups"]) == 2
    assert len(report["satisfied_groups"]) == 1
    assert len(report["missing_groups"]) == 1
    assert report["status"] == "INCOMPLETE"


async def test_parent_summaries_keep_their_own_document_and_source():
    class Parents:
        async def edges_from(self, tenant, nodes, *, kinds):
            return [edge("a", ContextGraphEdge.PARENT), edge("b", ContextGraphEdge.PARENT)]

        async def get_chunks(self, *args):
            return []

        async def chunks_for_nodes(self, *args):
            return []

        async def node_summaries(self, *args):
            return {"target_a": "Summary A", "target_b": "Summary B"}

    stage = ExpansionStage(factory(Parents()), settings=CFG)
    seeds = [seed("a"), seed("b")]
    added = await stage.expand(CTX, seeds, seeds, budget=2)
    assert [(c.payload["document_id"], c.expanded_from) for c in added] == [
        ("doc_a", "chk_a"),
        ("doc_b", "chk_b"),
    ]


async def test_expansion_batches_neighbours_and_edge_targets_in_one_read():
    def chunk(identifier, node, ordinal):
        return Chunk(
            chunk_id=identifier,
            node_id=node,
            document_id="doc_a",
            document_version_id="v1",
            tenant_id="t",
            ordinal=ordinal,
            text=identifier,
            contextual_text=identifier,
            text_hash=identifier,
        )

    chunks = [
        chunk("chk_a", "node_a", 0),
        chunk("next", "node_a", 1),
        chunk("footnote", "target_a", 0),
    ]
    reads = []

    class Neighbours(Documents):
        async def get_chunks(self, *args):
            return chunks[:1]

        async def chunks_for_nodes(self, tenant, nodes):
            reads.append(set(nodes))
            return [c for c in chunks if c.node_id in nodes]

        async def node_summaries(self, *args):
            return {}

    stage = ExpansionStage(factory(Neighbours()), settings=CFG)
    added = await stage.expand(CTX, [seed("a")], [seed("a")], budget=2)
    assert {c.record_id for c in added} == {"next", "footnote"}
    assert reads == [{"node_a", "target_a"}]
