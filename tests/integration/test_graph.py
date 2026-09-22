"""Knowledge graph end to end on PostgreSQL: document enrichment, memory facts, bounded
multi-hop traversal with visibility, temporal (as-of) view, and the retrieval stage that
turns graph hops into evidence chunks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole, ObservationKind, QueryType, Visibility
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
U2 = MemoryExecutionContext(tenant_id="acme", user_id="u2", workspace_id="ws1")


async def _ingest(container, uow_factory, ctx, *, visibility=None, salt=""):
    register_handlers(container)
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            ctx,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes() + salt.encode(),
            title="ACME FY26",
            visibility=visibility,
        )
        await uow.commit()
    await container.tasks.drain()  # parse
    await container.tasks.drain()  # index + graph enrichment
    return ack.document_id


async def _observe(container, uow_factory, ctx, content, kind=ObservationKind.MESSAGE):
    register_handlers(container)
    async with uow_factory() as uow:
        ack = await container.services["memory"].submit_observation(
            uow, ctx, kind=kind, content=content
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack


async def test_document_graph_and_multi_hop_stage(container, uow_factory) -> None:
    doc_id = await _ingest(container, uow_factory, U1)
    graph = container.services["graph"]
    entities, relations = await container.graph_store.count("acme")
    assert entities > 5 and relations > 10
    # explicit graph query: definition and co-occurrence facts carry page evidence
    answer = await graph.query(U1, entities=["Adjusted EBITDA"], hops=1)
    assert [e.canonical_name for e in answer.matched] == ["adjusted ebitda"]
    preds = {r.predicate for r in answer.relations}
    assert {"defined_in", "mentioned_in", "co_occurs_with"} <= preds
    defined = next(r for r in answer.relations if r.predicate == "defined_in")
    assert defined.evidence[0].page == 1 and defined.document_id == doc_id
    mentioned = next(r for r in answer.relations if r.predicate == "mentioned_in")
    assert {1, 11, 14, 20} <= set(mentioned.attributes["pages"])
    assert {r.evidence[0].page for r in answer.relations} >= {1, 11, 14, 20}
    # factual layer: values with period + currency, the FY25 comparative, exclusions, the
    # counterfactual kept apart, and aliases ("FY26 Adjusted EBITDA" resolved to the metric)
    by_pred: dict[str, list] = {}
    for r in answer.relations:
        by_pred.setdefault(r.predicate, []).append(r)
    ents = {e.entity_id: e for e in answer.entities}
    values = {(ents[r.object_id].name, r.attributes.get("period")) for r in by_pred["has_value"]}
    assert {("EUR 98 million", "FY26"), ("EUR 81 million", "FY25")} <= values
    excluded = {ents[r.object_id].canonical_name for r in by_pred["excludes"]}
    assert {
        "restructuring charges",
        "litigation settlement",
        "share-based compensation",
    } <= excluded
    counter = by_pred["would_have_value"][0]
    assert ents[counter.object_id].name == "EUR 91 million" and counter.attributes["hypothetical"]
    assert answer.matched[0].entity_type == "METRIC"
    assert [
        e.canonical_name for e in (await graph.query(U1, entities=["ARR"], hops=1)).matched
    ] == ["recurring revenue"]
    assert [
        e.canonical_name for e in (await graph.query(U1, entities=["ACME"], hops=1)).matched
    ] == ["acme corporation"]
    # two hops reach entities that never share a chunk with the seed
    two = await graph.query(U1, entities=["Adjusted EBITDA"], hops=2)
    assert two.visited > answer.visited and len(two.relations) > len(answer.relations)
    # the same doc re-indexed does not duplicate (deterministic ids + delete_for_document)
    indexer = container.services["indexer"]
    await indexer.rebuild_document("acme", doc_id)
    await graph.enrich_document("acme", doc_id)
    assert await container.graph_store.count("acme") == (entities, relations)

    # retrieval stage: multi-hop question gets graph facts + evidence-chunk expansion
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "Why did Adjusted EBITDA increase despite lower revenue?")
    assert res.routed.query_type is QueryType.DOCUMENT_MULTI_HOP
    assert "graph" in res.diagnostics["stages"] and res.diagnostics["graph"]["relations"] > 0
    facts = [c for c in res.candidates if c.kind == "fact"]
    assert facts and all(c.retrievers == ["graph"] for c in facts)
    assert all(c.payload["document_id"] == doc_id for c in facts)
    # graph-expanded chunks are marked and visibility-checked; pages 14 (restructuring) and 20
    # (footnote) are reachable through the graph regardless of dense/BM25 rank
    expanded = [c for c in res.candidates if c.expansion_edge == "GRAPH_EVIDENCE"]
    chunk_pages = {c.payload.get("page") for c in res.candidates if c.kind == "chunk"}
    assert {11, 14, 20, 1} <= chunk_pages, chunk_pages
    assert all(c.retrievers == ["graph"] for c in expanded)
    # bundle: facts land in graph_facts with relation citations
    bundle = await container.services["context_builder"].build(
        U1, "Why did Adjusted EBITDA increase despite lower revenue?"
    )
    assert bundle.graph_facts and bundle.graph_facts[0].citation.startswith("relation_id:rel_")
    assert bundle.graph_facts[0].evidence[0].source_type == "graph_fact"
    assert "## Facts" in bundle.render()
    # queries that do not need the graph skip the stage
    plain = await engine.retrieve(U1, "restructuring savings")
    assert "graph" not in plain.diagnostics


async def test_graph_visibility_and_isolation(container, uow_factory) -> None:
    await _ingest(container, uow_factory, U1)  # USER visibility (default without thread)
    graph = container.services["graph"]
    assert (await graph.query(U1, entities=["Adjusted EBITDA"])).relations
    assert (await graph.query(U2, entities=["Adjusted EBITDA"])).relations == []
    stranger = MemoryExecutionContext(tenant_id="globex", user_id="u1", workspace_id="ws1")
    assert (await graph.query(stranger, entities=["Adjusted EBITDA"])).relations == []
    # the retrieval stage cannot leak either: u2 gets no facts and no expansion chunks
    engine = container.services["retrieval"]
    res = await engine.retrieve(U2, "Why did Adjusted EBITDA increase despite lower revenue?")
    assert not [c for c in res.candidates if c.kind == "fact"]
    assert not [c for c in res.candidates if c.expansion_edge == "GRAPH_EVIDENCE"]
    # a workspace-shared copy widens the audience of the shared entities only
    async with uow_factory() as uow:
        authz = container.services["authz"]
        for user in ("u1", "u2"):
            await authz.grant_membership("acme", user, workspaces=["ws1"], revisions=uow.revisions)
        await uow.commit()
    await _ingest(container, uow_factory, U1, visibility=Visibility.WORKSPACE, salt="\n\nShared.\n")
    shared = await graph.query(U2, entities=["Adjusted EBITDA"])
    assert shared.relations and all("ws:acme/ws1" in r.visibility_keys for r in shared.relations)


async def test_memory_facts_supersession_and_as_of(container, uow_factory) -> None:
    thread = new_id("thread")
    ctx = U1.model_copy(
        update={"thread_id": thread, "session_id": new_id("session"), "turn_id": new_id("turn")}
    )
    async with uow_factory() as uow:
        await container.services["conversation"].append_message(
            uow, ctx, role=MessageRole.USER, content="hello"
        )
        await uow.commit()
    await _observe(container, uow_factory, ctx, "I work at ACME Corp.")
    graph = container.services["graph"]
    first = await graph.query(ctx, entities=["ACME Corp"])
    fact = next(r for r in first.relations if r.predicate == "works_at")
    assert fact.memory_id and fact.status == "CURRENT"
    # the fact reads as its own triple and date, not as the turn it came from: the bundle's
    # memories section already carries that turn verbatim
    day, _, triple = fact.fact_text.partition(" ")
    assert triple == "u1 works at acme corp" and day == fact.observed_at.date().isoformat()
    # supersession: the new employer is current; the old fact is closed, visible as-of the past
    await _observe(container, uow_factory, ctx, "I work at Globex now.")
    now = await graph.query(ctx, entities=["Globex", "ACME Corp"])
    current = [r for r in now.relations if r.predicate == "works_at"]
    assert len(current) == 1 and current[0].status == "CURRENT"
    assert "globex" in {
        e.canonical_name for e in now.entities if e.entity_id == current[0].object_id
    }
    past = await graph.query(
        ctx, entities=["ACME Corp"], as_of=datetime.now(UTC) - timedelta(seconds=30)
    )
    old = [r for r in past.relations if r.predicate == "works_at"]
    assert old and old[0].status == "SUPERSEDED" and old[0].valid_to is not None
    # entity/relation questions reach the graph stage and return the typed fact
    engine = container.services["retrieval"]
    res = await engine.retrieve(ctx, "who works for Globex?")
    assert res.routed.query_type is QueryType.ENTITY_RELATION
    facts = [c for c in res.candidates if c.kind == "fact"]
    assert any(
        c.payload["predicate"] == "works_at" and c.payload["status"] == "CURRENT" for c in facts
    )
    # forgetting the memory retires its facts
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, ctx, current[0].memory_id)
        await uow.commit()
    await container.tasks.drain()
    after = await graph.query(ctx, entities=["Globex"])
    assert not [r for r in after.relations if r.predicate == "works_at"]
