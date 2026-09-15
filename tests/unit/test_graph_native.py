"""Unit tests: native graph enrichment, in-memory store traversal (bounded, temporal,
visibility-filtered) and the retrieval stage's candidate mapping."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.adapters.graph.memory_store import MemoryGraphStore
from memory_service.config.settings import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, DocumentNode, DocumentVersion
from memory_service.domain.enums import ObservationKind, Representation
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.modules.graph.native import NativeGraphEnrichment, entity_id_for
from memory_service.modules.graph.retrieval import fact_candidate
from memory_service.modules.graph.service import query_terms
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory, keys_for, scope_for
from memory_service.ports.intelligence import Entity, Relation

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1")
NOW = datetime(2026, 9, 15, tzinfo=UTC)


async def _memory(text: str, ctx=CTX):
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
    obs = Observation(
        tenant_id=ctx.tenant_id,
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        thread_id=ctx.thread_id,
        workspace_id=ctx.workspace_id,
        principal_id=ctx.principal_id,
        message_id="msg_1",
    )
    cand = await native.classify((await native.extract(obs, ctx))[0], ctx)
    mem = build_memory(cand, ctx, now=NOW)
    mem.system_metadata["visibility_keys"] = keys_for(scope_for(cand, ctx), mem.visibility, ctx)
    return mem


async def test_memory_enrichment_yields_typed_fact_and_mentions() -> None:
    mem = await _memory("I work at ACME Corp.")
    entities, relations = await NativeGraphEnrichment().enrich_memory(mem, CTX)
    names = {e.canonical_name: e for e in entities}
    assert "user:u1" in names and "acme corp" in names
    assert names["user:u1"].entity_type == "USER" and names["acme corp"].entity_type == "ORG"
    fact = next(r for r in relations if r.predicate == "works_at")
    assert fact.subject_id == names["user:u1"].entity_id
    assert fact.object_id == names["acme corp"].entity_id
    assert fact.memory_id == mem.memory_id and fact.fact_text == mem.content
    assert fact.visibility_keys == mem.system_metadata["visibility_keys"]
    assert fact.evidence[0].message_id == "msg_1"
    # deterministic ids: re-enrichment produces the same ids (upsert, not duplicate)
    again_entities, again_relations = await NativeGraphEnrichment().enrich_memory(mem, CTX)
    assert {e.entity_id for e in again_entities} == {e.entity_id for e in entities}
    assert {r.relation_id for r in again_relations} == {r.relation_id for r in relations}
    assert names["acme corp"].entity_id == entity_id_for("acme", mem.scope.key(), "acme corp")


def _doc_fixture():
    version = DocumentVersion(document_id="doc_1", tenant_id="acme", parser="builtin")
    definition = DocumentNode(
        document_id="doc_1",
        document_version_id=version.document_version_id,
        tenant_id="acme",
        representation=Representation.PARAGRAPH,
        ordinal=0,
        depth=1,
        text="**Adjusted EBITDA** means earnings before interest, taxes, depreciation and amortisation.",
        page_start=1,
        page_end=1,
    )
    chunk_a = Chunk(
        node_id="nod_a",
        document_id="doc_1",
        document_version_id=version.document_version_id,
        tenant_id="acme",
        ordinal=0,
        text="Adjusted EBITDA increased to EUR 98 million. The increase reflects the Restructuring Programme savings.",
        text_hash="h1",
        contextual_text="x",
        page=11,
        entities=["Adjusted EBITDA", "Restructuring Programme"],
    )
    chunk_b = Chunk(
        node_id="nod_b",
        document_id="doc_1",
        document_version_id=version.document_version_id,
        tenant_id="acme",
        ordinal=1,
        text="The Restructuring Programme reduced headcount by 12%. Annualised savings are EUR 19 million.",
        text_hash="h2",
        contextual_text="y",
        page=14,
        entities=["Restructuring Programme"],
    )
    return version, [definition], [chunk_a, chunk_b]


async def test_document_enrichment_definitions_mentions_cooccurrence() -> None:
    version, nodes, chunks = _doc_fixture()
    entities, relations = await NativeGraphEnrichment().enrich_document(
        version,
        nodes,
        chunks,
        CTX,
        visibility_keys=["thread:acme/thr_1"],
        document_title="ACME FY26",
        scope_key="t=acme/l=THREAD/thread=thr_1",
    )
    names = {e.canonical_name: e for e in entities}
    assert {"adjusted ebitda", "restructuring programme", "doc:doc_1"} <= set(names)
    assert names["doc:doc_1"].entity_type == "DOCUMENT" and names["doc:doc_1"].name == "ACME FY26"
    preds = {
        (names_by_id := {e.entity_id: e.canonical_name for e in entities})[r.subject_id]: r
        for r in relations
        if r.predicate == "defined_in"
    }
    assert "adjusted ebitda" in preds and preds["adjusted ebitda"].evidence[0].page == 1
    co = next(r for r in relations if r.predicate == "co_occurs_with")
    assert {names_by_id[co.subject_id], names_by_id[co.object_id]} == {
        "adjusted ebitda",
        "restructuring programme",
    }
    assert co.evidence[0].chunk_id == chunks[0].chunk_id and co.evidence[0].page == 11
    mentioned = [r for r in relations if r.predicate == "mentioned_in"]
    assert {r.evidence[0].page for r in mentioned} == {11, 14}
    assert all(r.visibility_keys == ["thread:acme/thr_1"] for r in relations)


async def test_memory_store_traversal_bounds_visibility_and_time() -> None:
    store = MemoryGraphStore()
    keys = ["user:acme/u1"]

    def ent(name: str) -> Entity:
        return Entity(
            entity_id=f"ent_{name}",
            tenant_id="acme",
            name=name,
            canonical_name=name,
            visibility_keys=keys,
        )

    def rel(
        s: str, p: str, o: str, *, keys_=None, valid_from=None, valid_to=None, status="CURRENT"
    ) -> Relation:
        return Relation(
            relation_id=f"rel_{s}_{p}_{o}",
            tenant_id="acme",
            subject_id=f"ent_{s}",
            predicate=p,
            object_id=f"ent_{o}",
            visibility_keys=keys_ or keys,
            valid_from=valid_from,
            valid_to=valid_to,
            observed_at=NOW,
            status=status,
            evidence=[EvidenceRef(source_type="memory", source_id="mem_1", observed_at=NOW)],
            fact_text=f"{s} {p} {o}",
        )

    await store.upsert_entities([ent(n) for n in "abcd"] + [ent("secret")])
    await store.upsert_relations(
        [
            rel("a", "knows", "b"),
            rel("b", "knows", "c"),
            rel("c", "knows", "d"),
            rel("a", "hides", "secret", keys_=["user:acme/u2"]),
            rel(
                "a",
                "lived_in",
                "b",
                valid_from=NOW - timedelta(days=400),
                valid_to=NOW - timedelta(days=100),
                status="SUPERSEDED",
            ),
        ]
    )
    one = await store.neighborhood("acme", ["ent_a"], scope_keys=keys, hops=1)
    assert {r.relation_id for r in one.relations} == {"rel_a_knows_b"}
    two = await store.neighborhood("acme", ["ent_a"], scope_keys=keys, hops=2)
    assert {r.relation_id for r in two.relations} == {"rel_a_knows_b", "rel_b_knows_c"}
    assert "ent_secret" not in {e.entity_id for e in two.entities}
    # bounded: max_visited caps the frontier
    capped = await store.neighborhood("acme", ["ent_a"], scope_keys=keys, hops=3, max_visited=2)
    assert capped.visited <= 2
    # temporal view: as_of inside the superseded interval sees the old fact, not the current one
    past = await store.neighborhood(
        "acme", ["ent_a"], scope_keys=keys, hops=1, as_of=NOW - timedelta(days=200)
    )
    assert {r.relation_id for r in past.relations} == {"rel_a_knows_b", "rel_a_lived_in_b"}
    # a reader without the keys sees nothing at all
    assert (
        await store.neighborhood("acme", ["ent_a"], scope_keys=["user:acme/u9"], hops=2)
    ).relations == []
    assert (await store.neighborhood("globex", ["ent_a"], scope_keys=keys, hops=2)).relations == []
    # upsert is idempotent and merges audiences
    await store.upsert_entities([ent("a").model_copy(update={"visibility_keys": ["ws:acme/ws1"]})])
    assert set(store.entities["ent_a"].visibility_keys) == {"user:acme/u1", "ws:acme/ws1"}
    assert store.entities["ent_a"].mention_count == 2
    assert await store.supersede_for_memory("acme", "nope", at=NOW) == 0


def test_query_terms_and_fact_candidate() -> None:
    terms = query_terms("Why did Adjusted EBITDA increase despite lower revenue?")
    assert "adjusted ebitda" in terms and "ebitda" in terms and "why" not in terms
    r = Relation(
        relation_id="rel_x",
        tenant_id="acme",
        subject_id="e1",
        predicate="works_at",
        object_id="e2",
        observed_at=NOW,
        evidence=[
            EvidenceRef(
                source_type="document_chunk",
                source_id="chk_1",
                chunk_id="chk_1",
                page=3,
                document_id="doc_1",
                observed_at=NOW,
            )
        ],
        fact_text="",
    )
    c = fact_candidate(r, {"e1": "Amit", "e2": "ACME"})
    assert c.kind == "fact" and c.text == "Amit works at ACME" and c.retrievers == ["graph"]
    assert c.payload["chunk_id"] == "chk_1" and c.payload["page"] == 3
    assert c.representation is Representation.RELATION
