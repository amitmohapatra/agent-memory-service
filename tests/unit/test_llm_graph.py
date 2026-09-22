"""Unit tests: LLM-assisted relation extraction and entity resolution in the graph module
against a mocked gateway. Each use: model output applied, gateway failure and flag off fall
back to the native result, and names outside the native lexicon are rejected."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from memory_service.adapters.graph.memory_store import MemoryGraphStore
from memory_service.config.constants import GraphSettings
from memory_service.domain.documents import Chunk, DocumentNode, DocumentVersion
from memory_service.domain.enums import Representation
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.native import (
    LLM_MAX_PAIRS_PER_DOCUMENT,
    NativeGraphEnrichment,
    llm_predicate,
)
from memory_service.modules.graph.service import GraphService
from memory_service.ports.intelligence import Entity, Relation
from tests.support_llm import mocked_gateway
from tests.unit.test_graph_native import CTX, _memory

pytestmark = pytest.mark.unit

DECISION = (
    "I decided to pair Priya Sharma with Globex Corp on the Atlas Rollout for the whole "
    "of the next quarter."
)
KEYS = ["thread:acme/thr_1"]
SCOPE = "t=acme/l=THREAD/thread=thr_1"


def _names(entities: list[Entity]) -> dict[str, str]:
    return {e.entity_id: e.canonical_name for e in entities}


def _typed(
    entities: list[Entity], relations: list[Relation]
) -> dict[tuple[str, str, str], Relation]:
    by_id = _names(entities)
    return {
        (by_id[r.subject_id], r.predicate, by_id[r.object_id]): r
        for r in relations
        if r.predicate not in ("mentions", "co_occurs_with", "mentioned_in", "discusses")
    }


# -- relation_extraction: memories -----------------------------------------------------


async def test_memory_relations_from_model_join_native_mentions() -> None:
    mem = await _memory(DECISION)
    native_entities, native_relations = await NativeGraphEnrichment().enrich_memory(mem, CTX)
    assert {r.predicate for r in native_relations} == {"mentions"} and len(native_entities) >= 2
    reply = {
        "relations": [
            {
                "subject": "Priya Sharma",
                "predicate": "works at",
                "object": "Globex Corp",
                "confidence": 0.97,
            },
            {
                "subject": "Priya Sharma",
                "predicate": "Leads",
                "object": "Atlas Rollout",
                "confidence": 0.5,
            },
            {
                "subject": "Priya Sharma",
                "predicate": "works at",
                "object": "Globex Corp",
                "confidence": 0.9,
            },
        ]
    }
    with mocked_gateway([reply]) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        entities, relations = await provider.enrich_memory(mem, CTX)
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
    assert "Priya Sharma" in prompt and "Globex Corp" in prompt and DECISION in prompt
    assert {e.entity_id for e in entities} == {e.entity_id for e in native_entities}
    assert {r.relation_id for r in native_relations} < {r.relation_id for r in relations}
    typed = _typed(entities, relations)
    assert set(typed) == {
        ("priya sharma", "works_at", "globex corp"),
        ("priya sharma", "leads", "atlas rollout"),
    }
    works = typed[("priya sharma", "works_at", "globex corp")]
    assert works.confidence == 0.8 and works.attributes["extraction"] == "llm"
    assert works.memory_id == mem.memory_id and works.evidence == mem.evidence
    assert works.visibility_keys == mem.system_metadata["visibility_keys"]
    assert typed[("priya sharma", "leads", "atlas rollout")].confidence == 0.5
    # deterministic ids: re-enrichment upserts
    with mocked_gateway([reply]) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        _, again = await provider.enrich_memory(mem, CTX)
    assert {r.relation_id for r in again} == {r.relation_id for r in relations}


async def test_memory_relations_fall_back_when_gateway_fails() -> None:
    mem = await _memory(DECISION)
    _, native_relations = await NativeGraphEnrichment().enrich_memory(mem, CTX)
    with mocked_gateway(failing=True) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        _, relations = await provider.enrich_memory(mem, CTX)
        assert gw.route.called
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}


async def test_memory_relations_flag_off_never_calls_the_gateway() -> None:
    mem = await _memory(DECISION)
    _, native_relations = await NativeGraphEnrichment().enrich_memory(mem, CTX)
    with mocked_gateway(['{"relations": []}']) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["ambiguous_worthiness"]))
        _, relations = await provider.enrich_memory(mem, CTX)
        assert gw.route.call_count == 0
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}
    # a memory whose native triple is already typed is not sent to the model either
    typed_mem = await _memory("I work at ACME Corp.")
    with mocked_gateway(['{"relations": []}']) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        _, relations = await provider.enrich_memory(typed_mem, CTX)
        assert gw.route.call_count == 0
    assert "works_at" in {r.predicate for r in relations}


async def test_memory_relations_naming_unknown_entities_are_rejected() -> None:
    mem = await _memory(DECISION)
    native_entities, native_relations = await NativeGraphEnrichment().enrich_memory(mem, CTX)
    reply = {
        "relations": [
            {
                "subject": "Priya Sharma",
                "predicate": "works_at",
                "object": "Initech",
                "confidence": 0.9,
            },
            {
                "subject": "Hank Scorpio",
                "predicate": "leads",
                "object": "Globex Corp",
                "confidence": 0.9,
            },
            {
                "subject": "Priya Sharma",
                "predicate": "mentions",
                "object": "Globex Corp",
                "confidence": 0.9,
            },
            {
                "subject": "Priya Sharma",
                "predicate": "???",
                "object": "Globex Corp",
                "confidence": 0.9,
            },
            {
                "subject": "Globex Corp",
                "predicate": "is",
                "object": "Globex Corp",
                "confidence": 0.9,
            },
        ]
    }
    with mocked_gateway([reply]) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        entities, relations = await provider.enrich_memory(mem, CTX)
        assert gw.route.call_count == 1
    assert {e.entity_id for e in entities} == {e.entity_id for e in native_entities}
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}


def test_llm_predicate_normalisation() -> None:
    assert llm_predicate("Works At") == "works_at"
    assert llm_predicate(" headquartered-in ") == "headquartered_in"
    assert llm_predicate("co_occurs_with") is None
    assert llm_predicate("") is None and llm_predicate("1st") is None
    assert llm_predicate("x" * 60) is None


# -- relation_extraction: documents ----------------------------------------------------


def _doc(chunk_texts: list[tuple[str, list[str]]]):
    version = DocumentVersion(document_id="doc_1", tenant_id="acme", parser="builtin")
    node = DocumentNode(
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
    chunks = [
        Chunk(
            node_id=f"nod_{i}",
            document_id="doc_1",
            document_version_id=version.document_version_id,
            tenant_id="acme",
            ordinal=i,
            text=text,
            text_hash=f"h{i}",
            contextual_text="x",
            page=10 + i,
            entities=entities,
        )
        for i, (text, entities) in enumerate(chunk_texts)
    ]
    return version, [node], chunks


EBITDA_CHUNK = (
    "Adjusted EBITDA increased to EUR 98 million. The Restructuring Programme lifted "
    "Adjusted EBITDA in FY26.",
    ["Adjusted EBITDA", "Restructuring Programme"],
)


async def _enrich(provider: NativeGraphEnrichment, doc):
    version, nodes, chunks = doc
    return await provider.enrich_document(
        version,
        nodes,
        chunks,
        CTX,
        visibility_keys=KEYS,
        document_title="ACME FY26",
        scope_key=SCOPE,
    )


async def test_document_pair_relations_from_model_keep_co_occurrence() -> None:
    doc = _doc([EBITDA_CHUNK])
    native_entities, native_relations = await _enrich(NativeGraphEnrichment(), doc)
    reply = {
        "relations": [
            {
                "subject": "Restructuring Programme",
                "predicate": "Improved",
                "object": "Adjusted EBITDA",
                "confidence": 0.95,
            }
        ]
    }
    with mocked_gateway([reply]) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        entities, relations = await _enrich(provider, doc)
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
    assert "1. Adjusted EBITDA | Restructuring Programme" in prompt
    assert "The Restructuring Programme lifted Adjusted EBITDA in FY26." in prompt
    assert "EUR 98 million" not in prompt.split("context:")[1]
    assert {e.entity_id for e in entities} == {e.entity_id for e in native_entities}
    assert {r.relation_id for r in native_relations} < {r.relation_id for r in relations}
    by_id = _names(entities)
    co = [r for r in relations if r.predicate == "co_occurs_with"]
    assert len(co) == 1 and co[0].confidence == 0.4
    improved = next(r for r in relations if r.predicate == "improved")
    assert by_id[improved.subject_id] == "restructuring programme"
    assert by_id[improved.object_id] == "adjusted ebitda"
    assert improved.confidence == 0.8 and improved.attributes == {"page": 10, "extraction": "llm"}
    assert improved.evidence[0].chunk_id == doc[2][0].chunk_id and improved.evidence[0].page == 10
    assert improved.document_id == "doc_1"
    assert improved.fact_text.startswith("Restructuring Programme improved Adjusted EBITDA — The")


async def test_document_pair_relations_fall_back_and_respect_flag() -> None:
    doc = _doc([EBITDA_CHUNK])
    _, native_relations = await _enrich(NativeGraphEnrichment(), doc)
    with mocked_gateway(failing=True) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        _, relations = await _enrich(provider, doc)
        assert gw.route.called
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}
    with mocked_gateway(['{"relations": []}']) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["query_expansion"]))
        _, relations = await _enrich(provider, doc)
        assert gw.route.call_count == 0
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}


async def test_document_pair_relations_reject_unknown_entities_and_unasked_pairs() -> None:
    doc = _doc([EBITDA_CHUNK])
    native_entities, native_relations = await _enrich(NativeGraphEnrichment(), doc)
    reply = {
        "relations": [
            {
                "subject": "Globex",
                "predicate": "owns",
                "object": "Adjusted EBITDA",
                "confidence": 0.9,
            },
            {
                "subject": "Adjusted EBITDA",
                "predicate": "equals",
                "object": "EUR 98 million",
                "confidence": 0.9,
            },
            {
                "subject": "Adjusted EBITDA",
                "predicate": "co_occurs_with",
                "object": "Restructuring Programme",
                "confidence": 0.9,
            },
            {
                "subject": "Adjusted EBITDA",
                "predicate": "defined_in",
                "object": "ACME FY26",
                "confidence": 0.9,
            },
        ]
    }
    with mocked_gateway([reply]) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        entities, relations = await _enrich(provider, doc)
        assert gw.route.call_count == 1
    assert {e.entity_id for e in entities} == {e.entity_id for e in native_entities}
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}


async def test_document_pairs_sent_to_model_are_bounded() -> None:
    groups = [
        ["Alpha Corp", "Beta Corp", "Gamma Corp", "Delta Corp"],
        ["Epsilon Corp", "Zeta Corp", "Eta Corp", "Theta Corp"],
        ["Iota Corp", "Kappa Corp", "Lambda Corp", "Mu Corp"],
    ]
    doc = _doc(
        [(f"{', '.join(g[:-1])} and {g[-1]} signed the framework agreement.", g) for g in groups]
    )
    _, native_relations = await _enrich(NativeGraphEnrichment(), doc)
    assert (
        sum(1 for r in native_relations if r.predicate == "co_occurs_with")
        > LLM_MAX_PAIRS_PER_DOCUMENT
    )
    with mocked_gateway(['{"relations": []}']) as gw:
        provider = NativeGraphEnrichment(assist=gw.assist(uses=["relation_extraction"]))
        _, relations = await _enrich(provider, doc)
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
    numbered = [line for line in prompt.splitlines() if line[:1].isdigit()]
    assert len(numbered) == LLM_MAX_PAIRS_PER_DOCUMENT
    assert {r.relation_id for r in relations} == {r.relation_id for r in native_relations}


# -- entity_resolution -------------------------------------------------------------------

NOW = datetime(2026, 9, 15, tzinfo=UTC)
VIS = VisibilitySpecification(tenant_id="acme", keys=frozenset({"user:acme/u1"}))


def _entity(
    name: str,
    *,
    keys: list[str] | None = None,
    aliases: tuple[str, ...] = (),
    entity_type: str = "THING",
    mention_count: int = 1,
) -> Entity:
    canonical = name.casefold()
    return Entity(
        entity_id=f"ent_{canonical.replace(' ', '_')}",
        tenant_id="acme",
        name=name,
        canonical_name=canonical,
        entity_type=entity_type,
        aliases=list(aliases),
        visibility_keys=keys or ["user:acme/u1"],
        mention_count=mention_count,
    )


async def _service(assist=None) -> tuple[GraphService, MemoryGraphStore]:
    store = MemoryGraphStore()
    await store.upsert_entities(
        [
            _entity(
                "Adjusted EBITDA",
                aliases=("adjusted earnings",),
                entity_type="METRIC",
                mention_count=5,
            ),
            _entity("Restructuring Programme", entity_type="EVENT", mention_count=3),
            _entity("EUR 98 million", entity_type="MONEY", mention_count=9),
            _entity("Secret Plan", keys=["user:acme/u2"], mention_count=7),
        ]
    )
    await store.upsert_relations(
        [
            Relation(
                relation_id="rel_1",
                tenant_id="acme",
                subject_id="ent_adjusted_ebitda",
                predicate="driven_by",
                object_id="ent_restructuring_programme",
                visibility_keys=["user:acme/u1"],
                observed_at=NOW,
                evidence=[
                    EvidenceRef(source_type="document_chunk", source_id="c1", observed_at=NOW)
                ],
                fact_text="Adjusted EBITDA driven by Restructuring Programme",
            )
        ]
    )
    service = GraphService(
        cast(Any, None), store, None, cast(Any, None), settings=GraphSettings(), assist=assist
    )
    return service, store


async def test_memory_store_lists_visible_entities_most_mentioned_first() -> None:
    _, store = await _service()
    listed = await store.list_entities("acme", scope_keys=["user:acme/u1"], limit=2)
    assert [e.canonical_name for e in listed] == ["eur 98 million", "adjusted ebitda"]
    assert await store.list_entities("acme", scope_keys=[], limit=10) == []
    assert await store.list_entities("globex", scope_keys=["user:acme/u1"], limit=10) == []
    everything = await store.list_entities("acme", scope_keys=["user:acme/u1"], limit=10)
    assert "secret plan" not in {e.canonical_name for e in everything}


async def test_entity_resolution_maps_unmatched_names_to_candidates() -> None:
    reply = {
        "matches": [
            {"query_name": "adj. ebitda", "entity_name": "Adjusted EBITDA"},
            {"query_name": "the restructuring", "entity_name": "restructuring programme"},
        ]
    }
    with mocked_gateway([reply]) as gw:
        service, _ = await _service(gw.assist(uses=["entity_resolution"]))
        resolved = await service.resolve(CTX, ["adj. EBITDA", "the restructuring"], VIS)
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
    assert prompt.startswith("Query names: adj. ebitda; the restructuring\nCandidates:\n")
    assert (
        "- Adjusted EBITDA (aka adjusted earnings)" in prompt
        and "- Restructuring Programme" in prompt
    )
    assert "Secret Plan" not in prompt and "EUR 98 million" not in prompt
    assert {e.canonical_name for e in resolved} == {"adjusted ebitda", "restructuring programme"}
    # through query(): the question's terms are the query names, the seeds come from the
    # model and the neighbourhood from the store
    query_reply = {"matches": [{"query_name": "adj ebitda", "entity_name": "Adjusted EBITDA"}]}
    with mocked_gateway([query_reply]) as gw:
        service, _ = await _service(gw.assist(uses=["entity_resolution"]))
        answer = await service.query(CTX, query="adj. EBITDA?", visibility=VIS)
        assert gw.route.call_count == 1
        assert gw.prompts()[0]["messages"][1]["content"].startswith(
            "Query names: ebitda; adj ebitda; adj\n"
        )
    assert {e.canonical_name for e in answer.matched} == {"adjusted ebitda"}
    assert [r.predicate for r in answer.relations] == ["driven_by"]


async def test_entity_resolution_only_asks_for_the_unmatched_names() -> None:
    reply = {
        "matches": [{"query_name": "the restructuring", "entity_name": "Restructuring Programme"}]
    }
    with mocked_gateway([reply]) as gw:
        service, _ = await _service(gw.assist(uses=["entity_resolution"]))
        resolved = await service.resolve(CTX, ["Adjusted EBITDA", "the restructuring"], VIS)
        assert gw.route.call_count == 1
        assert gw.prompts()[0]["messages"][1]["content"].startswith(
            "Query names: the restructuring\n"
        )
        assert {e.canonical_name for e in resolved} == {
            "adjusted ebitda",
            "restructuring programme",
        }
        # every name matched natively: no call at all
        resolved = await service.resolve(CTX, ["Adjusted EBITDA", "adjusted earnings"], VIS)
        assert gw.route.call_count == 1
        assert [e.canonical_name for e in resolved] == ["adjusted ebitda"]


async def test_entity_resolution_falls_back_to_empty_when_gateway_fails() -> None:
    with mocked_gateway(failing=True) as gw:
        service, _ = await _service(gw.assist(uses=["entity_resolution"]))
        assert await service.resolve(CTX, ["adj. EBITDA"], VIS) == []
        assert gw.route.called
        answer = await service.query(CTX, query="adj. EBITDA?", visibility=VIS)
    assert answer.matched == [] and answer.relations == [] and answer.visited == 0


async def test_entity_resolution_flag_off_never_calls_the_gateway() -> None:
    reply = {"matches": [{"query_name": "adj. ebitda", "entity_name": "Adjusted EBITDA"}]}
    with mocked_gateway([reply]) as gw:
        service, _ = await _service(gw.assist(uses=["relation_extraction"]))
        assert await service.resolve(CTX, ["adj. EBITDA"], VIS) == []
        assert gw.route.call_count == 0
    service, _ = await _service()
    assert await service.resolve(CTX, ["adj. EBITDA"], VIS) == []


async def test_entity_resolution_rejects_names_outside_the_candidates() -> None:
    reply = {
        "matches": [
            {"query_name": "adj. ebitda", "entity_name": "Initech"},
            {"query_name": "adj. ebitda", "entity_name": "Secret Plan"},
            {"query_name": "adj. ebitda", "entity_name": "EUR 98 million"},
            {"query_name": "something else", "entity_name": "Adjusted EBITDA"},
        ]
    }
    with mocked_gateway([reply]) as gw:
        service, _ = await _service(gw.assist(uses=["entity_resolution"]))
        assert await service.resolve(CTX, ["adj. EBITDA"], VIS) == []
        assert gw.route.call_count == 1
