"""Unclassified questions can traverse known entities without English intent keywords."""

# ruff: noqa: RUF001 — literal multilingual fixtures.

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.retrieval.router import QueryRouter
from memory_service.ports.intelligence import Entity, Relation

pytestmark = pytest.mark.integration

QUERIES = (
    "Wer leitet den Lieferanten von Acme?",
    "Qui dirige le fournisseur d’Acme ?",
    "Кто руководит поставщиком Acme?",
    "谁负责Acme的供应商？",
    "Acmeの取引先を率いているのは誰ですか？",
    "Acme के आपूर्तिकर्ता का नेतृत्व कौन करता है?",
    "من يدير المورد الخاص بشركة Acme؟",
)


async def test_semantic_graph_returns_multihop_facts_without_widening_access(container):
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1")
    keys = ["user:acme/u1"]
    names = ("Acme", "Acme Report", "Westfalen", "Bergmann")
    entities = [
        Entity(
            entity_id=f"ent_multilingual_{i}",
            tenant_id="acme",
            name=name,
            canonical_name=name.casefold(),
            scope_key="user:acme/u1",
            visibility_keys=keys,
        )
        for i, name in enumerate(names)
    ]
    relations = [
        Relation(
            relation_id=f"rel_multilingual_{i}",
            tenant_id="acme",
            subject_id=entities[i].entity_id,
            predicate=predicate,
            object_id=entities[i + 1].entity_id,
            visibility_keys=keys,
            observed_at=datetime(2023, 1, 1, tzinfo=UTC),
            evidence=[
                EvidenceRef(
                    source_type="fixture",
                    source_id=f"source_{i}",
                    observed_at=datetime(2023, 1, 1, tzinfo=UTC),
                )
            ],
            fact_text=f"{names[i]} {predicate} {names[i + 1]}",
        )
        for i, predicate in enumerate(("mentioned_in", "names_supplier", "led_by"))
    ]
    await container.graph_store.upsert_entities(entities)
    await container.graph_store.upsert_relations(relations)
    engine = container.services["retrieval"]
    for question in QUERIES:
        result = await engine.retrieve(ctx, question)
        assert result.routed.query_type is QueryType.GENERAL_SEMANTIC
        assert "rel_multilingual_2" in {c.record_id for c in result.candidates if c.kind == "fact"}
        assert result.diagnostics["graph"]["visited"] <= 40
        assert "acme" in result.diagnostics["graph"]["matched"]
    # Paired mechanism control: the old keyword gate cannot reach this fact.
    engine.router = QueryRouter(semantic_graph=False)
    control = await engine.retrieve(ctx, QUERIES[3])
    assert not [c for c in control.candidates if c.kind == "fact"]
    engine.router = QueryRouter()
    for stranger in (
        ctx.model_copy(update={"user_id": "u2"}),
        ctx.model_copy(update={"tenant_id": "other"}),
    ):
        result = await engine.retrieve(stranger, QUERIES[3])
        assert not [c for c in result.candidates if c.kind == "fact"]
        assert result.diagnostics["graph"]["matched"] == []
    unknown = await engine.retrieve(ctx, "完全没有记录的机构叫什么名字？")
    assert unknown.diagnostics["graph"]["matched"] == []
    assert not [c for c in unknown.candidates if c.kind == "fact"]
