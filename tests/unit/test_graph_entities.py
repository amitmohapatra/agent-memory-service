"""Entity search, entity relations and entity summaries on the in-memory graph store (the
PostgreSQL store is held to the same behaviour in tests/integration/test_graph_entities.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.adapters.graph.memory_store import MemoryGraphStore
from memory_service.config.constants import GraphSettings
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.graph.summaries import EntitySummaries, deterministic_summary
from memory_service.modules.llm.assist import LLMAssist
from memory_service.ports.intelligence import Entity, Relation
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 15, tzinfo=UTC)
MINE = ["user:acme/u1"]
BOTH = ["user:acme/u1", "user:acme/u2"]


def entity(name: str, *, keys=MINE, mentions: int = 1, kind: str = "ORG") -> Entity:
    canonical = name.casefold()
    return Entity(
        entity_id=f"ent_{canonical.replace(' ', '_')}",
        tenant_id="acme",
        name=name,
        canonical_name=canonical,
        entity_type=kind,
        visibility_keys=keys,
        mention_count=mentions,
    )


def relation(
    rid: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    keys=MINE,
    status: str = "CURRENT",
    at: datetime = NOW,
    confidence: float = 0.8,
) -> Relation:
    return Relation(
        relation_id=rid,
        tenant_id="acme",
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        visibility_keys=keys,
        status=status,
        observed_at=at,
        confidence=confidence,
        evidence=[EvidenceRef(source_type="memory", source_id="mem_1", observed_at=at)],
    )


async def seeded() -> MemoryGraphStore:
    store = MemoryGraphStore()
    await store.upsert_entities(
        [
            entity("Acme Corp", keys=BOTH, mentions=9),
            entity("Acme Labs", mentions=2),
            entity("Globex", mentions=5),
            entity("Germany", mentions=1, kind="PLACE"),
            entity("Secret Project", keys=["user:acme/u2"], mentions=7, kind="EVENT"),
        ]
    )
    await store.upsert_relations(
        [
            relation("rel_1", "ent_acme_corp", "operates_in", "ent_germany", keys=BOTH),
            relation("rel_2", "ent_acme_corp", "acquired", "ent_globex", keys=MINE),
            relation(
                "rel_3",
                "ent_acme_corp",
                "headquartered_in",
                "ent_globex",
                keys=BOTH,
                status="SUPERSEDED",
                at=NOW - timedelta(days=1),
            ),
            relation("rel_4", "ent_acme_corp", "mentions", "ent_globex", keys=BOTH),
            relation("rel_5", "ent_secret_project", "run_by", "ent_acme_corp", keys=BOTH),
        ]
    )
    return store


async def test_search_matches_a_name_prefix_and_type_most_mentioned_first() -> None:
    store = await seeded()
    found = await store.search_entities("acme", scope_keys=MINE, prefix="acme")
    assert [e.name for e in found] == ["Acme Corp", "Acme Labs"]
    assert [
        e.name for e in await store.search_entities("acme", scope_keys=MINE, entity_type="PLACE")
    ] == ["Germany"]
    # invisible entities are never found, whatever the prefix
    assert await store.search_entities("acme", scope_keys=MINE, prefix="secret") == []
    assert await store.search_entities("acme", scope_keys=[], prefix="acme") == []


async def test_entity_relations_split_current_from_history_and_respect_visibility() -> None:
    store = await seeded()
    current = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=MINE, current=True, limit=10
    )
    assert {r.relation_id for r in current} == {"rel_1", "rel_2", "rel_4", "rel_5"}
    history = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=MINE, current=False, limit=10
    )
    assert [r.relation_id for r in history] == ["rel_3"]
    theirs = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=["user:acme/u2"], current=True, limit=10
    )
    assert "rel_2" not in {r.relation_id for r in theirs}, "u2 cannot read rel_2"


async def test_a_summary_uses_only_facts_every_reader_of_the_entity_can_read() -> None:
    """Acme Corp is visible to u1 and u2; rel_2 only to u1, so it must stay out of the
    summary u2 can read. Structural edges (mentions) and history are not summary facts."""
    store = await seeded()
    [facts] = await store.summary_sources("acme", ["ent_acme_corp"], limit=10)
    assert facts.facts == [("operates_in", "Germany")]
    assert deterministic_summary(facts) == "Acme Corp (ORG): operates in Germany"


async def test_summaries_are_written_once_and_rewritten_only_when_the_facts_change() -> None:
    store = await seeded()
    summaries = EntitySummaries(store, LLMAssist.disabled(), GraphSettings())
    assert await summaries.refresh("acme", ["ent_acme_corp", "ent_globex"]) == 2
    assert store.entities["ent_acme_corp"].summary == "Acme Corp (ORG): operates in Germany"
    assert store.entities["ent_globex"].summary == "Globex (ORG)"
    assert await summaries.refresh("acme", ["ent_acme_corp"]) == 0, "nothing changed"
    await store.upsert_relations(
        [relation("rel_6", "ent_acme_corp", "led_by", "ent_acme_labs", keys=BOTH)]
    )
    assert await summaries.refresh("acme", ["ent_acme_corp"]) == 1


async def test_the_model_rewrites_a_bounded_number_of_summaries_per_job() -> None:
    store = await seeded()
    await store.upsert_relations(
        [relation("rel_7", "ent_globex", "operates_in", "ent_germany", keys=MINE)]
    )
    settings = GraphSettings(entity_summary_model_calls_per_job=1)
    with mocked_gateway(["Acme Corp is an organisation operating in Germany."]) as gw:
        summaries = EntitySummaries(store, gw.assist(uses=["summaries"]), settings)
        assert await summaries.refresh("acme", ["ent_acme_corp", "ent_globex"]) == 2
        assert gw.route.call_count == 1, "the second entity keeps its deterministic summary"
        assert "Germany" in gw.prompts()[0]["messages"][1]["content"]
    assert store.entities["ent_acme_corp"].summary.startswith("Acme Corp is an organisation")
    assert store.entities["ent_globex"].summary == "Globex (ORG): operates in Germany"


async def test_a_failing_model_keeps_the_deterministic_summary() -> None:
    store = await seeded()
    with mocked_gateway(failing=True) as gw:
        summaries = EntitySummaries(store, gw.assist(uses=["summaries"]), GraphSettings())
        await summaries.refresh("acme", ["ent_acme_corp"])
    assert store.entities["ent_acme_corp"].summary == "Acme Corp (ORG): operates in Germany"
