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


THREAD = ["thread:acme/t1"]


def _cited(document_id: str) -> EvidenceRef:
    return EvidenceRef(
        source_type="document_chunk", source_id="chk_1", document_id=document_id, observed_at=NOW
    )


def _told_by(document_id: str, r: Relation) -> Relation:
    return r.model_copy(update={"document_id": document_id, "evidence": [_cited(document_id)]})


async def check_a_gone_documents_graph_is_forgotten(store) -> None:
    """``forget_document`` on any store (the PostgreSQL test calls this too): doc_a's facts
    and the ``invalidated_by`` edge quoting one go; Westfalen (only doc_a named it) and
    doc_a's own document entity (no relation) go; Acme and Germany, which a memory's fact
    still holds, stay without doc_a's evidence or summary; doc_b is untouched."""
    both = [*MINE, *THREAD]
    await store.upsert_entities(
        [
            entity("Acme Corp", keys=both).model_copy(
                update={"evidence": [_cited("doc_a"), _cited("doc_b")]}
            ),
            entity("Westfalen", keys=THREAD).model_copy(update={"evidence": [_cited("doc_a")]}),
            entity("Germany", keys=both, kind="PLACE"),
            entity("Globex"),
            entity("Report", keys=THREAD).model_copy(
                update={"entity_id": "ent_doc_a", "canonical_name": "doc:doc_a"}
            ),
        ]
    )
    await store.upsert_relations(
        [
            _told_by("doc_a", relation("rel_a1", "ent_acme_corp", "acquired", "ent_westfalen")),
            _told_by("doc_a", relation("rel_a2", "ent_acme_corp", "operates_in", "ent_germany")),
            relation("rel_m1", "ent_acme_corp", "operates_in", "ent_germany"),
            _told_by("doc_b", relation("rel_b1", "ent_acme_corp", "acquired", "ent_globex")),
        ]
    )
    await store.invalidate("rel_a2", reason="contradicted", at=NOW, by="rel_m1")
    await store.set_summary("acme", "ent_acme_corp", summary="Acme acquired Westfalen", source="x")

    forgotten = await store.forget_document("acme", "doc_a")

    assert forgotten.kept == ["ent_acme_corp", "ent_germany"]
    assert set(forgotten.audience) == set(both)
    left = await store.get_entities(
        "acme", ["ent_acme_corp", "ent_westfalen", "ent_germany", "ent_doc_a"], scope_keys=both
    )
    by_id = {e.entity_id: e for e in left}
    assert set(by_id) == {"ent_acme_corp", "ent_germany"}
    assert [e.document_id for e in by_id["ent_acme_corp"].evidence] == ["doc_b"]
    assert by_id["ent_acme_corp"].summary == ""
    [facts] = await store.summary_sources("acme", ["ent_acme_corp"], limit=10)
    assert facts.summary_source == ""
    held = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=both, current=True, limit=10
    )
    history = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=both, current=False, limit=10
    )
    assert {r.relation_id for r in held + history} == {"rel_m1", "rel_b1"}
    assert (
        await store.relations_for_document(
            "acme", "doc_a", scope_keys=both, include_invalidated=True
        )
        == []
    )
    [kept] = await store.relations_for_document("acme", "doc_b", scope_keys=both)
    assert kept.relation_id == "rel_b1"
    # forgetting again finds nothing: idempotent, as a retried job is
    again = await store.forget_document("acme", "doc_a")
    assert again.kept == [] and again.audience == []


async def test_a_gone_documents_graph_is_forgotten() -> None:
    """MEM-05: a deleted thread's document leaves nothing in the graph."""
    await check_a_gone_documents_graph_is_forgotten(MemoryGraphStore())


async def check_a_documents_audience_moves(store) -> None:
    """``move_document`` on any store (the PostgreSQL test calls this too): doc_a's facts
    carry exactly the new audience (the live thread that holds it), the deleted thread's key
    gone; every entity it named, and its own document entity, gains the new audience
    besides what it had; a memory's fact and doc_b's facts keep theirs."""
    held = ["thread:acme/t2"]
    await store.upsert_entities(
        [
            entity("Acme Corp", keys=MINE),
            entity("Westfalen", keys=THREAD),
            entity("Globex", keys=THREAD),
            entity("Report", keys=THREAD).model_copy(
                update={"entity_id": "ent_doc_a", "canonical_name": "doc:doc_a"}
            ),
        ]
    )
    await store.upsert_relations(
        [
            _told_by(
                "doc_a",
                relation("rel_a1", "ent_acme_corp", "acquired", "ent_westfalen", keys=THREAD),
            ),
            relation("rel_m1", "ent_acme_corp", "operates_in", "ent_westfalen"),
            _told_by(
                "doc_b", relation("rel_b1", "ent_westfalen", "acquired", "ent_globex", keys=THREAD)
            ),
        ]
    )

    assert await store.move_document("acme", "doc_a", held) == [
        "ent_acme_corp",
        "ent_doc_a",
        "ent_westfalen",
    ]

    [moved] = await store.relations_for_document("acme", "doc_a", scope_keys=held)
    assert moved.relation_id == "rel_a1" and moved.visibility_keys == held
    assert await store.relations_for_document("acme", "doc_a", scope_keys=THREAD) == []
    [other] = await store.relations_for_document("acme", "doc_b", scope_keys=THREAD)
    assert other.visibility_keys == THREAD
    found = await store.get_entities(
        "acme",
        ["ent_acme_corp", "ent_westfalen", "ent_globex", "ent_doc_a"],
        scope_keys=[*MINE, *THREAD, *held],
    )
    keys = {e.entity_id: sorted(e.visibility_keys) for e in found}
    assert keys == {
        "ent_acme_corp": sorted([*MINE, *held]),
        "ent_westfalen": sorted([*THREAD, *held]),
        "ent_globex": THREAD,
        "ent_doc_a": sorted([*THREAD, *held]),
    }
    mine = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=MINE, current=True, limit=10
    )
    assert [r.relation_id for r in mine] == ["rel_m1"]
    # moving again to the same audience changes nothing: idempotent, as a retried job is
    await store.move_document("acme", "doc_a", held)
    [again] = await store.relations_for_document("acme", "doc_a", scope_keys=held)
    assert again.visibility_keys == held


async def test_a_documents_audience_moves() -> None:
    """MEM-07: a kept document's facts move to the live threads that hold it."""
    await check_a_documents_audience_moves(MemoryGraphStore())
