"""Entity search, profiles, summaries and the bounded traversal on PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.errors import NotFound
from memory_service.domain.ids import new_id
from memory_service.modules.graph.summaries import EntitySummaries
from memory_service.modules.llm.assist import LLMAssist
from tests.integration.test_graph import U1, U2, _observe
from tests.unit.test_graph_entities import MINE, entity, relation, seeded

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 15, tzinfo=UTC)


async def _seed(store) -> None:
    memory = await seeded()
    await store.upsert_entities(list(memory.entities.values()))
    await store.upsert_relations(list(memory.relations.values()))


async def test_postgres_search_relations_and_summary_sources(container) -> None:
    store = container.graph_store
    await _seed(store)
    found = await store.search_entities("acme", scope_keys=MINE, prefix="acme")
    assert [e.name for e in found] == ["Acme Corp", "Acme Labs"]
    assert [
        e.name for e in await store.search_entities("acme", scope_keys=MINE, entity_type="PLACE")
    ] == ["Germany"]
    assert await store.search_entities("acme", scope_keys=MINE, prefix="secret") == []
    # a LIKE metacharacter in the prefix is a character, not a wildcard
    assert await store.search_entities("acme", scope_keys=MINE, prefix="acme%") == []
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
    assert "rel_2" not in {r.relation_id for r in theirs}
    [facts] = await store.summary_sources("acme", ["ent_acme_corp"], limit=10)
    assert facts.facts == [("operates_in", "Germany")] and facts.summary_source == ""
    summaries = EntitySummaries(store, LLMAssist.disabled(), container.tuning.graph)
    assert await summaries.refresh("acme", ["ent_acme_corp"]) == 1
    assert await summaries.refresh("acme", ["ent_acme_corp"]) == 0
    [stored] = await store.get_entities("acme", ["ent_acme_corp"], scope_keys=MINE)
    assert stored.summary == "Acme Corp (ORG): operates in Germany"


async def test_the_hop_limit_is_spent_on_new_edges(container) -> None:
    """A bounded hop used to fill its LIMIT with edges it then threw away: back to an entity
    an earlier hop had expanded, or to a neighbour the caller cannot see. Both are filtered
    in SQL now, so the low-ranked new edge still arrives."""
    store = container.graph_store
    await store.upsert_entities(
        [
            entity("A"),
            entity("B"),
            entity("C"),
            entity("D"),
            *(entity(f"X{i}", keys=["user:acme/u2"]) for i in range(20)),
        ]
    )
    await store.upsert_relations(
        [
            relation("rel_ab", "ent_a", "knows", "ent_b", confidence=0.95),
            relation("rel_ac", "ent_a", "knows", "ent_c", confidence=0.95),
            # twenty strong edges back to the seed, all found by the first hop's query
            *(
                relation(f"rel_ba{i}", "ent_b", f"likes_{i}", "ent_a", confidence=0.9)
                for i in range(20)
            ),
            # twenty strong edges to entities this reader cannot see
            *(
                relation(f"rel_cx{i}", "ent_c", f"owns_{i}", f"ent_x{i}", confidence=0.9)
                for i in range(20)
            ),
            relation("rel_cd", "ent_c", "knows", "ent_d", confidence=0.1),
        ]
    )
    hood = await store.neighborhood("acme", ["ent_a"], scope_keys=MINE, hops=2, max_visited=4)
    assert "rel_cd" in {r.relation_id for r in hood.relations}
    assert "ent_d" in {e.entity_id for e in hood.entities}
    assert not any(e.entity_id.startswith("ent_x") for e in hood.entities)


async def test_the_budgeted_traversal_is_bounded_by_the_server(container) -> None:
    """The retrieval-time walk runs on connections whose statement_timeout is the graph
    budget, and a walk the server stops surfaces as GraphBudgetExceededError."""
    from sqlalchemy import text

    from memory_service.adapters.graph.postgres_store import PostgresGraphStore
    from memory_service.ports.intelligence import GraphBudgetExceededError

    await _seed(container.graph_store)
    store = PostgresGraphStore(container.database.engine)  # the shipped budget
    starved = PostgresGraphStore(container.database.engine, budget_ms=1)
    try:
        async with store._budgeted() as s:
            assert await s.scalar(text("SHOW statement_timeout")) == "150ms"
        async with store.read() as s:
            assert await s.scalar(text("SHOW statement_timeout")) == "15s"
        same = await store.neighborhood("acme", ["ent_acme_corp"], scope_keys=MINE, budgeted=True)
        plain = await store.neighborhood("acme", ["ent_acme_corp"], scope_keys=MINE)
        assert same.relations and same == plain
        async with starved._budgeted() as s:
            with pytest.raises(Exception, match="statement timeout"):
                await s.execute(text("SELECT pg_sleep(0.05)"))
        from unittest.mock import patch

        from memory_service.adapters.graph import postgres_store

        slow = text("SELECT pg_sleep(0.05)")
        with (
            patch.object(postgres_store, "traversal_query", return_value=slow),
            pytest.raises(GraphBudgetExceededError) as stopped,
        ):
            await starved.neighborhood("acme", ["ent_a"], scope_keys=MINE, budgeted=True)
        assert stopped.value.budget_ms == 1
    finally:
        await store.close()
        await starved.close()


async def test_query_exposes_layers_and_knowledge_time(container, uow_factory) -> None:
    ctx = U1.model_copy(update={"thread_id": new_id("thread")})
    await _observe(container, uow_factory, ctx, "I work at ACME Corp.")
    graph = container.services["graph"]
    everything = await graph.query(ctx, entities=["ACME Corp"])
    assert {r.layer for r in everything.relations} >= {"entity"}
    typed = await graph.query(ctx, entities=["ACME Corp"], layers=["entity"])
    assert typed.relations and all(r.layer == "entity" for r in typed.relations)
    before = await graph.query(
        ctx, entities=["ACME Corp"], valid_at=datetime.now(UTC) - timedelta(days=1)
    )
    assert before.relations == [], "nothing had been asserted yet"


async def test_enrichment_fills_summaries_and_the_profile_reads_them(
    container, uow_factory
) -> None:
    ctx = U1.model_copy(update={"thread_id": new_id("thread")})
    await _observe(container, uow_factory, ctx, "I work at ACME Corp.")
    await _observe(container, uow_factory, ctx, "I work at Globex now.")
    graph = container.services["graph"]
    [user] = [e for e in await graph.search_entities(ctx, query="user:u1") if e.name == "user:u1"]
    assert "works at" in user.summary and "Globex" in user.summary.title()
    profile = await graph.profile(ctx, user.entity_id)
    [employer] = [r for r in profile.current if r.predicate == "works_at"]
    assert profile.names[employer.object_id].lower() == "globex"
    assert any(r.predicate == "works_at" for r in profile.history), "the old employer"
    with pytest.raises(NotFound):
        await graph.profile(U2.model_copy(update={"thread_id": ctx.thread_id}), user.entity_id)
    with pytest.raises(NotFound):
        await graph.profile(ctx, "ent_missing")


async def test_search_is_bounded_and_scoped(container, uow_factory) -> None:
    await _seed(container.graph_store)
    graph = container.services["graph"]
    assert [e.name for e in await graph.search_entities(U1, query="Acme", limit=1)] == ["Acme Corp"]
    assert await graph.search_entities(U2.model_copy(update={"tenant_id": "globex"})) == []
