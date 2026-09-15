"""LLM-assisted entity resolution against the PostgreSQL graph store: the bounded,
visibility-filtered candidate listing and the resolve path with a mocked gateway."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.service import GraphService
from memory_service.ports.intelligence import Entity, Relation
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
VIS = VisibilitySpecification(tenant_id="acme", keys=frozenset({"user:acme/u1"}))
NOW = datetime(2026, 9, 15, tzinfo=UTC)


def _entity(name: str, *, keys: list[str], mention_count: int = 1, **extra) -> Entity:
    canonical = name.casefold()
    return Entity(
        entity_id=f"ent_{canonical.replace(' ', '_')}",
        tenant_id="acme",
        name=name,
        canonical_name=canonical,
        visibility_keys=keys,
        mention_count=mention_count,
        **extra,
    )


async def _seed(store) -> None:
    mine = ["user:acme/u1"]
    await store.upsert_entities(
        [
            _entity(
                "Adjusted EBITDA",
                keys=mine,
                mention_count=5,
                entity_type="METRIC",
                aliases=["adjusted earnings"],
            ),
            _entity("Restructuring Programme", keys=mine, mention_count=3, entity_type="EVENT"),
            _entity("EUR 98 million", keys=mine, mention_count=9, entity_type="MONEY"),
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
                visibility_keys=mine,
                observed_at=NOW,
                evidence=[
                    EvidenceRef(source_type="document_chunk", source_id="c1", observed_at=NOW)
                ],
                fact_text="Adjusted EBITDA driven by Restructuring Programme",
            )
        ]
    )


async def test_postgres_store_lists_visible_entities_bounded(container) -> None:
    store = container.graph_store
    await _seed(store)
    listed = await store.list_entities("acme", scope_keys=["user:acme/u1"], limit=2)
    assert [e.canonical_name for e in listed] == ["eur 98 million", "adjusted ebitda"]
    everything = await store.list_entities("acme", scope_keys=["user:acme/u1"], limit=50)
    assert {e.canonical_name for e in everything} == {
        "eur 98 million",
        "adjusted ebitda",
        "restructuring programme",
    }
    assert await store.list_entities("acme", scope_keys=[], limit=50) == []
    assert await store.list_entities("globex", scope_keys=["user:acme/u1"], limit=50) == []
    assert [
        e.canonical_name for e in await store.list_entities("acme", scope_keys=["user:acme/u2"])
    ] == ["secret plan"]


async def test_entity_resolution_on_postgres(container) -> None:
    store = container.graph_store
    await _seed(store)

    def service(assist):
        return GraphService(
            container.services["uow_factory"],
            store,
            None,
            container.services["authz"],
            settings=container.settings.graph,
            assist=assist,
        )

    reply = {"matches": [{"query_name": "adj ebitda", "entity_name": "Adjusted EBITDA"}]}
    with mocked_gateway([reply]) as gw:
        answer = await service(gw.assist(uses=["entity_resolution"])).query(
            CTX, query="adj. EBITDA?", visibility=VIS
        )
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
    assert "- Adjusted EBITDA (aka adjusted earnings)" in prompt
    assert "Secret Plan" not in prompt and "EUR 98 million" not in prompt
    assert [e.canonical_name for e in answer.matched] == ["adjusted ebitda"]
    assert [r.predicate for r in answer.relations] == ["driven_by"]
    # another user's entities are neither offered nor resolvable, whatever the model says
    other = VisibilitySpecification(tenant_id="acme", keys=frozenset({"user:acme/u2"}))
    with mocked_gateway(
        [{"matches": [{"query_name": "adj. ebitda", "entity_name": "Adjusted EBITDA"}]}]
    ) as gw:
        svc = service(gw.assist(uses=["entity_resolution"]))
        assert await svc.resolve(CTX, ["adj. EBITDA"], other) == []
        assert "Adjusted EBITDA" not in gw.prompts()[0]["messages"][1]["content"]
    with mocked_gateway(failing=True) as gw:
        assert (
            await service(gw.assist(uses=["entity_resolution"])).resolve(CTX, ["adj. EBITDA"], VIS)
            == []
        )
    with mocked_gateway([reply]) as gw:
        assert await service(gw.assist(uses=[])).resolve(CTX, ["adj. EBITDA"], VIS) == []
        assert gw.route.call_count == 0
