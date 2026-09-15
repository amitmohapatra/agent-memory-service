"""Graph isolation gate (M8): traversal never returns an entity or relation outside the
reader's tenant + audience, for every reader configuration of the visibility oracle — and
2-hop traversal cannot "tunnel" through a visible entity into invisible relations."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest

from memory_service.adapters.graph.memory_store import MemoryGraphStore
from memory_service.domain.enums import Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.ports.intelligence import Entity, Relation
from tests.security.test_isolation import TENANTS, _keys, _oracle, _spec
from tests.security.test_retrieval_isolation import READERS

pytestmark = pytest.mark.security

NOW = datetime(2026, 9, 15, tzinfo=UTC)

OBJECTS = [
    {
        "tenant": tenant,
        "visibility": vis.value,
        "owner": owner,
        "user": user,
        "group": "legal",
        "thread": "thr1",
        "workspace": ws,
        "work": "w1",
        "agent_group": "crew",
    }
    for tenant, vis, owner, user, ws in itertools.product(
        TENANTS, Visibility, ["user:u1", "agent:research"], ["u1", "u2"], ["ws1", "ws2"]
    )
]


@pytest.fixture(scope="module")
async def store() -> tuple[MemoryGraphStore, dict[str, dict]]:
    """A hub entity per tenant (visible to everyone in the tenant) linked to one private-ish
    entity per visibility variant: a leak would surface as a 2-hop relation."""
    store = MemoryGraphStore()
    by_relation: dict[str, dict] = {}
    for tenant in TENANTS:
        hub = Entity(
            entity_id=f"ent_hub_{tenant}",
            tenant_id=tenant,
            name="Hub",
            canonical_name="hub",
            visibility_keys=[f"tenant:{tenant}"],
        )
        await store.upsert_entities([hub])
    for n, obj in enumerate(OBJECTS):
        keys = _keys(obj)
        leaf = Entity(
            entity_id=f"ent_leaf_{n}",
            tenant_id=obj["tenant"],
            name=f"Leaf {n}",
            canonical_name=f"leaf {n}",
            visibility_keys=keys,
        )
        await store.upsert_entities([leaf])
        rel = Relation(
            relation_id=f"rel_{n}",
            tenant_id=obj["tenant"],
            subject_id=f"ent_hub_{obj['tenant']}",
            predicate="links",
            object_id=leaf.entity_id,
            visibility_keys=keys,
            observed_at=NOW,
            evidence=[EvidenceRef(source_type="memory", source_id="m", observed_at=NOW)],
            fact_text=f"hub links leaf {n}",
        )
        await store.upsert_relations([rel])
        by_relation[rel.relation_id] = obj
    return store, by_relation


async def test_traversal_returns_only_authorized_relations(store) -> None:
    graph, by_relation = store
    total = 0
    for reader in READERS:
        spec = _spec(reader)
        hood = await graph.neighborhood(
            reader["tenant"], [f"ent_hub_{reader['tenant']}"], scope_keys=sorted(spec.keys), hops=2
        )
        leaked = [
            r.relation_id for r in hood.relations if not _oracle(reader, by_relation[r.relation_id])
        ]
        assert leaked == [], f"reader={reader} leaked={leaked[:5]}"
        allowed = {
            rid
            for rid, obj in by_relation.items()
            if obj["tenant"] == reader["tenant"] and spec.allows(obj["tenant"], _keys(obj))
        }
        assert {r.relation_id for r in hood.relations} == allowed
        for e in hood.entities:
            assert e.tenant_id == reader["tenant"] and spec.allows(e.tenant_id, e.visibility_keys)
        total += len(hood.relations)
    assert total > 0


async def test_cross_tenant_seed_is_inert(store) -> None:
    graph, _ = store
    from tests.security.test_isolation import _spec as spec_for

    reader = {
        "tenant": "acme",
        "user": "u1",
        "is_agent": False,
        "agent": "research",
        "groups": ["legal"],
        "threads": ["thr1"],
        "workspaces": ["ws1", "ws2"],
        "works": ["w1"],
        "agent_groups": ["crew"],
    }
    spec = spec_for(reader)
    # seeding with the OTHER tenant's hub id under acme's keys yields nothing
    hood = await graph.neighborhood(
        "acme", ["ent_hub_globex"], scope_keys=sorted(spec.keys), hops=2
    )
    assert hood.relations == [] and hood.entities == []
    hood = await graph.neighborhood(
        "globex", ["ent_hub_globex"], scope_keys=sorted(spec.keys), hops=2
    )
    assert hood.relations == [] and hood.entities == []
