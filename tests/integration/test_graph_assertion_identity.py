"""Real SQL must preserve qualified assertions while collapsing duplicate supports."""

from datetime import UTC, datetime

import pytest

from memory_service.domain.evidence import EvidenceRef
from memory_service.ports.intelligence import Entity, Relation

pytestmark = pytest.mark.integration


async def test_same_amount_in_two_periods_survives_sql_deduplication(container):
    store = container.graph_store
    keys = ["user:acme/u1"]
    await store.upsert_entities(
        [
            Entity(
                entity_id=name,
                tenant_id="acme",
                name=name,
                canonical_name=name,
                visibility_keys=keys,
            )
            for name in ("revenue", "eur412")
        ]
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        Relation(
            relation_id=f"rel_{number}",
            tenant_id="acme",
            subject_id="revenue",
            predicate="has_value",
            object_id="eur412",
            visibility_keys=keys,
            observed_at=now,
            attributes={"period": period},
            evidence=[
                EvidenceRef(source_type="document_chunk", source_id="doc_source", observed_at=now)
            ],
        )
        for number, period in enumerate(("FY25", "FY26", "FY26"))
    ]
    await store.upsert_relations(rows)
    answer = await store.neighborhood("acme", ["revenue"], scope_keys=keys, hops=2)
    assert len(answer.relations) == 2
    assert {r.attributes["period"] for r in answer.relations} == {"FY25", "FY26"}
    assert not (
        await store.neighborhood("acme", ["revenue"], scope_keys=["user:acme/u2"])
    ).relations
