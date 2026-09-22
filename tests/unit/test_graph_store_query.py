"""Unit tests: the neighbourhood hop's SQL order and its triple dedupe.

Hermetic - the statement is compiled, never executed, and the dedupe is a pure function
over rows. Traversal semantics (bounds, time, visibility) are covered against the in-memory
store in test_graph_native.py and against PostgreSQL in the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql

from memory_service.adapters.db.orm import GraphRelationRow
from memory_service.adapters.graph.postgres_store import (
    first_of_each_triple,
    neighborhood_query,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 15, tzinfo=UTC)


def _row(relation_id: str, subject: str, predicate: str, obj: str) -> GraphRelationRow:
    return GraphRelationRow(
        relation_id=relation_id,
        tenant_id="acme",
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        observed_at=NOW,
    )


def _sql() -> str:
    stmt = neighborhood_query(
        "acme",
        ["ent_a"],
        scope_keys=["user:acme/u1"],
        layers=None,
        as_of=None,
        valid_at=None,
        limit=600,
    )
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_the_hop_orders_by_more_than_confidence() -> None:
    # Every MENTIONS edge is written at the same capped confidence, so ORDER BY confidence
    # alone left the LIMIT to return whichever tied rows the scan reached first.
    order = _sql().split("ORDER BY", 1)[1]
    assert "graph_relations.confidence DESC" in order
    assert "mention_count" in order and "graph_relations.observed_at DESC" in order
    assert order.index("graph_relations.relation_id") > order.index("mention_count")
    assert "LIMIT" in order


def test_the_hop_reads_the_neighbour_end_for_its_mention_count() -> None:
    sql = _sql()
    # the tie-break is the mention count of the far end of the edge, whichever end that is
    assert "LEFT OUTER JOIN graph_entities" in sql
    assert "CASE WHEN" in sql and "coalesce" in sql.lower()


def test_identical_triples_are_kept_once_across_hops() -> None:
    seen: set[tuple[str, str, str]] = set()
    first_hop = [
        _row("rel_1", "ent_a", "mentions", "ent_b"),
        _row("rel_2", "ent_a", "mentions", "ent_b"),  # same triple, a second memory said it
        _row("rel_3", "ent_a", "knows", "ent_b"),
    ]
    assert [r.relation_id for r in first_of_each_triple(first_hop, seen)] == ["rel_1", "rel_3"]
    # the set carries over: the next hop cannot bring the same triple back
    second_hop = [
        _row("rel_4", "ent_a", "mentions", "ent_b"),
        _row("rel_5", "ent_b", "mentions", "ent_c"),
    ]
    assert [r.relation_id for r in first_of_each_triple(second_hop, seen)] == ["rel_5"]
    assert seen == {
        ("ent_a", "mentions", "ent_b"),
        ("ent_a", "knows", "ent_b"),
        ("ent_b", "mentions", "ent_c"),
    }
