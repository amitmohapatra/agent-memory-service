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


def test_the_hop_deduplicates_triples_before_the_limit_not_after() -> None:
    """The limit must land on distinct triples, or the busiest entity spends it all.

    A triple is written once per memory that states it, and every MENTIONS edge carries
    the same capped confidence, so the tie-break falls to the neighbour's mention count
    and one talkative entity's duplicates fill the limit. Measured on the benchmark
    corpus: 600 rows collapsed to 72 distinct triples against 125 reachable, with one
    triple holding 178 of the 600 slots. Deduplicating in Python afterwards cannot get
    those slots back.
    """
    sql = _sql()
    inner, outer = sql.split("ORDER BY")[1], sql.rsplit("ORDER BY", 1)[1]
    assert "DISTINCT ON" in sql
    # PostgreSQL requires the DISTINCT ON columns to lead the ordering that picks the
    # surviving row, and the rest of that ordering is what makes the choice deterministic
    assert inner.index("graph_relations.subject_id") < inner.index("graph_relations.confidence")
    assert "neighbour_mentions" in inner
    # ...and the limit applies to the outer ordering, which is the ranking the caller wants
    assert outer.index("confidence DESC") < outer.index("neighbour_mentions DESC")
    assert outer.index("observed_at DESC") < outer.index("relation_id")
    assert "LIMIT" in outer and "DISTINCT ON" not in outer


def test_the_surviving_row_of_a_triple_is_the_most_confident_one() -> None:
    order = _sql().split("DISTINCT ON")[1].split("ORDER BY", 1)[1]
    assert "graph_relations.confidence DESC" in order
    assert "neighbour_mentions" in order and "graph_relations.observed_at DESC" in order


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
