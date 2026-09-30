"""Unit tests: the neighbourhood hop's SQL order, its triple dedupe, and the one-statement
traversal that chains the hops.

Hermetic - the statements are compiled, never executed. Traversal semantics (bounds, time, visibility) are covered against the in-memory
store in test_graph_native.py and against PostgreSQL in the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql

from memory_service.adapters.graph.postgres_store import (
    neighborhood_query,
    traversal_params,
    traversal_query,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 15, tzinfo=UTC)


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
    assert "JOIN graph_entities" in sql
    assert "CASE WHEN" in sql and "coalesce" in sql.lower()


def test_edges_the_hop_would_drop_are_dropped_before_its_limit() -> None:
    """An invisible neighbour and an entity an earlier hop expanded used to be filtered after
    the LIMIT, so their edges filled it and starved the new ones."""
    stmt = neighborhood_query(
        "acme",
        ["ent_b"],
        scope_keys=["user:acme/u1"],
        layers=None,
        as_of=None,
        valid_at=None,
        limit=600,
        expanded=["ent_a"],
    )
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    inner = sql.split("LIMIT", maxsplit=1)[0]
    # the neighbour's audience is part of the join, before the limit
    assert "LEFT OUTER JOIN" not in inner
    assert inner.count("?|") == 2, "both the edge's and the neighbour's audience are filtered"
    assert "NOT IN" in inner, "edges back to an expanded entity are excluded in SQL"
    assert "NOT IN" not in _sql(), "the first hop has nothing to exclude"


def _traversal(hops: int = 3) -> str:
    stmt = traversal_query(hops)
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_the_whole_traversal_is_one_statement_of_chained_hops() -> None:
    sql = _traversal()
    for cte in ("v1", "h1", "f2", "v2", "h2", "f3", "v3", "h3", "f4", "v4"):
        assert f"{cte} AS" in sql, cte
    assert "h4 AS" not in sql
    # a hop runs only while the visited set is below the cap, and a frontier takes only
    # what the cap leaves, in the order the previous hop ranked its edges
    assert sql.count("< %(max_visited)s") == 3 and "greatest(" in sql
    assert "min(anon_" in sql and "row_number() OVER" in sql
    # the result keeps only edges whose two ends were visited, in (hop, rank) order
    assert sql.rstrip().endswith("rank")


def test_an_assertion_an_earlier_hop_returned_is_excluded_before_the_next_limit() -> None:
    """Hop two can reach a triple hop one already returned; excluded inside the hop, the
    limit is spent on edges that are new."""
    hop2 = _traversal().split("h2 AS", 1)[1].split("f3 AS", 1)[0]
    assert "NOT (EXISTS" in hop2 and "FROM h1" in hop2
    for field in (
        "subject_id",
        "predicate",
        "object_id",
        "document_id",
        "layer",
        "valid_from",
        "valid_to",
    ):
        assert f"h1.{field} IS NOT DISTINCT FROM graph_relations.{field}" in hop2, field
    assert "h1.attributes = graph_relations.attributes" in hop2
    hop3 = _traversal().split("h3 AS", 1)[1]
    assert "FROM h1" in hop3 and "FROM h2" in hop3
    assert "NOT (EXISTS" not in _traversal(hops=1), "the first hop has nothing earlier"


def test_the_traversal_is_built_once_per_shape_and_bound_per_query() -> None:
    """Its text depends on the shape only, so the server prepares it once per connection."""
    assert traversal_query(3) is traversal_query(3)
    assert traversal_query(3, as_of=True) is not traversal_query(3)
    sql = str(traversal_query(1, layers=True, as_of=True).compile(dialect=postgresql.dialect()))
    for name in ("tenant_id", "seeds", "keys", "max_visited", "layers", "as_of"):
        assert f"%({name})s" in sql, name
    params = traversal_params(
        "acme",
        ["ent_a"],
        scope_keys=["k1", "k2"],
        max_visited=40,
        layers=["entity"],
        as_of=NOW,
        valid_at=None,
    )
    assert params == {
        "tenant_id": "acme",
        "seeds": ["ent_a"],
        "keys": ["k1", "k2"],
        "max_visited": 40,
        "layers": ["entity"],
        "as_of": NOW,
    }


def test_sql_preserves_qualifiers_and_validity_before_the_limit():
    distinct = _sql().split("DISTINCT ON (", 1)[1].split(")", 1)[0]
    for field in ("attributes", "document_id", "layer", "valid_from", "valid_to"):
        assert f"graph_relations.{field}" in distinct
