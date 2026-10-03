"""The graph's memories fused by rank into the memory ranking (``GRAPH_WEIGHT``): ranks, not
scores, so the fusion holds whatever units the ranking's last step left the scores in."""

from __future__ import annotations

import pytest

from memory_service.modules.graph import retrieval as graph
from memory_service.modules.retrieval.engine import Candidate

pytestmark = pytest.mark.unit


def _c(rid: str, kind: str = "memory", score: float = 0.0) -> Candidate:
    return Candidate(record_id=rid, kind=kind, text=rid, score=score, retrievers=["bm25"])


def test_a_memory_the_traversal_ranks_first_moves_up_by_rank_not_score(monkeypatch) -> None:
    monkeypatch.setattr(graph, "GRAPH_WEIGHT", 1.0)
    # scores in tiny units, as the last fusion leaves them; c is third by rank
    candidates = [_c("a", score=0.03), _c("b", score=0.02), _c("chunk", kind="chunk"), _c("c")]
    graph._resort_memories(candidates, [], ["c"])
    # c: 1/(K+3) + 1/(K+1) beats a: 1/(K+1); the chunk keeps its place
    assert [x.record_id for x in candidates] == ["c", "a", "chunk", "b"]
    assert "graph" in candidates[0].retrievers


def test_a_memory_only_the_traversal_found_competes_on_its_graph_rank(monkeypatch) -> None:
    monkeypatch.setattr(graph, "GRAPH_WEIGHT", 0.25)
    candidates = [_c("a"), _c("b")]
    graph._resort_memories(candidates, [_c("new")], ["new"])
    # 0.25/(K+1) is below 1/(K+2): the ranked memories keep the places, the new one follows
    assert [x.record_id for x in candidates] == ["a", "b", "new"]


def test_the_traversal_order_follows_its_ranked_relations_without_repeats() -> None:
    class R:
        def __init__(self, memory_id: str | None, evidence: tuple = ()) -> None:
            self.memory_id, self.evidence = memory_id, evidence

    class Ev:
        source_type, source_id = "memory", "m3"

    ranked = [R("m1"), R("m2", (Ev(),)), R("m1"), R(None)]
    assert graph._memory_order(ranked, depth=10) == ["m1", "m2", "m3"]  # type: ignore[arg-type]
    assert graph._memory_order(ranked, depth=2) == ["m1", "m2"]  # type: ignore[arg-type]
