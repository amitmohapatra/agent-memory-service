"""The retrieval-time graph stage: facts from a bounded traversal, the evidence expansion
behind them, and the graph budget - enforced by the database, never by a client timer.

A client timer measured the client's own scheduling as much as the graph: on a loaded box a
traversal whose statement took 90 ms lost its facts after a "150 ms" wait that lasted 500.
The traversal is one bounded statement whose connection carries the budget as its
``statement_timeout``; the stage waits for its answer, and the store's
``GraphBudgetExceededError`` is the one way it is cut short.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.retrieval import GraphStage, graph_budget_expired_total
from memory_service.modules.graph.service import GraphAnswer
from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.router import QueryRouter
from memory_service.ports.intelligence import Entity, GraphBudgetExceededError, Relation

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1")
VISIBILITY = VisibilitySpecification(tenant_id="acme", keys=frozenset({"tenant:acme"}))


def _routed(query: str = "who leads the freight operator used by Acme?"):
    return QueryRouter().routed(
        query, QueryType.ENTITY_RELATION, identifiers=[], signals={}, has_thread=False
    )


def _answer() -> GraphAnswer:
    acme = Entity(entity_id="ent_acme", tenant_id="acme", name="Acme", canonical_name="acme")
    west = Entity(
        entity_id="ent_west", tenant_id="acme", name="Westfalen", canonical_name="westfalen"
    )
    return GraphAnswer(
        entities=[acme, west],
        relations=[
            Relation(
                relation_id="rel_1",
                tenant_id="acme",
                subject_id="ent_acme",
                predicate="uses",
                object_id="ent_west",
                confidence=0.9,
                fact_text="Acme uses Westfalen",
                observed_at=datetime.now(UTC),
                evidence=[
                    EvidenceRef(
                        source_type="memory", source_id="mem_1", observed_at=datetime.now(UTC)
                    )
                ],
            )
        ],
        matched=[acme],
        visited=7,
    )


class _Graph:
    """A GraphService whose traversal takes as long as the test says, or is stopped at the
    budget by its store."""

    def __init__(self, delay: float, *, exceeded: bool = False) -> None:
        self.delay = delay
        self.exceeded = exceeded
        self.started = 0
        self.finished = 0
        self.budgeted: list[bool] = []

    async def query(self, ctx: MemoryExecutionContext, **kwargs: Any) -> GraphAnswer:
        self.started += 1
        self.budgeted.append(kwargs.get("budgeted", False))
        await asyncio.sleep(self.delay)
        if self.exceeded:
            raise GraphBudgetExceededError(150)
        self.finished += 1
        return _answer()


class _NoUoW:
    def __call__(self):  # pragma: no cover - no expansion chunks in these answers
        raise AssertionError("no unit of work expected")


def _stage(delay: float = 0.0, *, exceeded: bool = False) -> tuple[GraphStage, _Graph]:
    graph = _Graph(delay, exceeded=exceeded)
    return GraphStage(graph, _NoUoW(), max_expansion_memories=0), graph  # type: ignore[arg-type]


async def test_a_prefetched_traversal_answers_with_its_facts() -> None:
    stage, graph = _stage()
    routed = _routed()
    diagnostics: dict[str, Any] = {}
    task = stage.prefetch(CTX, routed, VISIBILITY)
    out = await stage(CTX, routed, [], VISIBILITY, diagnostics, prefetched=task)

    assert [c.record_id for c in out] == ["rel_1"]
    assert diagnostics["graph"]["budget_expired"] is False
    assert diagnostics["graph"]["matched"] == ["acme"] and diagnostics["graph"]["visited"] == 7
    assert graph.finished == 1 and graph.budgeted == [True], "the stage's walk is budgeted"


async def test_a_slow_client_does_not_cost_the_answer_its_graph_facts() -> None:
    """No timer in the client: however long this process takes to get back to the
    traversal, the facts it returned are used."""
    stage, _ = _stage(delay=0.3)
    routed = _routed()
    diagnostics: dict[str, Any] = {}
    task = stage.prefetch(CTX, routed, VISIBILITY)
    out = await stage(CTX, routed, [], VISIBILITY, diagnostics, prefetched=task)
    assert [c.record_id for c in out] == ["rel_1"]
    assert diagnostics["graph"]["budget_expired"] is False


async def test_a_traversal_the_database_stopped_answers_without_graph_facts() -> None:
    stage, _ = _stage(exceeded=True)
    routed = _routed()
    diagnostics: dict[str, Any] = {}
    before = graph_budget_expired_total._value.get()
    existing = [Candidate(record_id="chk_1", kind="chunk", text="a passage", score=0.5)]

    task = stage.prefetch(CTX, routed, VISIBILITY)
    out = await stage(CTX, routed, existing, VISIBILITY, diagnostics, prefetched=task)

    assert [c.record_id for c in out] == ["chk_1"], "the ranked evidence still reaches the caller"
    assert diagnostics["graph"] == {"budget_expired": True, "budget_ms": 150}
    assert graph_budget_expired_total._value.get() == before + 1


async def test_a_stage_called_without_a_prefetch_runs_the_same_budgeted_walk() -> None:
    stage, graph = _stage(exceeded=True)
    diagnostics: dict[str, Any] = {}
    assert await stage(CTX, _routed(), [], VISIBILITY, diagnostics) == []
    assert diagnostics["graph"]["budget_expired"] is True and graph.budgeted == [True]


@pytest.mark.parametrize("budget", [0, 2, 6])
async def test_one_relation_cannot_overrun_the_evidence_expansion_budget(budget) -> None:
    stage, graph = _stage()
    stage.max_expansion_chunks = budget
    answer = _answer()
    answer.relations[0].evidence = [
        EvidenceRef(
            source_type="document_chunk",
            source_id=f"c{i}",
            chunk_id=f"c{i}",
            observed_at=datetime.now(UTC),
        )
        for i in range(20)
    ]
    graph.query = AsyncMock(return_value=answer)
    stage._expand = AsyncMock(return_value=[])
    await stage(CTX, _routed(), [], VISIBILITY, {})
    if budget:
        assert stage._expand.call_args.args[1] == [f"c{i}" for i in range(budget)]
    else:
        stage._expand.assert_not_called()


@pytest.mark.parametrize("budget", [0, 1, 6])
async def test_memory_pointers_are_deduplicated_bounded_and_skip_ranked_hits(budget):
    stage, graph = _stage()
    stage.max_expansion_memories = budget
    answer = _answer()
    answer.relations[0].memory_id = "mem_direct"
    answer.relations[0].evidence = [
        EvidenceRef(source_type="memory", source_id=identifier, observed_at=datetime.now(UTC))
        for identifier in ["mem_direct", "mem_ranked", *[f"mem_{i}" for i in range(20)]]
    ]
    graph.query = AsyncMock(return_value=answer)
    added = Candidate(
        record_id="mem_direct",
        kind="memory",
        text="source text",
        score=0.4,
        expansion_edge="GRAPH_EVIDENCE",
    )
    stage._expand_memories = AsyncMock(return_value=[added])
    ranked = Candidate(record_id="mem_ranked", kind="memory", text="ranked", score=1.0)
    out = await stage(CTX, _routed(), [ranked], VISIBILITY, {})
    if budget:
        assert (
            stage._expand_memories.call_args.args[1]
            == ["mem_direct", *[f"mem_{i}" for i in range(5)]][:budget]
        )
        assert [c.record_id for c in out] == ["mem_ranked", "mem_direct", "rel_1"]
    else:
        stage._expand_memories.assert_not_called()


async def test_a_route_without_a_graph_costs_nothing() -> None:
    stage, graph = _stage(delay=10.0)
    routed = QueryRouter().routed(
        "summarise the report", QueryType.GLOBAL_SUMMARY, identifiers=[], signals={}
    )
    assert routed.needs_graph is False
    diagnostics: dict[str, Any] = {}
    assert await stage(CTX, routed, [], VISIBILITY, diagnostics) == []
    assert stage.prefetch(CTX, routed, VISIBILITY) is None
    assert graph.started == 0 and diagnostics == {}


def test_the_budget_is_frozen_at_150ms() -> None:
    from memory_service.config.constants import GRAPH

    assert GRAPH.prefetch_budget_ms == 150
