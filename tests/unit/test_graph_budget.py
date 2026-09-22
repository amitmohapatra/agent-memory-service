"""The graph traversal has a wall budget, and expiry drops facts without cancelling anything.

The traversal is started as soon as the scope is known, so today it finishes underneath a
178 ms encoder and costs nothing. That is precisely why it needs a ceiling: the moment the
encoder is int8 ONNX at ~40 ms, or the graph is deep enough for a three-hop walk to outrun
it, an unbounded traversal is the tail of every entity, temporal and multi-hop question.

The expiry must end *this query's wait*, not the traversal. ``asyncio.wait_for`` cancels what
it waits on, and what it would cancel here is a statement holding a pooled connection: an
aborted connection is charged to every later request on that pool, which is a far worse trade
than one slow answer. So the wait is on a shield and the traversal is left to finish.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.retrieval import GraphStage, graph_budget_expired_total
from memory_service.modules.graph.service import GraphAnswer
from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.router import QueryRouter
from memory_service.ports.intelligence import Entity, Relation

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
    """A GraphService whose traversal takes as long as the test says."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.started = 0
        self.finished = 0

    async def query(self, ctx: MemoryExecutionContext, **kwargs: Any) -> GraphAnswer:
        self.started += 1
        await asyncio.sleep(self.delay)
        self.finished += 1
        return _answer()


class _NoUoW:
    def __call__(self):  # pragma: no cover - no expansion chunks in these answers
        raise AssertionError("no unit of work expected")


def _stage(delay: float, budget: float) -> tuple[GraphStage, _Graph]:
    graph = _Graph(delay)
    return (
        GraphStage(graph, _NoUoW(), budget_seconds=budget),  # type: ignore[arg-type]
        graph,
    )


async def test_a_traversal_inside_the_budget_answers_with_its_facts() -> None:
    stage, graph = _stage(delay=0.0, budget=0.5)
    routed = _routed()
    diagnostics: dict[str, Any] = {}
    task = stage.prefetch(CTX, routed, VISIBILITY)
    out = await stage(CTX, routed, [], VISIBILITY, diagnostics, prefetched=task)

    assert [c.record_id for c in out] == ["rel_1"]
    assert diagnostics["graph"]["budget_expired"] is False
    assert diagnostics["graph"]["matched"] == ["acme"] and diagnostics["graph"]["visited"] == 7
    assert graph.finished == 1


async def test_an_expired_budget_answers_without_graph_facts() -> None:
    stage, _ = _stage(delay=0.2, budget=0.01)
    routed = _routed()
    diagnostics: dict[str, Any] = {}
    before = graph_budget_expired_total._value.get()
    existing = [Candidate(record_id="chk_1", kind="chunk", text="a passage", score=0.5)]

    task = stage.prefetch(CTX, routed, VISIBILITY)
    out = await stage(CTX, routed, existing, VISIBILITY, diagnostics, prefetched=task)

    assert [c.record_id for c in out] == ["chk_1"], "the ranked evidence still reaches the caller"
    assert diagnostics["graph"] == {"budget_expired": True, "budget_ms": 10}
    assert graph_budget_expired_total._value.get() == before + 1
    await stage.drain()


async def test_an_expired_budget_never_cancels_the_traversal() -> None:
    """The statement keeps its connection to the end. Cancelling it would abort a pooled
    connection, which every later request on that pool pays for."""
    stage, graph = _stage(delay=0.05, budget=0.001)
    routed = _routed()
    task = stage.prefetch(CTX, routed, VISIBILITY)
    assert task is not None

    await stage(CTX, routed, [], VISIBILITY, {}, prefetched=task)
    assert not task.cancelled() and not task.done(), "the traversal was cut off mid-statement"
    await stage.drain()
    assert task.done() and not task.cancelled()
    assert graph.finished == 1, "the traversal did not run to completion"


async def test_a_traversal_that_fails_after_its_budget_is_not_an_unretrieved_exception() -> None:
    class _Broken(_Graph):
        async def query(self, ctx: MemoryExecutionContext, **kwargs: Any) -> GraphAnswer:
            await asyncio.sleep(0.05)
            raise RuntimeError("graph down")

    stage = GraphStage(_Broken(0.05), _NoUoW(), budget_seconds=0.001)  # type: ignore[arg-type]
    routed = _routed()
    diagnostics: dict[str, Any] = {}
    task = stage.prefetch(CTX, routed, VISIBILITY)
    await stage(CTX, routed, [], VISIBILITY, diagnostics, prefetched=task)
    assert diagnostics["graph"]["budget_expired"] is True
    await stage.drain()
    assert task is not None and task.exception() is not None


async def test_the_budget_also_bounds_a_stage_called_without_a_prefetch() -> None:
    """A caller that does not prefetch starts the traversal here; the wait is bounded the
    same way, because the reason for the ceiling is the wait, not who started it."""
    stage, _ = _stage(delay=0.2, budget=0.01)
    diagnostics: dict[str, Any] = {}
    out = await stage(CTX, _routed(), [], VISIBILITY, diagnostics)
    assert out == [] and diagnostics["graph"]["budget_expired"] is True
    await stage.drain()


async def test_a_route_without_a_graph_costs_nothing() -> None:
    stage, graph = _stage(delay=10.0, budget=0.01)
    routed = QueryRouter().routed(
        "summarise the report", QueryType.GLOBAL_SUMMARY, identifiers=[], signals={}
    )
    assert routed.needs_graph is False
    diagnostics: dict[str, Any] = {}
    assert await stage(CTX, routed, [], VISIBILITY, diagnostics) == []
    assert stage.prefetch(CTX, routed, VISIBILITY) is None
    assert graph.started == 0 and diagnostics == {}


def test_the_default_budget_comes_from_the_frozen_constant() -> None:
    from memory_service.config.constants import GRAPH

    stage = GraphStage(_Graph(0.0), _NoUoW())  # type: ignore[arg-type]
    assert stage.budget_seconds == pytest.approx(GRAPH.prefetch_budget_ms / 1000)


def test_the_budget_is_frozen_at_150ms() -> None:
    from memory_service.config.constants import GRAPH

    assert GRAPH.prefetch_budget_ms == 150
