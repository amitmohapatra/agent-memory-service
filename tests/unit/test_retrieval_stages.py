"""Post-stages that can start before the search do, and never outlive the request.

The graph traversal depends on the routed query and the caller's scope, not on the ranked
candidates, yet it ran after the search - sequential to work it never read. A stage may now
declare ``prefetch``; the engine starts it as soon as the scope is known and hands the task
back when the stage's turn comes. What it must never do is leak that task when the stage is
not reached.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from memory_service.domain.enums import QueryType
from tests.unit import test_llm_retrieval as base

CTX, QUERY, VISIBILITY, _engine = base.CTX, base.QUERY, base.VISIBILITY, base._engine
parts = base.parts  # the indexed two-chunk corpus fixture, shared with that module


class _Prefetching:
    """A stage shaped like GraphStage: work that needs only the route and the scope."""

    def __init__(self, events: list[str], *, hold: bool = False) -> None:
        self.events = events
        self.hold = hold  # a traversal that never finishes on its own
        self.task: asyncio.Task[str] | None = None
        self.received: Any = None

    def prefetch(self, ctx, routed, visibility) -> asyncio.Task[str]:
        async def work() -> str:
            self.events.append("prefetch:start")
            if self.hold:
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            self.events.append("prefetch:done")
            return "traversal"

        self.task = asyncio.ensure_future(work())
        return self.task

    async def __call__(self, ctx, routed, candidates, visibility, diagnostics, *, prefetched=None):
        self.received = prefetched
        self.events.append(f"stage:{await prefetched}")
        return candidates


class _Failing:
    async def __call__(self, ctx, routed, candidates, visibility, diagnostics):
        raise RuntimeError("stage failed")


async def test_a_prefetching_stage_starts_before_the_search_and_is_handed_its_task(parts):
    events: list[str] = []
    engine = _engine(parts)
    stage = _Prefetching(events)
    engine.post_stages["graph"] = stage
    hybrid = engine._hybrid

    async def spied(*args, **kwargs):
        events.append("search")
        return await hybrid(*args, **kwargs)

    engine._hybrid = spied  # type: ignore[method-assign]
    res = await engine.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)

    assert events.index("prefetch:start") < events.index("search"), "started under the search"
    assert events[-1] == "stage:traversal"
    assert stage.received is stage.task, "the stage is given the very task the engine started"
    assert res.diagnostics["stages"] == ["graph"]
    timings = res.diagnostics["timings_ms"]
    assert {"encode", "search", "graph"} <= set(timings), timings
    assert all(v >= 0 for v in timings.values())


async def test_a_prefetched_task_is_cancelled_when_an_earlier_stage_raises(parts):
    engine = _engine(parts)
    stage = _Prefetching([], hold=True)
    engine.post_stages["boom"] = _Failing()
    engine.post_stages["graph"] = stage
    with pytest.raises(RuntimeError, match="stage failed"):
        await engine.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
    assert stage.task is not None and stage.received is None, "the stage was never reached"
    await asyncio.sleep(0)
    assert stage.task.cancelled(), "the request ended; its prefetch must not run on"


async def test_a_stage_without_prefetch_is_called_exactly_as_before(parts):
    seen: dict[str, Any] = {}

    async def plain(ctx, routed, candidates, visibility, diagnostics):
        seen["kwargs_free"] = True
        seen["query_type"] = routed.query_type
        return candidates

    engine = _engine(parts)
    engine.post_stages["plain"] = plain
    res = await engine.retrieve(CTX, QUERY, kinds=("chunk",), visibility=VISIBILITY)
    assert seen == {"kwargs_free": True, "query_type": QueryType.GENERAL_SEMANTIC}
    assert res.diagnostics["stages"] == ["plain"]
