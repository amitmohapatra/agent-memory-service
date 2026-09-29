"""Typed connections against PostgreSQL: the edge survives the round trip, the endpoints keep
their status, and a second pass over the same memories asks nothing.

The unit tests pin the logic; this one pins the two things only a real store can answer - that
an edge written into ``system_metadata`` comes back on the next read, and that writing one
bumps the row revision without touching the audience or the temporal state.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory.connections import ConnectionService, edges_of
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
#: This shape is one the rule extractor turns into a SEMANTIC fact about "release review".
#: A shape it cannot parse is kept as a verbatim turn, which this pass never connects.
OLD = "The release review is on Tuesdays at 15:00."
NEW = "The release review is on Thursdays at 15:00."


async def _observe(container, uow_factory, content: str) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["memory"].submit_observation(
            uow, U1, kind=ObservationKind.MESSAGE, content=content
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _memories(container, uow_factory):
    async with uow_factory() as uow:
        return await container.services["memory"].list_memories(uow, U1)


async def test_an_edge_survives_the_round_trip_and_changes_no_state(container, uow_factory) -> None:
    await _observe(container, uow_factory, OLD)
    await _observe(container, uow_factory, NEW)
    before = await _memories(container, uow_factory)
    assert len(before) >= 2
    revisions = {m.memory_id: m.revision for m in before}

    reply = {"connections": [{"pair": 0, "kind": "relates", "why": "one review, two days"}]}
    with mocked_gateway([reply]) as gateway:
        service = ConnectionService(uow_factory, assist=gateway.assist(uses=["memory_connections"]))
        written = await service.connect_all()
        assert written, "the two facts share a subject, so the pass had a pair to ask about"
        assert gateway.route.call_count == 1

        after = {m.memory_id: m for m in await _memories(container, uow_factory)}
        left, right = written[0]["left"], written[0]["right"]
        assert [e["kind"] for e in edges_of(after[left])] == ["relates"]
        assert edges_of(after[left])[0]["memory_id"] == right
        assert edges_of(after[right])[0]["memory_id"] == left
        for memory_id in (left, right):
            assert after[memory_id].revision > revisions[memory_id], "an edge is a revision"
            assert after[memory_id].temporal.status.value == "CURRENT"
            assert after[memory_id].system_metadata["visibility_keys"], "audience is untouched"

        # The re-index the write queued is real work, and the second pass finds the pair
        # already connected: no model call, nothing written.
        await container.tasks.drain()
        assert await service.connect_all() == []
        assert gateway.route.call_count == 1


async def test_without_the_use_configured_the_pass_does_nothing(container, uow_factory) -> None:
    await _observe(container, uow_factory, OLD)
    await _observe(container, uow_factory, NEW)
    before = {m.memory_id: m.revision for m in await _memories(container, uow_factory)}

    with mocked_gateway([{"connections": []}]) as gateway:
        service = ConnectionService(uow_factory, assist=gateway.assist(uses=["reflection"]))
        assert await service.connect_all() == []
        assert gateway.route.call_count == 0

    assert {m.memory_id: m.revision for m in await _memories(container, uow_factory)} == before
