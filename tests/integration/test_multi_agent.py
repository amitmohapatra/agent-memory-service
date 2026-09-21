"""Multi-agent semantics (M11): private vs shared memory, run lineage (hand-off context flows
down, never up or sideways), cross-agent corroboration and conflict, and no chat pollution."""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    MemoryType,
    MessageKind,
    MessageRole,
    ObservationKind,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.ids import new_id
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration


def _user_ctx(**extra) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id="u1",
        workspace_id="ws1",
        thread_id=new_id("thread"),
        session_id=new_id("session"),
        turn_id=new_id("turn"),
        **extra,
    )


async def _thread(container, uow_factory, ctx) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["conversation"].append_message(
            uow, ctx, role=MessageRole.USER, content="Please prepare the FY26 brief."
        )
        await uow.commit()


async def _observe(
    container, uow_factory, ctx, content, *, kind=ObservationKind.AGENT_RESULT, hints=None
):
    async with uow_factory() as uow:
        ack = await container.services["memory"].submit_observation(
            uow, ctx, kind=kind, content=content, hints=hints
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack


async def _memory_texts(container, ctx, query):
    res = await container.services["retrieval"].retrieve(ctx, query, kinds=("memory",))
    return {c.text for c in res.candidates if c.kind == "memory"}


async def test_run_lineage_flows_down_not_up_or_sideways(container, uow_factory) -> None:
    user = _user_ctx()
    await _thread(container, uow_factory, user)
    planner = user.model_copy(update={"agent_id": "planner", "agent_run_id": new_id("agent_run")})
    writer = planner.model_copy(
        update={
            "agent_id": "writer",
            "agent_run_id": new_id("agent_run"),
            "parent_agent_run_id": planner.agent_run_id,
        }
    )
    reviewer = planner.model_copy(
        update={
            "agent_id": "reviewer",
            "agent_run_id": new_id("agent_run"),
            "parent_agent_run_id": planner.agent_run_id,
        }
    )
    grandchild = writer.model_copy(
        update={
            "agent_id": "formatter",
            "agent_run_id": new_id("agent_run"),
            "parent_agent_run_id": writer.agent_run_id,
        }
    )
    await _observe(container, uow_factory, planner, "Plan: split the brief into revenue and cost.")
    await _observe(container, uow_factory, writer, "Draft note: revenue section uses Table 1.")
    q = "plan for the brief sections"
    async with uow_factory() as uow:
        mine = await container.services["memory"].list_memories(uow, planner)
    assert mine and mine[0].visibility is Visibility.RUN
    assert f"run:acme/{planner.agent_run_id}" in mine[0].system_metadata["visibility_keys"]
    # children of the planner run read the planner's hand-off context
    assert "Plan: split the brief into revenue and cost." in await _memory_texts(
        container, writer, q
    )
    assert "Plan: split the brief into revenue and cost." in await _memory_texts(
        container, reviewer, q
    )
    # ... but not each other's (sideways), and the grandchild only sees its own parent's
    assert "Draft note: revenue section uses Table 1." not in await _memory_texts(
        container, reviewer, "draft note revenue table"
    )
    assert "Draft note: revenue section uses Table 1." in await _memory_texts(
        container, grandchild, "draft note revenue table"
    )
    assert "Plan: split the brief into revenue and cost." not in await _memory_texts(
        container, grandchild, q
    )
    # the user (upwards) sees no agent working notes; an unrelated agent's fresh run sees nothing
    assert await _memory_texts(container, user, q) == set()
    stranger_run = user.model_copy(
        update={"agent_id": "intern", "agent_run_id": new_id("agent_run")}
    )
    assert await _memory_texts(container, stranger_run, q) == set()
    # ... while the writing agent keeps its own notes across its later runs
    later_planner = user.model_copy(
        update={"agent_id": "planner", "agent_run_id": new_id("agent_run")}
    )
    assert "Plan: split the brief into revenue and cost." in await _memory_texts(
        container, later_planner, q
    )


async def test_shared_group_memory_corroboration_and_conflict(container, uow_factory) -> None:
    user = _user_ctx(agent_group_id="crew")
    await _thread(container, uow_factory, user)
    analyst = user.model_copy(update={"agent_id": "analyst", "agent_run_id": new_id("agent_run")})
    auditor = user.model_copy(update={"agent_id": "auditor", "agent_run_id": new_id("agent_run")})
    share = ProcessingHints(visibility=Visibility.AGENT_GROUP, memory_type=MemoryType.SHARED)
    # 1. corroboration: two agents report the same fact -> one memory, two contributors
    fact = "Revenue was EUR 412 million in FY26."
    await _observe(container, uow_factory, analyst, fact, kind=ObservationKind.EVENT, hints=share)
    await _observe(container, uow_factory, auditor, fact, kind=ObservationKind.EVENT, hints=share)
    async with uow_factory() as uow:
        shared = await container.services["memory"].list_memories(uow, auditor)
    revenue = next(m for m in shared if "412" in m.content)
    assert revenue.visibility is Visibility.AGENT_GROUP and revenue.reinforcement_count == 2
    # Agent principals are bound to the user they run for ("agent:<user>/<agent_id>"): the
    # agent_id arrives unauthenticated in the request body, so the same agent_id acting for
    # a different user must not be the same principal.
    assert revenue.system_metadata["contributors"] == ["agent:u1/auditor"]
    assert revenue.owner_principal == "agent:u1/analyst" and revenue.confidence >= 0.7
    # 2. conflict: a different value for a single-valued slot from another agent is kept as a
    #    contradiction, not silently superseded (no agent overrides another's finding)
    await _observe(
        container,
        uow_factory,
        analyst,
        "My manager is Dana.",
        kind=ObservationKind.EVENT,
        hints=share,
    )
    await _observe(
        container,
        uow_factory,
        auditor,
        "My manager is Lee.",
        kind=ObservationKind.EVENT,
        hints=share,
    )
    async with uow_factory() as uow:
        shared = await container.services["memory"].list_memories(uow, analyst)
    managers = [m for m in shared if m.predicate == "manager"]
    assert len(managers) == 2 and all(m.temporal.status is TemporalStatus.CURRENT for m in managers)
    assert {m.object for m in managers} == {"dana", "lee"}
    a, b = managers
    assert a.memory_id in b.temporal.contradicts or b.memory_id in a.temporal.contradicts
    # both are retrievable by any group member, flagged in the evidence report
    res = await container.services["retrieval"].retrieve(auditor, "who is my manager?")
    got = [
        c for c in res.candidates if c.kind == "memory" and c.payload.get("predicate") == "manager"
    ]
    assert len(got) == 2 and any(c.payload["contradicts"] for c in got)
    assert any("conflicting memories" in n for n in res.diagnostics["evidence"]["notes"])
    # the same agent correcting itself still supersedes
    await _observe(
        container,
        uow_factory,
        analyst,
        "Actually, my manager is now Sam.",
        kind=ObservationKind.EVENT,
        hints=share,
    )
    async with uow_factory() as uow:
        shared = await container.services["memory"].list_memories(uow, analyst)
    assert {m.object for m in shared if m.predicate == "manager"} == {"sam", "lee"}
    # outsiders of the agent group see none of it
    outsider = user.model_copy(update={"agent_group_id": None, "user_id": "u2"})
    assert await _memory_texts(container, outsider, "revenue FY26 manager") == set()


async def test_agent_chatter_does_not_pollute_user_history_or_memory(
    container, uow_factory
) -> None:
    user = _user_ctx()
    await _thread(container, uow_factory, user)
    agent = user.model_copy(update={"agent_id": "planner", "agent_run_id": new_id("agent_run")})
    conv = container.services["conversation"]
    async with uow_factory() as uow:
        await conv.append_message(
            uow,
            agent,
            role=MessageRole.AGENT,
            kind=MessageKind.INTERNAL,
            content="Thinking: my timezone is UTC and I prefer JSON output.",
        )
        await conv.append_message(
            uow, user, role=MessageRole.ASSISTANT, content="Here is the brief outline."
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    async with uow_factory() as uow:
        visible = await conv.list_messages(uow, user, user.thread_id)
        everything = await conv.list_messages(uow, user, user.thread_id, include_internal=True)
        user_memories = await container.services["memory"].list_memories(uow, user)
    assert [m.kind for m in visible] == [MessageKind.VISIBLE, MessageKind.VISIBLE]
    assert len(everything) == 3
    # nothing the agent "thought" became a USER memory of the human
    assert not any("timezone" in (m.predicate or "") for m in user_memories)
    bundle = await container.services["context_builder"].build(user, "what did I ask for?")
    assert "Thinking:" not in bundle.conversation.rendered
    assert all("Thinking:" not in m.text for m in bundle.memories)
