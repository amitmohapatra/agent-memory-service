"""Memory intelligence end to end: observation -> pipeline -> memories rows -> index ->
recall; supersession; forget; isolation of agent-private memories; expiry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    ObservationKind,
    QueryType,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.ids import new_id
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
U2 = MemoryExecutionContext(tenant_id="acme", user_id="u2", workspace_id="ws1")


async def _observe(
    container, uow_factory, ctx, content, *, kind=ObservationKind.MESSAGE, hints=None
):
    register_handlers(container)
    service = container.services["memory"]
    async with uow_factory() as uow:
        ack = await service.submit_observation(uow, ctx, kind=kind, content=content, hints=hints)
        await uow.commit()
    await container.tasks.drain()  # process_observation
    await container.tasks.drain()  # memory.index
    return ack


async def _memories(uow_factory, ctx, container, **kw):
    async with uow_factory() as uow:
        return await container.services["memory"].list_memories(uow, ctx, **kw)


async def test_observation_becomes_memories_and_is_recallable(container, uow_factory) -> None:
    ack = await _observe(
        container,
        uow_factory,
        U1,
        "My name is Amit and my timezone is Europe/Berlin. I prefer concise answers with code.",
    )
    assert ack.job_ids
    async with uow_factory() as uow:
        obs = await uow.observations.get("acme", ack.observation_id)
    assert obs is not None and obs.processed_at is not None
    mems = await _memories(uow_factory, U1, container)
    by_pred = {m.predicate: m for m in mems}
    assert {"name", "timezone", "prefers"} <= set(by_pred)
    tz = by_pred["timezone"]
    assert tz.memory_type is MemoryType.USER and tz.lifetime is Lifetime.LONG_TERM
    assert tz.visibility is Visibility.USER and tz.object == "europe/berlin"
    assert tz.evidence and tz.evidence[0].source_type == "message"
    assert tz.system_metadata["visibility_keys"] == ["user:acme/u1", "principal:acme/user:u1"]
    # indexed: recall over the memories collection with a memory-only query
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "what is my timezone?", kinds=("memory",))
    assert res.routed.query_type is QueryType.USER_MEMORY
    assert res.candidates and res.candidates[0].kind == "memory"
    assert any("Europe/Berlin" in c.text for c in res.candidates)
    # exact identifier lookup by memory id, with visibility enforced
    exact = await engine.retrieve(U1, f"show {tz.memory_id}")
    assert [c.record_id for c in exact.candidates] == [tz.memory_id]
    assert (await engine.retrieve(U2, f"show {tz.memory_id}")).candidates == []
    # another user sees nothing of u1's user-level memories
    assert (await engine.retrieve(U2, "what is my timezone?", kinds=("memory",))).candidates == []
    assert await _memories(uow_factory, U2, container) == []


async def test_reinforce_supersede_and_history(container, uow_factory) -> None:
    await _observe(container, uow_factory, U1, "My timezone is Europe/Berlin.")
    await _observe(container, uow_factory, U1, "my timezone is Europe/Berlin")  # same, reinforce
    mems = await _memories(uow_factory, U1, container)
    assert len(mems) == 1 and mems[0].reinforcement_count == 2
    first_id = mems[0].memory_id
    # new value for a single-valued slot -> supersede
    await _observe(container, uow_factory, U1, "Actually, my timezone is now America/New_York.")
    current = await _memories(uow_factory, U1, container)
    assert len(current) == 1 and current[0].object == "america/new_york"
    assert current[0].temporal.supersedes == first_id
    history = await _memories(uow_factory, U1, container, include_superseded=True)
    old = next(m for m in history if m.memory_id == first_id)
    assert old.temporal.status is TemporalStatus.SUPERSEDED
    assert old.temporal.superseded_by == current[0].memory_id
    assert old.temporal.valid_to is not None
    # only the current value is searchable
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "my timezone", kinds=("memory",))
    assert [c.record_id for c in res.candidates] == [current[0].memory_id]
    # distinct preferences are NOT merged (no false merge)
    await _observe(container, uow_factory, U1, "I prefer tabs over spaces.")
    await _observe(container, uow_factory, U1, "I prefer dark mode in the editor.")
    prefs = [m for m in await _memories(uow_factory, U1, container) if m.predicate == "prefers"]
    assert len(prefs) == 2
    # but an explicit replacement of a preference does supersede it
    await _observe(container, uow_factory, U1, "I prefer spaces instead of tabs now.")
    prefs = [m for m in await _memories(uow_factory, U1, container) if m.predicate == "prefers"]
    assert {p.object for p in prefs} == {"spaces instead of tabs", "dark mode in the editor"}


async def test_decisions_are_thread_scoped_and_shared_in_thread(container, uow_factory) -> None:
    thread = new_id("thread")
    a = U1.model_copy(
        update={"thread_id": thread, "session_id": new_id("session"), "turn_id": new_id("turn")}
    )
    b = U2.model_copy(
        update={"thread_id": thread, "session_id": new_id("session"), "turn_id": new_id("turn")}
    )
    async with uow_factory() as uow:  # create the thread + grant both users
        conv = container.services["conversation"]
        from memory_service.domain.enums import MessageRole

        await conv.append_message(uow, a, role=MessageRole.USER, content="kick-off")
        await uow.commit()
    await container.services["authz"].grant_thread(b, thread, workspace_id="ws1")
    await _observe(
        container,
        uow_factory,
        a,
        "We decided to use PostgreSQL instead of MongoDB.",
        kind=ObservationKind.DECISION,
    )
    mems = await _memories(uow_factory, a, container)
    decision = next(m for m in mems if m.predicate == "decided")
    assert decision.visibility is Visibility.THREAD and decision.scope.thread_id == thread
    assert decision.system_metadata["category"] == "decision"
    # thread participant b can recall it; a stranger outside the thread cannot
    engine = container.services["retrieval"]
    got = await engine.retrieve(b, "why did we decide on PostgreSQL?", kinds=("memory",))
    assert [c.record_id for c in got.candidates if c.kind == "memory"] == [decision.memory_id]
    stranger = U2.model_copy(update={"user_id": "u3"})
    assert (
        await engine.retrieve(stranger, "why did we decide on PostgreSQL?", kinds=("memory",))
    ).candidates == []


async def test_agent_private_memories_stay_private(container, uow_factory) -> None:
    thread = new_id("thread")
    user = U1.model_copy(
        update={"thread_id": thread, "session_id": new_id("session"), "turn_id": new_id("turn")}
    )
    async with uow_factory() as uow:  # the thread exists and the user owns it
        from memory_service.domain.enums import MessageRole

        await container.services["conversation"].append_message(
            uow, user, role=MessageRole.USER, content="plan the report"
        )
        await uow.commit()
    agent = user.model_copy(update={"agent_id": "planner", "agent_run_id": new_id("agent_run")})
    other_agent = user.model_copy(
        update={"agent_id": "writer", "agent_run_id": new_id("agent_run")}
    )
    await _observe(
        container,
        uow_factory,
        agent,
        "Intermediate plan: split the report into revenue and cost sections.",
        kind=ObservationKind.AGENT_RESULT,
    )
    mine = await _memories(uow_factory, agent, container)
    # The user's own turn is kept verbatim now and is visible inside the thread, so the agent
    # sees that too; this test is about the agent's own working note, which is the thing that
    # must not escape its run.
    notes = [m for m in mine if m.memory_type is MemoryType.AGENT]
    # working notes of a run stay with that run (RUN: the run, its children, the writer)
    assert len(notes) == 1 and notes[0].visibility is Visibility.RUN
    assert notes[0].lifetime is Lifetime.SHORT_TERM
    assert notes[0].system_metadata["expires_at"] is not None
    engine = container.services["retrieval"]
    q = "plan for the report sections"

    async def _sees_the_note(who) -> bool:
        """Whether this caller can retrieve the agent's private working note.

        Not "retrieves nothing": the user's own turn is kept verbatim and is visible to
        everyone in the thread, so both the other agent and the user legitimately match on
        it. What must not escape the run is the note itself.
        """
        found = (await engine.retrieve(who, q, kinds=("memory",))).candidates
        return any("Intermediate plan" in c.text for c in found)

    assert await _sees_the_note(agent)
    assert not await _sees_the_note(other_agent)
    assert not await _sees_the_note(user)
    # hints can widen visibility explicitly (shared with the thread)
    await _observe(
        container,
        uow_factory,
        agent,
        "Shared finding: revenue fell 4% because of Legacy Services.",
        kind=ObservationKind.AGENT_RESULT,
        hints=ProcessingHints(visibility=Visibility.THREAD, memory_type=MemoryType.SHARED),
    )
    assert (
        await engine.retrieve(other_agent, "why did revenue fall", kinds=("memory",))
    ).candidates
    assert (await engine.retrieve(user, "why did revenue fall", kinds=("memory",))).candidates


async def test_reading_an_agent_memory_requires_naming_the_agent(container, uow_factory) -> None:
    """An agent's memory is reachable only by that agent's principal — on every read path.

    Dropping ``agent_id`` is not a narrower view of the same identity, it is a *different*
    principal (``user:u1`` rather than ``agent:u1/planner``), so the listing (anchored on
    scope keys) returns nothing and the direct read (checked against visibility keys) is
    denied. Reproduced against the live service, where the two endpoints disagreeing looked
    like a bug until the missing query parameter turned out to be the whole difference:
    ``GET /v1/memories`` and ``GET /v1/memories/{id}`` both take ``?agent_id=``.
    """
    thread = new_id("thread")
    user = U1.model_copy(update={"thread_id": thread})
    agent = user.model_copy(update={"agent_id": "planner", "agent_run_id": new_id("agent_run")})
    await _observe(
        container,
        uow_factory,
        agent,
        "Intermediate plan: audit the Q3 supplier contracts first.",
        kind=ObservationKind.AGENT_RESULT,
    )
    mine = await _memories(uow_factory, agent, container)
    assert len(mine) == 1
    memory_id = mine[0].memory_id
    assert await _memories(uow_factory, user, container) == []

    service = container.services["memory"]
    async with uow_factory() as uow:
        assert (await service.get_memory(uow, agent, memory_id)).memory_id == memory_id
    async with uow_factory() as uow:
        with pytest.raises(ScopeDenied) as denied:
            await service.get_memory(uow, user, memory_id)
    # The denial names the caller's own principal, which is the answer to "why 403?" —
    # and says nothing about the memory, so it cannot confirm that it exists.
    assert denied.value.details == {"principal": user.principal_id}
    assert memory_id not in str(denied.value.details)


async def test_forget_and_expiry(container, uow_factory) -> None:
    await _observe(container, uow_factory, U1, "My favourite editor is neovim.")
    mems = await _memories(uow_factory, U1, container)
    fav = next(m for m in mems if m.predicate.startswith("favourite"))
    engine = container.services["retrieval"]
    assert (await engine.retrieve(U1, "favourite editor", kinds=("memory",))).candidates
    service = container.services["memory"]
    # another user cannot forget it (not even see it)
    async with uow_factory() as uow:
        with pytest.raises(ScopeDenied):
            await service.forget(uow, U2, fav.memory_id)
    async with uow_factory() as uow:
        await service.forget(uow, U1, fav.memory_id)
        await uow.commit()
    await container.tasks.drain()  # index removal
    assert await _memories(uow_factory, U1, container) == []
    assert (await engine.retrieve(U1, "favourite editor", kinds=("memory",))).candidates == []
    assert (await engine.retrieve(U1, f"show {fav.memory_id}")).candidates == []
    # short-term memory expiry: backdate expires_at and run the periodic job
    await _observe(container, uow_factory, U1, "Remind me to follow up with legal by Friday.")
    task = next(
        m for m in await _memories(uow_factory, U1, container) if m.memory_type is MemoryType.TASK
    )
    async with container.database.engine.begin() as conn:
        await conn.execute(
            text("UPDATE memories SET expires_at = :t WHERE memory_id = :id"),
            {"t": datetime.now(UTC) - timedelta(seconds=1), "id": task.memory_id},
        )
    await container.tasks.run_periodic("periodic.memory_expire")
    assert await _memories(uow_factory, U1, container) == []
    assert (await engine.retrieve(U1, "follow up with legal", kinds=("memory",))).candidates == []
    history = await _memories(uow_factory, U1, container, include_superseded=True)
    assert (
        next(m for m in history if m.memory_id == task.memory_id).temporal.status
        is TemporalStatus.EXPIRED
    )


async def test_pipeline_is_idempotent_on_replay(container, uow_factory) -> None:
    ack = await _observe(container, uow_factory, U1, "I work at ACME Corp.")
    pipeline = container.services["observation_pipeline"]
    again = await pipeline.run({"tenant_id": "acme", "observation_id": ack.observation_id})
    assert again == []
    assert len(await _memories(uow_factory, U1, container)) == 1


async def test_an_agent_repeating_itself_never_raises_confidence(container, uow_factory) -> None:
    """The recall loop, closed.

    Retrieval injects a memory into the prompt, the agent's answer restates it, the answer
    comes back as an observation, and the memory it came from is reinforced. Each turn added
    +0.05 with no new evidence, so anything the agent was once told drifted to certainty just
    by being mentioned again. The repeat is still recorded — it just cannot vouch for itself.
    """
    ctx = MemoryExecutionContext(
        tenant_id="acme", user_id="u1", workspace_id="ws1", agent_id="analyst"
    )
    answer = "The Q3 revenue figure is 4.2 million."
    await _observe(container, uow_factory, ctx, answer, kind=ObservationKind.AGENT_RESULT)
    first = [m for m in await _memories(uow_factory, ctx, container) if m.content == answer]
    assert len(first) == 1
    baseline = first[0].confidence

    for _ in range(6):
        await _observe(container, uow_factory, ctx, answer, kind=ObservationKind.AGENT_RESULT)

    after = next(m for m in await _memories(uow_factory, ctx, container) if m.content == answer)
    assert after.confidence == baseline, "six self-repeats must not make it more certain"
    assert after.reinforcement_count == 7, "the repeats are still counted"
    assert after.system_metadata.get("echoes") == 6, "...and named as echoes"


async def test_a_real_source_still_corroborates(container, uow_factory) -> None:
    """The guard is about the agent's own output, not about repetition."""
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u9", workspace_id="ws1")
    await _observe(container, uow_factory, ctx, "My timezone is Europe/Berlin.")
    before = (await _memories(uow_factory, ctx, container))[0].confidence
    await _observe(container, uow_factory, ctx, "my timezone is Europe/Berlin")
    after = (await _memories(uow_factory, ctx, container))[0]
    assert after.reinforcement_count == 2
    assert after.confidence > before, "a user saying it twice is still corroboration"
