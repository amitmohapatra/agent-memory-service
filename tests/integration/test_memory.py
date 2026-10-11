"""Memory intelligence end to end: observation -> pipeline -> memories rows -> index ->
recall; supersession; forget; isolation of agent-private memories; expiry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from benchmark.common import submit_observation
from sqlalchemy import text

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    ObservationKind,
    QueryType,
    StatementKind,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.ids import new_id
from memory_service.domain.memory import statement_kind_of
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
U2 = MemoryExecutionContext(tenant_id="acme", user_id="u2", workspace_id="ws1")


async def _observe(
    container, uow_factory, ctx, content, *, kind=ObservationKind.MESSAGE, hints=None
):
    register_handlers(container)
    async with uow_factory() as uow:
        ack = await submit_observation(uow, ctx, kind=kind, content=content, hints=hints)
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
    from memory_service.modules.context.builder import candidate_to_item

    retrieved = next(c for c in res.candidates if c.record_id == tz.memory_id)
    assert candidate_to_item(retrieved).evidence == tz.evidence
    # exact identifier lookup by memory id, with visibility enforced
    exact = await engine.retrieve(U1, f"show {tz.memory_id}")
    assert exact.candidates[0].record_id == tz.memory_id  # the exact hit leads
    assert candidate_to_item(exact.candidates[0]).evidence == tz.evidence
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


async def test_moving_supersedes_where_the_user_lived(container, uow_factory) -> None:
    """ "I moved to Austin" replaces "I live in Seattle": one current home, the old one kept as
    history, and only the new one recalled."""
    await _observe(container, uow_factory, U1, "I live in Seattle.")
    [seattle] = [
        m for m in await _memories(uow_factory, U1, container) if m.predicate == "lives_in"
    ]
    await _observe(container, uow_factory, U1, "I moved to Austin last month.")
    homes = [m for m in await _memories(uow_factory, U1, container) if m.predicate == "lives_in"]
    assert [(m.object or "").lower() for m in homes] == ["austin"]
    assert homes[0].temporal.supersedes == seattle.memory_id
    history = await _memories(uow_factory, U1, container, include_superseded=True)
    old = next(m for m in history if m.memory_id == seattle.memory_id)
    assert old.temporal.status is TemporalStatus.SUPERSEDED
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "where do I live", kinds=("memory",))
    texts = " ".join(c.text for c in res.candidates)
    assert "Austin" in texts
    assert seattle.memory_id not in {c.record_id for c in res.candidates}


async def test_a_standing_rule_does_not_expire(container, uow_factory) -> None:
    await _observe(container, uow_factory, U1, "Never suggest recipes with cilantro.")
    await _observe(container, uow_factory, U1, "Do not invent a sales number.")
    by_predicate = {m.predicate: m for m in await _memories(uow_factory, U1, container)}
    rule, instruction = by_predicate["rule"], by_predicate["instruction"]
    assert rule.lifetime is Lifetime.LONG_TERM
    assert rule.system_metadata.get("expires_at") is None, "a standing rule has no TTL"
    assert instruction.lifetime is Lifetime.SHORT_TERM
    assert instruction.system_metadata.get("expires_at") is not None


async def test_a_rule_scoped_to_a_request_is_stored_not_read_as_a_question(
    container, uow_factory
) -> None:
    """Audit case 8: "When I ask ..." opened like a question and nothing was stored at all."""
    for said in (
        "Whenever I ask for a stock audit, always format the response as a markdown table "
        "with columns for: SKU, Item Name, Current Stock, Reorder Threshold, and Action "
        "Required.",
        "When I ask for a sales report, always break it down by region.",
    ):
        await _observe(container, uow_factory, U1, said)
    rules = [m for m in await _memories(uow_factory, U1, container) if m.predicate == "rule"]
    assert len(rules) == 2, [m.content for m in rules]
    for rule in rules:
        assert rule.lifetime is Lifetime.LONG_TERM and rule.memory_type is MemoryType.PREFERENCE
        assert statement_kind_of(rule.system_metadata) is StatementKind.RULE
        trigger = rule.system_metadata["rule_trigger"].lower()
        assert trigger.startswith(("whenever i ask", "when i ask")), trigger


async def test_a_rule_with_text_before_never_keeps_its_exception(container, uow_factory) -> None:
    """Audit case 11: a rule failed when anything preceded "Never", and its exception clause
    was not kept."""
    await _observe(
        container,
        uow_factory,
        U1,
        "For my weekly category overviews: never include items with a stock level of zero "
        "unless I specifically type 'include out of stock'.",
    )
    [rule] = [m for m in await _memories(uow_factory, U1, container) if m.predicate == "rule"]
    assert rule.lifetime is Lifetime.LONG_TERM
    assert statement_kind_of(rule.system_metadata) is StatementKind.CONDITIONAL_RULE
    assert rule.system_metadata["rule_exception"] == (
        "unless I specifically type 'include out of stock'"
    )
    assert rule.system_metadata.get("expires_at") is None


async def test_a_decision_follows_its_author_and_is_shared_only_when_asked(
    container, uow_factory
) -> None:
    """A decision the user records is theirs, like anything else they say: it reaches their
    other conversations and not another person, even one granted the same thread. Sharing it
    with the thread's participants is a choice, ``visibility=THREAD``, and then it stays in
    that thread for its author too."""
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
    await _observe(
        container,
        uow_factory,
        a,
        "We decided to ship the Kafka migration on Friday.",
        kind=ObservationKind.DECISION,
        hints=ProcessingHints(visibility=Visibility.THREAD),
    )
    mems = await _memories(uow_factory, a, container)
    decided = {m.content: m for m in mems if m.predicate == "decided"}
    mine = decided["We decided to use PostgreSQL instead of MongoDB."]
    shared = decided["We decided to ship the Kafka migration on Friday."]
    assert mine.visibility is Visibility.USER and mine.scope.thread_id == thread
    assert shared.visibility is Visibility.THREAD and shared.scope.thread_id == thread
    assert mine.system_metadata["category"] == shared.system_metadata["category"] == "decision"
    engine = container.services["retrieval"]

    async def recalled(who, query: str) -> set[str]:
        got = await engine.retrieve(who, query, kinds=("memory",))
        return {c.record_id for c in got.candidates if c.kind == "memory"}

    elsewhere = U1.model_copy(update={"thread_id": new_id("thread")})
    assert mine.memory_id in await recalled(elsewhere, "why did we decide on PostgreSQL?")
    assert shared.memory_id not in await recalled(elsewhere, "when does the Kafka migration ship?")
    # the participant reads what was shared with the thread, and nothing that was a's own
    assert shared.memory_id in await recalled(b, "when does the Kafka migration ship?")
    readable_by_b = await recalled(b, "why did we decide on PostgreSQL?")
    assert mine.memory_id not in readable_by_b
    assert readable_by_b <= {m.memory_id for m in mems if m.visibility is Visibility.THREAD}
    stranger = U2.model_copy(update={"user_id": "u3"})
    for query in ("why did we decide on PostgreSQL?", "when does the Kafka migration ship?"):
        assert await recalled(stranger, query) == set()


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
    graph = container.services["graph"]
    before_expiry = await graph.query(U1, entities=[task.subject], hops=1)
    assert any(r.memory_id == task.memory_id for r in before_expiry.relations)
    async with container.database.engine.begin() as conn:
        await conn.execute(
            text("UPDATE memories SET expires_at = :t WHERE memory_id = :id"),
            {"t": datetime.now(UTC) - timedelta(seconds=1), "id": task.memory_id},
        )
    await container.tasks.run_periodic("periodic.memory_expire")
    await container.tasks.drain()  # durable search and graph projection cleanup
    after_expiry = await graph.query(U1, entities=[task.subject], hops=1)
    assert not any(r.memory_id == task.memory_id for r in after_expiry.relations)
    # the task lapsed; what the user said is still on record, as said and never as a task
    [said] = await _memories(uow_factory, U1, container)
    assert said.system_metadata["category"] == "verbatim_turn"
    assert said.content == "Remind me to follow up with legal by Friday."
    found = await engine.retrieve(U1, "follow up with legal", kinds=("memory",))
    assert [c.record_id for c in found.candidates] == [said.memory_id]
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
