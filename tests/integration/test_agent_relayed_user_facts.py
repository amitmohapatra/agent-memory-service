"""A human's own fact stays the human's, even when an agent posted the turn.

`_apply_hints` re-typed any USER/PREFERENCE candidate to AGENT whenever the *observation
carried an agent_id* - which is true of every request from an agent harness, including the
human's own turn relayed by it. The preference was then anchored at ScopeLevel.AGENT, given
Visibility.RUN, and owned by `agent:<user>/<agent>`, so its visibility keys were
`run:<tenant>/<run>` and `principal:<tenant>/agent:<user>/<agent>` - neither of which a
plain user read carries. The human could never see their own stated preference again.

ADR 0013 scopes the rule to "an agent-authored observation". The role was on the observation
the whole time and nothing read it.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageKind, MessageRole
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

PREFERENCE = "I prefer one-sentence answers."


def _ctx(**extra) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id="u1",
        workspace_id="ws1",
        thread_id=new_id("thread"),
        session_id=new_id("session"),
        turn_id=new_id("turn"),
        **extra,
    )


async def _say(container, uow_factory, ctx, content, *, role, kind=MessageKind.VISIBLE):
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["conversation"].append_message(
            uow, ctx, role=role, kind=kind, content=content
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _user_visible(container, ctx) -> list:
    """What a plain user read returns - no agent lineage on the context."""
    plain = ctx.model_copy(update={"agent_id": None, "agent_run_id": None})
    res = await container.services["retrieval"].retrieve(plain, "answer length preference")
    return [c.text for c in res.candidates if c.kind == "memory"]


async def test_a_preference_the_user_stated_survives_an_agent_relaying_it(
    container, uow_factory
) -> None:
    """The defect. PlanSmart always sets agent_id; the human still said this."""
    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    await _say(container, uow_factory, ctx, PREFERENCE, role=MessageRole.USER)
    texts = await _user_visible(container, ctx)
    assert any("one-sentence" in t for t in texts), (
        f"the user cannot see their own preference; retrieved {texts}"
    )


async def test_the_same_preference_with_no_agent_on_the_request_is_unchanged(
    container, uow_factory
) -> None:
    """The control: this always worked, and must keep working."""
    ctx = _ctx()
    await _say(container, uow_factory, ctx, PREFERENCE, role=MessageRole.USER)
    texts = await _user_visible(container, ctx)
    assert any("one-sentence" in t for t in texts)


async def test_the_agents_own_note_still_does_not_become_the_users_memory(
    container, uow_factory
) -> None:
    """The behaviour the rule exists for, which must not regress."""
    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    await _say(
        container,
        uow_factory,
        ctx,
        "I prefer to summarise before planning.",
        role=MessageRole.AGENT,
        kind=MessageKind.INTERNAL,
    )
    texts = await _user_visible(container, ctx)
    assert not any("summarise before planning" in t for t in texts), (
        f"the agent's own note leaked into the user's memory; retrieved {texts}"
    )


async def test_the_user_may_forget_what_an_agent_relayed_for_them(container, uow_factory) -> None:
    """``forget`` is allowed to "the user an agent acts for" (its docstring, USAGE §4): what a
    harness relays for the user is owned by ``agent:<user>/<agent>``, and the user themselves
    was refused ("only the owner (or a tenant admin)"), so nobody but the agent could retract
    the user's own words."""
    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    await _say(container, uow_factory, ctx, PREFERENCE, role=MessageRole.USER)
    plain = ctx.model_copy(update={"agent_id": None, "agent_run_id": None})
    service = container.services["memory"]
    async with uow_factory() as uow:
        mine = [m for m in await service.list_memories(uow, plain) if "one-sentence" in m.content]
    assert mine and all(m.owner_principal == "agent:u1/plansmart" for m in mine), mine
    for memory in mine:  # the reading, and the turn kept beside it when there is one
        async with uow_factory() as uow:
            await service.forget(uow, plain, memory.memory_id)
            await uow.commit()
    await container.tasks.drain()  # the index job removes them from search
    assert not any("one-sentence" in t for t in await _user_visible(container, ctx))


async def test_another_user_still_may_not_forget_it(container, uow_factory) -> None:
    from memory_service.domain.errors import ScopeDenied

    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    await _say(container, uow_factory, ctx, PREFERENCE, role=MessageRole.USER)
    plain = ctx.model_copy(update={"agent_id": None, "agent_run_id": None})
    service = container.services["memory"]
    async with uow_factory() as uow:
        mine = [m for m in await service.list_memories(uow, plain) if "one-sentence" in m.content]
    other = plain.model_copy(update={"user_id": "u2"})
    async with uow_factory() as uow:
        with pytest.raises(ScopeDenied):
            await service.forget(uow, other, mine[0].memory_id)


async def _readings(container, uow_factory, ctx, needle: str) -> list:
    """What was read out of a turn, the turn kept verbatim beside it left out."""
    async with uow_factory() as uow:
        held = await container.services["memory"].list_memories(uow, ctx)
    return [
        m
        for m in held
        if needle in m.content and m.system_metadata.get("category") != "verbatim_turn"
    ]


async def test_forgetting_a_fact_read_from_the_users_words_asks_first(
    container, uow_factory
) -> None:
    """MEM-05 (BUG-M1). docs/memory.md: forgetting what the user said asks a person first.
    The preference read out of the turn a harness relays is owned by ``agent:<user>/<agent>``
    and is no ``said`` turn, so the agent retracted, restated or restored it unasked: it is
    still the user's words, whoever wrote the row."""
    from memory_service.domain.errors import ScopeDenied  # noqa: PLC0415

    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    await _say(container, uow_factory, ctx, PREFERENCE, role=MessageRole.USER)
    readings = await _readings(container, uow_factory, ctx, "one-sentence")
    assert readings and all(m.owner_principal == "agent:u1/plansmart" for m in readings)
    service = container.services["memory"]
    for memory in readings:
        attempts = (
            lambda uow, m=memory: service.forget(uow, ctx, m.memory_id),
            lambda uow, m=memory: service.supersede(
                uow, ctx, m.memory_id, content="Answers of any length.", reason="tidy"
            ),
            lambda uow, m=memory: service.restore(
                uow, ctx, m.memory_id, container.services["forgetting"]
            ),
        )
        for attempt in attempts:
            async with uow_factory() as uow:
                with pytest.raises(ScopeDenied) as refused:
                    await attempt(uow)
            assert refused.value.details["reason"] == "users_own_statement"
    assert await _readings(container, uow_factory, ctx, "one-sentence") == readings


async def test_what_the_agent_noted_or_reported_it_still_forgets_unasked(
    container, uow_factory
) -> None:
    """The neighbour that must not change: an agent's own note (``memory_remember``) and
    what it reported as its result are its own to forget."""
    from benchmark.common import submit_observation  # noqa: PLC0415

    from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind  # noqa: PLC0415

    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    service = container.services["memory"]
    async with uow_factory() as uow:
        await service.remember(
            uow,
            ctx,
            content="Draft the summary before the plan.",
            memory_type=MemoryType.AGENT,
            lifetime=Lifetime.LONG_TERM,
        )
        await submit_observation(
            uow,
            ctx,
            kind=ObservationKind.AGENT_RESULT,
            content="Intermediate plan: split the report into revenue and cost sections.",
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    noted = await _readings(container, uow_factory, ctx, "Draft the summary")
    reported = await _readings(container, uow_factory, ctx, "Intermediate plan")
    assert noted and reported, (noted, reported)
    for memory in (*noted, *reported):
        async with uow_factory() as uow:
            assert await service.forget(uow, ctx, memory.memory_id) is not None
            await uow.commit()


async def test_a_reading_stored_before_its_origin_was_recorded_is_backfilled(
    container, uow_factory
) -> None:
    """Rows written before ``said_by_user`` was recorded: migration 0036 marks what was read
    out of a user's relayed turn (an agent-owned memory whose evidence is a USER message), and
    leaves the agent's own note and its reported result as they were."""
    import asyncio  # noqa: PLC0415

    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415
    from benchmark.common import submit_observation  # noqa: PLC0415

    from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind  # noqa: PLC0415
    from memory_service.modules.memory.pipeline import said_by_a_person  # noqa: PLC0415
    from tests.conftest import DB_URL  # noqa: PLC0415

    ctx = _ctx(agent_id="plansmart", agent_run_id=new_id("agent_run"))
    await _say(container, uow_factory, ctx, PREFERENCE, role=MessageRole.USER)
    async with uow_factory() as uow:
        await container.services["memory"].remember(
            uow,
            ctx,
            content="Draft the summary before the plan.",
            memory_type=MemoryType.AGENT,
            lifetime=Lifetime.LONG_TERM,
        )
        await submit_observation(
            uow, ctx, kind=ObservationKind.AGENT_RESULT, content="Intermediate plan: two parts."
        )
        await uow.commit()
    await container.tasks.drain()
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", DB_URL)

    async def said(needle: str) -> list[bool]:
        return [said_by_a_person(m) for m in await _readings(container, uow_factory, ctx, needle)]

    await asyncio.to_thread(command.downgrade, cfg, "0035_beliefs")  # the rows as stored before
    try:
        assert await said("one-sentence") == [False], "a pre-fix reading is the agent's"
    finally:
        await asyncio.to_thread(command.upgrade, cfg, "head")
    assert await said("one-sentence") == [True]
    assert await said("Draft the summary") == [False]
    assert await said("Intermediate plan") == [False]
