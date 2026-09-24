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
