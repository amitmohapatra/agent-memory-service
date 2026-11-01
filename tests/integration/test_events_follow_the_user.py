"""An event recorded in one of a user's conversations is read in all of them (ADR 0043).

"Change scheduled: host app-07 will be rebooted for kernel patching on Saturday at 02:00."
arrives as ``role: "EVENT"`` in one chat; "Will payroll be down this weekend?" is asked in a
new one, where "The payroll service runs on host app-07." was said weeks before. The event was
the first chat's (``THREAD``), so the second hop was out of reach. It now follows the user
the way their own words do (ADR 0036) - and only that user: another person, the same agent
acting for another person, an explicit ``THREAD`` and an event with no user keep the audiences
they had. A later event replaces an earlier one across conversations as a statement does
(ADR 0038).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from memory_service.domain.enums import MessageRole, ObservationKind, TemporalStatus, Visibility
from memory_service.domain.ids import new_id
from memory_service.domain.memory import replaced_by
from memory_service.domain.observation import ProcessingHints
from tests.integration.test_memory import _memories, _observe
from tests.integration.test_replacement_across_conversations import STATEMENTS, _scripted
from tests.integration.test_statements_follow_the_user import _chat, _context, _recalled, _say

pytestmark = pytest.mark.integration

HOST = "The payroll service runs on host app-07."
REBOOT = "Change scheduled: host app-07 will be rebooted for kernel patching on Saturday at 02:00."
ASK = "When is host app-07 rebooted for kernel patching?"


async def _event(container, uow_factory, ctx, content: str) -> None:
    await _say(container, uow_factory, ctx, content, role=MessageRole.EVENT)


async def _recorded(container, uow_factory, ctx, text: str, **kw):
    return [m for m in await _memories(uow_factory, ctx, container, **kw) if m.content == text]


async def test_an_event_in_one_chat_is_read_in_the_users_next(container, uow_factory) -> None:
    first, later = _chat(), _chat()
    await _say(container, uow_factory, first, HOST)
    await _event(container, uow_factory, first, REBOOT)

    assert "02:00" in await _recalled(container, later, ASK), "search, from a new conversation"
    assert "02:00" in (await _context(container, later, ASK)).render(), "and its context"

    recorded = await _recorded(container, uow_factory, first, REBOOT)
    assert recorded and all(m.visibility is Visibility.USER for m in recorded)
    assert all(
        m.system_metadata["visibility_keys"] == ["user:acme/u1", "principal:acme/user:u1"]
        for m in recorded
    )
    # the anchor stays where it was recorded (ADR 0036 decision 2)
    assert all(m.scope.thread_id == first.thread_id for m in recorded)


async def test_an_event_recorded_by_an_agent_follows_its_user(container, uow_factory) -> None:
    agent = _chat(agent_id="ops-assistant", agent_run_id=new_id("agent_run"))
    await _event(container, uow_factory, agent, REBOOT)
    assert "02:00" in await _recalled(container, _chat(), ASK), "the user, in a plain chat"
    again = _chat(agent_id="ops-assistant", agent_run_id=new_id("agent_run"))
    assert "02:00" in await _recalled(container, again, ASK), "the agent, acting for them again"


async def test_no_other_person_reads_a_users_event(container, uow_factory) -> None:
    await _event(container, uow_factory, _chat(), REBOOT)
    colleague = _chat("u2")  # same tenant, same workspace anchor, a different person
    assert "02:00" not in await _recalled(container, colleague, ASK)
    assert "02:00" not in (await _context(container, colleague, ASK)).render()


async def test_an_agents_other_users_never_read_one_users_event(container, uow_factory) -> None:
    for_u1 = _chat(agent_id="ops-assistant", agent_run_id=new_id("agent_run"))
    await _event(container, uow_factory, for_u1, REBOOT)
    for_u2 = _chat("u2", agent_id="ops-assistant", agent_run_id=new_id("agent_run"))
    assert "02:00" not in await _recalled(container, for_u2, ASK)
    assert "02:00" not in (await _context(container, for_u2, ASK)).render()


async def test_an_event_kept_to_its_conversation_stays_there(container, uow_factory) -> None:
    here = _chat()
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, here)
        await uow.commit()
    await _observe(
        container,
        uow_factory,
        here,
        REBOOT,
        kind=ObservationKind.EVENT,
        hints=ProcessingHints(visibility=Visibility.THREAD),
    )
    assert "02:00" in await _recalled(container, here, ASK)
    assert "02:00" not in await _recalled(container, _chat(), ASK)


async def test_an_event_without_a_user_keeps_its_audience(container, uow_factory) -> None:
    system = _chat(user_id=None)
    await _event(container, uow_factory, system, REBOOT)
    async with container.database.engine.connect() as conn:
        audiences = (
            await conn.execute(
                text("SELECT visibility FROM memories WHERE content = :c"), {"c": REBOOT}
            )
        ).scalars()
        assert set(audiences) == {Visibility.THREAD.value}, "the conversation's, as before"
    assert "02:00" not in await _recalled(container, _chat(), ASK)


# --- a later value replaces an earlier one, across conversations (ADR 0038) ------------------

OLD, _, WHO = STATEMENTS["en"]
#: one sentence: an event is kept sentence by sentence (no turn as said), and the scripted head
#: reads a contradiction only where one sentence names the later value
NEW = "We replaced Brightway with Corrimax as our supplier of shipping crates last week."


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (MessageRole.EVENT, MessageRole.EVENT),
        (MessageRole.USER, MessageRole.EVENT),
        (MessageRole.EVENT, MessageRole.USER),
    ],
    ids=["event-event", "statement-event", "event-statement"],
)
async def test_a_later_value_replaces_an_earlier_one_in_another_conversation(
    container, uow_factory, first, second
) -> None:
    _scripted(container, 0.97)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, OLD, role=first)
    await _say(container, uow_factory, may, NEW, role=second)

    old = await _recorded(container, uow_factory, march, OLD, include_superseded=True)
    assert old and all(m.temporal.status is TemporalStatus.SUPERSEDED for m in old)
    assert all(replaced_by(m.system_metadata) for m in old)
    recalled = await _recalled(container, _chat(), WHO)
    assert "Corrimax" in recalled and OLD not in recalled.split("\n")


async def test_another_persons_event_is_never_replaced(container, uow_factory) -> None:
    _scripted(container, 0.99)
    theirs, mine = _chat("u2"), _chat()
    await _event(container, uow_factory, theirs, OLD)
    await _event(container, uow_factory, mine, NEW)
    old = await _recorded(container, uow_factory, theirs, OLD)
    assert old and all(m.temporal.status is TemporalStatus.CURRENT for m in old)
    assert not any(replaced_by(m.system_metadata) for m in old)


@pytest.mark.parametrize("role", [MessageRole.EVENT, MessageRole.USER])
async def test_what_a_replaced_value_said_is_not_served_as_current(
    container, uow_factory, role
) -> None:
    """Three states of one order in three conversations: the earlier ones go to the history
    (an earlier value, ADR 0038), and no graph fact read from them is served beside the
    current one."""
    _scripted(container, 0.87, values=("has shipped", "was delivered"))
    chats = [_chat() for _ in range(3)]
    for chat, said in zip(
        chats,
        (
            "PO-88120 for 600 cases of canned tomatoes was confirmed by the supplier.",
            "PO-88120 has shipped from the supplier's warehouse.",
            "PO-88120 was delivered and received in full at DC-2.",
        ),
        strict=True,
    ):
        await _say(container, uow_factory, chat, said, role=role)
    bundle = await _context(container, chats[-1], "What's the status of PO-88120?")
    current = [i.text for i in (*bundle.memories, *bundle.knowledge, *bundle.graph_facts)]
    assert any("DC-2" in t for t in current), current
    assert not any("has shipped" in t for t in current), current
