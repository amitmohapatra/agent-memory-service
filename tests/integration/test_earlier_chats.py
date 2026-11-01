"""What the user's earlier chats said, the assistant's own answers included, recalled in a later
chat (ADR 0044): "what was that number you worked out on Monday?".

The user's memories hold their own words (ADR 0036), never the assistant's text, so a figure
the assistant computed in an earlier chat was reachable only by a message search the model had
to ask for by kind. Two paths now reach it, as ChatGPT's "reference chat history" and Claude's
"search past chats" do: the pushed context of a question that points to an earlier chat carries
the best matching lines of the user's earlier conversations, and ``memory_search`` reads them
when such a question names no kinds. Only the user's own conversations, in the request's
workspace, never a deleted one; a corrected answer is marked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

MONDAY = datetime(2026, 10, 26, 10, 0, tzinfo=UTC)
TUESDAY = datetime(2026, 10, 27, 15, 0, tzinfo=UTC)
ASKED = (
    "What reorder quantity should we place for the Merino Hoodie in the north region? Use the "
    "28-day average, a 14-day lead time and a 7-day review period."
)
WORKED_OUT = (
    "For the Merino Hoodie in the north, the 28-day average is 11 units a day, so 21 days of "
    "cover need 230 units; with 93 on hand the reorder quantity works out to 137 units."
)
FRIDAY_ASK = "What was that reorder number you worked out on Monday for the Merino Hoodie?"
#: the same agent's answer for another user
BOBS = "For your stores, the north reorder quantity for the Merino Hoodie is 555 units."


def _chat(user: str = "ann", workspace: str = "ws1") -> MemoryExecutionContext:
    """A new conversation of ``user`` with the same agent acting for them."""
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id=user,
        workspace_id=workspace,
        agent_id="analyst",
        agent_run_id="run_1",
        thread_id=new_id("thread"),
    )


async def _say(
    container, uow_factory, ctx, content: str, *, role=MessageRole.USER, when=None
) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["conversation"].append_message(
            uow, ctx, role=role, content=content, occurred_at=when
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _monday(container, uow_factory, user: str = "ann", workspace: str = "ws1"):
    """Monday's chat: the question and the assistant's worked-out figure; then Tuesday's chat
    about something else, which shares words but does not answer."""
    monday = _chat(user, workspace)
    await _say(container, uow_factory, monday, ASKED, when=MONDAY)
    await _say(container, uow_factory, monday, WORKED_OUT, role=MessageRole.ASSISTANT, when=MONDAY)
    tuesday = _chat(user, workspace)
    await _say(
        container,
        uow_factory,
        tuesday,
        "How many Cashmere Scarf units did we sell in the south last week?",
        when=TUESDAY,
    )
    await _say(
        container,
        uow_factory,
        tuesday,
        "The south sold 412 Cashmere Scarf units last week, the most of any region.",
        role=MessageRole.ASSISTANT,
        when=TUESDAY,
    )
    return monday


async def _context(container, ctx, query: str):
    return await container.services["context_builder"].build(ctx, query)


async def _searched(container, ctx, query: str) -> list[dict]:
    """``memory_search`` as the model calls it: the question's words, no kinds named."""
    found = await container.services["agent_tools"].call(ctx, "memory_search", {"query": query})
    await container.tasks.drain()
    return found


async def test_mondays_worked_out_figure_is_in_fridays_context(container, uow_factory) -> None:
    monday = await _monday(container, uow_factory)
    friday = _chat()

    bundle = await _context(container, friday, FRIDAY_ASK)
    rendered = bundle.render()
    [line] = [c for c in bundle.earlier_chats if "137" in c.text]
    assert line.role is MessageRole.ASSISTANT and line.thread_id == monday.thread_id
    assert line.said_at.date() == MONDAY.date() and not line.corrected
    assert "## Earlier conversations" in rendered
    said = next(row for row in rendered.splitlines() if "137 units" in row)
    assert "2026-10-26" in said and "Monday" in said, "dated, with its weekday"
    assert monday.thread_id in said and "the assistant said" in said, "labelled as its answer"
    assert "412" not in rendered, "another chat that shares words but does not answer"
    # the user's own words of that chat are a memory already: shown once, not again as a line
    assert all(c.role is MessageRole.ASSISTANT for c in bundle.earlier_chats)


async def test_memory_search_reads_the_earlier_answers_when_the_question_points_to_them(
    container, uow_factory
) -> None:
    monday = await _monday(container, uow_factory)
    found = await _searched(container, _chat(), FRIDAY_ASK)
    [item] = [i for i in found if "137" in i["text"]]
    assert item["kind"] == "message" and item["thread_id"] == monday.thread_id
    assert item["text"].startswith("ASSISTANT: ") and item["observed_on"] == "2026-10-26"


async def test_a_question_that_does_not_point_back_reads_no_earlier_chat(
    container, uow_factory
) -> None:
    """The cost stays where it was for every other question: no lines, no message search."""
    await _monday(container, uow_factory)
    friday = _chat()
    ask = "What reorder quantity should we place for the Merino Hoodie in the north?"

    bundle = await _context(container, friday, ask)
    assert bundle.earlier_chats == [] and "## Earlier conversations" not in bundle.render()
    assert {i["kind"] for i in await _searched(container, friday, ask)} <= {"memory", "chunk"}


async def test_another_user_of_the_same_agent_gets_nothing(container, uow_factory) -> None:
    await _monday(container, uow_factory)
    bob = _chat("bob")
    await _say(container, uow_factory, bob, "Merino Hoodie north reorder?", when=MONDAY)
    await _say(container, uow_factory, bob, BOBS, role=MessageRole.ASSISTANT, when=MONDAY)

    bobs_friday = _chat("bob")
    bundle = await _context(container, bobs_friday, FRIDAY_ASK)
    assert "137" not in bundle.render() and "555" in bundle.render(), "bob's own chat only"
    assert "137" not in str(await _searched(container, bobs_friday, FRIDAY_ASK))

    anns = await _context(container, _chat(), FRIDAY_ASK)
    assert "555" not in anns.render() and "555" not in str(
        await _searched(container, _chat(), FRIDAY_ASK)
    )


async def test_another_workspace_of_the_same_user_reads_nothing(container, uow_factory) -> None:
    """Like a project's chats: the earlier conversations of the request's workspace only."""
    await _monday(container, uow_factory, workspace="ws1")
    elsewhere = _chat(workspace="ws2")
    assert "137" not in (await _context(container, elsewhere, FRIDAY_ASK)).render()
    assert "137" not in str(await _searched(container, elsewhere, FRIDAY_ASK))
    explicit = await container.services["agent_tools"].call(
        elsewhere, "memory_search", {"query": FRIDAY_ASK, "kinds": ["message", "episode"]}
    )
    assert "137" not in str(explicit)


async def test_a_deleted_chat_is_never_used(container, uow_factory) -> None:
    monday = await _monday(container, uow_factory)
    friday = _chat()
    assert "137" in (await _context(container, friday, FRIDAY_ASK)).render()  # cached now

    async with uow_factory() as uow:
        await container.services["conversation"].delete_thread(uow, monday, monday.thread_id)
        await uow.commit()
    await container.tasks.drain()

    assert "137" not in (await _context(container, friday, FRIDAY_ASK)).render(), (
        "not from the cached bundle either"
    )
    assert "137" not in str(await _searched(container, friday, FRIDAY_ASK))
    assert "137" not in (await _context(container, _chat(), FRIDAY_ASK)).render()


async def test_a_corrected_answer_comes_back_marked_corrected(container, uow_factory) -> None:
    monday = await _monday(container, uow_factory)
    fix = (
        "That's wrong for the Merino Hoodie — the north lead time is 21 days, so the reorder "
        "is 186 units."
    )
    await _say(container, uow_factory, monday, fix, when=MONDAY)
    friday = _chat()

    bundle = await _context(container, friday, FRIDAY_ASK)
    [wrong] = [c for c in bundle.earlier_chats if "137" in c.text]
    assert wrong.corrected is True
    said = next(row for row in bundle.render().splitlines() if "137 units" in row)
    assert "corrected" in said, "the line says it was corrected"
    assert "186" in bundle.render(), "the correction itself reaches the context"

    [item] = [i for i in await _searched(container, friday, FRIDAY_ASK) if "137" in i["text"]]
    assert item["superseded"] is True
