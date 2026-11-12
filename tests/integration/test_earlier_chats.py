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
from memory_service.domain.conversation import MessageToolCall
from memory_service.domain.enums import MessageKind, MessageRole
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.retrieval import search

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


CARDIGAN = "Revenue for the Cashmere Cardigan Olive in Q3 2026 was £1,284.50."
CONTINUES = "And for the cardigan in that same colour but in cotton, same quarter?"


async def test_a_new_chat_that_continues_an_earlier_one_gets_its_answer(
    container, uow_factory
) -> None:
    """A new chat opened with "And for the cardigan in that same colour, same quarter?" has
    nothing of its own the words point at: the earlier chat has (break-and-fix bf3-mixed-05).
    The same words inside a chat that has its own earlier question read nothing more."""
    monday = _chat()
    await _say(
        container,
        uow_factory,
        monday,
        "What was revenue for the Cashmere Cardigan Olive in Q3 2026?",
    )
    await _say(container, uow_factory, monday, CARDIGAN, role=MessageRole.ASSISTANT)

    fresh = await _context(container, _chat(), CONTINUES)
    assert [c.text for c in fresh.earlier_chats] == [CARDIGAN]

    ongoing = _chat()
    await _say(container, uow_factory, ongoing, "What was revenue for denim jeans in Q3 2026?")
    await _say(container, uow_factory, ongoing, "£9,120.", role=MessageRole.ASSISTANT)
    assert (await _context(container, ongoing, CONTINUES)).earlier_chats == []
    plain = await _context(container, _chat(), "What was revenue for denim jeans in Q3 2026?")
    assert plain.earlier_chats == []


#: Monday's long tool result: every knitwear SKU's units, past what an answer repeats.
LIST = "sku,units\n" + "\n".join(f"SKU-{n:05d},{1000 - n}" for n in range(300))
TOP_ASK = "Which knitwear SKU sold the most units in Q3? Look at the whole list."
TOP_SAID = "SKU-00000 sold the most knitwear units in Q3: 1000."


async def _with_a_call(container, uow_factory, ctx) -> str:
    """Monday's question answered with one call: its whole result's message id."""
    register_handlers(container)
    conversation = container.services["conversation"]
    call = MessageToolCall(id="call_1", name="sales_summary", arguments={"group_by": "sku"})
    async with uow_factory() as uow:
        await conversation.append_message(uow, ctx, role=MessageRole.USER, content=TOP_ASK,
                                          occurred_at=MONDAY)  # fmt: skip
        await conversation.append_message(uow, ctx, role=MessageRole.ASSISTANT, content="",
                                          tool_calls=[call], occurred_at=MONDAY)  # fmt: skip
        result = await conversation.append_message(
            uow, ctx, role=MessageRole.TOOL, content=LIST, kind=MessageKind.INTERNAL,
            tool_call_id="call_1", occurred_at=MONDAY,
        )  # fmt: skip
        await conversation.append_message(uow, ctx, role=MessageRole.ASSISTANT, content=TOP_SAID,
                                          occurred_at=MONDAY)  # fmt: skip
        await uow.commit()
    await container.tasks.drain()
    assert result.message is not None
    return result.message.message_id


async def test_an_earlier_answer_comes_with_its_steps_and_each_result_reads_whole(
    container, uow_factory
) -> None:
    """ "Which tool did you use in our other chat?" and "number 7 on that list": the earlier
    answer's line carries its steps, each result named by its message, which the existing read
    (``get_message``: the thread's read rules) gives whole - to this user's agent, never to
    another user's."""
    monday = _chat()
    ref = await _with_a_call(container, uow_factory, monday)
    bundle = await _context(container, _chat(), "In our earlier chat, which tool gave the top SKU?")
    [line] = [c for c in bundle.earlier_chats if "SKU-00000" in c.text]
    assert line.steps.splitlines()[1] == f'1. sales_summary(group_by="sku"): result {ref}'
    assert f'  1. sales_summary(group_by="sku"): result {ref}' in bundle.render()
    service = container.services["conversation"]
    async with uow_factory() as uow:
        assert (await service.get_message(uow, _chat(), ref)).content == LIST
    with pytest.raises(ScopeDenied):
        async with uow_factory() as uow:
            await service.get_message(uow, _chat("bob"), ref)


#: a long worked answer: what it computes first, the asked-about figure in its middle, its
#: closing advice last; well past the section's cap for one answer
LONG_WORKED = (
    "Here is the full replenishment review for the north region, style by style. "
    + "The Cotton Tee holds steady at its usual weekly rate and needs no change this cycle. " * 9
    + "For the Merino Hoodie in the north, the reorder quantity works out to 137 units. "
    + "The Linen Shirt sells below plan and its next order can wait for the summer reset. " * 9
    + "Tell me if you want the same review for the south region."
)


async def test_a_long_answer_is_quoted_where_the_question_falls(container, uow_factory) -> None:
    """The figure asked about is in the middle of a long answer, past the two ends the section
    used to keep: the passage the question's words fall in is what is carried."""
    monday = _chat()
    await _say(container, uow_factory, monday, ASKED, when=MONDAY)
    await _say(container, uow_factory, monday, LONG_WORKED, role=MessageRole.ASSISTANT, when=MONDAY)
    assert len(LONG_WORKED) > 1500

    bundle = await _context(container, _chat(), FRIDAY_ASK)
    [line] = bundle.earlier_chats
    assert "the reorder quantity works out to 137 units." in line.text


async def _earlier_answers(container, uow_factory, count: int) -> None:
    """``count`` earlier chats, each asking for and answering one region's reorder number."""
    for n in range(count):
        chat = _chat()
        await _say(container, uow_factory, chat, f"Reorder for region {n}?", when=MONDAY)
        await _say(
            container,
            uow_factory,
            chat,
            f"For the Merino Hoodie, the reorder number worked out for region {n} is {300 + n}.",
            role=MessageRole.ASSISTANT,
            when=MONDAY,
        )


async def test_more_earlier_answers_come_when_the_context_has_room(
    container, uow_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five answers match the question alike. Three are pinned, as before; the others are
    added only from the budget the memories leave: the context stays within its budget and
    its memories are what they are without the further answers."""
    await _earlier_answers(container, uow_factory, 5)
    ask = "What was that reorder number you worked out for the Merino Hoodie?"
    builder = container.services["context_builder"]
    builder.cache = builder.semantic_cache = None

    roomy = await builder.build(_chat(), ask)
    assert len(roomy.earlier_chats) == 5
    tight = await builder.build(_chat(), ask, token_budget=260)
    assert tight.token_estimate <= 260 and len(tight.earlier_chats) < 5

    monkeypatch.setattr(search, "EARLIER_CHAT_MORE_LINES", search.EARLIER_CHAT_LINES)
    for budget, with_more in ((8000, roomy), (260, tight)):
        without = await builder.build(_chat(), ask, token_budget=budget)
        assert len(without.earlier_chats) <= 3
        assert [m.text for m in without.memories] == [m.text for m in with_more.memories]


async def test_an_answer_given_many_messages_ago_is_still_found(container, uow_factory) -> None:
    """The answer was given before the newest two hundred messages that share a word with the
    question: the section reads further back than those."""
    monday = _chat()
    await _say(container, uow_factory, monday, ASKED, when=MONDAY)
    await _say(container, uow_factory, monday, WORKED_OUT, role=MessageRole.ASSISTANT, when=MONDAY)
    later = _chat()
    async with uow_factory() as uow:
        for n in range(210):
            await container.services["conversation"].append_message(
                uow,
                later,
                role=MessageRole.ASSISTANT if n % 2 else MessageRole.USER,
                content=f"Note {n}: the Merino Scarf sold {n} units in the south.",
                occurred_at=TUESDAY,
            )
        await uow.commit()

    bundle = await _context(container, _chat(), FRIDAY_ASK)
    assert any("137" in line.text for line in bundle.earlier_chats)
