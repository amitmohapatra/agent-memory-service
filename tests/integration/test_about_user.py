"""What the user says of their own setup reaches every new conversation (the ``about_user``
profile section): their team's laptops, the currency of their books, what kind of business
they run. A request that shares no word with it ("How do I install the VPN client?") is still
answered for that setup, as an assistant's saved memories are. Bounded, the user's own and their
workspace's only, and a corrected value never comes back."""

from __future__ import annotations

import json

import pytest

from memory_service.domain.enums import Lifetime, MemoryType, MessageRole, Visibility
from tests.integration.test_replacement_across_conversations import _scripted
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration


async def _about(container, ctx, query: str) -> str:
    body = await container.services["context_builder"].build_api(ctx, query, output="full")
    profile = json.loads(body).get("profile") or []
    return "\n".join(b["text"] for b in profile if b["block"] == "about_user")


async def test_the_users_setup_reaches_a_request_that_shares_no_word_with_it(
    container, uow_factory
) -> None:
    march = _chat()
    await _say(container, uow_factory, march, "Our whole team runs Linux laptops now.")
    await _say(container, uow_factory, march, "Our accounts are kept in GBP.")

    about = await _about(container, _chat(), "How do I install the VPN client?")
    assert "Linux" in about and "GBP" in about
    # another user, and the same user in another workspace, read none of it
    assert await _about(container, _chat("u2"), "How do I install the VPN client?") == ""
    assert "Linux" not in await _about(container, _chat(workspace_id="ws2"), "VPN client?")


async def test_a_corrected_setup_is_never_shown_again(container, uow_factory) -> None:
    _scripted(container, 0.97, values=("GBP", "EUR"))
    await _say(container, uow_factory, _chat(), "Our accounts are kept in GBP.")
    await _say(container, uow_factory, _chat(), "Our accounts are kept in EUR since the move.")

    about = await _about(container, _chat(), "Draft a payment reminder for an overdue invoice.")
    assert "EUR" in about and "GBP" not in about


async def test_the_section_is_bounded(container, uow_factory) -> None:
    chat = _chat()
    things = [
        "vans",
        "forklifts",
        "scanners",
        "printers",
        "tablets",
        "carts",
        "ladders",
        "pallets",
        "trolleys",
        "shelves",
        "freezers",
        "conveyors",
        "lockers",
        "radios",
        "cameras",
        "sprinklers",
        "generators",
        "compressors",
        "scales",
        "bins",
    ]
    for n, what in enumerate(things):
        await _say(container, uow_factory, chat, f"Our depot has {n + 3} {what}.")
    await _say(container, uow_factory, chat, "Our depot is open " + "every working day " * 20)
    about = await _about(container, _chat(), "Plan next week's rota.")
    assert len(about.splitlines()) == 16, "the nearest sixteen of twenty-one"
    assert all(len(line) <= 200 for line in about.splitlines())


async def test_what_the_user_block_holds_is_not_listed_again(container, uow_factory) -> None:
    """A self-description kept as a USER memory is in the pinned ``user`` block already
    (``profile.refresh``): ``about_user`` does not repeat it, and its line goes to a setup
    statement only this section carries."""
    chat = _chat()
    for said in (
        "I'm a field engineer based in Leeds.",
        "I am vegetarian.",
        "We are a bakery with three shops.",
        "Our accounts are kept in GBP.",
        "Our whole team runs Linux laptops now.",
    ):
        await _say(container, uow_factory, chat, said)
    await container.services["profile"].refresh_user("acme", "u1")

    body = await container.services["context_builder"].build_api(
        _chat(), "Plan next week's rota.", output="full"
    )
    blocks = {b["block"]: b["text"] for b in json.loads(body).get("profile") or []}
    assert "field engineer" in blocks["user"]
    assert "field engineer" not in blocks["about_user"]
    assert "Linux" in blocks["about_user"] and "GBP" in blocks["about_user"]


#: what one person says of their own work over a first conversation (a field report): the
#: statement asked about later is one among many
SAID_OF_THEIR_WORK = (
    "Please keep your answers short and in bullet points.",
    "My buyer is Dana Lee.",
    "Never change a retail price without my approval.",
    "I work from the Chicago office on Tuesdays and Thursdays.",
    "I use a MacBook and two external monitors.",
    "Our team runs its weekly review on Monday mornings.",
    "I report to the divisional merchandise manager.",
    "Our fiscal year starts in February.",
    "I usually build my reports in Excel before sharing them.",
    "We sell through outlets and department stores as well.",
    "I cover the east region stores.",
    "I'm Alex. I plan Coach Womens Bags for the Retail channel.",
    "I prefer to see sell-through before margin in any summary.",
    "I am responsible for the spring assortment.",
    "My manager is Priya Raman.",
    "I keep my open-to-buy in a shared workbook.",
    "We lock the plan two weeks before each quarter.",
    "I check inventory positions every Friday.",
    "I am based in the central time zone.",
    "Our stores close their books on the last Saturday of the month.",
)


ALEX = "I'm Alex. I plan Coach Womens Bags for the Retail channel."


async def test_one_self_description_among_many_reaches_every_new_conversation(
    container, uow_factory
) -> None:
    """A field report: "I'm Alex. I plan Coach Womens Bags for the Retail channel." said among
    nineteen other things about the user's work, then "What do you remember about how I
    work?" in a new conversation of the same workspace. A broad question is close to none of
    them, so the ranked memories may carry nothing, and the section that carries the user's
    own setup had room for the four nearest: which four depended on the question. Both
    things the user said of themselves are in the context whatever is asked; the note the
    agent saved about them is not in their profile (ADR 0013); another workspace reads
    none of it."""
    first = _chat()
    for said in SAID_OF_THEIR_WORK:
        await _say(container, uow_factory, first, said)
        await _say(container, uow_factory, first, "Noted.", role=MessageRole.ASSISTANT)
    async with uow_factory() as uow:
        await container.services["memory"].remember(
            uow,
            first.model_copy(update={"agent_id": "assistant-agent"}),
            content="Alex plans Coach Womens Bags for the Retail channel",
            memory_type=MemoryType.USER,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.USER,
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()

    builder = container.services["context_builder"]
    for asked in (
        "What do you remember about how I work?",
        "How do I install the VPN client?",
        "Plan next week's rota.",
        "Draft a payment reminder for an overdue invoice.",
    ):
        bundle = await builder.build(_chat(), asked)
        sections = {block.block: block.text for block in bundle.profile}
        assert ALEX in sections.get("about_user", ""), asked
        assert "I'm Alex." in bundle.render(), asked
        assert "I plan Coach Womens Bags for the Retail channel." in bundle.render(), asked
        assert "Alex plans Coach Womens Bags" not in "\n".join(sections.values()), (
            "an agent's note is not the user's"
        )
    elsewhere = await _about(
        container, _chat(workspace_id="ws2"), "What do you remember about how I work?"
    )
    assert "Coach Womens Bags" not in elsewhere
