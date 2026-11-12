"""What the user says of their own setup reaches every new conversation (the ``about_user``
profile section): their team's laptops, the currency of their books, what kind of business
they run. A request that shares no word with it ("How do I install the VPN client?") is still
answered for that setup, as an assistant's saved memories are. Bounded, the user's own and their
workspace's only, and a corrected value never comes back."""

from __future__ import annotations

import json

import pytest

from memory_service.domain.enums import Lifetime, MemoryType, MessageRole, Visibility
from tests.integration.test_memory import _memories
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


SAM = "I'm Sam. I plan Mens Shoes for the Outlet channel."


async def test_another_users_self_description_in_the_same_workspace_is_not_in_mine(
    container, uow_factory
) -> None:
    """Two people in one workspace of one tenant, each in their own conversation with the
    default visibility: the section's search filters on what the asker may read and on the
    workspace, not on whose statement it is, so it is the statement's own visibility (USER:
    what a user says follows that user, ADR 0036) that keeps one person's "I'm Sam ..." out
    of the other's profile."""
    alex, sam = _chat("u1"), _chat("u2")
    await _say(container, uow_factory, alex, ALEX)
    await _say(container, uow_factory, sam, SAM)

    stored = [
        m
        for m in await _memories(uow_factory, sam, container)
        if m.content == SAM and (m.system_metadata or {}).get("self_description")
    ]
    assert stored, "the statement is kept as a self-description of the workspace"
    assert {m.visibility for m in stored} == {Visibility.USER}
    assert {m.system_metadata["self_description"] for m in stored} == {"ws1"}

    for asked in ("What do you remember about how I work?", "Which channel do I plan for?"):
        mine = await container.services["context_builder"].build(_chat("u1"), asked)
        sections = {block.block: block.text for block in mine.profile}
        assert ALEX in sections.get("about_user", ""), asked
        assert "Sam" not in mine.render(), asked
        theirs = await _about(container, _chat("u2"), asked)
        assert SAM in theirs and "Alex" not in theirs, asked


#: seventeen longer things a person says of their own work, each its own subject
LONG_SELF_DESCRIPTIONS = (
    "I work from the Chicago office on Tuesdays and Thursdays and from the Milwaukee "
    "distribution centre on the other days, because the regional planning team meets there.",
    "I use a MacBook with two external monitors at my desk and a tablet on the shop floor, "
    "so anything you send me has to be readable on a small screen as well.",
    "Our team runs its weekly review on Monday mornings with the merchandise managers of all "
    "four regions, and the agenda is fixed by the Friday before at the latest.",
    "I report to the divisional merchandise manager for handbags and small leather goods, who "
    "signs off every seasonal plan before it goes to the finance department.",
    "Our fiscal year starts in February and closes at the end of January, which is why the "
    "fourth quarter covers the whole of the holiday trading period for us.",
    "I usually build my reports in Excel before sharing them, with one sheet for each region "
    "and a summary sheet at the front that the directors read first.",
    "We sell through outlets and department stores as well as our own full-price boutiques, "
    "and each of the three has its own markdown calendar and clearance cadence.",
    "I cover the east region stores from Boston down to Washington, including the two flagship "
    "locations in Manhattan that carry the widest assortment of the whole chain.",
    "I'm Alex. I plan Coach Womens Bags for the Retail channel.",
    "I am responsible for the spring assortment of the leather category, from the first line "
    "review with the designers to the final buy quantities for every door.",
    "I keep my open-to-buy in a shared workbook that the allocation analysts update every "
    "morning with the receipts and the cancellations of the previous day.",
    "We lock the plan two weeks before each quarter, and after that any change to the receipt "
    "flow needs a written exception approved by the planning director.",
    "I check inventory positions every Friday afternoon against the sell-through of the week, "
    "store by store, before I decide which styles to replenish or transfer.",
    "Our stores close their books on the last Saturday of the month, so the monthly numbers I "
    "quote always follow the retail calendar and never the civil one.",
    "I am based in the central time zone although most of my stores are on the east coast, so "
    "my mornings start with the opening reports of the eastern doors.",
    "I travel to the showroom in New York for every market week, four times a year, and I am "
    "out of the office and slow to answer for the whole of those weeks.",
    "I mentor two assistant planners who joined the team last autumn, and I review their "
    "weekly recaps before they are sent to the merchandise managers.",
)


#: what the pinned ``user`` block keeps of a person: who they are
WHO_THEY_ARE = (
    "My name is Alex Morgan.",
    "My manager is Priya Raman.",
    "My assistant is Jordan Whitfield.",
    "My director is Helena Okafor.",
    "My deputy is Marcus Lindqvist.",
    "I am vegetarian.",
    "I am left-handed.",
    "I am fluent in Spanish and French.",
    "I am a senior merchandise planner.",
    "I am a member of the pricing committee.",
    "I am a certified first aider.",
    "I am on leave every August.",
)
RULES = (
    "Never change a retail price without my approval, not even in a draft, and never round a "
    "price to a whole dollar unless the pricing committee has signed the change off in writing.",
    "Always list my stores by their number, never by the name of the mall, and keep the "
    "flagship locations at the top of the list whatever the ranking criterion happens to be.",
    "Never send a weekly email to the directors before I have read it, and never copy the "
    "regional managers on a message that mentions margin, shrink or markdown budgets.",
    "Always put the totals of a report on its first page, with the figure of last year beside "
    "each one, and say in one line which calendar the comparison follows.",
    "Never quote a sell-through figure in a draft without the number of weeks it covers and "
    "the date of the inventory snapshot it was computed from.",
    "Always write dates as day, month and year in anything about my stores, because half of "
    "the team reads them the European way and the other half does not.",
)


async def _works_at_a_retailer(container, uow_factory) -> None:
    """One person's first conversation: who they are, six rules, a preference and seventeen
    things about their own work; the ``user`` block refreshed as the nightly job does."""
    first = _chat()
    for said in (
        *WHO_THEY_ARE,
        *RULES,
        "I prefer to see sell-through before margin in any weekly report.",
        *LONG_SELF_DESCRIPTIONS,
    ):
        await _say(container, uow_factory, first, said)
    await container.services["profile"].refresh_user("acme", "u1")


async def _context(container, budget: int):
    bundle = await container.services["context_builder"].build(
        _chat(), "Draft the weekly report for my stores.", token_budget=budget
    )
    return bundle, {block.block: block.text for block in bundle.profile}


async def test_a_small_budget_still_carries_the_user_block_and_the_rules(
    container, uow_factory
) -> None:
    """A harness that does not know its model's window asks for 2,000 tokens, of which the
    pinned sections may take half, and the profile is whole or cut to its standing sections.
    The room for more self-descriptions must not cost what a context of that size held: the
    user's name and their rules. The first lines of ``about_user`` are pinned as they were
    (four lines, 600 characters); the further ones take only the room the memories leave."""
    await _works_at_a_retailer(container, uow_factory)
    bundle, sections = await _context(container, 2000)
    assert "Alex Morgan" in sections.get("user", "")
    assert "Never change a retail price" in sections.get("standing_rules", "")
    assert "Always list my stores by their number" in sections.get("standing_rules", "")
    assert sections.get("about_user")
    assert bundle.token_estimate <= 2000

    roomy, sections = await _context(container, 8000)
    assert "Alex Morgan" in sections["user"]
    assert len(sections["about_user"].splitlines()) == 16
    assert [m.item_id for m in roomy.memories] == [m.item_id for m in bundle.memories], (
        "the memories are what they are without the further lines"
    )


async def test_further_self_descriptions_take_only_the_room_the_memories_leave(
    container, uow_factory
) -> None:
    """At 1,000 tokens the whole profile does not fit its half and the memories fill the
    rest: the rules that apply are still there, ``about_user`` is its pinned lines and no
    more, and the context is within its budget."""
    await _works_at_a_retailer(container, uow_factory)
    bundle, sections = await _context(container, 1000)
    assert "Never change a retail price" in sections.get("standing_rules", "")
    about = sections.get("about_user", "").splitlines()
    assert 0 < len(about) <= 4 and len(sections["about_user"]) <= 600
    assert bundle.token_estimate <= 1000
    assert bundle.memories, "and the memories took the rest"
