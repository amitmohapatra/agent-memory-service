"""What the user says of their own setup reaches every new conversation (the ``about_user``
profile section): their team's laptops, the currency of their books, what kind of business
they run. A request that shares no word with it ("How do I install the VPN client?") is still
answered for that setup, as an assistant's saved memories are. Bounded, the user's own and their
workspace's only, and a corrected value never comes back."""

from __future__ import annotations

import json

import pytest

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


async def test_the_section_is_small(container, uow_factory) -> None:
    chat = _chat()
    for n, what in enumerate(["vans", "forklifts", "scanners", "printers", "tablets", "carts"]):
        await _say(container, uow_factory, chat, f"Our depot has {n + 3} {what}.")
    about = await _about(container, _chat(), "Plan next week's rota.")
    assert 0 < len(about.splitlines()) <= 4 and len(about) <= 600


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
