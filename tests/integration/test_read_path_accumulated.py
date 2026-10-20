"""ADR 0041 through the real write and read paths: memory that is never cleared.

A user restates a value in several conversations, keeps standing rules from weeks ago, says
what they manage, and links a store to a region in one chat and the region to its buyer in
another. A new conversation reads the newest value as current (the older ones as history),
the rules and preferences that apply to the request as profile sections, and the buyer
through one more hop.
"""

from __future__ import annotations

import pytest

from tests.integration.test_statements_follow_the_user import _chat, _context, _say

pytestmark = pytest.mark.integration


def _block(bundle, name: str) -> str:
    return next((b.text for b in bundle.profile if b.block == name), "")


async def test_the_newest_value_is_current_and_the_older_ones_history(container, uow_factory):
    for value in ("40", "60", "75"):
        await _say(container, uow_factory, _chat(), f"The reorder point for MK-220 is {value}.")
    bundle = await _context(container, _chat(), "What's the reorder point for MK-220?")
    current = "\n".join(m.text for m in bundle.memories)
    assert "75" in current and "40" not in current and "60" not in current
    assert any("40" in h.text for h in bundle.history)


async def test_rules_and_preferences_that_apply_come_back_in_a_new_chat(container, uow_factory):
    await _say(
        container,
        uow_factory,
        _chat(),
        "Always answer stockout audits as a markdown table with columns SKU, store and days "
        "out; no prose.",
    )
    await _say(container, uow_factory, _chat(), "Never book deliveries on Sundays.")
    await _say(container, uow_factory, _chat(), "I care about fast-moving items at store 4779.")
    bundle = await _context(container, _chat(), "Run the stockout audits for store 4779.")
    rules = _block(bundle, "standing_rules")
    assert "markdown table" in rules and "Sundays" not in rules
    assert "store 4779" in _block(bundle, "preferences")


async def test_a_later_rule_on_the_same_subject_supersedes(container, uow_factory):
    await _say(
        container, uow_factory, _chat(), "Always answer stockout audits as a markdown table."
    )
    await _say(
        container, uow_factory, _chat(), "From now on, always answer stockout audits as a CSV."
    )
    rules = _block(
        await _context(container, _chat(), "stockout audits for region 5"), "standing_rules"
    )
    assert "CSV" in rules and "markdown" not in rules


async def test_the_second_hop_reaches_the_regions_buyer(container, uow_factory):
    await _say(container, uow_factory, _chat(), "Store 4779 is in Region X29.")
    await _say(container, uow_factory, _chat(), "The buyer for Region X29 is Priya Shah.")
    bundle = await _context(container, _chat(), "Who is the buyer for store 4779?")
    assert "Priya Shah" in "\n".join(m.text for m in bundle.memories)
