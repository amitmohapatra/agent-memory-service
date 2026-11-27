"""The stored ``user`` block is who the person is, sent with every context; their rules and
preferences come only when they apply to the request (``context.standing``), so no line is
in the prompt twice and a rule about something else is not in it at all."""

from __future__ import annotations

import json

import pytest

from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

ROLE = "I manage the Coach handbags account for the north region."
RULE = "Never create an allocation when ATA is 0."
PREFERENCE = "I prefer answers as a table with columns store, item, count."


async def test_the_user_block_is_who_they_are_and_each_statement_is_said_once(
    container, uow_factory
) -> None:
    earlier = _chat()
    for said in (ROLE, RULE, PREFERENCE, "The Coach tote is short at stores 12 and 40."):
        await _say(container, uow_factory, earlier, said)
    block = await container.services["profile"].refresh_user("acme", "u1")
    text = block.text if block is not None else ""
    assert "ATA" not in text and "as a table" not in text

    body = json.loads(
        await container.services["context_builder"].build_api(
            _chat(),
            "Which stores are short of the Coach tote? Show it as a table.",
            output="prompt",
        )
    )
    rendered = body["rendered"]
    assert rendered.count("as a table with columns") == 1, "said once, where it applies"
    profile = rendered.split("\n## ", 2)[0]
    # what goes with every request holds no rule about allocations (whether one ranks among
    # the memories is the encoder's; the suite's is a stand-in)
    assert "ATA is 0" not in profile
    assert rendered.count("Coach handbags account") == 1
