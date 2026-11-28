"""Each statement once, one pattern a line: a turn kept as said is its own line (not gathered
under a header as if "said" were a slot), and a graph fact read from a memory the context
already carries, or the graph's bookkeeping ("u1 mentions Priya"), is not shown again."""

from __future__ import annotations

import json

import pytest

from memory_service.domain.enums import Lifetime, MemoryType
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration


async def test_each_statement_is_in_the_context_once(container, uow_factory) -> None:
    earlier = _chat()
    for said in (
        "Never create an allocation when ATA is 0.",
        "I prefer answers as a table with columns store, item, count.",
        "The Coach tote is short of stock at stores 12 and 40.",
        "My dentist appointment moved to Tuesday.",
    ):
        await _say(container, uow_factory, earlier, said)
    async with uow_factory() as uow:
        await container.services["memory"].remember(
            uow,
            earlier,
            content="My name is Priya and I plan Coach bags for the north region.",
            memory_type=MemoryType.USER,
            lifetime=Lifetime.LONG_TERM,
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    body = json.loads(
        await container.services["context_builder"].build_api(
            _chat(), "Which stores are short of the Coach tote? Show it as a table.", output="full"
        )
    )
    rendered = json.loads(
        await container.services["context_builder"].build_api(
            _chat(),
            "Which stores are short of the Coach tote? Show it as a table?",
            output="prompt",
        )
    )["rendered"]
    assert "source statements" not in rendered, "a turn as said is its own line"
    assert "mentions" not in rendered, "no bookkeeping fact"
    shown = {m["id"] for m in body.get("memories") or []}
    for fact in body.get("graph_facts") or []:
        assert fact.get("memory_id") not in shown
    assert rendered.count("as a table with columns") == 1
    assert rendered.count("short of stock at stores 12 and 40") <= 1
