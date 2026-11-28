"""An agent's note for a period ends with it: ``memory_remember`` takes ``valid_to``, and once
it has passed the note is no longer found. "For this week only, no Acme orders" was kept as
the agent's note with no end, and would have been followed after the week."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

ME = MemoryExecutionContext(tenant_id="acme", user_id="priya", workspace_id="ws1", agent_id="a")


async def _found(tools, query: str) -> list[str]:
    return [i["text"] for i in await tools.call(ME, "memory_search", {"query": query, "k": 20})]


async def test_a_note_for_a_period_is_not_found_after_it(container) -> None:
    register_handlers(container)
    tools = container.services["agent_tools"]
    now = datetime.now(UTC)
    ended = "No Acme orders: Acme is on a shipping hold (last week only)."
    holds = "No Zenith orders: Zenith is on a shipping hold (this week only)."
    await tools.call(
        ME,
        "memory_remember",
        {"content": ended, "kind": "TASK", "valid_to": (now - timedelta(days=1)).isoformat()},
    )
    await tools.call(
        ME,
        "memory_remember",
        {"content": holds, "kind": "TASK", "valid_to": (now + timedelta(days=6)).isoformat()},
    )
    await container.tasks.drain()
    found = await _found(tools, "Which suppliers are on a shipping hold?")
    assert holds in found, found
    assert ended not in found, "past its end it is no longer found"
