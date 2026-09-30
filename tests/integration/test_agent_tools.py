"""Memory agent tools against PostgreSQL: each tool in the caller's scope, every call logged
as a pull, used ids marked, and the prefetch counts learned from them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole, TemporalStatus
from memory_service.domain.errors import NotFound, ValidationFailed
from memory_service.domain.pulls import PREFETCH_MIN_PULLS

pytestmark = pytest.mark.integration


def _agent(run: str = "run_1") -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", user_id="ann", agent_id="buyer", agent_run_id=run, thread_id="thr_at"
    )


async def _call(container, name: str, args: dict, run: str = "run_1"):
    result = await container.services["agent_tools"].call(_agent(run), name, args)
    await container.tasks.drain()
    return result


async def test_an_agent_remembers_finds_updates_and_forgets_through_its_tools(
    container, uow_factory
) -> None:
    stored = await _call(
        container,
        "memory_remember",
        {"content": "Ann's preferred supplier is Acme Paper", "kind": "PREFERENCE"},
    )
    found = await _call(container, "memory_search", {"query": "preferred supplier", "k": 3})
    assert found[0]["id"] == stored["id"] and found[0]["kind"] == "memory"
    assert found[0]["observed_at"]
    none_later = await _call(
        container,
        "memory_search",
        {
            "query": "preferred supplier",
            "time_from": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        },
    )
    assert none_later == []

    updated = await _call(
        container,
        "memory_update",
        {"id": stored["id"], "content": "Ann's preferred supplier is Globex", "reason": "switched"},
    )
    assert updated["supersedes"] == stored["id"]
    invalidated = await _call(
        container, "memory_update", {"id": updated["id"], "invalidate": True, "reason": "left"}
    )
    assert invalidated == {"id": updated["id"], "invalidated": True}
    async with uow_factory() as uow:
        gone = await uow.memories.get("acme", updated["id"])
        pulls = await uow.pulls.settled(before=datetime.now(UTC) + timedelta(hours=1), limit=50)
    assert gone is not None and gone.temporal.status is TemporalStatus.RETRACTED
    assert [p.tool for p in pulls] == [
        "memory_remember",
        "memory_search",
        "memory_search",
        "memory_update",
        "memory_update",
    ]
    first_search = pulls[1]
    assert first_search.result_ids[0] == stored["id"]
    assert first_search.used_ids == [stored["id"]], "the update acted on what the search found"

    with pytest.raises(ValidationFailed):
        await _call(container, "memory_update", {"id": stored["id"], "reason": "nothing to do"})
    with pytest.raises(NotFound):
        await _call(container, "no_such_tool", {})


async def test_history_profile_and_outcome_tools(container, uow_factory) -> None:
    conversation = container.services["conversation"]
    async with uow_factory() as uow:
        for text in ("We need A4 paper.", "Noted: 500 sheets."):
            await conversation.append_message(uow, _agent(), role=MessageRole.USER, content=text)
        await uow.commit()
    history = await _call(container, "history_search", {"query": "paper", "k": 5})
    assert [h["text"] for h in history] == ["We need A4 paper."]

    written = await _call(
        container, "profile_edit", {"block": "user", "old": "", "new": "name: Ann"}
    )
    edited = await _call(
        container, "profile_edit", {"block": "user", "old": "Ann", "new": "Ann Lee"}
    )
    assert written["text"] == "name: Ann" and edited["text"] == "name: Ann Lee"
    assert edited["version"] == written["version"] + 1

    assert await _call(container, "record_outcome", {"success": True}) == {
        "run_id": "run_1",
        "success": True,
    }
    assert await _call(container, "procedures_search", {"task": "order paper"}) == []
    hints = await _call(container, "tool_search", {"task": "order paper"})
    assert set(hints) == {"candidates", "plan", "next", "prefill", "missing"}


async def test_items_a_run_keeps_using_for_a_request_pattern_are_prefetched(
    container, uow_factory
) -> None:
    stored = await _call(
        container, "memory_remember", {"content": "The paper supplier is Acme", "kind": "SEMANTIC"}
    )
    agent_tools = container.services["agent_tools"]
    for i in range(PREFETCH_MIN_PULLS):
        run = f"run_{i}"
        await _call(container, "memory_search", {"query": f"supplier for order {i}"}, run=run)
        await agent_tools.used(_agent(run), [stored["id"]])
    assert await agent_tools.prefetched(_agent(), "supplier for order 99") == []
    assert await agent_tools.learn_prefetch() == 0, "pulls settle before they are learned"
    learned = await agent_tools.learn_prefetch(now=datetime.now(UTC) + timedelta(hours=1))
    assert learned == PREFETCH_MIN_PULLS + 1
    assert await agent_tools.prefetched(_agent(), "supplier for order 99") == [stored["id"]]
    other = MemoryExecutionContext(tenant_id="acme", user_id="bob", agent_id="buyer")
    assert await agent_tools.prefetched(other, "supplier for order 99") == []
