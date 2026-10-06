"""Memory agent tools against PostgreSQL: each tool in the caller's scope, every call logged
as a pull, used ids marked, and the prefetch counts learned from them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole
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
    assert found[0]["observed_on"]
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
        {"id": stored["id"], "content": "Ann's preferred supplier is Globex"},
    )
    assert updated["supersedes"] == stored["id"]
    forgotten = await _call(container, "memory_forget", {"id": updated["id"]})
    assert forgotten == {"id": updated["id"], "forgotten": True}
    async with uow_factory() as uow:
        pulls = await uow.pulls.settled(before=datetime.now(UTC) + timedelta(hours=1), limit=50)
    assert [p.tool for p in pulls] == [
        "memory_remember",
        "memory_search",
        "memory_search",
        "memory_update",
        "memory_forget",
    ]
    first_search = pulls[1]
    assert first_search.result_ids[0] == stored["id"]
    assert first_search.used_ids == [stored["id"]], "the update acted on what the search found"

    with pytest.raises(ValidationFailed):
        await _call(container, "memory_update", {"id": stored["id"]})
    with pytest.raises(NotFound):
        await _call(container, "no_such_tool", {})


async def test_message_search_profile_and_tool_search_tools(container, uow_factory) -> None:
    conversation = container.services["conversation"]
    async with uow_factory() as uow:
        for text in ("We need A4 paper.", "Noted: 500 sheets."):
            await conversation.append_message(uow, _agent(), role=MessageRole.USER, content=text)
        await uow.commit()
    history = await _call(
        container, "memory_search", {"query": "paper", "kinds": ["message"], "k": 5}
    )
    assert [h["text"] for h in history] == ["USER: We need A4 paper."]  # the speaker leads
    assert {h["kind"] for h in history} == {"message"}

    written = await _call(
        container, "profile_edit", {"block": "user", "old": "", "new": "name: Ann"}
    )
    edited = await _call(
        container, "profile_edit", {"block": "user", "old": "Ann", "new": "Ann Lee"}
    )
    assert written["text"] == "name: Ann" and edited["text"] == "name: Ann Lee"
    assert edited["version"] == written["version"] + 1

    hints = await _call(container, "tool_search", {"task": "order paper"})
    # nothing learned for this task: the tools that fit, and no plan (an absent key is none)
    assert set(hints) == {"tools"}


async def test_message_search_reads_every_conversation_of_the_user_and_no_one_elses(
    container, uow_factory
) -> None:
    """``threads="all"``: the messages of every thread the user owns, each with its thread,
    newest first among equals; another user's thread, a deleted one and a time outside the
    range are never read; without a user there is nothing to read across."""
    conversation = container.services["conversation"]

    def at(user: str | None, thread: str) -> MemoryExecutionContext:
        return MemoryExecutionContext(
            tenant_id="acme", user_id=user, agent_id="buyer", agent_run_id="run_1", thread_id=thread
        )

    week_ago = datetime.now(UTC) - timedelta(days=7)
    said = [
        (at("ann", "thr_june"), "My order number is 4471.", week_ago),
        (at("ann", "thr_july"), "Did order 4471 ship yet?", None),
        (at("ann", "thr_gone"), "Order 4471 was a gift.", None),
        (at("bob", "thr_bob"), "Bob's order number is 9002.", None),
    ]
    async with uow_factory() as uow:
        for ctx, text, when in said:
            await conversation.append_message(
                uow, ctx, role=MessageRole.USER, content=text, occurred_at=when
            )
        await uow.commit()
    async with uow_factory() as uow:
        await conversation.delete_thread(uow, at("ann", "thr_gone"), "thr_gone")
        await uow.commit()
    search = container.services["search"]

    here = await search.search(at("ann", "thr_now"), "order number", kinds=["message"], limit=5)
    assert here.items == [], "the current thread only, by default: this one is empty"
    found = await search.search(
        at("ann", "thr_now"), "order number", kinds=["message"], limit=5, threads="all"
    )
    assert [(i.text, i.thread_id) for i in found.items] == [
        ("USER: My order number is 4471.", "thr_june"),  # shares two words
        ("USER: Did order 4471 ship yet?", "thr_july"),
    ]
    recent = await search.search(
        at("ann", "thr_now"),
        "order",
        kinds=["message"],
        limit=5,
        threads="all",
        observed=(datetime.now(UTC) - timedelta(days=1), None),
    )
    assert [i.thread_id for i in recent.items] == ["thr_july"]
    nobody = await search.search(
        at(None, "thr_now"), "order", kinds=["message"], limit=5, threads="all"
    )
    assert nobody.items == []

    tool = await container.services["agent_tools"].call(
        at("ann", "thr_now"),
        "memory_search",
        {"query": "order number", "kinds": ["message"], "threads": "all"},
    )
    assert [h["thread_id"] for h in tool] == ["thr_june", "thr_july"]

    # bounded by the user's most recently active threads, not by all they ever said
    async with uow_factory() as uow:
        newest = await uow.messages.owned_recent("acme", "ann", limit=10, threads=1)
    assert {m.thread_id for m in newest} == {"thr_july"}


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
    bundle = await container.services["context_builder"].build(_agent(), "supplier for order 99")
    assert stored["id"] in [m.item_id for m in bundle.memories]
