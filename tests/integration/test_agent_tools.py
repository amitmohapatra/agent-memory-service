"""Memory agent tools against PostgreSQL: each tool in the caller's scope."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole, TemporalStatus
from memory_service.domain.errors import NotFound, ScopeDenied, ValidationFailed

pytestmark = pytest.mark.integration


def _agent(run: str = "run_1") -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", user_id="ann", agent_id="buyer", agent_run_id=run, thread_id="thr_at"
    )


async def _call(container, name: str, args: dict, run: str = "run_1"):
    result = await container.services["agent_tools"].call(_agent(run), name, args)
    await container.tasks.drain()
    return result


async def test_an_agent_remembers_finds_updates_and_forgets_through_its_tools(container) -> None:
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

    with pytest.raises(ValidationFailed):
        await _call(container, "memory_update", {"id": stored["id"]})
    with pytest.raises(NotFound):
        await _call(container, "no_such_tool", {})


async def test_an_agent_run_cannot_rewrite_or_forget_what_the_user_said(
    container, uow_factory
) -> None:
    """N2: a run answering "What is 12 times 7?" called memory_update on the user's "I raised
    PO-77812 ..." and replaced it with its answer, kept as the user's own statement. The tool
    refuses and says what to do instead; the user's words stay as they were."""
    from memory_service.modules.jobs.registry import register_handlers

    said = "I raised PO-77812 for 2,000 corrugated boxes; it's late."
    user = _agent().model_copy(update={"agent_id": None, "agent_run_id": None})
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["conversation"].append_message(
            uow, user, role=MessageRole.USER, content=said
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    found = await _call(container, "memory_search", {"query": "PO-77812 corrugated boxes"})
    mine = next(f for f in found if f["kind"] == "memory" and said in f.get("text", ""))

    for name, args in (
        ("memory_update", {"id": mine["id"], "content": "12 times 7 = 84"}),
        ("memory_forget", {"id": mine["id"]}),
    ):
        with pytest.raises(ScopeDenied, match="memory_remember"):
            await _call(container, name, args)
    async with uow_factory() as uow:
        kept = await container.services["memory"].get_memory(uow, user, mine["id"])
    assert kept.content == said and kept.temporal.status is TemporalStatus.CURRENT
    assert kept.owner_principal == "user:ann"
    # the user still corrects their own words
    async with uow_factory() as uow:
        new = await container.services["memory"].supersede(
            uow, user, mine["id"], content="PO-77812 arrived today.", reason="updated"
        )
        await uow.commit()
    assert new.owner_principal == "user:ann"


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


async def test_message_search_reads_this_conversation_then_the_users_earlier_ones(
    container, uow_factory
) -> None:
    """What was said: this thread's messages first, then the user's earlier conversations
    (newest first among equals), each with its thread; another user's thread, a deleted one
    and a time outside the range are never read; without a user, this thread's only."""
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
        (at("ann", "thr_now"), "Which order was that?", None),
        (at(None, "thr_anon"), "An order with no user.", None),
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

    found = await search.search(at("ann", "thr_now"), "order", kinds=["message"], limit=5)
    assert [(i.text, i.thread_id) for i in found.items] == [
        ("USER: Which order was that?", "thr_now"),  # this conversation first
        ("USER: Did order 4471 ship yet?", "thr_july"),  # then the newest earlier one
        ("USER: My order number is 4471.", "thr_june"),
    ], "never bob's, never the deleted thread"
    number = await search.search(at("ann", "thr_now"), "order number", kinds=["message"], limit=5)
    assert number.items[0].thread_id == "thr_june", "most words shared first, wherever said"
    recent = await search.search(
        at("ann", "thr_now"),
        "order",
        kinds=["message"],
        limit=5,
        observed=(datetime.now(UTC) - timedelta(days=1), None),
    )
    assert [i.thread_id for i in recent.items] == ["thr_now", "thr_july"]
    alone = await search.search(at(None, "thr_anon"), "order", kinds=["message"], limit=5)
    assert [i.thread_id for i in alone.items] == ["thr_anon"], "no user: this thread only"

    tool = await container.services["agent_tools"].call(
        at("ann", "thr_now"), "memory_search", {"query": "order number", "kinds": ["message"]}
    )
    assert tool[0]["thread_id"] == "thr_june"
    assert "bob" not in str(tool).casefold()

    # earlier conversations give only what shares a word with the query, never filler
    unrelated = await search.search(at("ann", "thr_now"), "weather", kinds=["message"], limit=5)
    assert [i.thread_id for i in unrelated.items] == ["thr_now"], "this thread's fallback only"

    # bounded by the user's most recently active threads, filtered in the database
    async with uow_factory() as uow:
        newest = await uow.messages.owned_recent(
            "acme", "ann", workspace_id=None, limit=10, threads=1, words=["order"]
        )
        assert {m.thread_id for m in newest} == {"thr_now"}
        assert (
            await uow.messages.owned_recent(
                "acme", "ann", workspace_id=None, limit=10, threads=5, words=[]
            )
            == []
        )
        assert (
            await uow.messages.owned_recent(
                "acme", "nobody", workspace_id=None, limit=10, threads=5, words=["x"]
            )
            == []
        )
        literal = await uow.messages.owned_recent(
            "acme", "ann", workspace_id=None, limit=10, threads=5, words=["4_71", "44%"]
        )
        assert literal == [], "LIKE wildcards in a word are matched literally"
