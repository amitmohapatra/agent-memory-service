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
    for ctx, text, when in said:  # one write each: each thread active after the one before
        async with uow_factory() as uow:
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


async def test_the_last_and_the_first_thing_said_are_read_by_time_not_by_words(
    container, uow_factory
) -> None:
    """ "My last question", "the first thing I asked", "your last answer" are questions about
    order: ``order`` newest or oldest reads the messages by when they were said, whatever the
    query's words. This conversation only with ``here``; one speaker's with ``said_by``;
    never another user's, never a deleted conversation's."""
    conversation = container.services["conversation"]

    def at(user: str | None, thread: str) -> MemoryExecutionContext:
        return MemoryExecutionContext(
            tenant_id="acme", user_id=user, agent_id="buyer", agent_run_id="run_1", thread_id=thread
        )

    now = datetime.now(UTC)
    said = [
        (at("ann", "thr_old"), MessageRole.USER, "Show the allocation for store 12.", -3000),
        (at("ann", "thr_old"), MessageRole.ASSISTANT, "Store 12 has 40 units allocated.", -2990),
        (at("ann", "thr_old"), MessageRole.USER, "Now move 10 units to store 40.", -2980),
        (at("ann", "thr_old"), MessageRole.ASSISTANT, "Done: 10 units moved to store 40.", -2970),
        (at("bob", "thr_bob"), MessageRole.USER, "Bob asked something last of all.", -10),
        (at("ann", "thr_gone"), MessageRole.USER, "A question in a deleted chat.", -20),
        (at("ann", "thr_new"), MessageRole.USER, "What is the last question I asked?", -5),
    ]
    for ctx, role, text, seconds in said:
        async with uow_factory() as uow:
            await conversation.append_message(
                uow, ctx, role=role, content=text, occurred_at=now + timedelta(seconds=seconds)
            )
            await uow.commit()
    async with uow_factory() as uow:
        await conversation.delete_thread(uow, at("ann", "thr_gone"), "thr_gone")
        await uow.commit()

    async def search(thread: str, **args) -> list[str]:
        found = await container.services["agent_tools"].call(
            at("ann", thread), "memory_search", {"query": "my last question", **args}
        )
        assert {item["kind"] for item in found} <= {"message"}
        return [item["text"] for item in found]

    # by words, the ranking does not know what "last" means: it is not what answers this
    by_words = await search("thr_new", kinds=["message"], k=1)
    assert by_words == ["USER: What is the last question I asked?"]

    # by time, across the user's conversations: the question before this one is exact
    newest = await search("thr_new", order="newest", said_by="user", k=2)
    assert newest == [
        "USER: What is the last question I asked?",
        "USER: Now move 10 units to store 40.",
    ]
    assert await search("thr_new", order="newest", said_by="assistant", k=1) == [
        "ASSISTANT: Done: 10 units moved to store 40."
    ]
    # the first thing said, in a conversation: only that conversation's messages
    assert await search("thr_old", order="oldest", here=True, k=1) == [
        "USER: Show the allocation for store 12."
    ]
    assert await search("thr_old", order="newest", here=True, k=2) == [
        "ASSISTANT: Done: 10 units moved to store 40.",
        "USER: Now move 10 units to store 40.",
    ]
    assert await search("thr_new", order="oldest", here=True, k=5) == [
        "USER: What is the last question I asked?"
    ]
    # the oldest across the user's conversations, and a time range narrows either order
    assert (await search("thr_new", order="oldest", k=1))[0].endswith("allocation for store 12.")
    recent = await search(
        "thr_new", order="newest", k=5, time_from=(now - timedelta(seconds=60)).isoformat()
    )
    assert recent == ["USER: What is the last question I asked?"]
    everything = await search("thr_new", order="newest", k=20)
    assert (
        len(everything) == 5 and "Bob" not in str(everything) and "deleted" not in str(everything)
    )

    # a conversation that does not exist gives nothing; a caller with no user reads its own
    # conversation only, and is refused another person's
    assert await search("thr_none", order="newest", here=True) == []
    async with uow_factory() as uow:
        await conversation.append_message(
            uow, at(None, "thr_anon_time"), role=MessageRole.USER, content="Said with no user."
        )
        await uow.commit()
    tools = container.services["agent_tools"]
    args = {"query": "last", "order": "newest", "k": 5}
    anonymous = await tools.call(at(None, "thr_anon_time"), "memory_search", args)
    assert [item["text"] for item in anonymous] == ["USER: Said with no user."]
    with pytest.raises(ScopeDenied):
        await tools.call(at(None, "thr_new"), "memory_search", args)
    with pytest.raises(ValidationFailed):
        await search("thr_new", order="latest")


async def test_the_graphs_facts_are_a_kind_a_search_can_ask_for(container, uow_factory) -> None:
    """``fact``: the relations the knowledge graph holds of what the query names, on demand -
    a context shows them only when its question names an entity. Only facts that say
    something: where a name appears ("u1 mentions Dana") is the graph's own bookkeeping."""
    from tests.integration.test_statements_follow_the_user import _chat, _say

    earlier = _chat()
    for said in ("The Coach tote is short of stock at stores 12 and 40.", "Dana manages store 40."):
        await _say(container, uow_factory, earlier, said)
    tools = container.services["agent_tools"]
    ask = "Which stores are short of stock of the Coach tote?"

    facts = await tools.call(_chat(), "memory_search", {"query": ask, "kinds": ["fact"]})
    assert facts and {item["kind"] for item in facts} == {"fact"}
    assert any("short of stock at stores 12 and 40" in item["text"] for item in facts)
    assert not any(" mentions " in item["text"] for item in facts)
    assert all(item["observed_on"] for item in facts)

    both = await tools.call(_chat(), "memory_search", {"query": ask, "kinds": ["memory", "fact"]})
    assert {item["kind"] for item in both} == {"memory", "fact"}
    # unless asked for, a search returns none, as before
    default = await tools.call(_chat(), "memory_search", {"query": ask})
    assert "fact" not in {item["kind"] for item in default}
    # another user's graph is never read
    theirs = await tools.call(_chat("u2"), "memory_search", {"query": ask, "kinds": ["fact"]})
    assert theirs == []
