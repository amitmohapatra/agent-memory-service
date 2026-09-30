"""The record path on PostgreSQL: messages without session or turn ids, stated memories stored
synchronously and deduplicated, and bi-temporal supersession."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    MessageRole,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import Conflict, NotFound, ScopeDenied
from memory_service.domain.ids import new_id, thread_session_id, thread_turn_id
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
U2 = MemoryExecutionContext(tenant_id="acme", user_id="u2", workspace_id="ws1")


async def _say(container, uow_factory, ctx, role: MessageRole, content: str):
    async with uow_factory() as uow:
        result = await container.services["conversation"].append_message(
            uow, ctx, role=role, content=content
        )
        await uow.commit()
    return result.ack


async def test_a_message_needs_only_its_thread(container, uow_factory) -> None:
    thread = new_id("thread")
    ctx = U1.model_copy(update={"thread_id": thread})
    first = await _say(container, uow_factory, ctx, MessageRole.USER, "How much was revenue?")
    answer = await _say(container, uow_factory, ctx, MessageRole.ASSISTANT, "EUR 412 million.")
    second = await _say(container, uow_factory, ctx, MessageRole.USER, "And EBITDA?")

    session = thread_session_id("acme", thread)
    assert {first.session_id, answer.session_id, second.session_id} == {session}
    # a question opens the next turn; what is said while answering it joins that turn
    assert first.turn_id == answer.turn_id == thread_turn_id("acme", thread, 1)
    assert second.turn_id == thread_turn_id("acme", thread, 2)
    assert [first.sequence, answer.sequence, second.sequence] == [1, 2, 3]
    async with uow_factory() as uow:
        turn = await uow.turns.get("acme", second.turn_id)
    assert turn is not None and turn.sequence == 2 and turn.session_id == session

    # a caller that names its lineage keeps it
    named = ctx.model_copy(update={"session_id": new_id("session"), "turn_id": new_id("turn")})
    explicit = await _say(container, uow_factory, named, MessageRole.USER, "Thanks.")
    assert (explicit.session_id, explicit.turn_id) == (named.session_id, named.turn_id)
    # ...and a reply without a turn does not join a turn of another session
    reply = await _say(container, uow_factory, ctx, MessageRole.ASSISTANT, "You're welcome.")
    assert reply.turn_id == thread_turn_id("acme", thread, 4)


async def test_a_stated_memory_is_stored_now_and_once(container, uow_factory) -> None:
    register_handlers(container)
    service = container.services["memory"]
    async with uow_factory() as uow:
        ack = await service.remember(
            uow,
            U1,
            content="Prefers metric units.",
            memory_type=MemoryType.PREFERENCE,
            lifetime=Lifetime.LONG_TERM,
            entities=["Metric System"],
        )
        await uow.commit()
    assert not ack.deduplicated and ack.job_ids
    # readable before any job ran: the statement is the memory
    async with uow_factory() as uow:
        stored = await service.get_memory(uow, U1, ack.memory_id)
    assert stored.content == "Prefers metric units." and stored.subject == "user:u1"
    assert stored.memory_type is MemoryType.PREFERENCE and stored.visibility is Visibility.USER
    assert stored.evidence[0].source_type == "statement"
    assert stored.system_metadata["category"] == "stated"

    async with uow_factory() as uow:
        again = await service.remember(
            uow,
            U1,
            content="Prefers  metric units.",  # the same content, normalised
            memory_type=MemoryType.PREFERENCE,
            lifetime=Lifetime.LONG_TERM,
        )
        await uow.commit()
    assert again.deduplicated and again.memory_id == ack.memory_id and again.job_ids == []

    await container.tasks.drain()  # memory.index: search + graph
    found = await container.services["retrieval"].retrieve(U1, "metric units", kinds=("memory",))
    assert ack.memory_id in {c.record_id for c in found.candidates}
    entities = await container.services["graph"].search_entities(U1, query="metric")
    assert "metric system" in {e.canonical_name for e in entities}, "declared entities are linked"


async def test_supersede_closes_the_old_version_and_keeps_it_readable(
    container, uow_factory
) -> None:
    register_handlers(container)
    service = container.services["memory"]
    async with uow_factory() as uow:
        ack = await service.remember(
            uow,
            U1,
            content="The release review is on Tuesdays.",
            memory_type=MemoryType.SEMANTIC,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.USER,
        )
        await uow.commit()
    with pytest.raises(ScopeDenied):
        async with uow_factory() as uow:
            await service.supersede(
                uow, U2, ack.memory_id, content="Wednesdays.", reason="not yours"
            )
    before = datetime.now(UTC)
    async with uow_factory() as uow:
        new = await service.supersede(
            uow, U1, ack.memory_id, content="The release review is on Thursdays.", reason="moved"
        )
        await uow.commit()
    async with uow_factory() as uow:
        old = await service.get_memory(uow, U1, ack.memory_id)
        current = await service.get_memory(uow, U1, new.memory_id)
    assert old.temporal.status is TemporalStatus.SUPERSEDED
    assert old.temporal.superseded_by == new.memory_id and old.temporal.valid_to >= before
    assert old.system_metadata["supersede_reason"] == "moved"
    assert current.temporal.status is TemporalStatus.CURRENT
    assert current.temporal.supersedes == ack.memory_id
    assert current.visibility is Visibility.USER and current.owner_principal == "user:u1"
    async with uow_factory() as uow:
        assert await uow.memories.visibility_keys("acme", new.memory_id) == (
            await uow.memories.visibility_keys("acme", ack.memory_id)
        ), "the new version keeps the old audience"
    with pytest.raises(Conflict):
        async with uow_factory() as uow:
            await service.supersede(uow, U1, ack.memory_id, content="Fridays.", reason="stale")

    await container.tasks.drain()
    listed = await container.services["retrieval"].retrieve(U1, "release review", kinds=("memory",))
    ids = {c.record_id for c in listed.candidates}
    assert new.memory_id in ids and ack.memory_id not in ids, "only the current version"

    async with uow_factory() as uow:
        await service.forget(uow, U1, new.memory_id)
        await uow.commit()
    with pytest.raises(NotFound):
        async with uow_factory() as uow:
            await service.get_memory(uow, U1, new.memory_id)
