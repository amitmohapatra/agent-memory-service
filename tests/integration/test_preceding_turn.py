"""A message remembers the conversation turn said just before it (index_preceding_turn)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from benchmark.common import submit_observation

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

ASK = MemoryExecutionContext(tenant_id="acme", user_id="melanie", workspace_id="ws1")
REPLY = MemoryExecutionContext(tenant_id="acme", user_id="caroline", workspace_id="ws1")
ELSEWHERE = MemoryExecutionContext(tenant_id="acme", user_id="melanie", workspace_id="ws2")
WHEN = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)


async def _say(container, uow_factory, ctx, content, *, at=WHEN):
    register_handlers(container)
    async with uow_factory() as uow:
        ack = await submit_observation(uow, ctx, content=content, occurred_at=at)
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack


async def _verbatim(container, uow_factory, ctx):
    async with uow_factory() as uow:
        mems = await container.services["memory"].list_memories(uow, ctx)
    return [m for m in mems if m.system_metadata.get("category") == "verbatim_turn"]


async def test_a_reply_carries_the_question_it_answers(container, uow_factory) -> None:
    asked = await _say(container, uow_factory, ASK, "Did you go camping last weekend?")
    # another workspace speaking in between is a different conversation
    await _say(container, uow_factory, ELSEWHERE, "The quarterly report is due on Friday.")
    await _say(container, uow_factory, REPLY, "Yes, with my kids at the lake, we loved it.")
    [reply] = await _verbatim(container, uow_factory, REPLY)
    prior = reply.system_metadata["preceding_turn"]
    assert prior["text"] == "Did you go camping last weekend?"
    assert prior["speaker"] == "melanie"
    async with uow_factory() as uow:
        obs = await uow.observations.get("acme", asked.observation_id)
    assert obs is not None
    assert prior["source_id"] == (obs.message_id or obs.observation_id)
    # the payload carries the link retrieval scores neighbours by
    engine = container.services["retrieval"]
    res = await engine.retrieve(REPLY, "camping with kids at the lake", kinds=("memory",))
    linked = [c for c in res.candidates if c.payload.get("preceding_source_id")]
    assert linked and linked[0].payload["preceding_source_id"] == prior["source_id"]


async def test_a_turn_from_another_session_is_not_its_context(container, uow_factory) -> None:
    await _say(container, uow_factory, ASK, "Did you go camping last weekend?")
    await _say(
        container,
        uow_factory,
        REPLY,
        "Yes, with my kids at the lake, we loved it.",
        at=WHEN + timedelta(days=3),
    )
    [reply] = await _verbatim(container, uow_factory, REPLY)
    assert "preceding_turn" not in reply.system_metadata


async def test_a_reply_is_found_by_its_own_words_and_by_the_question_it_answers(
    container, uow_factory
) -> None:
    """Two keys (ADR 0025): the reply alone, and the reply read after its question. A word
    only the question said reaches the reply through the second key and not the first."""
    await _say(container, uow_factory, ASK, "Did you go camping last weekend?")
    await _say(container, uow_factory, REPLY, "Yes, with my kids at the lake, we loved it.")
    [reply] = await _verbatim(container, uow_factory, REPLY)
    engine = container.services["retrieval"]
    result = await engine.retrieve(REPLY, "camping", kinds=("memory",))
    assert reply.memory_id in {c.record_id for c in result.candidates}
    [hit] = [c for c in result.candidates if c.record_id == reply.memory_id]
    assert "bm25_ctx" in hit.retrievers and "bm25" not in hit.retrievers
    own = await engine.retrieve(REPLY, "kids at the lake", kinds=("memory",))
    [hit] = [c for c in own.candidates if c.record_id == reply.memory_id]
    assert {"bm25", "bm25_ctx"} <= set(hit.retrievers)
