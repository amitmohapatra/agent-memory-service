"""The gap-fill call through durable ingestion: stored, scoped, recalled, replay-safe and
forgettable; and nothing at all without a model."""

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.gapfill import GapFill
from tests.integration.test_memory import U1, U2, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

FIRST = "I finally adopted a dog last month."
SECOND = "Her name is Biscuit, a beagle mix."
FACT = "u1 adopted a beagle mix named Biscuit."
OPS = {
    "ops": [
        {
            "op": "add",
            "fact": FACT,
            "quote": "Her name is Biscuit, a beagle mix.",
            "type": "fact",
            "subject": "",
            "date": "",
            "confidence": 0.9,
            "target": 0,
        }
    ]
}
USER = ProcessingHints(visibility=Visibility.USER)


def _shape(memories):
    return sorted(
        (m.content, m.memory_type.value, m.system_metadata.get("category"), m.visibility.value)
        for m in memories
    )


async def test_a_gap_fill_fact_is_stored_recalled_replay_safe_and_forgettable(
    container, uow_factory
):
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"ops": []}, OPS]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, FIRST, hints=USER)
        ack = await _observe(container, uow_factory, U1, SECOND, hints=USER)
        sent = gateway.prompts()[-1]["messages"][1]["content"]
    assert FIRST in sent  # the turn before it is read with it
    memories = await _memories(uow_factory, U1, container)
    fact = next(m for m in memories if m.content == FACT)
    assert fact.system_metadata.get("provider") == "llm"
    turn = next(m for m in memories if m.content == SECOND)
    assert fact.visibility is turn.visibility and fact.evidence == turn.evidence
    assert await pipeline.run({"tenant_id": "acme", "observation_id": ack.observation_id}) == []
    assert len(await _memories(uow_factory, U1, container)) == len(memories)
    engine = container.services["retrieval"]
    found = await engine.retrieve(U1, "What breed is my dog Biscuit?", kinds=("memory",))
    assert any(
        fact.memory_id in (c.record_id, *c.payload.get("duplicates", ())) for c in found.candidates
    )
    assert (await engine.retrieve(U2, f"show {fact.memory_id}")).candidates == []
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, U1, fact.memory_id)
        await uow.commit()
    await container.tasks.drain()
    assert fact.memory_id not in {m.memory_id for m in await _memories(uow_factory, U1, container)}


async def test_without_a_model_the_memories_are_the_rules_alone(container, uow_factory):
    """The no-model path is the rules' path: wired or not, the gap-fill adds nothing and
    asks nothing."""
    pipeline = container.services["observation_pipeline"]
    assert pipeline.gapfill is not None
    for text in (FIRST, SECOND):
        await _observe(container, uow_factory, U1, text, hints=USER)
    wired = _shape(await _memories(uow_factory, U1, container))
    pipeline.gapfill = None
    for text in (FIRST, SECOND):
        await _observe(container, uow_factory, U2, text, hints=USER)
    unwired = _shape(await _memories(uow_factory, U2, container))
    assert wired == unwired
    assert all(category != "source_fact" for _, _, category, _ in wired)


async def test_a_rule_reading_the_model_read_again_is_held_once(container, uow_factory):
    """vdate-4 of the hard eval: the rules read "The audit is due the Friday before
    Christmas." and the model the same words with the day resolved: one memory, the
    model's."""
    said = "The audit is due the Friday before Christmas."
    resolved = "The audit is due on 2026-12-18."
    ops = {
        "ops": [
            {
                "op": "add",
                "fact": resolved,
                "quote": said,
                "type": "plan",
                "subject": "audit",
                "date": "2026-12-18",
                "confidence": 0.9,
                "target": 0,
            }
        ]
    }
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([ops]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, said, hints=USER)
    readings = [
        m.content
        for m in await _memories(uow_factory, U1, container)
        if m.system_metadata.get("category") != "verbatim_turn" and "audit" in m.content
    ]
    assert readings == [resolved]


async def test_a_reply_the_model_read_whole_is_not_kept_again_with_its_question(
    container, uow_factory
):
    """spread-2 of the hard eval: "In Tacoma. It handles all the West Coast returns." after
    "Where is it?" was kept with the question beside it, and the model read both sentences
    with "it" resolved: the same words held twice. A reply the model did not read whole
    keeps the question beside it (tests/integration/test_referred_turn.py)."""
    from datetime import UTC, datetime, timedelta

    from tests.integration.test_referred_turn import _contents, _turn

    reply = "In Tacoma. It handles all the West Coast returns."
    quotes = ("In Tacoma.", "It handles all the West Coast returns.")
    facts = ("Ada's second warehouse is in Tacoma.", "Ada's Tacoma warehouse handles returns.")
    ops = {
        "ops": [
            {
                "op": "add",
                "fact": fact,
                "quote": quote,
                "type": "fact",
                "subject": "",
                "date": "",
                "confidence": 0.9,
                "target": 0,
            }
            for fact, quote in zip(facts, quotes, strict=True)
        ]
    }
    at = datetime(2026, 10, 30, 9, tzinfo=UTC)
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"ops": []}, ops]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _turn(container, uow_factory, "We opened a second warehouse.", at=at, role="user")
        await _turn(
            container, uow_factory, "Where is it?", at=at + timedelta(minutes=1), role="assistant"
        )
        await _turn(container, uow_factory, reply, at=at + timedelta(minutes=2), role="user")
    held = await _contents(container, uow_factory)
    assert not [c for c in held if "(in reply to:" in c], held
    assert set(facts) <= set(held)
