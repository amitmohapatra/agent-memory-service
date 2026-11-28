"""What a model read from a message is a key to the message, not its stand-in: a reading the
search finds without its turn comes as the turn kept as said, in the reading's place, with the
reading among its duplicates. A context with the model on carried "u1 adopted a beagle mix
named Biscuit." - "model-extracted, unverified" - where the turn said it whole, and three
readings of one message each took a line of their own. LongMemEval (Wu et al., 2024): facts
as keys find the round; the round as said is what is read."""

from __future__ import annotations

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.gapfill import GapFill
from memory_service.modules.retrieval.engine import memory_candidate
from tests.integration.test_gapfill_memory import FIRST, OPS, SECOND
from tests.integration.test_memory import U1, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

FACT = OPS["ops"][0]["fact"]


async def _stored(container, uow_factory):
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"kind": "statement", "ops": []}, OPS]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        for said in (FIRST, SECOND):
            await _observe(
                container, uow_factory, U1, said, hints=ProcessingHints(visibility=Visibility.USER)
            )
    memories = await _memories(uow_factory, U1, container)
    return (
        next(m for m in memories if m.content == FACT),
        next(m for m in memories if m.content == SECOND),
    )


async def test_a_reading_found_alone_comes_as_its_turn(container, uow_factory) -> None:
    fact, turn = await _stored(container, uow_factory)
    engine = container.services["retrieval"]
    visibility = await engine.authz.visibility(U1, uow_factory=uow_factory)
    diagnostics: dict = {}
    found = memory_candidate(fact, retriever="dense", score=0.4)
    [said] = await engine._said_for_readings(U1, [found], visibility, diagnostics)
    assert said.record_id == turn.memory_id and said.text == SECOND
    assert fact.memory_id in said.payload["duplicates"] and said.score == found.score
    assert diagnostics["said_for_readings"] == 1


async def test_an_exact_lookup_keeps_the_reading_it_names(container, uow_factory) -> None:
    fact, _ = await _stored(container, uow_factory)
    engine = container.services["retrieval"]
    found = await engine.retrieve(U1, f"show {fact.memory_id}")
    assert found.candidates[0].record_id == fact.memory_id


async def test_the_context_shows_the_turn_and_not_the_reading(container, uow_factory) -> None:
    await _stored(container, uow_factory)
    bundle = await container.services["context_builder"].build(U1, "What breed is my dog?")
    rendered = bundle.render()
    assert SECOND in rendered and FACT not in rendered, rendered
