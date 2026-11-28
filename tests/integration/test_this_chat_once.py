"""This conversation's own turns reach the model through its window and summary, not a second
time as memories; another conversation's turns, and what was read from this one's, do."""

from __future__ import annotations

import pytest

from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

TURNS = ("We are repricing the Globex quotes this week.", "Quote Q-1183 is the first one.")


async def test_this_chats_turns_are_not_memories_of_it(container, uow_factory) -> None:
    here = _chat()
    for said in TURNS:
        await _say(container, uow_factory, here, said)
    builder = container.services["context_builder"]
    same = await builder.build(here, "Which Globex quote do we reprice first?")
    assert not any(t in m.text for m in same.memories for t in TURNS)
    assert any(TURNS[1] in m.text for m in same.conversation.messages)
    other = await builder.build(_chat(), "Which Globex quote do we reprice first?")
    assert any(TURNS[1] in m.text for m in other.memories), "another chat recalls them"


async def test_a_summary_written_without_a_model_is_not_carried(container, uow_factory) -> None:
    """Without a model the "summary" is the earlier turns pieced together - the turns again:
    the context carries the latest turns in the window and no such summary; a search finds the
    rest."""
    from memory_service.domain.profile import ThreadSummary
    from memory_service.modules.conversation.summary import EXTRACTIVE

    here = _chat()
    await _say(container, uow_factory, here, TURNS[0])
    async with uow_factory() as uow:
        await uow.summaries.add(
            ThreadSummary(
                tenant_id="acme",
                thread_id=here.thread_id,
                version=1,
                text="user: " + TURNS[0],
                covers_to_sequence=1,
                model=EXTRACTIVE,
            )
        )
        await uow.commit()
    bundle = await container.services["context_builder"].build(here, "What are we repricing?")
    assert bundle.thread_summary is None
    assert any(TURNS[0] in m.text for m in bundle.conversation.messages), "the window has it"
