"""Thread summaries and profile blocks against PostgreSQL: the jobs and their triggers."""

from __future__ import annotations

import json

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, MessageRole
from memory_service.domain.errors import Conflict
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.conversation.summary import SUMMARY_EVERY, ThreadSummaries
from memory_service.modules.profile.service import ProfileService
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

ANN = MemoryExecutionContext(tenant_id="acme", user_id="ann", thread_id="thr_sum")


async def _say(container, uow_factory, n: int, start: int = 0) -> None:
    conversation = container.services["conversation"]
    for i in range(start, start + n):
        async with uow_factory() as uow:
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await conversation.append_message(uow, ANN, role=role, content=f"Message {i}. More.")
            await uow.commit()


async def test_every_summary_every_messages_the_summary_rolls_forward(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, SUMMARY_EVERY - 1)
    async with uow_factory() as uow:
        assert await uow.summaries.latest("acme", "thr_sum") is None
    await _say(container, uow_factory, 1, start=SUMMARY_EVERY - 1)
    await container.tasks.drain()
    async with uow_factory() as uow:
        first = await uow.summaries.latest("acme", "thr_sum")
    assert first is not None and first.version == 1 and first.model == "extractive"
    assert first.covers_to_sequence == SUMMARY_EVERY
    assert first.text.splitlines()[0] == "user: Message 0."

    await _say(container, uow_factory, SUMMARY_EVERY, start=SUMMARY_EVERY)
    await container.tasks.drain()
    async with uow_factory() as uow:
        second = await uow.summaries.latest("acme", "thr_sum")
    assert second is not None and second.version == 2
    assert second.covers_to_sequence == 2 * SUMMARY_EVERY
    assert second.text.startswith(first.text), "rolled forward from the previous version"


async def test_with_a_model_the_previous_summary_and_the_new_messages_are_folded(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, 3)
    with mocked_gateway([json.dumps({"summary": "Ann asked about three things."})]) as gateway:
        summaries = ThreadSummaries(uow_factory, gateway.assist(uses=["summaries"]))
        stored = await summaries.refresh("acme", "thr_sum", principal_id="user:ann")
        prompt = gateway.prompts()[0]["messages"][1]["content"]
    assert stored is not None and stored.text == "Ann asked about three things."
    assert stored.model == "test/strong" and "Previous summary:\n(none)" in prompt
    again = await summaries.refresh("acme", "thr_sum", principal_id="user:ann")
    assert again == stored, "nothing new: nothing written"


async def _remember(container, uow_factory, content: str, memory_type: MemoryType) -> None:
    async with uow_factory() as uow:
        await container.services["memory"].remember(
            uow, ANN, content=content, memory_type=memory_type, lifetime=Lifetime.LONG_TERM
        )
        await uow.commit()
    await container.tasks.drain()  # memory.index -> profile.refresh


async def test_the_user_block_is_kept_from_what_the_user_said_and_an_edit_is_kept(
    container, uow_factory
) -> None:
    await _remember(container, uow_factory, "I prefer email over phone", MemoryType.PREFERENCE)
    profile: ProfileService = container.services["profile"]
    async with uow_factory() as uow:
        [block] = await profile.blocks(uow, ANN)
    assert block.block == "user" and block.text == "- I prefer email over phone"
    assert block.source == "learned"

    async with uow_factory() as uow:
        before = await uow.revisions.get_many("acme", [(RevisionKind.USER, "ann")])
        edited = await profile.edit(uow, ANN, "user", "email", "e-mail")
        await uow.commit()
        after = await uow.revisions.get_many("acme", [(RevisionKind.USER, "ann")])
    assert edited.text == "- I prefer e-mail over phone" and edited.version == 2
    assert after != before

    await _remember(container, uow_factory, "My desk is in Berlin", MemoryType.USER)
    async with uow_factory() as uow:
        [block] = await profile.blocks(uow, ANN)
    assert block.text == "- I prefer e-mail over phone\n- My desk is in Berlin"
    assert block.source == "edited" and block.version == 3

    with pytest.raises(Conflict):
        async with uow_factory() as uow:
            await profile.edit(uow, ANN, "user", "phone number", "x")
