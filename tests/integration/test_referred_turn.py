"""A short turn that refers back to the agent's answer ("Approve it", "go with that") keeps what
it refers to: the answer's headline beside the user's words, so a later chat recalls "you
approved 19 units", not "you approved it" (break-and-fix round 2: the figure was lost)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.modules.conversation.service import TASK_PROCESS_OBSERVATION
from memory_service.modules.jobs.registry import register_handlers
from memory_service.ports.tasks import JobSpec, Queue

pytestmark = pytest.mark.integration

USER = MemoryExecutionContext(tenant_id="acme", user_id="ada", workspace_id="ws1")
WHEN = datetime(2026, 10, 30, 9, tzinfo=UTC)
ASKED = (
    "Reorder qty of the Cotton Cardigan Burgundy for the north? reorder qty = average daily "
    "units sold over 2026-09-03..2026-09-30 (28 days) x (supplier lead time + 7 review days) - "
    "units on hand, rounded up, never below 0 (north only)."
)
LATER = (
    "Today's stock is in. How far is my approved quantity for the Cotton Cardigan Burgundy from "
    "what the formula gives today, and what would today's number be if the lead time were 5 "
    "days longer?"
)
ANSWER = (
    "**Reorder quantity for Cotton Cardigan Burgundy (SKU-00116), north: 19 units**\n\n"
    "- North units sold, 3-30 Sep 2026: 15, so 0.5357 a day\n- Supplier lead time: 38 days"
)


async def _turn(container, uow_factory, content: str, *, at: datetime, role: str) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        observation = Observation(
            **USER.provenance(),
            kind=ObservationKind.MESSAGE,
            content=content,
            content_hash=content_hash(content),
            hints=ProcessingHints(),
            occurred_at=at,
            custom_metadata={"role": role},
        )
        await uow.observations.add(observation)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_PROCESS_OBSERVATION,
                queue=Queue.CHAT_FAST,
                payload={"tenant_id": "acme", "observation_id": observation.observation_id},
                idempotency_key=f"obs:{observation.observation_id}",
                tenant_id="acme",
            )
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _contents(container, uow_factory) -> list[str]:
    async with uow_factory() as uow:
        return [m.content for m in await container.services["memory"].list_memories(uow, USER)]


async def test_approve_it_keeps_what_the_agents_answer_said(container, uow_factory) -> None:
    await _turn(container, uow_factory, ANSWER, at=WHEN, role="ASSISTANT")
    await _turn(
        container,
        uow_factory,
        "Approve it as my north order for this item.",
        at=WHEN + timedelta(minutes=1),
        role="USER",
    )
    kept = await _contents(container, uow_factory)
    referred = [c for c in kept if "Approve it" in c and "19 units" in c]
    assert referred, kept
    assert "**" not in referred[0]  # the headline, as text


async def test_a_turn_that_refers_to_nothing_before_it_is_kept_as_said(
    container, uow_factory
) -> None:
    await _turn(container, uow_factory, "Approve it.", at=WHEN, role="USER")
    assert all("in reply to" not in c for c in await _contents(container, uow_factory))


async def test_a_statement_that_stands_alone_is_not_tied_to_the_answer(
    container, uow_factory
) -> None:
    await _turn(container, uow_factory, ANSWER, at=WHEN, role="ASSISTANT")
    await _turn(
        container,
        uow_factory,
        "Our autumn budget for knitwear is 40,000 pounds.",
        at=WHEN + timedelta(minutes=1),
        role="USER",
    )
    assert all("19 units" not in c for c in await _contents(container, uow_factory))


async def test_a_later_chat_recalls_the_figure_it_approved(container, uow_factory) -> None:
    """The recall the stored turn exists for (break-and-fix re-measure, Claude and Kimi on
    ``bf2-mixed-01``): the next chat recalls "Approve it ... 19 units". The bare turn and the
    referring one are one statement (``memory_ranking.statements``), and the bare one stood
    for it, so the context said "Approve it as my north order for this item." and no figure."""
    from memory_service.domain.enums import MessageRole
    from tests.integration.test_statements_follow_the_user import _chat, _recalled, _say

    chat = _chat("ada")
    await _say(container, uow_factory, chat, ASKED)
    await _say(container, uow_factory, chat, ANSWER, role=MessageRole.ASSISTANT)
    await _say(container, uow_factory, chat, "Approve it as my north order for this item.")
    recalled = await _recalled(container, _chat("ada"), LATER)
    approved = [line for line in recalled.splitlines() if line.startswith("Approve it")]
    assert approved and all("19 units" in line for line in approved), recalled
