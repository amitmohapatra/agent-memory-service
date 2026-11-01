"""A short follow-up ("who is in charge of it?") names nothing the search can find; the
thread's previous question does. Its context is searched again with that question when its
own search finds nothing close (``builder.FOLLOW_UP_FLOOR``); a longer question, or one asked
outside a thread, is searched as asked."""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, MessageRole
from memory_service.domain.ids import new_id
from memory_service.modules.context import builder
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]

GOLD = "Marta Silva is running the Lisbon offsite and approves its budget."
MEMORIES = (
    "The Lisbon offsite is booked at Hotel Avenida from 12 to 14 May.",
    GOLD,
    "Our packaging shortlist is Boxly, CartonCo and PakRight.",
    "The Q3 board deck is due to Priya by 20 September.",
    "I am allergic to penicillin.",
)
ASKED = "Where is the Lisbon offsite booked?"
FOLLOW_UP = "Who is in charge of it?"


def _ctx(thread: str | None) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", user_id="u-follow", workspace_id="ws1", thread_id=thread
    )


async def _seed(container, *said: tuple[MessageRole, str]) -> str:
    """The user's memories, and a thread holding what was ``said`` in it."""
    thread = new_id("thread")
    async with container.services["uow_factory"]() as uow:
        for text in MEMORIES:
            await container.services["memory"].remember(
                uow,
                _ctx(None),
                content=text,
                memory_type=MemoryType.SEMANTIC,
                lifetime=Lifetime.LONG_TERM,
            )
        conversation = container.services["conversation"]
        await conversation.create_thread(uow, _ctx(thread))
        for role, text in said:
            await conversation.append_message(uow, _ctx(thread), role=role, content=text)
        await uow.commit()
    await container.tasks.drain()
    return thread


def _packed(bundle) -> list[str]:
    """What the context packed, closest to the question first."""
    items = (*bundle.memories, *bundle.knowledge, *bundle.graph_facts)
    return [item.text for item in sorted(items, key=lambda item: -item.relevance)]


def _rank(bundle) -> int | None:
    return next((i for i, text in enumerate(_packed(bundle), 1) if GOLD in text), None)


async def test_a_follow_up_is_searched_with_the_question_before_it(container) -> None:
    thread = await _seed(
        container,
        (MessageRole.USER, ASKED),
        (MessageRole.ASSISTANT, "Hotel Avenida, 12 to 14 May."),
    )
    contexts = container.services["context_builder"]
    alone = await contexts.build(_ctx(None), FOLLOW_UP, window=False)
    bundle = await contexts.build(_ctx(thread), FOLLOW_UP, window=False)
    # the gold ranks closer with the question before it (the hash stand-in's ranks: 5 -> 3;
    # the frozen encoder's on benchmark/follow_ups.py are in the module's docstring there)
    assert (_rank(bundle) or 99) < (_rank(alone) or 99)
    assert bundle.diagnostics["follow_up"] == ASKED
    assert bundle.query == FOLLOW_UP  # the coverage and the route stay the question's own


async def test_the_question_itself_appended_first_is_not_its_own_previous(container) -> None:
    """A caller may append the question before it asks for its context."""
    thread = await _seed(
        container,
        (MessageRole.USER, ASKED),
        (MessageRole.ASSISTANT, "Hotel Avenida."),
        (MessageRole.USER, FOLLOW_UP),
    )
    bundle = await container.services["context_builder"].build(
        _ctx(thread), FOLLOW_UP, window=False
    )
    assert bundle.diagnostics["follow_up"] == ASKED


async def test_a_longer_question_is_searched_as_asked(container) -> None:
    thread = await _seed(container, (MessageRole.USER, ASKED))
    longer = "Who is in charge of it, and who signs off the budget for the event?"
    assert len(longer.split()) > builder.FOLLOW_UP_WORDS
    bundle = await container.services["context_builder"].build(_ctx(thread), longer, window=False)
    assert "follow_up" not in bundle.diagnostics


async def test_a_question_whose_own_search_is_close_is_searched_as_asked(
    container, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gated on a weak first search: a short question on a new topic keeps its ranking."""
    thread = await _seed(container, (MessageRole.USER, ASKED))
    # every search counts as close: the floor sits under any similarity, a cosine being
    # negative for a candidate pointing away from the question (0.0 left those weak)
    monkeypatch.setattr(builder, "FOLLOW_UP_FLOOR", -1.0)
    bundle = await container.services["context_builder"].build(
        _ctx(thread), FOLLOW_UP, window=False
    )
    assert "follow_up" not in bundle.diagnostics


async def test_a_follow_up_about_two_answers_back_is_searched_with_both_questions(
    container,
) -> None:
    """ "Which of the two should I trust?" after two questions: the first one named what the
    follow-up points at (the break-and-fix round's bf1-mixed-06: the user's note on a lead
    time, found for the first question, was gone two questions later). The questions before
    it, newest first, within ``FOLLOW_UP_CHARS``."""
    second = "And the dates?"
    thread = await _seed(
        container,
        (MessageRole.USER, ASKED),
        (MessageRole.ASSISTANT, "Hotel Avenida."),
        (MessageRole.USER, second),
        (MessageRole.ASSISTANT, "12 to 14 May."),
    )
    bundle = await container.services["context_builder"].build(
        _ctx(thread), FOLLOW_UP, window=False
    )
    assert bundle.diagnostics["follow_up"] == f"{second} {ASKED}"
