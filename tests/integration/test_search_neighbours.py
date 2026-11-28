"""A turn a search finds comes with the conversation around it: the turn it replies to
(``before``) and the reply to it (``after``), as a reader of the conversation has them. A
found turn alone ("We went to the Tate Modern together.") does not say who "we" are or what
came of it; an agent searching on demand got isolated lines where an attached context had
the conversation."""

from __future__ import annotations

import pytest

from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

SAID = (
    "My sister Ana visited last weekend.",
    "We went to the Tate Modern together.",
    "She loved the Rothko room most.",
)


async def test_a_found_turn_comes_with_the_turn_before_and_the_reply(
    container, uow_factory
) -> None:
    here = _chat()
    for said in SAID:
        await _say(container, uow_factory, here, said)
    tools = container.services["agent_tools"]
    found = await tools.call(here, "memory_search", {"query": "Tate Modern", "k": 5})
    [turn] = [i for i in found if i["text"] == SAID[1]]
    assert SAID[0] in turn["before"] and SAID[2] in turn["after"], turn


async def test_a_neighbour_the_asker_may_not_read_is_left_out(container, uow_factory) -> None:
    """The turn around a found one is shown only to an asker who may read it: another
    person's private turn next to a shared one stays theirs."""
    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, U2, _observe

    await _observe(container, uow_factory, U1, "My salary is 90k.")  # u1's own
    await _observe(
        container,
        uow_factory,
        U1,
        "The team offsite is in Lisbon in May.",
        hints=ProcessingHints(visibility=Visibility.TENANT),
    )
    tools = container.services["agent_tools"]
    found = await tools.call(U2, "memory_search", {"query": "team offsite Lisbon", "k": 5})
    shared = [i for i in found if "offsite" in i["text"]]
    assert shared and all("90k" not in (i.get("before") or "") for i in shared)


async def test_a_neighbour_names_who_said_it(container, uow_factory) -> None:
    """A turn shared with the tenant keeps no user id of its own; its speaker is its subject
    (``user:u1``), as the context names it. "That's awesome, Caroline!" with no speaker
    left an agent unable to tell whose line it was - what a question about the wrong person
    turns on."""
    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, U2, _observe

    shared = ProcessingHints(visibility=Visibility.TENANT)
    await _observe(
        container, uow_factory, U1, "The team offsite is in Lisbon in May.", hints=shared
    )
    await _observe(
        container, uow_factory, U1, "We booked the old tram tour for day two.", hints=shared
    )
    tools = container.services["agent_tools"]
    found = await tools.call(U2, "memory_search", {"query": "old tram tour", "k": 5})
    [turn] = [i for i in found if "tram" in i["text"]]
    assert turn["before"] == f"{U1.user_id}: The team offsite is in Lisbon in May.", turn


async def test_a_found_turn_names_who_said_it(container, uow_factory) -> None:
    """A found turn says who said it, as the context's line does ("caroline: ..."): an
    agent searching on demand gave Melanie's "I'm a fan of Bach and Mozart" to Caroline, and
    said why - "the entry doesn't name the speaker"."""
    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, U2, _observe

    shared = ProcessingHints(visibility=Visibility.TENANT)
    await _observe(container, uow_factory, U1, "I'm a fan of Bach and Mozart.", hints=shared)
    tools = container.services["agent_tools"]
    found = await tools.call(U2, "memory_search", {"query": "Bach and Mozart", "k": 5})
    [turn] = [i for i in found if "Bach" in i["text"]]
    assert turn["speaker"] == U1.user_id, turn


async def test_a_search_without_k_returns_what_a_context_would_carry(
    container, uow_factory
) -> None:
    """Without ``k`` a search returns what memory ranks for the request, as deep as a context
    would carry it - not eight items. An agent searching on demand saw 20 lines in two
    searches where the pushed context carried a hundred, and found every piece of a
    many-part answer half as often."""
    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, _observe

    shared = ProcessingHints(visibility=Visibility.TENANT)
    for n in range(30):
        await _observe(
            container, uow_factory, U1, f"Trip note {n}: we hiked trail number {n}.", hints=shared
        )
    tools = container.services["agent_tools"]
    narrowed = await tools.call(U1, "memory_search", {"query": "hiking trips", "k": 5})
    whole = await tools.call(U1, "memory_search", {"query": "hiking trips"})
    assert len(narrowed) == 5
    assert len([i for i in whole if "Trip note" in i["text"]]) >= 25, len(whole)


async def test_a_found_turn_says_the_dates_it_names(container, uow_factory) -> None:
    """A found turn carries its relative dates resolved against the day it was said, as the
    context's line does ("yesterday = 2023-05-07"): the agent searching on demand otherwise
    works them out itself, from the day alone."""
    from datetime import UTC, datetime, timedelta

    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, _observe

    shared = ProcessingHints(visibility=Visibility.TENANT)
    await _observe(container, uow_factory, U1, "I went to a pottery class yesterday.", hints=shared)
    tools = container.services["agent_tools"]
    found = await tools.call(U1, "memory_search", {"query": "pottery class", "k": 5})
    [turn] = [i for i in found if "pottery" in i["text"]]
    yesterday = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    assert turn["dates"] == f"yesterday = {yesterday}", turn


async def test_a_turn_fetched_by_id_keeps_the_dates_it_names(container, uow_factory) -> None:
    """A turn the engine fetches by id - what a reading finds, a replacement, a derived
    memory's source - carries its resolved dates as an indexed hit does: the context printed
    "(yesterday = ...)" for a turn found by search and nothing for the same turn fetched."""
    from memory_service.domain.enums import Visibility
    from memory_service.domain.memory import VERBATIM_TURN
    from memory_service.domain.observation import ProcessingHints
    from memory_service.modules.retrieval.engine import memory_candidate
    from tests.integration.test_memory import U1, _memories, _observe

    shared = ProcessingHints(visibility=Visibility.TENANT)
    await _observe(container, uow_factory, U1, "I went to a pottery class yesterday.", hints=shared)
    [turn] = [
        m
        for m in await _memories(uow_factory, U1, container)
        if m.system_metadata.get("category") == VERBATIM_TURN
    ]
    fetched = memory_candidate(turn, retriever="exact", score=1.0)
    assert fetched.payload["dated_mentions"] == turn.system_metadata["dated_mentions"]


async def test_a_search_can_keep_one_persons_words(container, uow_factory) -> None:
    """``said_by`` keeps what one person said in a ranked search too: another person's id
    as items name them (``speaker``), ``user`` for the person asking. A conversation of two
    people about one thing gave one's words to the other; the agent could not ask for one
    person's alone."""
    from memory_service.domain.enums import Visibility
    from memory_service.domain.observation import ProcessingHints
    from tests.integration.test_memory import U1, U2, _observe

    shared = ProcessingHints(visibility=Visibility.TENANT)
    await _observe(container, uow_factory, U1, "I'm a fan of Bach and Mozart.", hints=shared)
    await _observe(container, uow_factory, U2, "I love Bach's cello suites.", hints=shared)
    tools = container.services["agent_tools"]
    theirs = await tools.call(U1, "memory_search", {"query": "Bach", "said_by": U2.user_id})
    mine = await tools.call(U1, "memory_search", {"query": "Bach", "said_by": "user"})
    assert theirs and {i.get("speaker") for i in theirs} == {U2.user_id}, theirs
    assert mine and {i.get("speaker") for i in mine} == {U1.user_id}, mine
