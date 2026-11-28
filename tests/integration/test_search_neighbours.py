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
