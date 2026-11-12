"""Whose words a message is: its role says who spoke.

An assistant's message was read as something a person said whenever it carried no agent run
id (a chat application posting both sides of a conversation, an import of one): the
assistant's "Never change a retail price without checking the pricing calendar first." became
the user's standing rule, its "I prefer ..." their preference, and its advice their facts. The
harness sends a run id with every assistant message, and those formed nothing already. Now a
message whose role is the assistant's or an agent's forms no memory with or without a run id;
it stays in the transcript and is found by a message search, as before.
"""

from __future__ import annotations

import pytest

from memory_service.domain.enums import MemoryType, MessageRole
from tests.integration.test_memory import _memories
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

#: what the user says, and what the assistant says beside it in the same shapes: a rule, a
#: self-description and a preference
USER_SAID = (
    "Never send a weekly email to the directors before I have read it.",
    "I'm Alex. I plan Coach Womens Bags for the Retail channel.",
    "I prefer short answers in bullet points.",
)
ASSISTANT_SAID = (
    "Never change a retail price without checking the pricing calendar first.",
    "I'm Sam. I plan Mens Shoes for the Outlet channel.",
    "I prefer to see sell-through before margin in any summary.",
)
#: a word of each assistant sentence that no user sentence has
ASSISTANTS_WORDS = ("pricing calendar", "Sam", "Mens Shoes", "sell-through")
ASKED = "Draft the weekly email about the retail price summary."


@pytest.mark.parametrize("run_id", [None, "run_1"])
async def test_an_assistants_message_is_not_the_users_statement(
    container, uow_factory, run_id: str | None
) -> None:
    lineage = {"agent_id": "analyst", "agent_run_id": run_id} if run_id else {}
    chat = _chat(**lineage)
    for said, replied in zip(USER_SAID, ASSISTANT_SAID, strict=True):
        await _say(container, uow_factory, chat, said)
        await _say(container, uow_factory, chat, replied, role=MessageRole.ASSISTANT)
    await container.services["profile"].refresh_user("acme", "u1")

    memories = await _memories(uow_factory, chat, container)
    theirs = [m.content for m in memories if any(w in m.content for w in ASSISTANTS_WORDS)]
    assert theirs == [], "nothing is formed from what the assistant said"
    # the user's words beside them still are: a rule, a self-description, a preference
    kept = {m.content: m for m in memories}
    assert kept[USER_SAID[0]].memory_type is MemoryType.PREFERENCE
    assert kept[USER_SAID[1]].system_metadata.get("self_description") == "ws1"
    assert kept[USER_SAID[2]].memory_type is MemoryType.PREFERENCE

    bundle = await container.services["context_builder"].build(_chat(**lineage), ASKED)
    sections = {block.block: block.text for block in bundle.profile}
    profile = "\n".join(sections.values())
    assert not any(word in profile for word in ASSISTANTS_WORDS)
    assert USER_SAID[0] in sections["standing_rules"]
    assert USER_SAID[1] in sections["about_user"]
    assert USER_SAID[2] in sections["user"]

    # still in the transcript, and found by a message search
    found = await container.services["agent_tools"].call(
        _chat(**lineage),
        "memory_search",
        {"query": "Mens Shoes for the Outlet channel", "kinds": ["message"]},
    )
    assert any(i["kind"] == "message" and ASSISTANT_SAID[1] in i["text"] for i in found)
