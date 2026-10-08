"""Single-fact recall across conversations, through the real write and read paths.

The user says "The master lock code for the hazardous materials cage in Warehouse 3 is
8492." in one chat and asks "I need to get into the hazmat cage in Warehouse 3. What's the
code?" in a new one. The statement used to be the first chat's (THREAD) and the second chat
could not see it. What a user says now follows that user - and only that user: another
person, a workspace the statement was not shared with, and an agent's private notes keep
the audiences they had.
"""

# ruff: noqa: RUF001 - the Hindi fixtures are literal text in their own script

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    MessageRole,
    ObservationKind,
    Visibility,
)
from memory_service.domain.ids import new_id
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers
from tests.integration.test_memory import _memories, _observe

pytestmark = pytest.mark.integration

CODE = "The master lock code for the hazardous materials cage in Warehouse 3 is 8492."
ASK = "I need to get into the hazmat cage in Warehouse 3. What's the code?"


def _chat(user: str = "u1", **extra) -> MemoryExecutionContext:
    """A new conversation of ``user``: its own thread, session and turn."""
    fields = {
        "tenant_id": "acme",
        "user_id": user,
        "workspace_id": "ws1",
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    return MemoryExecutionContext(**{**fields, **extra})


async def _say(container, uow_factory, ctx, content: str, role=MessageRole.USER) -> None:
    """``POST /v1/messages``: the message, its thread, and the memories made from it."""
    register_handlers(container)
    async with uow_factory() as uow:
        await container.services["conversation"].append_message(
            uow, ctx, role=role, content=content
        )
        await uow.commit()
    await container.tasks.drain()  # memory.process_observation
    await container.tasks.drain()  # memory.index


async def _recalled(container, ctx, query: str) -> str:
    found = await container.services["retrieval"].retrieve(ctx, query, kinds=("memory",))
    return "\n".join(c.text for c in found.candidates)


async def _context(container, ctx, query: str):
    return await container.services["context_builder"].build(ctx, query)


async def test_a_code_said_in_one_chat_is_answered_in_the_next(container, uow_factory) -> None:
    monday, tuesday = _chat(), _chat()
    await _say(container, uow_factory, monday, CODE)
    await _say(container, uow_factory, monday, "Got it, noted.", role=MessageRole.ASSISTANT)

    bundle = await _context(container, tuesday, ASK)
    assert "8492" in bundle.render(), "the new conversation's context carries the code"
    assert "8492" in await _recalled(container, tuesday, ASK)

    said = [m for m in await _memories(uow_factory, monday, container) if "8492" in m.content]
    # the rule's reading (a fact about the cage) and the user's words as said, both theirs
    assert {m.system_metadata["category"] for m in said} == {"fact", "verbatim_turn"}
    assert all(m.visibility is Visibility.USER for m in said)
    assert all(m.content == CODE for m in said), "the exact sentence, never a paraphrase"
    assert all(
        m.system_metadata["visibility_keys"] == ["user:acme/u1", "principal:acme/user:u1"]
        for m in said
    )
    # one statement takes one slot in the bundle, not two
    assert sum("8492" in item.text for item in bundle.memories) == 1


async def test_a_bundle_cached_in_another_chat_learns_the_code(container, uow_factory) -> None:
    """The second chat's bundle was built (and cached) before the code was said elsewhere;
    the user's revision moves with the write, so the next build is not the stale one."""
    first, second = _chat(), _chat()
    before = await _context(container, second, ASK)
    assert "8492" not in before.render()
    await _say(container, uow_factory, first, CODE)
    after = await _context(container, second, ASK)
    assert not after.cache_hit and "8492" in after.render()


async def test_another_person_never_reads_it(container, uow_factory) -> None:
    await _say(container, uow_factory, _chat(), CODE)
    colleague = _chat("u2")  # same tenant, same workspace anchor, a different person
    assert "8492" not in await _recalled(container, colleague, ASK)
    assert "8492" not in (await _context(container, colleague, ASK)).render()


async def test_the_agent_acting_for_the_user_reads_it(container, uow_factory) -> None:
    await _say(container, uow_factory, _chat(), CODE)
    agent = _chat(agent_id="floor-assistant", agent_run_id=new_id("agent_run"))
    assert "8492" in await _recalled(container, agent, ASK)


async def test_a_turn_relayed_by_a_harness_is_still_the_users(container, uow_factory) -> None:
    """An agent harness posts the human's turn with its own lineage; the words are the
    user's, so they follow the user rather than becoming the agent's run notes."""
    relayed = _chat(agent_id="floor-assistant", agent_run_id=new_id("agent_run"))
    await _say(container, uow_factory, relayed, CODE)
    assert "8492" in await _recalled(container, _chat(), ASK), "the user, in a plain chat"


async def test_keeping_a_statement_to_its_conversation_is_still_possible(
    container, uow_factory
) -> None:
    here = _chat()
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, here)
        await uow.commit()
    await _observe(
        container, uow_factory, here, CODE, hints=ProcessingHints(visibility=Visibility.THREAD)
    )
    assert "8492" in await _recalled(container, here, ASK)
    assert "8492" not in await _recalled(container, _chat(), ASK)


async def test_what_the_assistant_said_stays_in_its_conversation(container, uow_factory) -> None:
    chat = _chat()
    await _say(
        container, uow_factory, chat, "Got it. The dock door code is 7731.", MessageRole.ASSISTANT
    )
    assert all(
        m.visibility is Visibility.THREAD
        for m in await _memories(uow_factory, chat, container)
        if "7731" in m.content
    )
    assert "7731" not in await _recalled(container, _chat(), "what is the dock door code?")


async def test_a_workspace_share_is_still_explicit(container, uow_factory) -> None:
    tenancy = container.services["tenancy"]
    async with uow_factory() as uow:
        await tenancy.create_workspace(uow, "acme", name="Warehouse ops", workspace_id="ops")
        for member in ("user:u1", "user:u2"):
            await tenancy.set_member(uow, "acme", "ops", member, role="member", added_by="test")
        await uow.commit()
    author = _chat(workspace_id="ops")
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, author)
        await uow.commit()
    await _observe(container, uow_factory, author, CODE)  # said, not shared
    rota = "The forklift charging rota for the ops team is posted on the break room door."
    await _observe(
        container, uow_factory, author, rota, hints=ProcessingHints(visibility=Visibility.WORKSPACE)
    )
    teammate = _chat("u2", workspace_id="ops")
    assert "break room door" in await _recalled(container, teammate, "where is the charging rota?")
    assert "8492" not in await _recalled(container, teammate, ASK), "membership is not consent"
    outsider = _chat("u3", workspace_id="ws1")
    assert "break room door" not in await _recalled(container, outsider, "charging rota")


async def test_an_agents_private_and_run_notes_stay_where_they_were(container, uow_factory) -> None:
    run = _chat(agent_id="planner", agent_run_id=new_id("agent_run"))
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, run)
        await uow.commit()
    await _observe(
        container,
        uow_factory,
        run,
        "Scratch: the cage code lookup failed on the first attempt.",
        kind=ObservationKind.AGENT_RESULT,
    )
    notes = [
        m for m in await _memories(uow_factory, run, container) if m.memory_type is MemoryType.AGENT
    ]
    assert [m.visibility for m in notes] == [Visibility.RUN]
    query = "did the cage code lookup fail?"
    assert "Scratch" in await _recalled(container, run, query)
    assert "Scratch" not in await _recalled(container, _chat(), query), "not the user"
    other = _chat(agent_id="planner", agent_run_id=new_id("agent_run"))
    assert "Scratch" not in await _recalled(container, other, query), "not a later run"
    async with uow_factory() as uow:
        await container.services["memory"].remember(
            uow,
            run,
            content="Private: retry the cage lookup with the warehouse prefix.",
            memory_type=MemoryType.SEMANTIC,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.PRIVATE,
        )
        await uow.commit()
    await container.tasks.drain()
    assert "warehouse prefix" in await _recalled(container, other, "retry the cage lookup")
    assert "warehouse prefix" not in await _recalled(container, _chat(), "retry the cage lookup")


@pytest.mark.parametrize(
    ("said", "asked", "code"),
    [
        (
            "Der Generalcode für das Schloss am Gefahrstoffkäfig in Lager 3 lautet 8492.",
            "Ich muss in den Gefahrstoffkäfig in Lager 3. Wie lautet der Code?",
            "8492",
        ),
        (
            "वेयरहाउस 3 में खतरनाक सामग्री वाले केज के मास्टर लॉक का कोड 8492 है।",
            "मुझे वेयरहाउस 3 के हैज़मैट केज में जाना है। कोड क्या है?",
            "8492",
        ),
    ],
    ids=["de", "hi"],
)
async def test_the_same_in_another_language(container, uow_factory, said, asked, code) -> None:
    """No rule reads German or Hindi; the user's own words carry the code across chats."""
    await _say(container, uow_factory, _chat(), said)
    tomorrow = _chat()
    assert code in await _recalled(container, tomorrow, asked)
    assert code in (await _context(container, tomorrow, asked)).render()
    assert code not in await _recalled(container, _chat("u2"), asked)


async def test_forgetting_or_correcting_the_code_takes_every_copy(container, uow_factory) -> None:
    """The fact and the turn are one statement: forgetting either forgets both, and a
    correction closes both, so the old code is never answered again."""
    said, later = _chat(), _chat()
    await _say(container, uow_factory, said, CODE)
    fact, turn = sorted(
        (m for m in await _memories(uow_factory, said, container) if m.content == CODE),
        key=lambda m: m.system_metadata["category"],
    )
    assert (fact.system_metadata["category"], turn.system_metadata["category"]) == (
        "fact",
        "verbatim_turn",
    )
    service = container.services["memory"]
    async with uow_factory() as uow:
        await service.supersede(
            uow, said, turn.memory_id, content=CODE.replace("8492", "5170"), reason="rotated"
        )
        await uow.commit()
    await container.tasks.drain()
    recalled = await _recalled(container, later, ASK)
    assert "5170" in recalled and "8492" not in recalled
    [current] = [m for m in await _memories(uow_factory, said, container) if "5170" in m.content]
    async with uow_factory() as uow:
        await service.forget(uow, said, current.memory_id)
        await uow.commit()
    await container.tasks.drain()
    recalled = await _recalled(container, later, ASK)
    assert "5170" not in recalled and "8492" not in recalled
