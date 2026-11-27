"""A person's message is read by the model first, when one answers: what the model says the
message teaches is what is kept of it, beside the turn as said. The word patterns read a
message only when no model answers."""

import json

import pytest

from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.context.standing import is_rule
from memory_service.modules.memory.gapfill import GapFill
from tests.integration.test_memory import U1, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

USER = ProcessingHints(visibility=Visibility.USER)
NOTHING = {"ops": []}
#: a skill's text pasted into the chat: its instructions are the skill's, not the person's
SKILL = (
    "# Read-Only Mode\nUse this skill when the user wants to look at plans without changing "
    "them.\nNever call a tool that writes. Always tell the user which plan you read. "
    "Do not delete a plan in this mode."
)
OWN = "I am a demand planner at Acme Retail. My email is alex@example.com. I prefer weekly digests."
#: another person's email, shared to be summarised
EMAIL = (
    "Can you summarise this email from Priya?\n\n"
    "Hi team, I am the regional manager at Castor Supply. My email is priya@example.com. "
    "I prefer daily digests over weekly ones."
)
RULE = "Always answer in German."


def op(fact: str, quote: str, kind: str) -> dict:
    return {
        "op": "add",
        "fact": fact,
        "quote": quote,
        "type": kind,
        "subject": "",
        "date": "",
        "confidence": 0.9,
        "target": 0,
    }


OWN_READ = {
    "ops": [
        op(
            "u1 is a demand planner at Acme Retail.",
            "I am a demand planner at Acme Retail.",
            "profile",
        ),
        op("u1's email is alex@example.com.", "My email is alex@example.com.", "profile"),
        op("u1 prefers weekly digests.", "I prefer weekly digests.", "preference"),
    ]
}


def _category(memory) -> str | None:
    return memory.system_metadata.get("category")


def _readings(memories) -> list:
    """What was read from the messages, without the turns as said."""
    return [m for m in memories if _category(m) != "verbatim_turn"]


@pytest.mark.parametrize("message", [SKILL, "for now skip consent", EMAIL])
async def test_a_message_the_model_reads_nothing_from_leaves_the_turn_alone(
    container, uow_factory, message
):
    """Pasted material and an instruction for the moment: the model says they teach nothing
    of the person, so no rule, preference or attribute of theirs is made - the patterns that
    would have made one do not read the message."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([NOTHING]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, message, hints=USER)
        assert gateway.route.call_count == 1
    memories = await _memories(uow_factory, U1, container)
    assert memories and _readings(memories) == []
    assert not any(is_rule(m.system_metadata) for m in memories)
    assert not any(m.system_metadata.get("self_description") for m in memories)


async def test_the_patterns_read_a_message_no_model_answered(container, uow_factory):
    """The model cannot be reached: the message is read as it is without one."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway(failing=True) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, SKILL, hints=USER)
    memories = await _memories(uow_factory, U1, container)
    assert "rule" in {_category(m) for m in memories}


async def test_what_a_person_said_of_themselves_stands_beside_text_they_shared(
    container, uow_factory
):
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([OWN_READ, NOTHING]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, OWN, hints=USER)
        await _observe(container, uow_factory, U1, EMAIL, hints=USER)
    readings = _readings(await _memories(uow_factory, U1, container))
    assert sorted(m.content for m in readings) == sorted(o["fact"] for o in OWN_READ["ops"])
    assert not any("priya" in m.content.lower() or "Castor" in m.content for m in readings)


async def test_a_rule_the_person_gives_is_their_rule_once(container, uow_factory):
    """The model reads the standing instruction: it is the person's lasting rule, kept once
    (the patterns' reading of the same sentence is not kept beside it)."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"ops": [op(RULE, RULE, "rule")]}]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, RULE, hints=USER)
    memories = await _memories(uow_factory, U1, container)
    [rule] = [m for m in memories if is_rule(m.system_metadata)]
    assert rule.system_metadata.get("provider") == "llm"  # the model's reading, not a pattern's
    assert (rule.content, _category(rule)) == (RULE, "rule")
    assert (rule.memory_type, rule.lifetime) == (MemoryType.PREFERENCE, Lifetime.LONG_TERM)
    assert rule.visibility is Visibility.USER


async def test_a_preference_and_an_attribute_are_read_as_the_patterns_would_file_them(
    container, uow_factory
):
    """So what is read of a person reaches the profile and the preferences as before."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([OWN_READ]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, OWN, hints=USER)
    filed = {
        m.content: (m.memory_type, _category(m))
        for m in _readings(await _memories(uow_factory, U1, container))
    }
    assert filed == {
        "u1 is a demand planner at Acme Retail.": (MemoryType.USER, "attribute"),
        "u1's email is alex@example.com.": (MemoryType.USER, "attribute"),
        "u1 prefers weekly digests.": (MemoryType.PREFERENCE, "preference"),
    }


async def test_the_model_is_given_the_message_not_a_reading_of_it(container, uow_factory):
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([NOTHING]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, RULE, hints=USER)
        sent = json.loads(gateway.prompts()[-1]["messages"][1]["content"])
    assert sent["message"] == RULE and "rules" not in sent


@pytest.mark.parametrize(
    ("message", "kept"),
    [
        ("What time is it?", False),  # nothing said
        ("Call me Al.", True),  # a short opening message: read by the patterns
    ],
)
async def test_a_message_not_asked_about_costs_no_call(container, uow_factory, message, kept):
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([NOTHING]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, message, hints=USER)
        assert gateway.route.call_count == 0
    assert bool(await _memories(uow_factory, U1, container)) is kept
