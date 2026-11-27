"""Where a model is allowed to read a person's message, it reads it first and nothing else
does: what it says the message teaches is what is kept of it, and the message itself is kept
beyond its conversation only when the model says it is a statement. A message it does not
answer for is read again later, never by the word patterns."""

import json

import pytest

from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind, Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.context.standing import is_rule
from memory_service.modules.memory.gapfill import GapFill
from memory_service.modules.memory.pipeline import KEPT_UNREAD, LATER
from tests.integration.test_memory import U1, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

USER = ProcessingHints(visibility=Visibility.USER)
_KIND = ObservationKind.MESSAGE
NOTHING = {"kind": "statement", "ops": []}
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
        "until": "",
        "target": 0,
    }


OWN_READ = {
    "kind": "statement",
    "ops": [
        op(
            "u1 is a demand planner at Acme Retail.",
            "I am a demand planner at Acme Retail.",
            "profile",
        ),
        op("u1's email is alex@example.com.", "My email is alex@example.com.", "profile"),
        op("u1 prefers weekly digests.", "I prefer weekly digests.", "preference"),
    ],
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


class Model:
    """The model as the pipeline reaches it: no answer the first ``silent`` times, then
    ``replies`` in order (the last one again after them)."""

    def __init__(self, replies=(), *, silent: int = 0) -> None:
        self.replies, self.silent, self.calls = list(replies), silent, 0

    def wants(self, use) -> bool:
        return True

    async def structured(self, use, **_):
        self.calls += 1
        if self.calls <= self.silent:
            return None
        return self.replies[min(self.calls - self.silent, len(self.replies)) - 1]


async def _observation(uow_factory, ack):
    async with uow_factory() as uow:
        return await uow.observations.get("acme", ack.observation_id)


async def _status(container, ack) -> str | None:
    """How the observation was marked when it was processed."""
    from sqlalchemy import text

    async with container.database.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT status FROM observations WHERE observation_id = :o"),
            {"o": ack.observation_id},
        )
        return row.scalar_one()


async def test_a_message_its_model_does_not_answer_for_is_read_later_by_the_model(
    container, uow_factory
):
    """The model is down for two tries: no pattern reads the message meanwhile - the
    patterns would make the pasted rule the person's - and the third try's reading is the
    one kept."""
    pipeline = container.services["observation_pipeline"]
    model = Model([OWN_READ], silent=2)
    pipeline.gapfill = GapFill(model)  # type: ignore[arg-type]
    ack = await _observe(container, uow_factory, U1, OWN, hints=USER)
    assert model.calls == 3
    readings = _readings(await _memories(uow_factory, U1, container))
    assert sorted(m.content for m in readings) == sorted(o["fact"] for o in OWN_READ["ops"])
    assert {m.system_metadata.get("provider") for m in readings} == {"llm"}
    assert (await _observation(uow_factory, ack)).processed_at is not None


async def test_nothing_is_formed_from_a_message_while_it_waits_for_its_model(
    container, uow_factory
):
    from memory_service.modules.memory.pipeline import UNREAD

    pipeline = container.services["observation_pipeline"]
    pipeline.gapfill = GapFill(Model(silent=99))  # type: ignore[arg-type]
    from benchmark.common import submit_observation

    from memory_service.modules.jobs.registry import register_handlers

    register_handlers(container)
    async with uow_factory() as uow:
        ack = await submit_observation(uow, U1, kind=_KIND, content=SKILL, hints=USER)
        await uow.commit()
    assert await pipeline.run({"tenant_id": "acme", "observation_id": ack.observation_id}) == []
    assert await _memories(uow_factory, U1, container) == []
    assert (await _observation(uow_factory, ack)).processed_at is None
    later = [j.payload for j in container.tasks.payloads.values() if j.payload.get(UNREAD)]
    assert later == [{"tenant_id": "acme", "observation_id": ack.observation_id, UNREAD: 1}]


async def test_a_message_no_model_ever_answered_for_is_kept_as_said_unread(container, uow_factory):
    """About a day and a half of tries, then the message is kept as it was said, with
    nothing read from it: no rule of the person from the skill they pasted."""
    pipeline = container.services["observation_pipeline"]
    model = Model(silent=99)
    pipeline.gapfill = GapFill(model)  # type: ignore[arg-type]
    ack = await _observe(container, uow_factory, U1, SKILL, hints=USER)
    assert model.calls == 1 + len(LATER)
    memories = await _memories(uow_factory, U1, container)
    assert memories and _readings(memories) == []
    assert not any(is_rule(m.system_metadata) for m in memories)
    assert await _status(container, ack) == KEPT_UNREAD


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("delete this plan", "request"),
        ("ok for now skip the consent stage and create the allocation directly", "request"),
        (SKILL, "material"),
    ],
)
async def test_a_request_and_shared_text_are_not_kept_beyond_their_conversation(
    container, uow_factory, message, kind
):
    """They stay in the transcript, where the conversation reads them; they are no memory
    of the person in their next one."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"kind": kind, "ops": []}]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        ack = await _observe(container, uow_factory, U1, message, hints=USER)
    assert await _memories(uow_factory, U1, container) == []
    assert await _status(container, ack) == "NO_MEMORY"


async def test_a_statement_is_kept_as_said_beside_what_is_read_from_it(container, uow_factory):
    said = "Create the allocation, and remember that I manage store KS3183."
    read = op("u1 manages store KS3183.", "I manage store KS3183", "profile")
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"kind": "statement", "ops": [read]}]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, said, hints=USER)
    kept = {(_category(m), m.content) for m in await _memories(uow_factory, U1, container)}
    assert kept == {("verbatim_turn", said), ("attribute", "u1 manages store KS3183.")}


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
    with mocked_gateway([{"kind": "statement", "ops": [op(RULE, RULE, "rule")]}]) as gateway:
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
        ("Never write.", True),  # a short opening message: kept as said, read by nothing
    ],
)
async def test_a_message_not_asked_about_costs_no_call(container, uow_factory, message, kept):
    """Where a model is allowed no pattern stands in for it, for a message it is not asked
    about either: the short "Never write." is no rule of the person."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([NOTHING]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, message, hints=USER)
        assert gateway.route.call_count == 0
    memories = await _memories(uow_factory, U1, container)
    assert bool(memories) is kept and _readings(memories) == []
    assert not any(is_rule(m.system_metadata) for m in memories)


# --------------------------------------------------------------------------- for a period

TODAY_ONLY = "For today only, use the Retail-Outlet channel for allocations."


async def _rules_shown(container, ask: str) -> str:
    from tests.integration.test_statements_follow_the_user import _chat

    body = await container.services["context_builder"].build_api(_chat(), ask, output="full")
    blocks = json.loads(body).get("profile") or []
    return "\n".join(b["text"] for b in blocks if b["block"] == "standing_rules")


async def test_an_instruction_for_a_period_holds_until_its_last_day_and_then_lapses(
    container, uow_factory
):
    """ "For today only ..." was either kept for good (the patterns: "use" is no rule word, the
    turn stays) or not at all (an instruction for the moment). Read with its last day, it is
    the person's rule in their other conversations today, says until when, and is in no
    context once the day is over."""
    from datetime import UTC, datetime, timedelta

    from memory_service.domain.enums import TemporalStatus

    today = datetime.now(UTC).date()
    pipeline = container.services["observation_pipeline"]
    reading = {
        "kind": "statement",
        "ops": [op(TODAY_ONLY, TODAY_ONLY, "rule") | {"until": today.isoformat()}],
    }
    with mocked_gateway([reading]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, TODAY_ONLY, hints=USER)
    await container.tasks.drain()
    [rule] = [m for m in await _memories(uow_factory, U1, container) if is_rule(m.system_metadata)]
    ends = datetime.fromisoformat(rule.system_metadata["expires_at"])
    assert ends == datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=1)
    assert rule.temporal.valid_to == ends

    ask = "Create an allocation, which channel do I use?"
    shown = await _rules_shown(container, ask)
    assert TODAY_ONLY in shown and f"until {today.isoformat()}" in shown

    # the day is over: the hourly expiry retires it, and no later context carries it
    async with uow_factory() as uow:
        expired = await uow.memories.expire_due(now=ends + timedelta(minutes=1))
        await uow.commit()
    assert [memory_id for _, memory_id in expired] == [rule.memory_id]
    async with uow_factory() as uow:
        [gone] = await uow.memories.get_many(U1.tenant_id, [rule.memory_id])
    assert gone.temporal.status is TemporalStatus.EXPIRED


async def test_a_period_already_over_keeps_nothing_beyond_the_conversation(container, uow_factory):
    """The model names a last day before the day it was said: nothing is made of it."""
    pipeline = container.services["observation_pipeline"]
    reading = {
        "kind": "request",
        "ops": [op(TODAY_ONLY, TODAY_ONLY, "rule") | {"until": "2020-01-01"}],
    }
    with mocked_gateway([reading]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, TODAY_ONLY, hints=USER)
    memories = await _memories(uow_factory, U1, container)
    assert not any(is_rule(m.system_metadata) for m in memories)
    assert _readings(memories) == []
