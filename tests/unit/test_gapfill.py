"""The model fills what the rules missed: one call a message, every op checked in code."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime

import pytest

from memory_service.domain.enums import MemoryType
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory import gapfill
from memory_service.modules.memory.gapfill import GapFill, Turn, candidate, checked, worth_asking
from tests.support_llm import mocked_gateway
from tests.unit.test_memory_native import CTX, _obs

pytestmark = pytest.mark.unit

SAID = datetime(2026, 10, 1, 12, tzinfo=UTC)


def op(**fields):
    return {
        "op": "add",
        "fact": "Ana adopted a beagle named Biscuit.",
        "quote": "Her name is Biscuit",
        "type": "fact",
        "subject": "Biscuit",
        "date": "",
        "confidence": 0.9,
        "target": 0,
        **fields,
    }


async def fill(replies, *, message="Her name is Biscuit, a beagle mix.", rules=(), known=()):
    with mocked_gateway(replies) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]))
        out = await filler.fill(
            "t1",
            message=message,
            speaker="Ana",
            said_at=SAID,
            recent=[Turn("Ana", "I finally adopted a dog last month.")],
            rules=list(rules),
            known=list(known),
        )
        return out, gateway


async def test_one_call_reads_the_message_with_its_turns_rules_and_known():
    out, gateway = await fill([{"ops": [op()]}], rules=["Her name is Biscuit."])
    assert [f.fact for f in out] == ["Ana adopted a beagle named Biscuit."]
    assert gateway.route.call_count == 1
    sent = gateway.prompts()[0]["messages"]
    assert sent[0]["content"].startswith(gapfill.SYSTEM)  # fixed: a cacheable prefix
    payload = json.loads(sent[1]["content"])
    assert payload["turns"] == [{"speaker": "Ana", "text": "I finally adopted a dog last month."}]
    assert payload["rules"] == ["Her name is Biscuit."]
    assert payload["said_on"] == "2026-10-01 (Thursday)"


def test_a_quote_that_is_not_in_the_sources_drops_the_op():
    assert checked(op(quote="Her name is Rex"), ["Her name is Biscuit."], SAID, []) is None


def test_a_number_the_quote_does_not_state_drops_the_op():
    item = op(fact="Ana paid 300 dollars for Biscuit.")
    assert checked(item, ["Her name is Biscuit."], SAID, []) is None


def test_a_relative_date_is_the_librarys_day_not_the_models():
    item = op(fact="Ana got her licence on 2026-09-20.", quote="three days ago", date="2026-09-20")
    got = checked(item, ["I got my licence three days ago."], SAID, [])
    assert got is not None and got.day == date(2026, 9, 28)
    assert candidate(got, speaker="user:ana", evidence=[]).content == (
        "Ana got her licence on 2026-09-28."
    )


def test_a_day_the_library_cannot_read_is_the_models_when_near():
    item = op(
        fact="Ana flies to Goa on 2026-11-01.",
        quote="the week before Diwali",
        date="2026-11-01",
        type="plan",
    )
    got = checked(item, ["We fly to Goa the week before Diwali."], SAID, [])
    assert got is not None and got.day == date(2026, 11, 1)
    made = candidate(got, speaker="user:ana", evidence=[])
    assert made.memory_type is MemoryType.TASK  # classified as the rules' readings are
    assert made.valid_from == datetime(2026, 11, 1, tzinfo=UTC)
    far = op(fact="Ana flies on 2040-11-01.", quote="the week before Diwali", date="2040-11-01")
    assert checked(far, ["We fly the week before Diwali."], SAID, []) is None


def test_an_update_of_a_memory_not_shown_is_an_add():
    got = checked(op(op="update", target=3), ["Her name is Biscuit."], SAID, [])
    assert got is not None and got.op == "add" and got.target is None


def test_an_opinion_is_a_belief_and_an_implied_fact_is_less_sure():
    item = op(type="opinion", confidence=1.0)
    got = checked(item, ["Her name is Biscuit."], SAID, [])
    assert got is not None and got.memory_type is MemoryType.BELIEF and got.confidence == 0.9


async def test_what_the_rules_already_read_is_not_added_twice():
    same = op(fact="Her name is Biscuit.")
    out, _ = await fill([{"ops": [same, same]}], rules=["Her  name is biscuit."])
    assert out == []


@pytest.mark.parametrize(
    ("sentences", "recent", "asked"),
    [
        (["What time is it?"], [], False),
        (["Thanks!"], [Turn("Ana", "I moved.")], False),
        (["Yes, book it."], [Turn("assistant", "Shall I book the Granvia?")], True),
        (["I moved to Pune."], [], True),
    ],
)
def test_a_message_is_asked_about_by_its_shape(sentences, recent, asked):
    assert worth_asking(sentences, recent, "Ana") is asked


async def test_calls_in_flight_are_bounded_per_tenant():
    running = 0
    peak = 0

    class Assist:
        def wants(self, use):
            return True

        async def structured(self, *args, **kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return {"ops": []}

    filler = GapFill(Assist())  # type: ignore[arg-type]
    await asyncio.gather(
        *(
            filler.fill(
                "t1", message="I moved.", speaker="Ana", said_at=SAID, recent=[], rules=[], known=[]
            )
            for _ in range(3 * gapfill.PER_TENANT)
        )
    )
    assert peak == gapfill.PER_TENANT


async def test_a_person_s_message_is_the_gap_fill_s_unless_a_server_reads_it():
    with mocked_gateway() as gateway:
        assist = gateway.assist(uses=["contextual_extraction"])
        said = _obs("I moved to Pune.")
        assert GapFill(assist).wanted(said, CTX)
        assert not GapFill(assist).wanted(
            said.model_copy(update={"custom_metadata": {"role": "assistant"}}), CTX
        )
        assert not GapFill(assist, server_reads=lambda o, c: True).wanted(said, CTX)
        assert not GapFill(LLMAssist.disabled()).wanted(said, CTX)


async def test_a_rule_reading_said_again_with_the_speakers_name_is_not_added():
    restated = op(fact="Ana's pallet supplier is Uline.", quote="Our pallet supplier is Uline.")
    resolved = op(fact="Ana's dog is named Biscuit.", quote="Her name is Biscuit.")
    out, _ = await fill(
        [{"ops": [restated, resolved]}],
        message="Our pallet supplier is Uline. Her name is Biscuit.",
        rules=["Our pallet supplier is Uline.", "Her name is Biscuit."],
    )
    assert [f.fact for f in out] == ["Ana's dog is named Biscuit."]
