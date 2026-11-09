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


# ------------------------------------------------- one memory per fact (the hard eval's duplicates)


def _known(text: str):
    from memory_service.domain.enums import Lifetime, Visibility
    from memory_service.domain.evidence import EvidenceRef
    from memory_service.modules.memory.pipeline import build_memory
    from memory_service.ports.intelligence import MemoryCandidate

    said = EvidenceRef(source_type="message", source_id="m0", observed_at=SAID)
    return build_memory(
        MemoryCandidate(
            content=text,
            memory_type=MemoryType.SEMANTIC,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.USER,
            category="fact",
            evidence=[said],
        ),
        CTX,
        now=SAID,
    )


async def test_a_change_the_rules_already_read_is_not_stored_again():
    """contra-4: the rules read "Forklift 4 is back in service since this morning." and closed
    the earlier value; the model's contradict op restating it was a second memory of it."""
    back = op(
        op="contradict",
        fact="Forklift 4 is back in service.",
        quote="Forklift 4 is back in service",
        target=1,
    )
    out, _ = await fill(
        [{"ops": [back]}],
        message="Forklift 4 is back in service since this morning.",
        rules=["Forklift 4 is back in service since this morning."],
        known=[_known("Forklift 4 is out of service.")],
    )
    assert out == []


async def test_an_add_of_the_words_an_update_rests_on_is_the_update():
    """update-6 / hi-update: "Priya handed on-call over to Marcus this week." read as the new
    value of the lead (update) and again as an event (add) was two memories of one change.
    The event says the update, so the update keeps its target and takes the event's words;
    an add of other words, or one neither says, stays."""
    quote = "Priya handed on-call over to Marcus this week"
    update = op(op="update", fact="Our on-call lead is now Marcus.", quote=quote, target=1)
    event = op(fact="Priya handed on-call duty over to Marcus.", quote=quote)
    other = op(fact="Ana's team has a new pager.", quote="We also got a new pager")
    entails = {"Priya handed on-call duty over to Marcus.": {"Our on-call lead is now Marcus."}}
    with mocked_gateway([{"ops": [update, event, other]}]) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]), nli=_Entails(entails))
        out = await filler.fill(
            "t1",
            message=f"{quote}. We also got a new pager.",
            speaker="Ana",
            said_at=SAID,
            recent=[],
            rules=[],
            known=[_known("Our on-call lead is Priya.")],
        )
    assert [(f.op, f.fact) for f in out] == [
        ("update", "Priya handed on-call duty over to Marcus."),
        ("add", "Ana's team has a new pager."),
    ]
    assert out[0].target is not None and out[0].target.content == "Our on-call lead is Priya."


@pytest.mark.parametrize(
    ("other", "contradiction", "kept"),
    [
        # the move contradicts where the speaker lived: the change told again
        ("hi-update अगस्त 2026 में बेंगलुरु चला गया।", 0.93, 1),
        # a fact of the same words that leaves where they lived open stays
        ("hi-update को बेंगलुरु में नई नौकरी मिली।", 0.39, 2),
    ],
)
async def test_an_add_that_contradicts_what_a_change_of_its_words_replaces_is_the_change(
    other, contradiction, kept
):
    """hi-update: "मैं पिछले महीने बेंगलुरु चला गया।" read as what holds now (contradict: lives
    in Bengaluru, not Pune) and again as the move (add) was two memories of one change, in
    every model run. The head does not read the move as saying where the speaker lives
    (entailment 0.04, as the frozen head scores it; 0.99 for the English), but it does read it
    as contradicting the memory the change replaces (0.93) - the prompt's own rule: a change
    of a KNOWN memory is an update or a contradict, never an add."""
    quote = "मैं पिछले महीने बेंगलुरु चला गया"
    pune = "hi-update पुणे में रहता है।"
    now = op(op="contradict", fact="hi-update अब बेंगलुरु में रहता है।", quote=quote, target=1)
    add = op(fact=other, quote=quote, subject="")
    with mocked_gateway([{"ops": [now, add]}]) as gateway:
        filler = GapFill(
            gateway.assist(uses=["contextual_extraction"]),
            nli=_Entails({}, {(other, pune): (0.01, contradiction)}),
        )
        out = await filler.fill(
            "t1",
            message=f"{quote}।",
            speaker="hi-update",
            said_at=SAID,
            recent=[],
            rules=[],
            known=[_known(pune)],
        )
    assert len(out) == kept, out
    assert out[0].op == "contradict" and out[0].target is not None
    assert out[0].target.content == pune
    assert out[0].fact == (other if kept == 1 else now["fact"])


@pytest.mark.parametrize(("entailment", "folded"), [(0.8951, True), (0.30, False)])
async def test_an_add_the_head_reads_as_saying_a_change_of_its_words_is_the_change(
    entailment, folded
):
    """update-6: "Priya handed the on-call duty over to Marcus in the week of 2026-09-30."
    says "Our on-call lead is Marcus." at 0.895 as the frozen head reads it - entailment its
    reading, but just under the bar for dropping a fact. The change takes the add's words, so
    nothing is dropped: the head's reading is enough; below it both stay."""
    quote = "Priya handed on-call over to Marcus this week"
    update = op(op="update", fact="Our on-call lead is Marcus.", quote=quote, target=1)
    event = op(fact="Priya handed the on-call duty over to Marcus.", quote=quote)
    scored = {(event["fact"], update["fact"]): (entailment, 0.21)}
    with mocked_gateway([{"ops": [update, event]}]) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]), nli=_Entails({}, scored))
        out = await filler.fill(
            "t1",
            message=f"{quote}.",
            speaker="Ana",
            said_at=SAID,
            recent=[],
            rules=[],
            known=[_known("Our on-call lead is Priya.")],
        )
    facts = [(f.op, f.fact) for f in out]
    if folded:
        assert facts == [("update", event["fact"])]
    else:
        assert facts == [("update", update["fact"]), ("add", event["fact"])]


@pytest.mark.parametrize(
    ("reading", "quotes", "covered"),
    [
        # vdate-4: the model read the same words and resolved the day
        (
            "The audit is due the Friday before Christmas.",
            ["The audit is due the Friday before Christmas"],
            True,
        ),
        # opinion-1: the model read the sentence as its two facts
        (
            "I think the new CRM is clunky, but the team already migrated to it in May.",
            ["I think the new CRM is clunky", "the team already migrated to it in May"],
            True,
        ),
        # a fill that read only part of the reading leaves the rest to it
        (
            "Forklift 4 is back in service since this morning.",
            ["Forklift 4 is back in service"],
            False,
        ),
        # a fill of other words
        ("Our biggest client is renewing.", ["The renewal is for three years"], False),
    ],
)
def test_a_rule_reading_whose_words_the_model_read_is_the_model_s(reading, quotes, covered):
    fills = [
        checked(op(fact=f"{q} (resolved).", quote=q), [reading, *quotes], SAID, [], "Ana")
        for q in quotes
    ]
    assert all(fills)
    assert (gapfill.covered([reading], [f for f in fills if f]) == [reading]) is covered


class _Entails:
    """The NLI head as the frozen one reads these pairs: premise -> hypotheses it entails,
    and the (entailment, contradiction) it gives the pairs scored."""

    representative = True

    def __init__(
        self,
        entails: dict[str, set[str]],
        scored: dict[tuple[str, str], tuple[float, float]] | None = None,
    ) -> None:
        self.entails = entails
        self.scored = scored or {}

    async def entail_groups(self, groups):
        from memory_service.ports.models import NLIScore

        out = []
        for premises, hyp in groups:
            row = []
            for p in premises:
                e, c = self.scored.get(
                    (p, hyp), (0.98 if hyp in self.entails.get(p, set()) else 0.01, 0.0)
                )
                row.append(NLIScore(entailment=e, neutral=max(0.0, 1 - e - c), contradiction=c))
            out.append(row)
        return out


@pytest.mark.parametrize(
    ("facts", "entails", "kept"),
    [
        # coref-1: the job says where he works
        (
            [
                "Arjun moved to Toronto for a job at SickKids.",
                "Arjun works at SickKids in Toronto.",
            ],
            {
                "Arjun moved to Toronto for a job at SickKids.": {
                    "Arjun works at SickKids in Toronto."
                }
            },
            ["Arjun moved to Toronto for a job at SickKids."],
        ),
        # implied-3: learning Portuguese since moving there says the move
        (
            ["Ana moved to Lisbon.", "Ana has been learning Portuguese since moving to Lisbon."],
            {"Ana has been learning Portuguese since moving to Lisbon.": {"Ana moved to Lisbon."}},
            ["Ana has been learning Portuguese since moving to Lisbon."],
        ),
        # two facts neither of which says the other: both kept
        (
            ["Ana's cat is named Pimienta.", "Pimienta is a Siamese cat."],
            {},
            ["Ana's cat is named Pimienta.", "Pimienta is a Siamese cat."],
        ),
    ],
)
async def test_a_fact_another_fact_of_the_message_says_is_not_added_too(facts, entails, kept):
    message = "Since moving I have news. " + " ".join(facts)
    ops = [op(fact=f, quote=message) for f in facts]
    with mocked_gateway([{"ops": ops}]) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]), nli=_Entails(entails))
        out = await filler.fill(
            "t1", message=message, speaker="Ana", said_at=SAID, recent=[], rules=[], known=[]
        )
    assert [f.fact for f in out] == kept


async def test_a_fact_the_rules_reading_of_the_message_says_is_not_added():
    """implied-1 and opinion-1 of the hard eval: the rules kept "I'll have to skip the wine
    tasting, I'm eight months pregnant." and the model added "The speaker is eight months
    pregnant.", which that reading says; a fact it does not say is added."""
    said = "I'll have to skip the wine tasting, I'm eight months pregnant."
    pregnant = op(fact="The speaker is eight months pregnant.", quote="I'm eight months pregnant")
    due = op(fact="Ana's baby is due in November.", quote="I'm eight months pregnant")
    with mocked_gateway([{"ops": [pregnant, due]}]) as gateway:
        filler = GapFill(
            gateway.assist(uses=["contextual_extraction"]),
            nli=_Entails({said: {"The speaker is eight months pregnant."}}),
        )
        out = await filler.fill(
            "t1", message=said, speaker="Ana", said_at=SAID, recent=[], rules=[said], known=[]
        )
    assert [f.fact for f in out] == ["Ana's baby is due in November."]
