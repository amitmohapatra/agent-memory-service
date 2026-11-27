"""The model reads a person's message: one call a message, every op checked in code."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime

import pytest

from memory_service.domain.enums import MemoryType, StatementKind
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
        "until": "",
        "target": 0,
        **fields,
    }


async def fill(replies=(), *, message="Her name is Biscuit, a beagle mix.", known=(), **how):
    with mocked_gateway(replies, **how) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]))
        read = await filler.fill(
            "t1",
            message=message,
            speaker="Ana",
            said_at=SAID,
            recent=[Turn("Ana", "I finally adopted a dog last month.")],
            known=list(known),
        )
        out = None if read is None else read.fills
        return out, gateway


async def test_one_call_reads_the_message_with_its_turns_and_known():
    out, gateway = await fill([{"kind": "statement", "ops": [op()]}])
    assert [f.fact for f in out] == ["Ana adopted a beagle named Biscuit."]
    assert gateway.route.call_count == 1
    sent = gateway.prompts()[0]["messages"]
    assert sent[0]["content"].startswith(gapfill.SYSTEM)  # fixed: a cacheable prefix
    payload = json.loads(sent[1]["content"])
    assert payload["turns"] == [{"speaker": "Ana", "text": "I finally adopted a dog last month."}]
    assert payload["message"] == "Her name is Biscuit, a beagle mix."
    assert "rules" not in payload and "RULES" not in gapfill.SYSTEM  # no reading before its own
    assert payload["said_on"] == "2026-10-01 (Thursday)"


async def test_no_answer_is_not_an_answer_of_nothing():
    """A model that answers "nothing" has read the message; one that cannot be reached, or
    may not read it, has not - the message is then the patterns' to read."""
    assert (await fill([{"kind": "statement", "ops": []}]))[0] == []
    assert (await fill(failing=True))[0] is None
    with mocked_gateway([{"kind": "statement", "ops": [op()]}]) as gateway:
        unused = GapFill(gateway.assist(uses=[]))
        said = await unused.fill(
            "t1", message="x", speaker="Ana", said_at=SAID, recent=[], known=[]
        )
        assert said is None and gateway.route.call_count == 0


@pytest.mark.parametrize("reply", [{}, {"ops": None}, {"ops": "nothing"}, {"facts": []}])
async def test_a_reply_that_is_not_the_list_asked_for_is_no_answer(reply):
    class Assist:
        async def structured(self, *args, **kwargs):
            return reply

    said = await GapFill(Assist()).fill(  # type: ignore[arg-type]
        "t1", message="I moved.", speaker="Ana", said_at=SAID, recent=[], known=[]
    )
    assert said is None


def test_a_rule_the_speaker_gives_is_their_lasting_rule_in_the_words_given():
    rule = "Always answer in German."
    got = checked(op(fact=rule, quote=rule, type="rule", subject=""), [rule], SAID, [], "Ana")
    assert got is not None and got.kind == "rule"
    made = candidate(got, speaker="user:ana", evidence=[])
    assert (made.memory_type, made.category) == (MemoryType.PREFERENCE, "rule")
    assert (made.predicate, made.object, made.statement_kind) == ("rule", rule, StatementKind.RULE)
    assert made.subject == "user:ana" and made.provider == "llm"


@pytest.mark.parametrize(
    ("kind", "filed"),
    [
        ("profile", (MemoryType.USER, "attribute")),
        ("preference", (MemoryType.PREFERENCE, "preference")),
        ("fact", (MemoryType.SEMANTIC, "source_fact")),
        ("event", (MemoryType.EPISODIC, "source_fact")),
    ],
)
def test_what_is_read_of_the_speaker_is_filed_where_a_context_reads_it(kind, filed):
    got = checked(op(type=kind), ["Her name is Biscuit."], SAID, [])
    assert got is not None
    made = candidate(got, speaker="user:ana", evidence=[])
    assert (made.memory_type, made.category) == filed
    assert made.statement_kind is None and made.predicate == "said"


def test_a_type_the_model_made_up_drops_the_op():
    assert checked(op(type="secret"), ["Her name is Biscuit."], SAID, []) is None


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


async def test_what_a_memory_holds_is_not_added_again_and_a_fact_is_added_once():
    same = op(fact="Her name is Biscuit.")
    out, _ = await fill(
        [{"kind": "statement", "ops": [same, same]}], known=[_known("Her  name is biscuit.")]
    )
    assert out == []
    out, _ = await fill([{"kind": "statement", "ops": [same, same]}])
    assert [f.fact for f in out] == ["Her name is Biscuit."]


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
            return {"kind": "statement", "ops": []}

    filler = GapFill(Assist())  # type: ignore[arg-type]
    await asyncio.gather(
        *(
            filler.fill("t1", message="I moved.", speaker="Ana", said_at=SAID, recent=[], known=[])
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
    with mocked_gateway([{"kind": "statement", "ops": [update, event, other]}]) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]), nli=_Entails(entails))
        read = await filler.fill(
            "t1",
            message=f"{quote}. We also got a new pager.",
            speaker="Ana",
            said_at=SAID,
            recent=[],
            known=[_known("Our on-call lead is Priya.")],
        )
        out = None if read is None else read.fills
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
    with mocked_gateway([{"kind": "statement", "ops": [now, add]}]) as gateway:
        filler = GapFill(
            gateway.assist(uses=["contextual_extraction"]),
            nli=_Entails({}, {(other, pune): (0.01, contradiction)}),
        )
        read = await filler.fill(
            "t1",
            message=f"{quote}।",
            speaker="hi-update",
            said_at=SAID,
            recent=[],
            known=[_known(pune)],
        )
        out = None if read is None else read.fills
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
    with mocked_gateway([{"kind": "statement", "ops": [update, event]}]) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]), nli=_Entails({}, scored))
        read = await filler.fill(
            "t1",
            message=f"{quote}.",
            speaker="Ana",
            said_at=SAID,
            recent=[],
            known=[_known("Our on-call lead is Priya.")],
        )
        out = None if read is None else read.fills
    facts = [(f.op, f.fact) for f in out]
    if folded:
        assert facts == [("update", event["fact"])]
    else:
        assert facts == [("update", update["fact"]), ("add", event["fact"])]


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
    with mocked_gateway([{"kind": "statement", "ops": ops}]) as gateway:
        filler = GapFill(gateway.assist(uses=["contextual_extraction"]), nli=_Entails(entails))
        read = await filler.fill(
            "t1", message=message, speaker="Ana", said_at=SAID, recent=[], known=[]
        )
        out = None if read is None else read.fills
    assert [f.fact for f in out] == kept


async def test_a_provider_that_keeps_no_turn_as_said_reads_the_message_itself():
    """The pipeline asks the model first only where the turn as said can be kept beside its
    reading (``NativeMemoryIntelligence.said``): another provider's extraction stands."""
    from memory_service.config.constants import MemoryIntelligenceSettings
    from memory_service.modules.memory.pipeline import ObservationPipeline

    class Provider:  # extract / classify / consolidate only
        pass

    class Wanted:
        def wanted(self, *_):
            return True

    pipeline = ObservationPipeline(
        None,  # type: ignore[arg-type]
        Provider(),  # type: ignore[arg-type]
        settings=MemoryIntelligenceSettings(),
        gapfill=Wanted(),  # type: ignore[arg-type]
    )
    assert await pipeline._read(_obs("I moved to Pune."), CTX, None) is None


@pytest.mark.parametrize(
    ("reply", "kept"),
    [
        ({"kind": "statement", "ops": []}, True),
        ({"kind": "request", "ops": []}, False),
        ({"kind": "material", "ops": []}, False),
        ({"ops": []}, True),  # a reply that does not say: kept, as a turn always was
        ({"kind": "chatter", "ops": []}, True),
    ],
)
async def test_only_a_statement_is_kept_beyond_its_conversation(reply, kept):
    class Assist:
        async def structured(self, *args, **kwargs):
            return reply

    read = await GapFill(Assist()).fill(  # type: ignore[arg-type]
        "t1", message="delete this plan", speaker="Ana", said_at=SAID, recent=[], known=[]
    )
    assert read is not None and read.kept is kept and read.fills == []


def test_the_model_is_asked_what_the_message_is():
    assert gapfill.SCHEMA["required"] == ["kind", "ops"]
    assert gapfill.SCHEMA["properties"]["kind"]["enum"] == list(gapfill.KINDS)


# --------------------------------------------------------------------------- for a period


async def test_what_is_given_for_a_period_carries_its_last_day():
    """ "For today only ..." is no rule for every time and no instruction for this conversation
    alone: it holds in the speaker's other conversations until the day it names, then lapses."""
    message = "For this week only, answer me in German."
    reading = {
        "kind": "statement",
        "ops": [
            op(
                fact="Ana wants answers in German this week.",
                quote="For this week only, answer me in German.",
                type="rule",
                subject="",
                until="2026-10-04",
            )
        ],
    }
    [found], _ = await fill([reading], message=message)
    assert found.until == date(2026, 10, 4)
    made = gapfill.candidate(found, speaker="Ana", evidence=[])
    # through its last day: it lapses when the next one starts
    assert made.valid_to == datetime(2026, 10, 5, tzinfo=UTC)
    # no period named: no end
    [plain], _ = await fill([{"kind": "statement", "ops": [op()]}])
    assert plain.until is None
    assert gapfill.candidate(plain, speaker="Ana", evidence=[]).valid_to is None


@pytest.mark.parametrize("until", ["2026-09-30", "next friday", "2026-13-45", "9999-01-01"])
async def test_a_period_that_is_over_or_cannot_be_read_is_not_kept_as_if_it_had_no_end(until):
    """Said for a period: kept to that period or not at all. A day before the day it was
    said, a phrase that is no day, or a year no period reaches would leave an instruction
    for a while standing for good."""
    reading = {"kind": "statement", "ops": [op(until=until)]}
    out, _ = await fill([reading])
    assert out == []


def test_the_day_said_is_a_period_still_running():
    assert gapfill._until("2026-10-01", SAID) == date(2026, 10, 1)
    assert gapfill._until("  ", SAID) is None
    assert gapfill._until("2026-09-30", SAID) is gapfill._PAST
    assert "until" in gapfill.SCHEMA["properties"]["ops"]["items"]["required"]
