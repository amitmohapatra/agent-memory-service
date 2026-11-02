# ruff: noqa: RUF001 - literal multilingual fixtures.
"""Model-first extraction (``Tenant.model_first_extraction``, ADR 0045): with the switch on and a
model that may be asked, one ``contextual_extraction`` call reads every statement of a user's
message, and its facts go through the same consolidation as the rules' readings. With it off,
nothing changes.

The model is scripted (``tests.support_llm``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.enums import DedupDecision
from memory_service.domain.observation import Observation
from memory_service.modules.llm.assist import with_language_rule
from memory_service.modules.memory import source_facts
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from tests.support_llm import mocked_gateway
from tests.unit.test_memory_native import AGENT, CTX
from tests.unit.test_memory_native import _obs as native_obs

pytestmark = pytest.mark.unit

#: a Monday
SAID = datetime(2023, 5, 8, 15, 30, tzinfo=UTC)


def _obs(text: str, *, agent: bool = False) -> Observation:
    observed = native_obs(text, ctx=AGENT if agent else CTX)
    role = {"custom_metadata": {"role": "assistant"}} if agent else {}
    return observed.model_copy(update={"occurred_at": SAID, **role})


def _fact(text: str, *, slot: str = "none", value: str = "", kind: str = "fact", cited=(0,)):
    return {"text": text, "kind": kind, "slot": slot, "value": value, "sentences": list(cited)}


def _reply(*facts: dict, times: list[dict] | None = None) -> dict:
    return {"facts": list(facts), "times": times or []}


async def _extract(native, text: str, *, model_first: bool) -> list:
    return [
        await native.classify(c, CTX)
        for c in await native.extract(_obs(text), CTX, model_first=model_first)
    ]


def _native(gateway) -> NativeMemoryIntelligence:
    return NativeMemoryIntelligence(
        MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
    )


# --------------------------------------------------------------------------- switch off


@pytest.mark.parametrize(
    "text",
    [
        "Caroline went to the LGBTQ support group yesterday.",  # English, no rule, one sentence
        "My sister visited. She stayed for a week and we went hiking.",  # two unread sentences
        "Ich wohne jetzt in Köln. Ich arbeite bei Siemens.",  # another language
        "I live in Berlin.",  # a rule reads it
    ],
)
async def test_switched_off_the_calls_and_candidates_are_todays(text: str) -> None:
    """Off is the default: ``model_first=False`` and no argument ask the model exactly the same
    (nothing, the narrative call or the source-facts call, with today's prompt) and keep the
    same candidates, dated by the library."""
    reply = {"units": [{"start": 0, "end": 1}], "facts": []}
    runs = []
    for kwargs in ({}, {"model_first": False}):
        with mocked_gateway([reply]) as gateway:
            native = _native(gateway)
            found = await native.extract(_obs(text), CTX, **kwargs)
            runs.append(([c.model_dump() for c in found], gateway.prompts()))
    assert runs[0] == runs[1]
    candidates, prompts = runs[0]
    assert all(c["dated_mentions"] is None for c in candidates)
    for prompt in prompts:
        assert "said_on" not in prompt["messages"][1]["content"]
    if text.startswith("Ich"):
        assert prompts[0]["messages"][0]["content"] == with_language_rule(source_facts._SYSTEM)


# --------------------------------------------------------------------------- switch on


async def test_one_call_reads_a_message_no_rule_reads() -> None:
    """A one-sentence English message no rule reads: today no model call at all (two unread
    sentences are needed); model-first one call, told the day it was said. The fact, its date
    resolved, is appended to the key the turn is found by, not a second memory."""
    text = "Caroline went to the LGBTQ support group yesterday."
    reply = _reply(
        _fact("Caroline went to the LGBTQ support group on 2023-05-07.", kind="event"),
        times=[{"text": "yesterday", "kind": "relative_date", "date": "2023-05-07"}],
    )
    with mocked_gateway([reply]) as gateway:
        found = await _extract(_native(gateway), text, model_first=True)
        assert gateway.route.call_count == 1
        sent = json.loads(gateway.prompts()[0]["messages"][1]["content"])
    assert sent["said_on"] == "2023-05-08 (Monday)"
    assert [s["text"] for s in sent["sentences"]] == [text]
    assert [c.category for c in found] == ["event", "verbatim_turn"]
    turn = next(c for c in found if c.category == "verbatim_turn")
    assert turn.restatement == "Caroline went to the LGBTQ support group on 2023-05-07."
    assert turn.dated_mentions == [{"text": "yesterday", "date": "2023-05-07"}]
    stored = build_memory(turn, CTX, now=SAID)
    assert stored.system_metadata["dated_mentions"] == [{"text": "yesterday", "date": "2023-05-07"}]


async def test_a_statement_the_rules_read_keeps_one_reading() -> None:
    """The rules read "I live in Berlin" as the lives_in slot: the model's reading of the same
    slot is not stored beside it. A one-valued slot the rules did not read is a memory; a fact
    without one goes to the turn's key."""
    text = "I live in Berlin. My sister moved to Lisbon to work as a nurse. I'm at Bosch now."
    reply = _reply(
        _fact("The user lives in Berlin", slot="lives_in", value="Berlin", kind="profile"),
        _fact("The user's sister works as a nurse in Lisbon", cited=(1,)),
        _fact("The user works at Bosch", slot="works_at", value="Bosch", cited=(2,)),
    )
    with mocked_gateway([reply]) as gateway:
        found = await _extract(_native(gateway), text, model_first=True)
    lives = [c for c in found if c.predicate == "lives_in"]
    assert [(c.category, c.object) for c in lives] == [("attribute", "berlin")]
    facts = [c for c in found if c.category == "source_fact"]
    assert [(c.content, c.predicate, c.object) for c in facts] == [
        ("The user works at Bosch", "works_at", "Bosch")
    ]
    turn = next(c for c in found if c.category == "verbatim_turn")
    assert turn.restatement == "The user's sister works as a nurse in Lisbon"


async def test_untrusted_output_is_checked_before_it_is_stored() -> None:
    """A fact stating a number the message does not; a time not in the message, of the wrong
    kind, unparseable or years away: none is kept. A slot whose value is not copied from the
    message stays a plain fact (it could otherwise replace an earlier value)."""
    text = "We met Priya last Friday to plan the launch."
    reply = _reply(
        _fact("The launch budget is 4000 EUR"),
        _fact("The user met Priya on 2023-05-05 to plan the launch", kind="event"),
        _fact("The user works at Priya's company", slot="works_at", value="Acme"),
        times=[
            {"text": "last Friday", "kind": "relative_date", "date": "2023-05-05"},
            {"text": "next Friday", "kind": "relative_date", "date": "2023-05-12"},
            {"text": "last Friday", "kind": "duration", "date": ""},
            {"text": "the launch", "kind": "relative_date", "date": "soon"},
            {"text": "Friday", "kind": "relative_date", "date": "2041-05-05"},
        ],
    )
    with mocked_gateway([reply]) as gateway:
        found = await _extract(_native(gateway), text, model_first=True)
    assert not [c for c in found if c.category == "source_fact"]
    turn = next(c for c in found if c.category == "verbatim_turn")
    assert turn.restatement == (
        "The user met Priya on 2023-05-05 to plan the launch The user works at Priya's company"
    )
    assert turn.dated_mentions == [{"text": "last Friday", "date": "2023-05-05"}]


@pytest.mark.parametrize("failing", [True, False], ids=["gateway-down", "not-allowed"])
async def test_without_an_answer_the_rules_and_the_library_read_the_message(failing) -> None:
    """No model answer (down, or the policy does not allow the use): exactly what the rules
    read, the turn dated by the library, and no second model call."""
    text = "I live in Berlin. We went hiking yesterday."
    with mocked_gateway(failing=failing) as gateway:
        assist = gateway.assist(uses=["contextual_extraction"] if failing else [])
        native = NativeMemoryIntelligence(MemoryIntelligenceSettings(), assist=assist)
        first = await native.extract(_obs(text), CTX, model_first=True)
        calls = gateway.route.call_count
    with mocked_gateway() as gateway:
        native = NativeMemoryIntelligence(MemoryIntelligenceSettings(), assist=gateway.assist([]))
        rules = await native.extract(_obs(text), CTX)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in rules]
    assert calls <= 3, "one call, retried by the transport at most"


async def test_an_agent_s_message_is_not_sent() -> None:
    with mocked_gateway([_reply()]) as gateway:
        native = _native(gateway)
        await native.extract(
            _obs("Thinking: the user lives in Berlin.", agent=True),
            AGENT,
            model_first=True,
        )
        assert gateway.route.call_count == 0


# --------------------------------------------------------------------------- consolidation


async def _stored(native, text: str, *, model_first: bool) -> list:
    found = await _extract(native, text, model_first=model_first)
    return [build_memory(c, CTX, now=SAID) for c in found if c.category != "verbatim_turn"]


async def _decisions(native, earlier: list, text: str, *, model_first: bool) -> list:
    out = []
    for cand in await _extract(native, text, model_first=model_first):
        if cand.category == "verbatim_turn":
            continue
        decided = await native.consolidate(cand, earlier, CTX, model_first=model_first)
        out.append((cand.content, decided.decision))
    return out


async def test_a_paraphrase_of_a_model_fact_is_not_a_second_memory() -> None:
    """The same fact said again in other words: model-first, the second reading reinforces
    the first; off, it is a second memory, as today."""
    first = _reply(_fact("Der Nutzer wohnt in Köln", slot="lives_in", value="Köln", kind="profile"))
    again = _reply(_fact("Der Nutzer lebt in Köln", slot="lives_in", value="Köln", kind="profile"))
    for model_first, expected in ((True, DedupDecision.REINFORCE), (False, DedupDecision.CREATE)):
        with mocked_gateway([{**first, "units": []}, {**again, "units": []}]) as gateway:
            native = _native(gateway)
            earlier = await _stored(native, "Ich wohne in Köln.", model_first=model_first)
            got = await _decisions(native, earlier, "Ich lebe in Köln.", model_first=model_first)
        assert [d for content, d in got if "Köln" in content] == [expected], got


async def test_a_model_fact_the_source_already_states_is_ignored() -> None:
    """ "I live in Berlin" was read by the rules; a later message the rules cannot read
    ("Berlin is where I live these days") gives the model the same slot and value: it is not
    stored beside the user's own words."""
    reply = _reply(_fact("The user lives in Berlin", slot="lives_in", value="Berlin"))
    with mocked_gateway([reply]) as gateway:
        native = _native(gateway)
        earlier = await _stored(native, "I live in Berlin.", model_first=False)
        got = await _decisions(
            native, earlier, "Berlin is where I've been living for ages.", model_first=True
        )
    assert ("The user lives in Berlin", DedupDecision.IGNORE) in got


async def test_a_new_value_the_model_reads_replaces_the_old_one() -> None:
    """Stale values: the rules read "I live in Berlin"; "Ich wohne jetzt in München" is read by
    the model only. Its lives_in replaces Berlin, as the rule reading of the same words would;
    off, the old value stays current beside it, as today."""
    reply = _reply(_fact("Der Nutzer wohnt in München", slot="lives_in", value="München"))
    for model_first, expected in (
        (True, DedupDecision.SUPERSEDE),
        (False, DedupDecision.CREATE),
    ):
        with mocked_gateway([{**reply, "units": []}]) as gateway:
            native = _native(gateway)
            earlier = await _stored(native, "I live in Berlin.", model_first=False)
            got = await _decisions(
                native, earlier, "Ich wohne jetzt in München.", model_first=model_first
            )
        assert ("Der Nutzer wohnt in München", expected) in got, got


@pytest.mark.parametrize(
    ("text", "said", "kept"),
    [
        # a bare weekday: past or coming is not in the words, and the nollm contract leaves
        # it unresolved (``temporal``); resolved, a reminder "on Thursday" outranked a
        # standing rule in its conversation on the capability eval
        ("Remind me to water the plants on Thursday.", "Thursday", []),
        ("Erinnere mich am Donnerstag an die Pflanzen.", "Donnerstag", []),
        # with its direction the rules read it: the library's day is kept
        (
            "We met last Thursday.",
            "last Thursday",
            [{"text": "last Thursday", "date": "2023-05-04"}],
        ),
    ],
)
async def test_a_weekday_keeps_the_library_s_reading(text: str, said: str, kept: list) -> None:
    reply = _reply(times=[{"text": said, "kind": "relative_date", "date": "2023-05-11"}])
    with mocked_gateway([reply]) as gateway:
        found = await _extract(_native(gateway), text, model_first=True)
    turn = next(c for c in found if c.category == "verbatim_turn")
    assert turn.dated_mentions == kept
