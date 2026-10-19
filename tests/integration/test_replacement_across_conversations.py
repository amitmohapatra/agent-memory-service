"""A value the user replaces in a later conversation, through the real write and read paths
(ADR 0038).

"Our main supplier of shipping crates is Brightway." in March, then "We ended the
Brightway agreement ... Shipping crates now come from Corrimax only." in May: since a user's words follow the user (ADR 0036) both were current in every conversation,
and the old supplier kept being offered. Consolidation compares a statement only with its own
conversation, and a turn kept verbatim is never superseded by it - in another language there is
no reading at all. The hermetic suite's NLI is a stand-in that cannot read a contradiction, so
these tests give the pipeline a scripted head; the thresholds and the anchors are the real
ones.
"""

# ruff: noqa: RUF001 - the Hindi fixtures are literal text in their own script

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from memory_service.domain.enums import MessageRole, TemporalStatus, Visibility
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.feedback import FeedbackTargetKind, FeedbackVerdict, ProjectionAction
from memory_service.domain.memory import earlier_value, replaced_by
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.replacement import CrossConversationReplacement
from memory_service.ports.models import NLIScore
from tests.integration.test_feedback_projection import _admin, _submit
from tests.integration.test_memory import _memories, _observe
from tests.integration.test_statements_follow_the_user import _chat, _recalled, _say

pytestmark = pytest.mark.integration

STATEMENTS = {
    "en": (
        "Our main supplier of shipping crates is Brightway.",
        "We ended the Brightway agreement last week over delivery times. "
        "Shipping crates now come from Corrimax only.",
        "Who supplies our shipping crates?",
    ),
    "hi": (
        "शिपिंग क्रेट का हमारा मुख्य सप्लायर Brightway है।",
        "पिछले हफ़्ते डिलीवरी में देरी के कारण हमने Brightway के साथ समझौता ख़त्म कर दिया। "
        "अब शिपिंग क्रेट सिर्फ़ Corrimax से आते हैं।",
        "हमारे शिपिंग क्रेट कौन देता है?",
    ),
}


class ScriptedNLI:
    """A trained head's interface: a premise naming a later value contradicts a hypothesis
    naming only an earlier one (``values`` in the order they were said), at ``score``;
    everything else is neutral. ``always`` contradicts every pair (to show what the words
    alone keep apart); ``fail`` raises, as a broken or saturated head does."""

    representative = True

    def __init__(
        self,
        score: float,
        values: Sequence[str] = ("Brightway", "Corrimax"),
        *,
        always: bool = False,
        fail: bool = False,
    ) -> None:
        self.score, self.values, self.always, self.fail = score, list(values), always, fail
        self.calls = 0

    def _contradicts(self, premise: str, hypothesis: str) -> bool:
        if self.always:
            return True
        for later, value in enumerate(self.values):
            if value in premise:
                earlier = self.values[:later]
                if any(e in hypothesis for e in earlier) and value not in hypothesis:
                    return True
        return False

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        self.calls += 1
        if self.fail:
            raise DependencyUnavailable("the NLI head's queue is full")
        out = []
        for premises, hypothesis in groups:
            row = []
            for premise in premises:
                c = self.score if self._contradicts(premise, hypothesis) else 0.02
                row.append(NLIScore(entailment=0.0, neutral=1 - c, contradiction=c))
            out.append(row)
        return out


class TenantModel:
    """The tenant's model through ``LLMAssist``'s interface (``conflict_adjudication``
    allowed): its verdict, or ``None`` - what ``structured`` returns when the call fails."""

    def __init__(self, verdict: str | None = "update") -> None:
        self.verdict, self.calls = verdict, 0

    def wants(self, use: str) -> bool:
        return use == "conflict_adjudication"

    async def structured(self, use: str, **_: object) -> dict | None:
        self.calls += 1
        return {"verdict": self.verdict} if self.verdict else None


#: every test runs twice: without a model (nollm) and with the tenant's model allowed (llm)
_MODE: dict[str, TenantModel | None] = {"assist": None}


@pytest.fixture(autouse=True, params=["nollm", "llm"])
def mode(request):
    _MODE["assist"] = TenantModel() if request.param == "llm" else None
    yield request.param
    _MODE["assist"] = None


def _scripted(container, score: float, *, assist: TenantModel | None = None, **kw) -> ScriptedNLI:
    nli = ScriptedNLI(score, **kw)
    pipeline = container.services["observation_pipeline"]
    pipeline.replacement = CrossConversationReplacement(
        nli,
        assist=assist or _MODE["assist"],  # type: ignore[arg-type]
    )
    return nli


async def _full(container, ctx, query: str) -> dict:
    body = await container.services["context_builder"].build_api(ctx, query, output="full")
    return json.loads(body)


async def _old_forms(container, uow_factory, ctx, text: str):
    return [
        m
        for m in await _memories(uow_factory, ctx, container, include_superseded=True)
        if m.content == text
    ]


@pytest.mark.parametrize("lang", ["en", "hi"])
async def test_a_value_replaced_in_a_later_conversation_is_closed(
    container, uow_factory, lang
) -> None:
    old, new, ask = STATEMENTS[lang]
    _scripted(container, 0.97)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new)

    forms = await _old_forms(container, uow_factory, march, old)
    assert forms, "the old statement is kept, as history"
    assert all(m.temporal.status is TemporalStatus.SUPERSEDED for m in forms)
    assert all(replaced_by(m.system_metadata) for m in forms)
    # every form: the turn, and in English the fact read from it
    if lang == "en":
        assert {m.system_metadata["category"] for m in forms} == {"fact", "verbatim_turn"}

    for where in (may, _chat()):  # the same conversation, and a new one
        recalled = await _recalled(container, where, ask)
        assert "Corrimax" in recalled
        assert old not in recalled.split("\n")


@pytest.mark.parametrize("lang", ["en", "hi"])
async def test_a_weaker_contradiction_is_served_apart_as_an_earlier_value(
    container, uow_factory, lang
) -> None:
    old, new, ask = STATEMENTS[lang]
    _scripted(container, 0.87)  # above the earlier-value bar, below the supersede bar
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new)

    forms = await _old_forms(container, uow_factory, march, old)
    assert forms and all(m.temporal.status is TemporalStatus.CURRENT for m in forms)
    assert all(earlier_value(m.system_metadata) for m in forms)

    bundle = await _full(container, _chat(), ask)
    texts = [m["text"] for m in bundle.get("memories", [])]
    assert any("Corrimax" in t for t in texts), "the current value is served"
    assert old not in texts, "never among the current evidence"
    history = bundle.get("history", [])
    assert old in [m["text"] for m in history], "but kept, in the history section"
    assert all(m.get("superseded") is True for m in history)

    rendered = await container.services["context_builder"].build_api(_chat(), ask)
    text = json.loads(rendered)["rendered"]
    current, heading, earlier = text.partition("## Earlier values (since replaced)")
    assert heading and old in earlier and old not in current


async def test_forgetting_the_new_statement_reinstates_the_old_one(container, uow_factory) -> None:
    old, new, ask = STATEMENTS["en"]
    _scripted(container, 0.97)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new)
    [turn] = [
        m
        for m in await _memories(uow_factory, may, container)
        if m.content == new and m.system_metadata.get("category") == "verbatim_turn"
    ]
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, may, turn.memory_id)
        await uow.commit()
    await container.tasks.drain()

    forms = await _old_forms(container, uow_factory, march, old)
    assert forms and all(m.temporal.status is TemporalStatus.CURRENT for m in forms)
    assert not any(replaced_by(m.system_metadata) for m in forms)
    assert "Brightway" in await _recalled(container, _chat(), ask)


async def test_a_turn_replaced_in_the_same_conversation_closes_too(container, uow_factory) -> None:
    """B1's claim ("a value replaced later in the conversation acts on both") held only for a
    turn whose reading has the same words; a turn of two sentences stayed current."""
    _scripted(container, 0.97)
    chat = _chat()
    old = "Our main supplier of shipping crates is Brightway. We reorder every Monday."
    await _say(container, uow_factory, chat, old)
    await _say(container, uow_factory, chat, STATEMENTS["en"][1])
    [turn] = [
        m
        for m in await _memories(uow_factory, chat, container, include_superseded=True)
        if m.content == old
    ]
    assert turn.temporal.status is TemporalStatus.SUPERSEDED


async def test_without_a_trained_head_both_stay_current(container, uow_factory) -> None:
    """The hermetic stand-in cannot read a contradiction: nothing is replaced on its say."""
    old, new, _ = STATEMENTS["en"]
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new)
    forms = await _old_forms(container, uow_factory, march, old)
    assert forms and all(m.temporal.status is TemporalStatus.CURRENT for m in forms)
    assert not any(replaced_by(m.system_metadata) for m in forms)


async def test_another_persons_statement_is_never_replaced(container, uow_factory) -> None:
    old, new, _ = STATEMENTS["en"]
    _scripted(container, 0.99)
    theirs, mine = _chat("u2"), _chat()
    await _say(container, uow_factory, theirs, old)
    await _say(container, uow_factory, mine, new)
    forms = await _old_forms(container, uow_factory, theirs, old)
    assert forms and all(m.temporal.status is TemporalStatus.CURRENT for m in forms)


async def test_the_assistants_words_are_never_a_replacement(container, uow_factory) -> None:
    old, new, _ = STATEMENTS["en"]
    _scripted(container, 0.99)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new, role=MessageRole.ASSISTANT)
    forms = await _old_forms(container, uow_factory, march, old)
    assert forms and all(m.temporal.status is TemporalStatus.CURRENT for m in forms)


# --------------------------------------------------------------------------- edges


def _status(forms) -> set[TemporalStatus]:
    return {m.temporal.status for m in forms}


async def test_rejecting_the_new_statement_reinstates_the_old_one(container, uow_factory) -> None:
    """Feedback that withdraws the newer statement (an owner's REJECT, approved by a tenant
    admin) restores what it replaced, as forgetting it does."""
    old, new, _ = STATEMENTS["en"]
    _scripted(container, 0.97)
    owner = _chat("alice", workspace_id="fin")
    later = _chat("alice", workspace_id="fin")
    await _say(container, uow_factory, owner, old)
    await _say(container, uow_factory, later, new)
    assert _status(await _old_forms(container, uow_factory, owner, old)) == {
        TemporalStatus.SUPERSEDED
    }
    [turn] = [
        m
        for m in await _memories(uow_factory, later, container)
        if m.content == new and m.system_metadata.get("category") == "verbatim_turn"
    ]
    await _admin(container, uow_factory)
    rejected = await _submit(
        container,
        uow_factory,
        later,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=turn.memory_id,
        verdict=FeedbackVerdict.REJECT,
    )
    assert rejected.projection is not None
    assert rejected.projection.action is ProjectionAction.MEMORY_RETRACTED
    assert _status(await _old_forms(container, uow_factory, owner, old)) == {TemporalStatus.CURRENT}


async def test_three_successive_values(container, uow_factory) -> None:
    """A -> B -> C: each replaces the one before; the first stays closed by the second."""
    values = ("Brightway", "Corrimax", "CrateHub")
    _scripted(container, 0.97, values=values)
    a = "Our main supplier of shipping crates is Brightway."
    b = "We dropped Brightway. Our crate supplier is Corrimax now."
    c = "We dropped Corrimax. Our crate supplier is CrateHub now."
    first, second, third = _chat(), _chat(), _chat()
    await _say(container, uow_factory, first, a)
    await _say(container, uow_factory, second, b)
    await _say(container, uow_factory, third, c)
    a_forms = await _old_forms(container, uow_factory, first, a)
    b_forms = await _old_forms(container, uow_factory, second, b)
    assert _status(a_forms) == {TemporalStatus.SUPERSEDED}
    assert _status(b_forms) == {TemporalStatus.SUPERSEDED}
    b_turn = next(m for m in b_forms if m.system_metadata.get("category") == "verbatim_turn")
    assert all(replaced_by(m.system_metadata) == b_turn.memory_id for m in a_forms)
    recalled = await _recalled(container, _chat(), "Who supplies our shipping crates?")
    assert "CrateHub" in recalled
    assert a not in recalled.split("\n") and b not in recalled.split("\n")


async def test_the_same_subject_in_another_language(container, uow_factory) -> None:
    """Said in English, replaced in Hindi: the name carries across the scripts."""
    _scripted(container, 0.97)
    old = STATEMENTS["en"][0]
    english, hindi = _chat(), _chat()
    await _say(container, uow_factory, english, old)
    await _say(container, uow_factory, hindi, STATEMENTS["hi"][1])
    assert _status(await _old_forms(container, uow_factory, english, old)) == {
        TemporalStatus.SUPERSEDED
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "Delivery bike #3 stays in the yard for the week.",
            "Delivery bike #4 got new brakes and is out on routes again.",
        ),
        (
            "ट्रॉली #3 इस हफ़्ते गोदाम में रहेगी।",
            "ट्रॉली #4 की मरम्मत हो गई है और वह फिर से काम पर है।",
        ),
    ],
    ids=["en", "hi"],
)
async def test_different_identifiers_are_never_replaced(
    container, uow_factory, old: str, new: str
) -> None:
    """Even a head that reads every pair as a contradiction cannot close bike #3 with a
    statement about bike #4."""
    nli = _scripted(container, 0.99, always=True)
    before, after = _chat(), _chat()
    await _say(container, uow_factory, before, old)
    await _say(container, uow_factory, after, new)
    forms = await _old_forms(container, uow_factory, before, old)
    assert forms and _status(forms) == {TemporalStatus.CURRENT}
    assert not any(replaced_by(m.system_metadata) for m in forms)
    assert nli.calls == 0, "two things of one kind are never even asked about"


@pytest.mark.parametrize("visibility", [Visibility.THREAD, Visibility.WORKSPACE])
async def test_a_statement_kept_to_its_conversation_or_team_is_not_replaced(
    container, uow_factory, visibility: Visibility
) -> None:
    """Only what follows the user (USER) is compared across conversations: an explicit
    THREAD statement stays its conversation's, a WORKSPACE one the team's."""
    old, new, _ = STATEMENTS["en"]
    _scripted(container, 0.99)
    here = _chat()
    async with uow_factory() as uow:
        await container.services["conversation"].create_thread(uow, here)
        await uow.commit()
    await _observe(container, uow_factory, here, old, hints=ProcessingHints(visibility=visibility))
    await _say(container, uow_factory, _chat(), new)
    forms = [
        m
        for m in await _memories(uow_factory, here, container, include_superseded=True)
        if m.content == old
    ]
    assert forms and _status(forms) == {TemporalStatus.CURRENT}


async def test_concurrent_statements_about_one_subject(container, uow_factory) -> None:
    """Two later statements written at once: the old value is replaced (once), both new
    statements stay current, and nothing fails."""
    import asyncio

    old, new, _ = STATEMENTS["en"]
    _scripted(container, 0.97)
    first = _chat()
    await _say(container, uow_factory, first, old)
    other = "We no longer buy crates from Brightway. Corrimax supplies them now."
    a, b = _chat(), _chat()
    await asyncio.gather(
        _say(container, uow_factory, a, new), _say(container, uow_factory, b, other)
    )
    forms = await _old_forms(container, uow_factory, first, old)
    assert _status(forms) == {TemporalStatus.SUPERSEDED}
    current = {m.content for m in await _memories(uow_factory, a, container)}
    current |= {m.content for m in await _memories(uow_factory, b, container)}
    assert new in current and other in current


async def test_a_failing_head_costs_no_message(container, uow_factory) -> None:
    """The NLI head refusing (a full queue) or failing: the new statement is stored, the old
    one stays current, and the next statement is compared again."""
    old, new, _ = STATEMENTS["en"]
    nli = _scripted(container, 0.97, fail=True)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new)
    assert nli.calls >= 1
    assert new in {m.content for m in await _memories(uow_factory, may, container)}
    assert _status(await _old_forms(container, uow_factory, march, old)) == {TemporalStatus.CURRENT}


PLAIN_NEW = "Our shipping crates come from Corrimax and Brightway."


@pytest.mark.parametrize(("verdict", "marked"), [("update", True), (None, False)])
async def test_the_model_settles_the_heads_unsure_band(
    container, uow_factory, verdict, marked
) -> None:
    """A related pair the head reads at 0.6 (no change cue): only the tenant's model can mark
    it, as an earlier value; a failing model call (``None``) marks nothing and costs no
    message."""
    old = STATEMENTS["en"][0]
    model = TenantModel(verdict)
    _scripted(container, 0.6, assist=model)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, PLAIN_NEW)
    assert model.calls >= 1
    forms = await _old_forms(container, uow_factory, march, old)
    assert forms and _status(forms) == {TemporalStatus.CURRENT}
    assert all(earlier_value(m.system_metadata) for m in forms) is marked
    assert PLAIN_NEW in {m.content for m in await _memories(uow_factory, may, container)}


# --------------------------------------------------------------------------- slots


async def test_a_slot_read_from_both_statements_relates_them(container, uow_factory) -> None:
    """ "I live in Leeds. The commute is short." then "I moved to Bristol last month.": no name
    in common and no encoder. The rules read the user's ``lives_in`` from both; consolidation
    closes the old reading (a single-valued slot), and the slot it held relates the turn it
    came from, which the head's 0.6 then marks as an earlier value."""
    old, new = "I live in Leeds. The commute is short.", "I moved to Bristol last month."
    _scripted(container, 0.6, values=("Leeds", "Bristol"))
    before, after = _chat(), _chat()
    await _say(container, uow_factory, before, old)
    await _say(container, uow_factory, after, new)
    [turn] = await _old_forms(container, uow_factory, before, old)
    assert turn.temporal.status is TemporalStatus.CURRENT
    assert earlier_value(turn.system_metadata)
    bundle = await _full(container, _chat(), "Where do I live?")
    current = [m["text"] for m in bundle.get("memories", [])]
    current += [b["text"] for b in bundle.get("profile", [])]
    assert any("Bristol" in t for t in current) and old not in current
    assert old in [m["text"] for m in bundle.get("history", [])]


async def test_another_subjects_slot_is_left_alone(container, uow_factory) -> None:
    """The same slot of two subjects is two slots: a user's move says nothing about where
    another user lives."""
    old, new = "I live in Leeds. The commute is short.", "I moved to Bristol last month."
    _scripted(container, 0.6, values=("Leeds", "Bristol"))
    theirs, mine = _chat("u2"), _chat()
    await _say(container, uow_factory, theirs, old)
    await _say(container, uow_factory, mine, new)
    forms = await _old_forms(container, uow_factory, theirs, old)
    assert forms and not any(replaced_by(m.system_metadata) for m in forms)


async def test_an_earlier_value_found_alone_brings_its_replacement(container, uow_factory) -> None:
    """A question in the old statement's words: the search finds only the earlier value, and
    the statement that replaced it is served with it, as current, before it."""
    old = "Our main supplier of shipping crates is Brightway."
    new = "We ended the Brightway agreement. Corrimax took over in May."
    _scripted(container, 0.87)
    march, may = _chat(), _chat()
    await _say(container, uow_factory, march, old)
    await _say(container, uow_factory, may, new)
    bundle = await _full(container, _chat(), "Who is our main supplier of shipping crates?")
    assert new in [m["text"] for m in bundle.get("memories", [])]
    assert old in [m["text"] for m in bundle.get("history", [])]
