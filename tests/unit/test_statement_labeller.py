"""The statement labeller (ADR 0035): the lexicon path in every pack language, the NLI and
LLM tiers against a scripted head and a mocked gateway, and what the write path stores.

The real head is exercised by ``tests/eval/test_statement_kinds_gate.py`` (``models``)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings, StatementLabellerSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, ObservationKind, StatementKind
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import statement_kind_of
from memory_service.domain.observation import Observation
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.statements import (
    Lexicon,
    StatementLabeller,
    load_packs,
    most_salient,
)
from memory_service.ports.models import NLIScore
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit

K = StatementKind
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1")

CASE_8 = (
    "Whenever I ask for a stock audit, always format the response as a markdown table with "
    "columns for: SKU, Item Name, Current Stock, Reorder Threshold, and Action Required."
)
CASE_11 = (
    "Never include items with a stock level of zero in my weekly category overviews unless "
    "I specifically type 'include out of stock'."
)


def _kind(text: str) -> StatementKind | None:
    return Lexicon.default().label(text).kind


# --------------------------------------------------------------------------- lexicon


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (CASE_8, K.RULE),
        ("When I ask for a stock audit, always format the response as a markdown table.", K.RULE),
        (CASE_11, K.CONDITIONAL_RULE),
        # case 11 with text before "never": it used to be no rule at all
        ("For weekly overviews, never include items with zero stock.", K.RULE),
        ("Important: never share supplier pricing with store staff.", K.RULE),
        ("All seasonal holiday merchandise must be routed to Overflow Storage Facility B.", K.RULE),
        (
            "If a delivery is more than two hours late, always notify the store manager.",
            K.CONDITIONAL_RULE,
        ),
        (
            "Only escalate a ticket to engineering if it has been open for 48 hours.",
            K.CONDITIONAL_RULE,
        ),
        ("No, our system was updated.", K.CORRECTION),
        (
            "We now use the /shrinkage command to write off damage so it hits the right ledger.",
            K.CORRECTION,
        ),
        ("Actually, we terminated our contract with Uline yesterday due to pricing.", K.CORRECTION),
        ("Our new exclusive pallet supplier is PackagingCorp.", K.LIFECYCLE),
        (
            "The temporary refrigeration unit has been dismantled and the project is closed.",
            K.LIFECYCLE,
        ),
        ("Forklift #4 has been repaired and is back on the floor.", K.STATUS),
        ("Log an incident: Forklift #4 is out of service for maintenance.", K.STATUS),
        ("The master lock code for the hazmat cage in Warehouse 3 is 8492.", K.FACT),
        ("Store 41 is the largest location in the northern region.", K.FACT),
        # questions, greetings and one-off requests carry no kind - never a rule
        ("Do you always round prices?", None),
        ("Why is the walk-in freezer always so loud?", None),
        ("Whenever you get a chance, could you send me the report?", None),
        ("Give me a stock audit for the office supplies category.", None),
        ("Never mind.", None),
        ("Thanks, that's helpful!", None),
    ],
)
def test_english_statements(text: str, kind: StatementKind | None) -> None:
    assert _kind(text) is kind, text


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Gabelstapler 4 ist wegen Wartung außer Betrieb.", K.STATUS),
        ("Eigentlich haben wir den Vertrag mit Uline gestern gekündigt.", K.CORRECTION),
        ("Nunca me sugieras recetas con cilantro.", K.RULE),
        (
            "Nunca incluyas artículos sin existencias a menos que escriba 'incluir agotados'.",
            K.CONDITIONAL_RULE,
        ),
        ("La sudadera azul se ha descatalogado.", K.LIFECYCLE),
        ("¡Buenos días!", None),
        ("لا تقترح عليّ أبدًا وصفات تحتوي على الكزبرة.", K.RULE),
        ("الرافعة الشوكية رقم 4 خارج الخدمة للصيانة.", K.STATUS),
        ("في الواقع، أنهينا عقدنا مع Uline أمس.", K.CORRECTION),
        ("मुझे कभी भी धनिया वाली रेसिपी मत सुझाना।", K.RULE),
        # "hamesha ke liye" is "for good", not "always": closed for good is a lifecycle
        ("स्टोर 22 शनिवार को हमेशा के लिए बंद हो गया।", K.LIFECYCLE),
        ("कोल्ड रूम हमेशा इतना शोर क्यों करता है?", None),
    ],
)
def test_other_languages(text: str, kind: StatementKind | None) -> None:
    assert _kind(text) is kind, text


@pytest.mark.parametrize(
    ("text", "maybe"),
    [
        # a team habit or a policy? "We always look forward to our camping trip" is not one
        ("We never ship hazardous goods on Fridays.", K.RULE),
        # a standing rule, or advice ("If SF is your thing, check out The Expanse")?
        ("If the forklift is out of service, route heavy pallets to Dock 3.", K.CONDITIONAL_RULE),
        # verb-first languages: the standing word follows a verb the packs cannot list
        ("Wenn ich nach einer Bestandsprüfung frage, verwende immer eine Tabelle.", K.RULE),
        (
            "Benutze immer den Expressversand, außer die Bestellung liegt unter 50 Euro.",
            K.CONDITIONAL_RULE,
        ),
        ("أجب دائمًا باللغة العربية.", K.RULE),
    ],
)
def test_what_the_words_cannot_settle_is_left_to_the_model(text: str, maybe: StatementKind) -> None:
    """Unsure, the lexicon answers FACT - a durable rule nobody gave is worse than a rule
    kept only as the turn it was said in - and names the rule the head should check."""
    label = Lexicon.default().label(text)
    assert (label.kind, label.decided, label.maybe) == (K.FACT, False, maybe), text


@pytest.mark.parametrize(
    "text",
    [
        "Always happy to help.",
        "Rock concerts always have such an electrifying atmosphere!",
        "Keep going and never give up!",
        "It must be exciting to see it all come together.",
        "In the future, I'm aiming to work on projects that make a difference.",
        "No, I haven't tried it yet.",
        "Thanks for reminding me to focus on progress, not perfection.",
        "We ended up with some awesome sponsors.",
        "Wow, that sunset is stunning!",
        "I'm stuck with my music at the moment.",
    ],
)
def test_chat_is_not_read_as_an_instruction_or_a_change(text: str) -> None:
    """Sentences of the kind LoCoMo's dialogues are full of (a precision check on text the
    packs were not written from): none is a rule, a correction, a status or a lifecycle."""
    assert Lexicon.default().label(text).kind in (K.FACT, None), text


def test_a_rule_keeps_its_trigger_and_its_exception() -> None:
    audit = Lexicon.default().label(CASE_8)
    assert audit.trigger == "Whenever I ask for a stock audit" and audit.exception is None
    overview = Lexicon.default().label(CASE_11)
    assert overview.exception == "unless I specifically type 'include out of stock'"
    late = Lexicon.default().label(
        "If a delivery is more than two hours late, always notify the store manager."
    )
    assert late.trigger == "If a delivery is more than two hours late"
    assert Lexicon.default().label("From now on, reply in German.").instruction == (
        "reply in German"
    )


def test_when_opens_a_question_only_before_an_auxiliary() -> None:
    lexicon = Lexicon.default()
    assert not lexicon.is_question("When I ask for a stock audit, always use a table.")
    assert lexicon.is_question("When is the next delivery")
    assert lexicon.is_question("What's the code for the hazmat cage")
    assert lexicon.is_question("¿Cuál es el código?") and lexicon.is_question("ما هو الرمز؟")


def test_domain_words_come_from_the_packs() -> None:
    """Retail words are data: without the retail pack "out of stock" is a plain fact."""
    text = "Blue hoodies in size M are out of stock at the downtown store."
    assert Lexicon(("generic", "retail")).label(text).kind is K.STATUS
    assert Lexicon(("generic",)).label(text).kind is K.FACT
    assert load_packs(("generic", "retail"))["status"]["de"]


def test_an_unknown_pack_field_is_refused(tmp_path, monkeypatch) -> None:
    from memory_service.modules.memory import statements

    (tmp_path / "bad.json").write_text('{"languages": {"en": {"stauts": ["x"]}}}')
    monkeypatch.setattr(statements, "LEXICON_DIR", tmp_path)
    with pytest.raises(ValueError, match="stauts"):
        load_packs(("bad",))


def test_a_turn_carries_its_most_telling_kind() -> None:
    assert most_salient([K.FACT, K.CORRECTION, K.STATUS]) is K.CORRECTION
    assert most_salient([None, K.FACT]) is K.FACT and most_salient([None]) is None


# --------------------------------------------------------------------------- model tiers


class ScriptedNLI:
    """A trained head's interface with scripted entailment: ``(sentence, kind) -> score``."""

    representative = True

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []
        cfg = StatementLabellerSettings()
        self.hypotheses: dict[str, str] = {v: k for k, v in cfg.hypotheses.items()}
        self.hypotheses[cfg.screen_hypothesis] = "CHANGE"

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        out = []
        for premises, hypothesis in groups:
            kind = self.hypotheses[hypothesis]
            row = []
            for premise in premises:
                self.pairs.append((premise, hypothesis))
                e = self.scores.get((premise, kind), 0.01)
                row.append(NLIScore(entailment=e, neutral=1 - e, contradiction=0.0))
            out.append(row)
        return out


AMBIGUOUS_RULE = "Formatiere Bestandsprüfungen immer als Markdown-Tabelle."
#: a statement no German cue marks: the head screens it for a change, then checks the kinds
PLAIN = "Der Kühler in Gang 5 macht seit heute ein mahlendes Geräusch."
CHANGED = {(PLAIN, "CHANGE"): 0.9}


async def test_the_head_decides_only_what_the_lexicon_left_open() -> None:
    nli = ScriptedNLI({(AMBIGUOUS_RULE, K.RULE): 0.97, (PLAIN, K.STATUS): 0.99, **CHANGED})
    labeller = StatementLabeller(nli=nli)  # type: ignore[arg-type]
    labels = await labeller.label([CASE_11, AMBIGUOUS_RULE, PLAIN, "Thanks!"])
    assert [label.kind for label in labels] == [K.CONDITIONAL_RULE, K.RULE, K.STATUS, None]
    assert [label.source for label in labels] == ["lexicon", "nli", "nli", "lexicon"]
    assert {premise for premise, _ in nli.pairs} == {AMBIGUOUS_RULE, PLAIN}, "one batch, open only"


async def test_a_statement_costs_one_pair_unless_something_changed() -> None:
    """Outside English an open statement is screened with one pair; only a change is checked
    against the kinds. An English statement no cue marks is a fact with no pair at all."""
    nli = ScriptedNLI({})
    labeller = StatementLabeller(nli=nli)  # type: ignore[arg-type]
    [german, english] = await labeller.label([PLAIN, "The cooler in aisle 5 is quite old."])
    assert german.kind is K.FACT and english.kind is K.FACT
    assert len(nli.pairs) == 1 and nli.pairs[0][0] == PLAIN


async def test_a_standing_word_the_head_rejects_is_a_fact() -> None:
    labeller = StatementLabeller(nli=ScriptedNLI({}))  # type: ignore[arg-type]
    [label] = await labeller.label([AMBIGUOUS_RULE])
    assert label.kind is K.FACT and label.source == "nli"


async def test_a_stand_in_head_is_never_read_as_a_classifier() -> None:
    from memory_service.adapters.models.nli import LexicalNLI

    labeller = StatementLabeller(nli=LexicalNLI())  # type: ignore[arg-type]
    assert labeller.nli is None
    [label] = await labeller.label([PLAIN])
    assert label.kind is K.FACT and label.source == "lexicon"


async def test_the_model_is_asked_only_when_the_head_is_unsure_and_must_be_confirmed() -> None:
    # above the confirmation bar, below the decision bar
    unsure = {(PLAIN, K.STATUS): 0.6, **CHANGED}
    with mocked_gateway([{"labels": [{"index": 0, "kind": "status"}]}]) as gw:
        labeller = StatementLabeller(
            nli=ScriptedNLI(unsure),  # type: ignore[arg-type]
            assist=gw.assist(uses=["contextual_extraction"]),
        )
        [label] = await labeller.label([PLAIN])
        assert label.kind is K.STATUS and label.source == "llm"
        assert gw.route.call_count == 1
    with mocked_gateway([{"labels": [{"index": 0, "kind": "lifecycle"}]}]) as gw:
        labeller = StatementLabeller(
            nli=ScriptedNLI(unsure),  # type: ignore[arg-type]
            assist=gw.assist(uses=["contextual_extraction"]),
        )
        [label] = await labeller.label([PLAIN])
        assert label.kind is K.FACT, "a proposal the head does not confirm is not taken"
    with mocked_gateway([{"labels": [{"index": 0, "kind": "status"}]}]) as gw:
        sure = {(PLAIN, K.STATUS): 0.99, **CHANGED}
        labeller = StatementLabeller(
            nli=ScriptedNLI(sure),  # type: ignore[arg-type]
            assist=gw.assist(uses=["contextual_extraction"]),
        )
        await labeller.label([PLAIN])
        assert gw.route.call_count == 0, "a sure head needs no model call"


async def test_a_failing_gateway_keeps_the_head_s_answer() -> None:
    with mocked_gateway(failing=True) as gw:
        labeller = StatementLabeller(
            nli=ScriptedNLI({(PLAIN, K.STATUS): 0.6, **CHANGED}),  # type: ignore[arg-type]
            assist=gw.assist(uses=["contextual_extraction"]),
        )
        [label] = await labeller.label([PLAIN])
        assert label.kind is K.FACT


# --------------------------------------------------------------------------- write path


def _obs(text: str) -> Observation:
    return Observation(
        tenant_id=CTX.tenant_id,
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        user_id=CTX.user_id,
        thread_id=CTX.thread_id,
        workspace_id=CTX.workspace_id,
        principal_id=CTX.principal_id,
        message_id="msg_1",
    )


async def _extract(text: str, labeller: StatementLabeller | None = None):
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings(), labeller=labeller)
    return [await native.classify(c, CTX) for c in await native.extract(_obs(text), CTX)]


@pytest.mark.parametrize(
    "text",
    [CASE_8, "When I ask for a stock audit, always format the response as a markdown table."],
)
async def test_case_8_is_stored_as_a_rule(text: str) -> None:
    [rule] = await _extract(text)
    assert rule.predicate == "rule" and rule.memory_type is MemoryType.PREFERENCE
    assert rule.statement_kind is K.RULE
    assert rule.rule_trigger and rule.rule_trigger.lower().endswith("i ask for a stock audit")


@pytest.mark.parametrize(
    "text",
    [CASE_11, "For my weekly category overviews, never include items with a stock level of zero."],
)
async def test_case_11_is_a_rule_wherever_never_sits(text: str) -> None:
    [rule] = await _extract(text)
    assert rule.predicate == "rule" and rule.category == "rule"
    assert rule.rule_exception == (
        "unless I specifically type 'include out of stock'" if "unless" in text else None
    )
    memory = build_memory(rule, CTX, now=datetime.now(UTC))
    assert statement_kind_of(memory.system_metadata) is rule.statement_kind
    if rule.rule_exception:
        assert memory.system_metadata["rule_exception"] == rule.rule_exception


async def test_a_rule_in_another_language_is_kept_as_a_rule() -> None:
    [rule] = await _extract("Nunca me sugieras recetas con cilantro.")
    assert rule.predicate == "rule" and rule.statement_kind is K.RULE


async def test_every_candidate_carries_its_kind_and_the_turn_its_most_telling_one() -> None:
    cands = await _extract(
        "No, our system was updated. We now use the /shrinkage command to write off damage."
    )
    turn = next(c for c in cands if c.category == "verbatim_turn")
    assert turn.statement_kind is K.CORRECTION
    assert all(c.statement_kind is K.CORRECTION for c in cands)
    [status] = await _extract("Forklift #4 has been repaired and is back on the floor.")
    assert status.statement_kind is K.STATUS


async def test_the_head_reaches_extraction_for_user_messages_only() -> None:
    nli = ScriptedNLI({(PLAIN, K.STATUS): 0.99, **CHANGED})
    cands = await _extract(PLAIN, StatementLabeller(nli=nli))  # type: ignore[arg-type]
    assert {c.statement_kind for c in cands} == {K.STATUS}
    assert nli.pairs
