"""The statement labeller (ADR 0037): the lexicon path in every pack language, the NLI and
LLM tiers against a scripted head and a mocked gateway, and what the write path stores.

The real head is exercised by ``tests/eval/test_statement_kinds_gate.py`` (``models``)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings, StatementLabellerSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind, StatementKind
from memory_service.domain.errors import DependencyUnavailable
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
        ("Wenn ich nach einer Bestandsprüfung frage, verwende immer eine Tabelle.", K.RULE),
        (
            "Benutze immer den Expressversand, außer die Bestellung liegt unter 50 Euro.",
            K.CONDITIONAL_RULE,
        ),
        ("Gabelstapler 4 ist wegen Wartung außer Betrieb.", K.STATUS),
        ("Nunca me sugieras recetas con cilantro.", K.RULE),
        (
            "Nunca incluyas artículos sin existencias a menos que escriba 'incluir agotados'.",
            K.CONDITIONAL_RULE,
        ),
        ("La sudadera azul se ha descatalogado.", K.LIFECYCLE),
        ("¡Buenos días!", None),
        ("لا تقترح عليّ أبدًا وصفات تحتوي على الكزبرة.", K.RULE),
        ("الرافعة الشوكية رقم 4 خارج الخدمة للصيانة.", K.STATUS),
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
        ("أجب دائمًا باللغة العربية.", K.RULE),
        (
            "Send alerts if the client's location deviates from their usual routine.",
            K.CONDITIONAL_RULE,
        ),
    ],
)
def test_what_the_words_cannot_settle_is_left_to_the_model(text: str, maybe: StatementKind) -> None:
    """Unsure, the lexicon keeps the sentence's reading without the rule - a durable rule
    nobody gave is worse than a rule kept only as the turn it was said in - and names the
    rule the head should check."""
    label = Lexicon.default().label(text)
    assert (label.decided, label.maybe) == (False, maybe), text
    # a request stays one ("Send alerts if ..." stores nothing new); anything else is a fact
    assert label.kind in (K.FACT, None), text


@pytest.mark.parametrize(
    ("text", "otherwise"),
    [
        # a discourse word alone does not correct: the head decides, else the sentence stays
        # what it surely is
        ("Actually, we terminated our contract with Uline yesterday due to pricing.", K.LIFECYCLE),
        ("Eigentlich haben wir den Vertrag mit Uline gestern gekündigt.", K.LIFECYCLE),
        ("في الواقع، أنهينا عقدنا مع Uline أمس.", K.LIFECYCLE),
        ("Actually I love hiking.", K.FACT),
        ("Eigentlich wohne ich in Berlin.", K.FACT),
        ("Realmente me gusta mucho el café.", K.FACT),
    ],
)
def test_a_discourse_word_alone_is_left_to_the_head(text: str, otherwise: StatementKind) -> None:
    label = Lexicon.default().label(text)
    assert (label.kind, label.decided, label.maybe) == (otherwise, False, K.CORRECTION), text


@pytest.mark.parametrize(
    "text",
    [
        "Actually, I have two kids, not three.",
        "Eigentlich ist es Dienstag, nicht Montag.",
        "En realidad la tasa es del 5%, no del 10%.",
    ],
)
def test_a_discourse_word_with_a_contrast_corrects(text: str) -> None:
    assert _kind(text) is K.CORRECTION, text


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


@pytest.mark.parametrize(
    "text",
    [
        # a condition trailing a clause no instruction word opens: the sentence describes
        "Life is so much more meaningful when we spend time together.",
        "Seeing their faces light up when they hit the court was priceless.",
        # a possessive opens a noun phrase, a negated copula a description
        "Your determination never ceases to amaze me.",
        "Writing isn't always easy but moments like these make me appreciate it.",
        # a standing word before a third-person verb: a habit, not an instruction
        "Nature always cheers me up and makes me feel grateful.",
        # "new" alone says nothing began
        "I'm playing this new RPG that has a really cool story and world.",
    ],
)
def test_a_description_costs_no_model_pass(text: str) -> None:
    """The sentence's shape settles these: a fact, decided, with no NLI pair."""
    label = Lexicon.default().label(text)
    assert (label.kind, label.decided, label.maybe) == (K.FACT, True, None), text


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        # the recipient written onto the verb, a feminine imperative ending
        ("إذا تأخرت الشحنة، أبلغيني فورًا.", K.CONDITIONAL_RULE),
        ("احرصي دائمًا على تحديث المخزون.", K.RULE),
        # the personal "a": the recipient of a Spanish instruction
        ("Si un envío se retrasa, avisa al cliente.", K.CONDITIONAL_RULE),
    ],
)
def test_word_forms_are_read_as_families(text: str, kind: StatementKind) -> None:
    label = Lexicon.default().label(text)
    assert (label.kind, label.decided) == (kind, True), text


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
    """Retail words are data: without the retail pack "on backorder" is a plain fact."""
    text = "SKU 10442 is on backorder until next month."
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
        # one hypothesis may serve several kinds (both rule kinds ask "an instruction?")
        self.hypotheses: dict[str, list[str]] = {}
        for kind, hypothesis in cfg.hypotheses.items():
            self.hypotheses.setdefault(hypothesis, []).append(kind)

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        out = []
        for premises, hypothesis in groups:
            kinds = self.hypotheses[hypothesis]
            row = []
            for premise in premises:
                self.pairs.append((premise, hypothesis))
                e = max(self.scores.get((premise, kind), 0.01) for kind in kinds)
                row.append(NLIScore(entailment=e, neutral=1 - e, contradiction=0.0))
            out.append(row)
        return out


AMBIGUOUS_RULE = "Formatiere Bestandsprüfungen immer als Markdown-Tabelle."
#: a word that only suggests a status ("stopped"): the head confirms it with one pair
PLAIN = "The cooler in aisle 5 stopped this morning."
CFG = StatementLabellerSettings()
SURE = CFG.thresholds[K.STATUS] + 0.05
#: above the bar at which a model proposal is confirmed, below the one at which the head decides
UNSURE = (CFG.llm_confirm_min + CFG.thresholds[K.STATUS]) / 2


async def test_the_head_decides_only_what_the_lexicon_left_open() -> None:
    nli = ScriptedNLI({(AMBIGUOUS_RULE, K.RULE): 0.97, (PLAIN, K.STATUS): SURE})
    labeller = StatementLabeller(nli=nli)  # type: ignore[arg-type]
    labels = await labeller.label([CASE_11, AMBIGUOUS_RULE, PLAIN, "Thanks!"])
    assert [label.kind for label in labels] == [K.CONDITIONAL_RULE, K.RULE, K.STATUS, None]
    assert [label.source for label in labels] == ["lexicon", "nli", "nli", "lexicon"]
    assert {premise for premise, _ in nli.pairs} == {AMBIGUOUS_RULE, PLAIN}, "one batch, open only"


async def test_a_sentence_costs_at_most_one_pair() -> None:
    """The head scores only the kind the lexicon suspects, once; a statement no cue marks
    costs nothing, in any language."""
    nli = ScriptedNLI({})
    labeller = StatementLabeller(nli=nli)  # type: ignore[arg-type]
    [suspected, plain, german] = await labeller.label(
        [PLAIN, "The cooler in aisle 5 is quite old.", "Der Kühler in Gang 5 ist sehr alt."]
    )
    assert {suspected.kind, plain.kind, german.kind} == {K.FACT}
    assert len(nli.pairs) == 1 and nli.pairs[0] == (PLAIN, CFG.hypotheses[K.STATUS])


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
    unsure = {(PLAIN, K.STATUS): UNSURE}
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
        sure = {(PLAIN, K.STATUS): SURE}
        labeller = StatementLabeller(
            nli=ScriptedNLI(sure),  # type: ignore[arg-type]
            assist=gw.assist(uses=["contextual_extraction"]),
        )
        await labeller.label([PLAIN])
        assert gw.route.call_count == 0, "a sure head needs no model call"


async def test_a_failing_gateway_keeps_the_head_s_answer() -> None:
    with mocked_gateway(failing=True) as gw:
        labeller = StatementLabeller(
            nli=ScriptedNLI({(PLAIN, K.STATUS): UNSURE}),  # type: ignore[arg-type]
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
    nli = ScriptedNLI({(PLAIN, K.STATUS): SURE})
    cands = await _extract(PLAIN, StatementLabeller(nli=nli))  # type: ignore[arg-type]
    assert {c.statement_kind for c in cands} == {K.STATUS}
    assert nli.pairs


class _YesHead(ScriptedNLI):
    """A trained head that entails every suspected kind: the upper bound of what it confirms."""

    def __init__(self) -> None:
        super().__init__({})

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        return [
            [NLIScore(entailment=0.99, neutral=0.01, contradiction=0.0) for _ in p]
            for p, _ in groups
        ]


@pytest.mark.parametrize(
    "text",
    [
        # one-off conditional requests, and conditional instructions that do not say they
        # are for every time: never a lasting rule, whatever the head says
        "Please send the report when it's ready.",
        "Let me know when the shipment arrives.",
        "Tell me if the price drops.",
        "Call me when you get this.",
        "Remind me when it's time to leave.",
        "If a delivery is late, notify me.",
        "Don't use bullet points unless I ask.",
        "Don't reorder seasonal items unless the buyer approves it.",
        "Avísame cuando llegues.",
        "Si el pedido llega tarde, avísame.",
        "Sag mir Bescheid, wenn du da bist.",
        "Wenn die Lieferung zu spät ist, sag mir Bescheid.",
        "إذا تأخرت الشحنة، أبلغني.",
        "अगर कीमत गिरे तो मुझे बताना।",
        # first-person habits and promises with a standing word
        "Siempre consultaré con el banco antes de tomar decisiones importantes.",
        "Nunca uso efectivo en la tienda.",
        "A partir de ahora, siempre validaré tus datos personales antes de proceder.",
        "Nunca responderá sobre tus transacciones personales sin tu autorización.",
        "Müsste ich immer vor jeder Neuanschaffung ein Fahrzeugprüfergebnis einholen.",
    ],
)
async def test_nothing_that_does_not_say_it_is_standing_is_kept_as_a_lasting_rule(
    text: str,
) -> None:
    for labeller in (StatementLabeller(), StatementLabeller(nli=_YesHead())):  # type: ignore[arg-type]
        for cand in await _extract(text, labeller):
            # stored as before the labeller (main's own patterns decide): never a rule
            assert cand.category != "rule" and cand.predicate != "rule", (text, cand)


@pytest.mark.parametrize(
    "text",
    ["Don't use bullet points unless I ask.", "Don't reorder seasonal items unless it rains."],
)
async def test_an_unmarked_conditional_imperative_stays_a_short_term_instruction(text: str) -> None:
    cands = await _extract(text, StatementLabeller(nli=_YesHead()))  # type: ignore[arg-type]
    [cand] = [c for c in cands if c.category != "verbatim_turn"]  # the turn is kept beside it
    assert (cand.category, cand.lifetime) == ("instruction", Lifetime.SHORT_TERM)
    assert cand.statement_kind is K.CONDITIONAL_RULE, "the kind is kept, as metadata only"


@pytest.mark.parametrize(
    "text",
    [
        "Always reply in German.",
        "Whenever I mention a new account, remind me of the security measures.",
        "Every time a delivery is late, notify the supplier.",
        "Siempre que puedas, usa tablas.",
        "Wenn die Kasse ausfällt, erinnere mich bitte immer an die Bezahlmöglichkeiten.",
        "Nunca me sugieras recetas con cilantro.",
        "हर बार जब मैं दवाओं की जानकारी मांगूं तो सटीक जानकारी दें।",
    ],
)
async def test_a_rule_that_says_it_is_standing_is_lasting(text: str) -> None:
    labeller = StatementLabeller(nli=_YesHead())  # type: ignore[arg-type]
    rules = [c for c in await _extract(text, labeller) if c.category == "rule"]
    assert rules and rules[0].lifetime is Lifetime.LONG_TERM, text
    assert rules[0].statement_kind in (K.RULE, K.CONDITIONAL_RULE)


class _BrokenHead(ScriptedNLI):
    def __init__(self, exc: Exception) -> None:
        super().__init__({})
        self.exc = exc

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        raise self.exc


@pytest.mark.parametrize(
    "exc", [RuntimeError("onnx session failed"), DependencyUnavailable("model queue is full")]
)
async def test_a_failing_head_keeps_the_lexicon_s_labels_and_the_message(exc: Exception) -> None:
    from memory_service.observability.metrics import statement_labeller_fallback_total

    before = statement_labeller_fallback_total.labels(tier="nli")._value.get()
    labeller = StatementLabeller(nli=_BrokenHead(exc))  # type: ignore[arg-type]
    labels = await labeller.label([AMBIGUOUS_RULE, "Our supplier is Uline."])
    assert [label.kind for label in labels] == [K.FACT, K.FACT]
    assert statement_labeller_fallback_total.labels(tier="nli")._value.get() == before + 1
    # and extraction stores the turn as if no head had been there
    cands = await _extract(AMBIGUOUS_RULE, labeller)
    assert any(c.category == "verbatim_turn" for c in cands)


def test_a_run_on_sentence_is_a_fact_unread() -> None:
    text = "no, " * 50_000
    started = time.perf_counter()
    label = Lexicon.default().label(text)
    assert label.kind is K.FACT and label.decided
    assert not Lexicon.default().is_question(text)
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Don" + chr(0x2019) + "t ever use tables.", K.RULE),  # a typographic apostrophe
        ("".join(chr(0xFF00 + ord(c) - 0x20) for c in "Never") + " suggest recipes.", K.RULE),
        ("Siempre consultaré con el banco antes de tomar decisiones.", K.FACT),
        ("Which is why I moved to Berlin last year.", K.FACT),
        ("Did an analysis on this series and I think it went ok!", K.FACT),
        ("No, gracias.", None),
        ("Call me when you get this.", None),
        ("Avísame cuando llegues.", None),
    ],
)
def test_normalised_forms_questions_and_one_off_requests(
    text: str, kind: StatementKind | None
) -> None:
    assert _kind(text) is kind, text
