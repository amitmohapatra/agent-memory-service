"""Answer coverage (ADR 0040): a question asking for a contact detail or a secret that no
evidence about what it names states, or for a month the evidence never names, is not
answered by that evidence."""

# ruff: noqa: RUF001 - the Hindi and Arabic fixtures are literal text in their own script

from __future__ import annotations

import pytest

from memory_service.config.constants import RetrievalSettings
from memory_service.domain.context_bundle import ContextItem, EvidenceReport
from memory_service.domain.enums import EvidenceStatus, Representation
from memory_service.modules.context.builder import ContextBuilder
from memory_service.modules.context.coverage import Asked, asked, months_named, unanswered
from memory_service.modules.context.evidence import NO_SHARED_TERM

pytestmark = pytest.mark.unit

STORES = ["The alarm code at Store 31 is 4471.", "Store 33 opens at 7 am on weekdays."]


@pytest.mark.parametrize(
    ("question", "kinds"),
    [
        ("What's the alarm code for Store 33?", {Asked.SECRET}),
        ("Wie lautet der Alarmcode für Filiale 33?", {Asked.SECRET}),
        ("What is the cell phone number for John Doe?", {Asked.PHONE}),
        ("Give me the Lakeside Dairy rep's email address.", {Asked.EMAIL}),
        ("أعطني رمز إنذار المتجر 33.", {Asked.SECRET}),
        ("कोर राउटर का एडमिन पासवर्ड क्या है?", {Asked.SECRET}),
        # writing a message is not asking for an address
        ("Draft a reorder email to our pallet supplier.", set()),
        ("हमारे सप्लायर को ईमेल ड्राफ़्ट करो।", set()),
        # a word inside another word, or the verb "to phone"
        ("नया हेडफ़ोन फ़िक्स्चर आया है।", set()),
        ("पता करो कि PO-77812 के लिए किसे फ़ोन करूँ।", set()),
        ("شاحنة التبريد من FreshWay عند البوابة.", set()),
        ("Who leads the spill response now?", set()),
    ],
)
def test_what_a_question_asks_for(question: str, kinds: set[Asked]) -> None:
    assert asked(question) == kinds


@pytest.mark.parametrize(
    "question",
    [
        "What's the alarm code for Store 33?",
        "Gib mir den Alarmcode von Filiale 33.",
        "Dame el código de alarma de la Tienda 33.",
        "أعطني رمز إنذار المتجر 33.",
    ],
)
def test_a_code_held_for_another_thing_does_not_answer(question: str) -> None:
    evidence = {
        "en": STORES,
        "de": ["Der Alarmcode in Filiale 31 ist 4471.", "Filiale 33 öffnet werktags um 7 Uhr."],
        "es": [
            "El código de la alarma de la Tienda 31 es 4471.",
            "La Tienda 33 abre a las 7 de la mañana entre semana.",
        ],
        "ar": [
            "رمز الإنذار في المتجر 31 هو 4471.",
            "يفتح المتجر 33 الساعة 7 صباحًا في أيام الأسبوع.",
        ],
    }
    lang = next(k for k, v in evidence.items() if k in _lang(question))
    assert (
        unanswered(question, evidence[lang]) == "no evidence states the code or password asked for"
    )


def _lang(question: str) -> str:
    if "Filiale" in question:
        return "de"
    if "Tienda" in question:
        return "es"
    if "المتجر" in question:
        return "ar"
    return "en"


def test_a_code_held_for_the_thing_asked_about_answers() -> None:
    assert unanswered("What's the alarm code for Store 31?", STORES) is None
    assert unanswered("Riverside gate code?", ["The gate code at Riverside is 5512#."]) is None


def test_a_person_the_evidence_holds_no_number_for() -> None:
    evidence = [
        "The OSHA inspection of Warehouse 3 is scheduled for the 24th.",
        "Our own safety manager is Rita Gomez, extension 300.",
    ]
    assert unanswered("What is the cell phone number for OSHA inspector John Doe?", evidence)
    assert unanswered("What's Rita Gomez's phone extension?", evidence) is None


def test_an_address_that_is_not_there() -> None:
    evidence = ["Orders to Lakeside go in through their supplier portal."]
    assert unanswered("What's the email address of our Lakeside rep?", evidence)
    assert unanswered("Lakeside rep email?", ["Lakeside rep: jo@lakeside.example"]) is None


def test_a_password_word_is_not_a_password() -> None:
    evidence = [
        "The core router is a Cisco ISR 4331 in rack B2 of the server room.",
        "Admin credentials live in the team password vault, not in chat.",
    ]
    assert unanswered("What's the admin password for the core router?", evidence)


def test_a_month_the_evidence_never_names() -> None:
    evidence = ["March shrink at Store 12 came in at 1.8% of sales."]
    assert unanswered("What was the shrink rate at Store 12 in April?", evidence)
    assert unanswered("What was the shrink rate at Store 12 in March?", evidence) is None
    # evidence that names no month is not judged by the month asked about
    assert unanswered("April shrink at Store 12?", ["Store 12 shrink is 1.8%."]) is None
    assert months_named("Der Umsatz im März") == {2}
    assert months_named("You may go") == frozenset()


def test_a_question_that_asks_for_no_such_value_is_not_judged() -> None:
    assert unanswered("How do I override a price at the register?", ["Use code PO-9."]) is None
    assert unanswered("What's the code?", []) is None


def _builder(floor: float) -> ContextBuilder:
    builder = object.__new__(ContextBuilder)
    builder.retrieval_cfg = RetrievalSettings(related_evidence_relevance=floor)
    return builder


def _item(relevance: float) -> ContextItem:
    return ContextItem.model_construct(
        item_id="m1",
        representation=Representation.MEMORY,
        text="Store 33 opens at 7 am.",
        relevance=relevance,
    )


NO_WORD = EvidenceReport(status=EvidenceStatus.INSUFFICIENT, notes=[NO_SHARED_TERM])


def test_a_related_memory_is_evidence_though_it_shares_no_word() -> None:
    report = _builder(0.3)._related_evidence(NO_WORD, [_item(0.45)], [])
    assert report.status is EvidenceStatus.COMPLETE
    assert NO_SHARED_TERM not in report.notes


@pytest.mark.parametrize(
    ("floor", "relevance", "knowledge", "report"),
    [
        (0.3, 0.25, False, NO_WORD),  # not close enough
        (0.0, 0.9, False, NO_WORD),  # switched off
        (0.3, 0.9, True, NO_WORD),  # a document bundle keeps its companion checks
        (
            0.3,
            0.9,
            False,
            EvidenceReport(status=EvidenceStatus.INSUFFICIENT),
        ),  # empty, not unrelated
    ],
)
def test_otherwise_the_overlap_rule_stands(
    floor: float, relevance: float, knowledge: bool, report: EvidenceReport
) -> None:
    items = [_item(relevance)]
    out = _builder(floor)._related_evidence(report, items, items if knowledge else [])
    assert out.status is EvidenceStatus.INSUFFICIENT


def test_a_german_name_run_with_a_marked_word_is_a_name() -> None:
    evidence = [
        "Die OSHA-Inspektion von Lager 3 ist für den 24. angesetzt.",
        "Unsere eigene Sicherheitsbeauftragte ist Rita Gomez, Durchwahl 300.",
    ]
    question = "Wie lautet die Handynummer des OSHA-Sicherheitsinspektors John Doe?"
    assert unanswered(question, evidence) == "no evidence states the phone number asked for"
    # a run of nouns no word of which is marked stays no name
    spill = "Ein Kanister Abflussreiniger ist umgekippt, welche Durchwahl hat der Zuständige?"
    assert unanswered(spill, ["Zuständig ist jetzt Kenji Ito, Durchwahl 233."]) is None
