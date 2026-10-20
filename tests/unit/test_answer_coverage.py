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


def test_an_item_naming_no_month_is_about_the_month_it_is_dated_in() -> None:
    evidence = ["James adopted a puppy named Ned.", "In June James visited Canada."]
    april, june = frozenset({3}), frozenset({5})
    question = "What did James adopt in April 2022?"
    assert unanswered(question, evidence)  # no dates known: only June is named
    assert unanswered(question, evidence, [april, june]) is None
    assert unanswered(question, evidence, [june, june])
    # a month the text names wins over when it was said
    march = ["March shrink at Store 12 came in at 1.8% of sales."]
    assert unanswered("What was the shrink rate at Store 12 in April?", march, [april])


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


def _said(text: str, observed_at: str) -> ContextItem:
    return ContextItem.model_construct(
        item_id="m1",
        representation=Representation.MEMORY,
        text=text,
        relevance=0.6,
        attributes={"observed_at": observed_at},
    )


def test_the_bundle_dates_each_item_by_when_it_was_said() -> None:
    complete = EvidenceReport(status=EvidenceStatus.COMPLETE)
    shown = [
        _said("James adopted a puppy named Ned.", "2022-04-12T10:00:00+00:00"),
        _said("In June James visited Canada.", "2022-07-02T10:00:00+00:00"),
    ]
    question = "What did James adopt in April 2022?"
    assert _builder(0.4)._coverage(complete, question, shown).status is EvidenceStatus.COMPLETE
    shown[0] = _said("James adopted a puppy named Ned.", "2022-05-12T10:00:00+00:00")
    out = _builder(0.4)._coverage(complete, question, shown)
    assert out.status is EvidenceStatus.INSUFFICIENT
    assert out.notes == ["the evidence names other months than the one asked about"]


@pytest.mark.parametrize(
    "question",
    [
        "I'm opening Store 33 tomorrow morning and need to disarm the alarm.",
        "Öffne morgen die 33, Alarm?",
        "Router login, quick.",
        "Acceso al router, rápido.",
        "Maile dem Betreuer von Lakeside.",
    ],
)
def test_disarming_logging_in_and_mailing_ask_for_a_value(question: str) -> None:
    assert asked(question)


def test_the_post_and_signing_up_ask_for_nothing() -> None:
    assert not asked("Did the mail come today?")
    assert not asked("Ich muss mich bei einem neuen Zahnarzt anmelden.")


# ------------------------------------------------------------------ a code nothing names


@pytest.mark.parametrize(
    "question",
    [
        "What is the DC stock and ATA for style ZZ421722?",
        "Wie hoch sind Lagerbestand und ATA für Style ZZ421722?",
        "¿Cuál es el stock en el centro y el ATA del estilo ZZ421722?",
        "ما المخزون في مركز التوزيع وATA للموديل ZZ421722؟",  # noqa: RUF001
        "स्टाइल ZZ421722 का DC स्टॉक और ATA क्या है?",
    ],
)
def test_a_code_no_evidence_names_is_unanswered(question):
    evidence = ["For style RT335327 these stores have zero on-hand: 1641, 4307, 4610."]
    assert unanswered(question, evidence) == "no evidence names zz421722"


@pytest.mark.parametrize(
    ("question", "item"),
    [
        ("What is the reorder point for PX-90?", "The reorder point for PX90 is 120 units."),
        ("What is the reorder point for px90?", "Reorder point for PX-90: 120 units."),
        ("Who covers site SITE-9154 now?", "Hiro Abernathy now covers site SITE-9154."),
        ("What happened in 2023?", "We opened the Leeds store."),  # a year is no code
        ("Is B2 open?", "Aisle B2 is open."),  # too short to be a code
    ],
)
def test_a_code_the_evidence_names_however_written_is_answered(question, item):
    assert unanswered(question, [item]) is None


@pytest.mark.parametrize(
    ("question", "rule"),
    [
        (
            "Write the service report for WO-4120.",
            "Every service report I ask for must end with a parts-used list and a signature line.",
        ),
        (
            "Schreib den Servicebericht für WO-4120.",
            "Jeder Servicebericht, um den ich bitte, endet mit einer Liste der verbauten Teile.",
        ),
        (
            "¿Qué debo volver a pedir para CN-330?",
            "Cuando pida una recomendación de pedido, redondea a cajas completas.",
        ),
        (
            "ما هو الانكماش الأسبوع الماضي في HB-77؟",  # noqa: RUF001
            "أبلغ دائمًا عن الانكماش بالوحدات وبالقيمة.",
        ),
        (
            "KB-2291 को trd-01 पर इस हफ़्ते कब लगाएँ?",
            "ट्रेडिंग सर्वर पर 09:00 से 11:00 के बीच कभी रखरखाव मत रखो।",
        ),
    ],
)
def test_a_code_of_a_kind_the_store_holds_none_of_is_a_value_the_user_supplies(question, rule):
    # a request that names a work order, SKU or patch the rule applies to: no refusal
    assert unanswered(question, [rule]) is None


def test_a_code_of_another_kind_than_those_found_is_no_unknown_thing():
    question = "Die Pumpen in Riverside nehmen nicht den SK-12-Satz, oder? Welchen dann?"
    evidence = [
        "Die Station Riverside betreibt zwei PX-400-Pumpen.",
        "Pumpenmodell PX-400 braucht bei jedem Service den Dichtungssatz SK-19.",
    ]
    assert unanswered(question, evidence) is None


def test_a_code_found_but_not_shown_is_a_miss_not_an_unknown_thing():
    question = "Who is the dispatcher for site SITE-7786?"
    shown = ["The technician for site SITE-2860 is Zeno Borisova."]
    assert unanswered(question, shown) == "no evidence names site7786"
    found = ["Site SITE-7786 is in zone ZN-V59."]
    assert unanswered(question, shown, retrieved=found) is None
