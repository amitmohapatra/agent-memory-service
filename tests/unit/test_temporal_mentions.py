"""A relative date is resolved against the day it was said, in the language it was said in,
and an absolute date or a bare number is left alone."""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.script import Script, detect_script
from memory_service.modules.memory.temporal import DatedMention, resolve_dated_mentions

pytestmark = pytest.mark.unit

#: a Monday
BASE = datetime(2023, 5, 8, 15, 30, tzinfo=UTC)


def _resolve(text: str) -> list[DatedMention]:
    return resolve_dated_mentions(text, base=BASE, script=detect_script(text))


def test_english_relative_expressions_resolve_against_the_base() -> None:
    found = {
        m.text.casefold(): m.date for m in _resolve("We met yesterday and again three days ago.")
    }
    assert found["yesterday"] == "2023-05-07"
    assert found["three days ago"] == "2023-05-05"


def test_last_and_next_resolve_and_absolute_dates_are_left_alone() -> None:
    found = {
        m.text.casefold(): m.date
        for m in _resolve("I moved last week; the lease started on 8 May 2023.")
    }
    assert found["last week"] == "2023-05-01..2023-05-07"  # the Monday-to-Sunday before
    assert not any("2023" in text for text in found), found
    assert _resolve("The invoice was for 98 million and 3 items.") == []


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("Wir haben uns gestern getroffen.", "gestern", "2023-05-07"),
        ("Nos vimos ayer en la oficina.", "ayer", "2023-05-07"),
        ("Nos vimos hace 2 días.", "hace 2 días", "2023-05-06"),
        ("Мы встречались вчера.", "вчера", "2023-05-07"),
        ("我们昨天见过面。", "昨天", "2023-05-07"),
        ("Ne-am întâlnit ieri.", "ieri", "2023-05-07"),
    ],
)
def test_the_languages_the_service_is_measured_in(text: str, phrase: str, expected: str) -> None:
    found = {m.text.casefold(): m.date for m in _resolve(text)}
    assert found.get(phrase.casefold()) == expected, found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # BASE is Monday 2023-05-08
        ("I went there last Friday.", {"last friday": "2023-05-05"}),
        ("The show is next Monday.", {"next monday": "2023-05-15"}),
        ("We went camping last weekend.", {"last weekend": "2023-05-06..2023-05-07"}),
        ("Are you free this weekend?", {"this weekend": "2023-05-13..2023-05-14"}),
        ("I painted it this morning.", {"this morning": "2023-05-08"}),
        ("I started last month.", {"last month": "2023-04-01..2023-04-30"}),
        ("We moved last year.", {"last year": "2022-01-01..2022-12-31"}),
        ("I signed up a few days ago.", {"a few days ago": "2023-05-02..2023-05-06"}),
        ("We met a couple of weeks ago.", {"a couple of weeks ago": "2023-04-17..2023-04-28"}),
        # past or coming is not in the words: left unresolved rather than guessed
        ("We met on Friday.", {}),
        ("Let's go next weekend.", {}),
        ("I swam a lot last summer.", {}),
    ],
)
def test_english_weekdays_weekends_and_periods_resolve_by_rule(
    text: str, expected: dict[str, str]
) -> None:
    assert {m.text.casefold(): m.date for m in _resolve(text)} == expected


def test_a_script_without_a_parser_and_an_empty_text_resolve_nothing() -> None:
    assert resolve_dated_mentions("ᚠᚢᚦ", base=BASE, script=Script.OTHER) == []
    assert resolve_dated_mentions("   ", base=BASE, script=Script.LATIN) == []
    assert resolve_dated_mentions("42", base=BASE, script=Script.NONE) == []


def test_mentions_are_bounded_and_deduplicated() -> None:
    text = (
        "We met yesterday. She called yesterday too. He moved last month. They left last year. "
        "The lease started three weeks ago. I resigned last week. We fly tomorrow."
    )
    found = resolve_dated_mentions(text, base=BASE, script=Script.LATIN, limit=4)
    assert len(found) == 4 and len({m.text.casefold() for m in found}) == 4
    assert found[0].text.casefold() == "yesterday"


def test_a_weekday_phrase_is_resolved_by_rule_and_never_guessed() -> None:
    """``last Tuesday`` is resolved by an English rule, not by dateparser's ``absolute-time``
    parser, which reads the word "we" as a Wednesday (ADR 0024, decision 7). A bare weekday
    stays unresolved: whether "on Friday" is past or coming is not in the words."""
    assert {m.text.casefold(): m.date for m in _resolve("we met last Tuesday")} == {
        "last tuesday": "2023-05-02"
    }
    assert _resolve("I saw her on Friday.") == []
    found = {m.text.casefold(): m.date for m in _resolve("We met last Tuesday, three days ago.")}
    assert found == {"last tuesday": "2023-05-02", "three days ago": "2023-05-05"}


def test_the_renderer_prints_the_resolved_date_beside_the_text() -> None:
    from memory_service.domain.context_bundle import ContextItem, _memory_line
    from memory_service.domain.enums import Representation

    text = "We met three days ago at the office."
    # the pair the resolver actually produces for this text and this base
    assert _resolve(text) == [DatedMention(text="three days ago", date="2023-05-05")]
    item = ContextItem(
        item_id="mem_1",
        representation=Representation.MEMORY,
        text=text,
        citation="memory_id:mem_1",
        attributes={
            "observed_at": "2023-05-08T15:30:00+00:00",
            "subject": "user:caroline",
            "dated_mentions": [m.as_dict() for m in _resolve(text)],
        },
    )
    assert _memory_line(item, "m1") == (
        "- [m1] 2023-05-08 Mon caroline: We met three days ago at the office. "
        "(three days ago = 2023-05-05)"
    )
    assert _memory_line(item, "m1", body="(see Most relevant)").endswith("(see Most relevant)")


@pytest.mark.parametrize(
    "text",
    [
        "SUP-20's lead time is 32 days, from my call with them today.",
        "Delivery takes 3 weeks, I heard today.",
        "We launch in 2 months, they told me today.",
        "Today I learned the trial lasts 2 weeks.",
    ],
)
def test_a_duration_beside_today_is_never_a_date(text: str) -> None:
    """A length of time ("32 days", "3 weeks", "in 2 months") is no day: read as one, "32 days"
    beside "today" became "= 2026-10-01" in a recalled note (break-and-fix re-measure)."""
    found = {m.text.casefold().strip(" ,."): m.date for m in _resolve(text)}
    assert set(found) == {"today"}, found


def test_a_count_with_its_direction_is_still_a_date() -> None:
    found = {
        m.text.casefold(): m.date
        for m in _resolve("We met 3 days ago; review in two weeks from today.")
    }
    assert found["3 days ago"] == "2023-05-05"
    assert found["in two weeks from today"] == "2023-05-22"


#: A length of time against a day, in every pack language (``lexicon/generic.json``) and in
#: French: the dates each text should resolve to. A count with no direction of its own ("32
#: Tage", "durante tres semanas", "in 2 months", "دو महीने में") is a length of time; one that
#: says its direction ("vor 3 Tagen", "hace dos semanas", "قبل 3 أيام", "3 दिन पहले", "from
#: today") is a day. BASE is Monday 2023-05-08. Where dateparser itself misses a day the text
#: names, the case says so (``LIBRARY_GAPS``): it is measured, not hidden, and an upgrade that
#: reads it fails the strict xfail until the mark is removed.
DURATION_OR_DATE: list[tuple[str, str, frozenset[str]]] = [
    (lang, text, frozenset(dates))
    for lang, text, dates in [
        ("en", "The trial lasts fourteen days.", ()),
        ("en", "We rented the van for three weeks.", ()),
        ("en", "The warranty runs 2 years and the audit took 6 hours.", ()),
        ("en", "Shipping is 10 days, the supplier said today.", ("2023-05-08",)),
        ("en", "In 3 weeks the store opens, I was told today.", ("2023-05-08",)),
        ("en", "The lease started on 8 May 2023 and lasts 12 months.", ()),
        ("en", "Two weeks ago the shipment arrived.", ("2023-04-24",)),
        ("en", "We changed prices a month ago.", ("2023-04-08",)),
        ("en", "Inventory is 10 days from today.", ("2023-05-18",)),
        ("en", "Next Monday we restock aisle 4.", ("2023-05-15",)),
        ("en", "Yesterday the supplier said lead time is 45 days.", ("2023-05-07",)),
        ("en", "Today I heard the probation is 6 months.", ("2023-05-08",)),
        ("de", "Die Lieferzeit beträgt 32 Tage, das habe ich heute erfahren.", ("2023-05-08",)),
        ("de", "Die Testphase dauert vierzehn Tage.", ()),
        ("de", "Wir mieten den Transporter für drei Wochen.", ()),
        ("de", "In drei Wochen starten wir, sagte er heute.", ("2023-05-08",)),
        ("de", "Das Projekt dauert zwei Monate.", ()),
        ("de", "Die Garantie gilt 2 Jahre.", ()),
        ("de", "Der Mietvertrag begann am 8. Mai 2023 und läuft 12 Monate.", ()),
        ("de", "Wir haben uns vor 3 Tagen getroffen.", ("2023-05-05",)),
        ("de", "Vor zwei Wochen kam die Lieferung an.", ("2023-04-24",)),
        ("de", "Vor einem Monat haben wir die Preise geändert.", ("2023-04-08",)),
        ("de", "Gestern sagte der Lieferant, die Lieferzeit sei 45 Tage.", ("2023-05-07",)),
        ("de", "Heute habe ich gehört, dass die Probezeit 6 Monate dauert.", ("2023-05-08",)),
        ("es", "El plazo de entrega es de 32 días, me lo dijeron hoy.", ("2023-05-08",)),
        ("es", "La prueba dura catorce días.", ()),
        ("es", "Alquilamos la furgoneta durante tres semanas.", ()),
        ("es", "Lanzamos dentro de 2 meses, me dijeron hoy.", ("2023-05-08",)),
        ("es", "El proyecto tarda dos meses.", ()),
        ("es", "La garantía es de 2 años.", ()),
        ("es", "El contrato empezó el 8 de mayo de 2023 y dura 12 meses.", ()),
        ("es", "Nos vimos hace 3 días.", ("2023-05-05",)),
        ("es", "Hace dos semanas llegó el envío.", ("2023-04-24",)),
        ("es", "Hace un mes cambiamos los precios.", ("2023-04-08",)),
        ("es", "Ayer el proveedor dijo que el plazo es de 45 días.", ("2023-05-07",)),
        ("es", "Hoy supe que el periodo de prueba dura 6 meses.", ("2023-05-08",)),
        ("ar", "مدة التسليم 32 يومًا، عرفت ذلك اليوم.", ("2023-05-08",)),
        ("ar", "تستمر التجربة أربعة عشر يومًا.", ()),
        ("ar", "استأجرنا الشاحنة لمدة ثلاثة أسابيع.", ()),
        ("ar", "سنطلق المنتج خلال 3 أشهر، أخبروني اليوم.", ("2023-05-08",)),
        ("ar", "يستغرق المشروع 3 أشهر.", ()),
        ("ar", "الضمان لمدة 2 سنوات.", ()),
        ("ar", "بدأ العقد في 8 مايو 2023 ومدته 12 شهرًا.", ()),
        ("ar", "التقينا قبل 3 أيام.", ("2023-05-05",)),
        ("ar", "وصلت الشحنة قبل 2 أسابيع.", ("2023-04-24",)),
        ("ar", "غيرنا الأسعار قبل شهر.", ("2023-04-08",)),
        ("ar", "أمس قال المورد إن مدة التسليم 45 يومًا.", ("2023-05-07",)),
        ("ar", "اليوم عرفت أن فترة التجربة 6 أشهر.", ("2023-05-08",)),
        ("hi", "डिलीवरी में 32 दिन लगते हैं, आज पता चला।", ("2023-05-08",)),
        ("hi", "ट्रायल चौदह दिन चलता है।", ()),
        ("hi", "हमने वैन तीन हफ़्ते के लिए किराए पर ली।", ()),
        ("hi", "हम 2 महीने में लॉन्च करेंगे, आज बताया गया।", ("2023-05-08",)),
        ("hi", "प्रोजेक्ट में 2 महीने लगेंगे।", ()),
        ("hi", "वारंटी 2 साल की है।", ()),
        ("hi", "अनुबंध 8 मई 2023 को शुरू हुआ और 12 महीने चलेगा।", ()),
        ("hi", "हम 3 दिन पहले मिले थे।", ("2023-05-05",)),
        ("hi", "2 हफ़्ते पहले शिपमेंट आया।", ("2023-04-24",)),
        ("hi", "एक महीने पहले हमने कीमतें बदलीं।", ("2023-04-08",)),
        ("hi", "आज सप्लायर ने कहा कि लीड टाइम 45 दिन है।", ("2023-05-08",)),
        ("hi", "आज मुझे पता चला कि ट्रायल 6 महीने का है।", ("2023-05-08",)),
        ("fr", "Le délai de livraison est de 32 jours, appris aujourd'hui.", ("2023-05-08",)),
        ("fr", "Nous avons loué le camion pendant trois semaines.", ()),
        ("fr", "Dans 2 mois nous lançons, m'a-t-on dit aujourd'hui.", ("2023-05-08",)),
        ("fr", "Nous nous sommes vus il y a 3 jours.", ("2023-05-05",)),
    ]
]


LIBRARY_GAPS: dict[str, str] = {
    "Inventory is 10 days from today.": "dateparser reads '10 days from today' as today",
    "Hace dos semanas llegó el envío.": "dateparser's search finds no date in 'Hace dos semanas'",
    "التقينا قبل 3 أيام.": "dateparser's Arabic locale has no 'قبل N' (ago)",
    "وصلت الشحنة قبل 2 أسابيع.": "dateparser's Arabic locale has no 'قبل N' (ago)",
    "غيرنا الأسعار قبل شهر.": "dateparser's Arabic locale has no 'قبل N' (ago)",
    "2 हफ़्ते पहले शिपमेंट आया।": "dateparser's search finds no date in '2 हफ़्ते पहले'",
    "एक महीने पहले हमने कीमतें बदलीं।": "dateparser's search finds no date in 'एक महीने पहले'",
    "Nous nous sommes vus il y a 3 jours.": "dateparser's search returns 'a 3 jours', without 'il y'",
}


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        pytest.param(
            *case,
            id=f"{case[0]}-{i}",
            marks=[pytest.mark.xfail(strict=True, reason=LIBRARY_GAPS[case[1]])]
            if case[1] in LIBRARY_GAPS
            else [],
        )
        for i, case in enumerate(DURATION_OR_DATE)
    ],
)
def test_a_length_of_time_is_told_from_a_day_in_every_language(
    lang: str, text: str, expected: frozenset[str]
) -> None:
    assert {m.date for m in _resolve(text)} == expected, [m.as_dict() for m in _resolve(text)]
