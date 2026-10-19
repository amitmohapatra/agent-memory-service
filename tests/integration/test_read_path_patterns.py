"""ADR 0041 against the shapes production memory takes, through the real write and read paths,
in five languages: a list said once, a correction said in another chat, a format rule and a
guardrail that must reach a new chat whole, one word naming two things, and the same user's
memory never reaching another user, nor a forgotten one coming back."""

# ruff: noqa: RUF001 - the Arabic and Hindi fixtures are literal text in their own script

from __future__ import annotations

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from tests.integration.test_statements_follow_the_user import _chat, _context, _say

pytestmark = pytest.mark.integration


def _rules(bundle) -> str:
    return next((b.text for b in bundle.profile if b.block == "standing_rules"), "")


def _current(bundle) -> str:
    return "\n".join(m.text for m in bundle.memories)


# ------------------------------------------------------------------ a list said once

LIST = {
    "en": ("The pilot branches are {}.", "Which are the pilot branches?"),
    "de": ("Die Pilotfilialen sind {}.", "Welche sind die Pilotfilialen?"),
    "es": ("Las sucursales piloto son {}.", "¿Cuáles son las sucursales piloto?"),
    "ar": ("الفروع التجريبية هي {}.", "ما هي الفروع التجريبية؟"),
    "hi": ("पायलट शाखाएँ {} हैं।", "पायलट शाखाएँ कौन सी हैं?"),
}


@pytest.mark.parametrize("size", [6, 10, 20])
@pytest.mark.parametrize("lang", sorted(LIST))
async def test_a_list_comes_back_whole(container, uow_factory, lang, size):
    codes = [f"BR-{700 + n}" for n in range(size)]
    said, asked = LIST[lang]
    await _say(container, uow_factory, _chat(), said.format(", ".join(codes)))
    rendered = (await _context(container, _chat(), asked)).render()
    assert all(code in rendered for code in codes)


# ------------------------------------------------------------------ a correction in a new chat

CORRECTION = {
    "en": (
        "Order the restock from depot DP-11.",
        "Correction: do not order from depot DP-11 (it is closed); use depot DP-22 instead.",
        "Which depot should I order the restock from?",
    ),
    "de": (
        "Bestelle den Nachschub bei Depot DP-11.",
        "Korrektur: nicht bei Depot DP-11 bestellen (es ist geschlossen); nimm stattdessen Depot DP-22.",
        "Bei welchem Depot soll ich den Nachschub bestellen?",
    ),
    "es": (
        "Pide la reposición al depósito DP-11.",
        "Corrección: no pidas al depósito DP-11 (está cerrado); usa el depósito DP-22 en su lugar.",
        "¿A qué depósito pido la reposición?",
    ),
    "ar": (
        "اطلب إعادة التخزين من المستودع DP-11.",
        "تصحيح: لا تطلب من المستودع DP-11 (إنه مغلق)؛ استخدم المستودع DP-22 بدلاً منه.",
        "من أي مستودع أطلب إعادة التخزين؟",
    ),
    "hi": (
        "रीस्टॉक डिपो DP-11 से मँगाओ।",
        "सुधार: डिपो DP-11 से मत मँगाओ (वह बंद है); इसके बजाय डिपो DP-22 इस्तेमाल करो।",
        "रीस्टॉक किस डिपो से मँगाऊँ?",
    ),
}


@pytest.mark.parametrize("lang", sorted(CORRECTION))
async def test_a_correction_reaches_a_new_chat_with_its_reason(container, uow_factory, lang):
    # the write path's replacement as a deployment runs it, with its NLI head (scripted here:
    # a premise naming the later depot contradicts a hypothesis naming only the earlier one)
    from memory_service.modules.memory.replacement import CrossConversationReplacement
    from tests.integration.test_replacement_across_conversations import ScriptedNLI

    container.services["observation_pipeline"].replacement = CrossConversationReplacement(
        ScriptedNLI(0.95, values=("DP-11", "DP-22"))
    )
    first, correction, asked = CORRECTION[lang]
    await _say(container, uow_factory, _chat(), first)
    await _say(container, uow_factory, _chat(), correction)
    bundle = await _context(container, _chat(), asked)
    rendered = bundle.render()
    assert "DP-22" in rendered
    # the reason travels with the correction
    assert any("DP-22" in line and "DP-11" in line for line in rendered.splitlines())
    # the first instruction is never served as current on its own
    assert not [m.text for m in bundle.memories if "DP-11" in m.text and "DP-22" not in m.text]


# ------------------------------------------------------------------ format rule, guardrail

FORMAT = {
    "en": (
        "Always answer weekly reports as a table with columns: Item, Branch, Count, Value, "
        "Owner, Status. No prose before the table.",
        "Prepare the weekly report for branch 7.",
    ),
    "de": (
        "Beantworte Wochenberichte immer als Tabelle mit den Spalten: Item, Branch, Count, "
        "Value, Owner, Status. Kein Text vor der Tabelle.",
        "Erstelle den Wochenbericht für Filiale 7.",
    ),
    "es": (
        "Responde siempre los informes semanales como una tabla con las columnas: Item, "
        "Branch, Count, Value, Owner, Status. Sin texto antes de la tabla.",
        "Prepara el informe semanal de la sucursal 7.",
    ),
    "ar": (
        "أجب دائمًا عن التقارير الأسبوعية كجدول بالأعمدة: Item, Branch, Count, Value, Owner, "
        "Status. لا نص قبل الجدول.",
        "جهّز التقرير الأسبوعي للفرع 7.",
    ),
    "hi": (
        "साप्ताहिक रिपोर्ट का जवाब हमेशा इन कॉलम वाली तालिका में दो: Item, Branch, Count, "
        "Value, Owner, Status. तालिका से पहले कोई गद्य नहीं।",
        "शाखा 7 की साप्ताहिक रिपोर्ट तैयार करो।",
    ),
}
COLUMNS = ("Item", "Branch", "Count", "Value", "Owner", "Status")


@pytest.mark.parametrize("lang", sorted(FORMAT))
async def test_a_format_rule_comes_back_whole_in_a_new_chat(container, uow_factory, lang):
    rule, asked = FORMAT[lang]
    await _say(container, uow_factory, _chat(), rule)
    rules = _rules(await _context(container, _chat(), asked))
    assert all(column in rules for column in COLUMNS), rules


GUARD = {
    "en": (
        "Never schedule a delivery when the stock count is 0.",
        "Schedule a delivery of item LMP-140 to branch 7.",
    ),
    "de": (
        "Plane nie eine Lieferung, wenn der Bestand 0 ist.",
        "Plane eine Lieferung von Artikel LMP-140 an Filiale 7.",
    ),
    "es": (
        "Nunca programes una entrega cuando el stock sea 0.",
        "Programa una entrega del artículo LMP-140 a la sucursal 7.",
    ),
    "ar": (
        "لا تجدول أبدًا توصيلة عندما يكون المخزون 0.",
        "جدول توصيلة للصنف LMP-140 إلى الفرع 7.",
    ),
    "hi": (
        "जब स्टॉक 0 हो तो कभी भी डिलीवरी शेड्यूल मत करो।",
        "आइटम LMP-140 की डिलीवरी शाखा 7 के लिए शेड्यूल करो।",
    ),
}


@pytest.mark.parametrize("lang", sorted(GUARD))
async def test_a_guardrail_is_in_the_context_of_a_request_it_constrains(
    container, uow_factory, lang
):
    rule, asked = GUARD[lang]
    await _say(container, uow_factory, _chat(), rule)
    await _say(container, uow_factory, _chat(), "Branch 7 opens at 8 am.")
    assert rule.split()[1][:4].lower() in _rules(await _context(container, _chat(), asked)).lower()


async def test_a_lasting_preference_with_a_guardrail_reaches_a_later_new_chat(
    container, uow_factory
):
    said = "From now on, permanently prefer depot DP-22, and never finalize an order without my approval."
    await _say(container, uow_factory, _chat(), said)
    for n in range(5):
        await _say(container, uow_factory, _chat(), f"Branch {n} opens at {n + 7} am.")
    assert "never finalize" in _rules(
        await _context(container, _chat(), "Finalize the order for branch 3.")
    )


async def test_a_later_rule_supersedes_and_a_rule_keeps_its_exception(container, uow_factory):
    await _say(container, uow_factory, _chat(), "Always answer weekly reports as a list.")
    await _say(
        container,
        uow_factory,
        _chat(),
        "From now on, always answer weekly reports as a table, unless I type: prose please.",
    )
    rules = _rules(await _context(container, _chat(), "Prepare the weekly report for branch 7."))
    assert "table" in rules and "unless I type: prose please" in rules and "as a list" not in rules


# ------------------------------------------------------------------ one word, two things


async def test_one_word_for_two_things_follows_the_qualified_one(container, uow_factory):
    for said in (
        "The featured item this week is LMP-140.",
        "The clearance item is TBL-220.",
        "Branch 7 holds 40 units of LMP-140.",
        "Branch 9 holds 60 units of TBL-220.",
    ):
        await _say(container, uow_factory, _chat(), said)
    bundle = await _context(container, _chat(), "Which branch holds the featured item this week?")
    current = _current(bundle)
    assert "Branch 7 holds 40 units of LMP-140." in current


async def test_the_second_hop_and_rules_stay_within_the_readers_memory(container, uow_factory):
    await _say(container, uow_factory, _chat("u1"), "Branch 512 is in District D7.")
    await _say(container, uow_factory, _chat("u2"), "The dispatcher for District D7 is Priya Shah.")
    await _say(container, uow_factory, _chat("u2"), "Never schedule a delivery on a holiday.")
    bundle = await _context(container, _chat("u1"), "Who is the dispatcher for branch 512?")
    assert "Priya Shah" not in bundle.render()
    rules = _rules(await _context(container, _chat("u1"), "Schedule a delivery on a holiday"))
    assert "holiday" not in rules


async def test_a_thread_scoped_statement_stays_in_its_thread(container, uow_factory):
    from tests.integration.test_memory import _observe

    await _observe(
        container,
        uow_factory,
        _chat("u1"),
        "Never schedule a delivery on a holiday.",
        hints=ProcessingHints(visibility=Visibility.THREAD),
    )
    rules = _rules(await _context(container, _chat("u1"), "Schedule a delivery on a holiday"))
    assert "holiday" not in rules


async def test_a_forgotten_rule_never_comes_back(container, uow_factory):
    from tests.integration.test_memory import _memories

    ctx = _chat("u1")
    await _say(container, uow_factory, ctx, "Never schedule a delivery on a holiday.")
    for m in await _memories(uow_factory, ctx, container):
        async with uow_factory() as uow:
            await container.services["memory"].forget(uow, ctx, m.memory_id)
            await uow.commit()
    await container.tasks.drain()
    rules = _rules(await _context(container, _chat("u1"), "Schedule a delivery on a holiday"))
    assert "holiday" not in rules
