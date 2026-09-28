"""The script of a text is decided by its letters' Unicode names, the same way every time."""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import pytest

from memory_service.domain.script import Script, detect_script, script_of

pytestmark = pytest.mark.unit

#: the twelve XQuAD languages, one sentence each
BY_LANGUAGE = {
    "en": ("My office is in Berlin.", Script.LATIN),
    "de": ("Mein Büro befindet sich in Berlin.", Script.LATIN),
    "es": ("Mi oficina está en Madrid.", Script.LATIN),
    "ro": ("Biroul meu este în București.", Script.LATIN),
    "tr": ("Benim ofisim İstanbul şehrinde bulunuyor.", Script.LATIN),
    "vi": ("Văn phòng của tôi ở Hà Nội.", Script.LATIN),
    "el": ("Το γραφείο μου βρίσκεται στην Αθήνα.", Script.GREEK),
    "ru": ("Мой офис находится в Москве.", Script.CYRILLIC),
    "hi": ("मेरा कार्यालय दिल्ली में है।", Script.DEVANAGARI),
    "ar": ("يقع مكتبي في القاهرة.", Script.ARABIC),
    "zh": ("我的办公室在北京。", Script.HAN),
    "th": ("สำนักงานของฉันอยู่ในกรุงเทพ", Script.THAI),
}


@pytest.mark.parametrize(("language", "case"), BY_LANGUAGE.items(), ids=list(BY_LANGUAGE))
def test_the_twelve_languages(language: str, case: tuple[str, Script]) -> None:
    text, expected = case
    assert detect_script(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("こんにちは、東京です。", Script.KANA),  # kana outnumber the two kanji
        ("東京都渋谷区の事務所", Script.HAN),  # kanji outnumber the kana
        ("서울에 있는 사무실", Script.HANGUL),
        ("המשרד שלי בתל אביב", Script.HEBREW),
        ("আমার অফিস ঢাকায়", Script.BENGALI),
        ("2024-05-08 10:00", Script.NONE),
        ("", Script.NONE),
        ("!!! ??? ...", Script.NONE),
        ("ＡＢＣ", Script.LATIN),  # width variants keep their script
    ],
)
def test_other_scripts_and_the_edge_cases(text: str, expected: Script) -> None:
    assert detect_script(text) is expected


def test_mixed_text_takes_the_dominant_script_by_letter_count() -> None:
    assert detect_script("Встреча с Acme в Москве") is Script.CYRILLIC
    assert detect_script("Meeting with Компания in Berlin next week") is Script.LATIN


def test_a_tie_goes_to_the_earlier_member() -> None:
    assert detect_script("ab" + "яб") is Script.LATIN
    assert detect_script("яб" + "ab") is Script.LATIN


def test_the_sample_bounds_the_work_and_is_deterministic() -> None:
    text = "a" * 600 + "я" * 600
    assert detect_script(text) is Script.LATIN
    assert detect_script(text, sample=2000) is Script.LATIN  # 600 each: the tie rule
    assert detect_script(text) == detect_script(text)


def test_a_letter_of_an_unnamed_script_is_other() -> None:
    assert script_of("ᚠ") is Script.OTHER  # runic
    assert detect_script("ᚠᚢᚦ") is Script.OTHER
