"""The language a text is written in, decided the same way every time, with no model.

Stored on every observation, memory and chunk at write (``lang``), and read by the two places
whose behaviour depends on it: memory extraction (the rules are English, so other languages
go to the model when a key can pay for it) and the query router (its cue patterns are
English, so another language is never routed by them).

The answer is an ISO 639-1 code, or ``und`` for a text with no letters. A script that belongs
to one language decides it outright (kana is Japanese, hangul Korean, Devanagari Hindi);
Han without kana is Chinese; a shared script is split by the letters only one of its
languages uses (``ß``, ``ñ``, the Turkish dotless i, the Ukrainian ``ї``) and by counts of
the commonest function words. Latin text with no evidence either way is English when it is
plain ASCII - the English rules are harmless on a product name - and ``und`` otherwise.

Ideographic and syllabic letters are weighted as words (about three Latin letters each), so
"私はAcme Corporationで働いています" is Japanese, not English. Cost: one pass over at most
``SAMPLE_LETTERS`` letters plus one over at most ``SAMPLE_WORDS`` words.
"""

from __future__ import annotations

import re
from typing import Final

from memory_service.domain.script import Script, dominant, script_counts

UNDETERMINED: Final = "und"
ENGLISH: Final = "en"
#: Words examined for the function-word vote.
SAMPLE_WORDS: Final = 200
#: Letters of these scripts stand for a word or a syllable, not a sound.
_WORD_SCRIPTS: Final = frozenset({Script.HAN, Script.KANA, Script.HANGUL})
_WORD_WEIGHT: Final = 3
#: A script only one language here writes in.
_BY_SCRIPT: Final[dict[Script, str]] = {
    Script.KANA: "ja",
    Script.HAN: "zh",
    Script.HANGUL: "ko",
    Script.DEVANAGARI: "hi",
    Script.BENGALI: "bn",
    Script.THAI: "th",
    Script.GREEK: "el",
    Script.HEBREW: "he",
    Script.ARABIC: "ar",
    Script.CYRILLIC: "ru",
}
#: Letters one language of a shared script uses and its neighbours do not.
_MARKERS: Final[dict[Script, dict[str, str]]] = {
    Script.CYRILLIC: {"uk": "іїєґ"},
    Script.ARABIC: {"fa": "پچژگ", "ur": "ٹڈڑںے"},
    Script.LATIN: {
        "de": "ß",
        "es": "ñ¿¡",
        "pt": "ãõ",
        "fr": "œæ",
        "tr": "ığş",
        "pl": "łąęśźżń",
        "vi": "ơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ",
    },
}
#: The commonest function words of each Latin-script language, disjoint across languages
#: (a word two languages share, like "de" or "was", votes for neither).
_FUNCTION_WORDS: Final[dict[str, frozenset[str]]] = {
    lang: frozenset(words.split())
    for lang, words in {
        "en": "the and is are to of you my it that for with have this not we they be at "
        "what where who when why how which does did do i i'm your from will would there",
        "de": "der die das und ist nicht ich du sie es ein eine mit auf für den dem zu von wir "
        "mein meine habe bin im auch aber wo wer wie wann warum wohnt arbeitet sind",
        "es": "el la los las y es que en un una por para con mi yo se del al está soy tengo "
        "pero muy dónde qué quién cuándo cómo vive trabaja son",
        "fr": "le les et est je tu il elle nous vous pas pour avec dans du des mon ma suis "
        "qui sur au où quand comment pourquoi quel quelle habite travaille sont",
        "it": "il è di che per non sono mi ho della gli io ma anche dove chi quando come "
        "perché vive lavora",
        "pt": "o os é um uma não eu meu minha do da em no na mas muito sou tenho onde quem "
        "quando como mora trabalha são",
        "nl": "het een van ik niet dat op te voor zijn ook maar wij waar wie wanneer hoe "
        "woont werkt",
    }.items()
}
_DISJOINT: Final[dict[str, frozenset[str]]] = {
    lang: words.difference(*(other for name, other in _FUNCTION_WORDS.items() if name != lang))
    for lang, words in _FUNCTION_WORDS.items()
}
_WORD: Final = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def detect_language(text: str) -> str:
    """The ISO 639-1 code of ``text``'s language, or ``und`` when it has no letters."""
    counts = script_counts(text)
    if not counts:
        return UNDETERMINED
    script = _weighted_dominant(counts)
    if script in (Script.HAN, Script.KANA):
        return "ja" if counts.get(Script.KANA) else "zh"
    if script is Script.LATIN:
        return _latin(text)
    marked = _marked(text, script)
    if marked is not None:
        return marked
    return _BY_SCRIPT.get(script, UNDETERMINED)


def writing_script(text: str) -> Script:
    """The script ``text`` is written in, counting an ideograph or syllable as a word: the
    Japanese of "私はAcme Corporationで働いています", where ``detect_script`` (which counts
    letters, for the dense-space choice) says Latin."""
    return _weighted_dominant(script_counts(text))


def _weighted_dominant(counts: dict[Script, int]) -> Script:
    return dominant(
        {
            script: n * (_WORD_WEIGHT if script in _WORD_SCRIPTS else 1)
            for script, n in counts.items()
        }
    )


def is_english(lang: str) -> bool:
    """Whether the English rules are the right reader of text in ``lang``."""
    return lang == ENGLISH


def english_evidence(text: str) -> int:
    """How many English-only function words ``text`` uses: the check that a model asked to
    answer in the source language did not answer in English instead."""
    return sum(
        match.group() in _DISJOINT[ENGLISH]
        for index, match in enumerate(_WORD.finditer(text.casefold()))
        if index < SAMPLE_WORDS
    )


def _marked(text: str, script: Script) -> str | None:
    for lang, letters in _MARKERS.get(script, {}).items():
        if any(letter in text for letter in letters):
            return lang
    return None


def _latin(text: str) -> str:
    lowered = text.casefold()
    votes: dict[str, int] = {}
    for lang, letters in _MARKERS[Script.LATIN].items():
        if any(letter in lowered for letter in letters):
            votes[lang] = votes.get(lang, 0) + 2
    for index, match in enumerate(_WORD.finditer(lowered)):
        if index >= SAMPLE_WORDS:
            break
        word = match.group()
        for lang, words in _DISJOINT.items():
            if word in words:
                votes[lang] = votes.get(lang, 0) + 1
    if votes:
        # English first among equals: its rules are the only ones that run without a model
        return max(votes, key=lambda lang: (votes[lang], lang == ENGLISH))
    if any(letter in lowered for letter in "äöü"):
        return "de"
    return ENGLISH if text.isascii() else UNDETERMINED
