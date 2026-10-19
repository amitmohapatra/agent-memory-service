"""Answer coverage: does the evidence hold the value the question asks for (ADR 0040)?

``overlaps`` asks whether any evidence shares a word with the question. Since a user's
statements follow them into every conversation (ADR 0036), a question nearly always finds
statements about the same things, so that rule almost never fires on a question memory
cannot answer: "What's the alarm code for Store 33?" finds "The alarm code at Store 31 is
4471" and "Store 33 opens at 7 am"; "What is the cell number for John Doe?" finds "Our safety
manager is Rita Gomez, extension 300".

This is the coverage check - the slot the question asks about, filled for the thing it asks
about - deterministic, in the five pack languages, and only ever saying "not answered":

* **A contact detail or a secret asked for** - a phone number, an e-mail address, a code or
  password (``_ASKS``), read from the question's words in any pack language (a question may
  switch languages). A question that asks for none of these is not checked by this rule.
* **Held for the thing asked about** - an item of evidence states such a value (``_holds``:
  a phone number or an extension, an address, a code word with a code) and names what the
  question names: every identifier ("Store 33", "L-7781") and one of its names ("John Doe").
* **A month asked about** - a question that names a month, where the evidence names months
  but never that one ("April shrink" / "March shrink came in at 1.8%"), is not answered. An
  item whose text names no month is about the month it is dated in (``dated``: when it was
  said, and the days its relative dates resolved to), as the rendered bundle prints that date
  beside it: "James adopted a puppy", said in April, answers "What did James adopt in
  April?" though another item names June.

When the evidence does not hold what was asked, the bundle is INSUFFICIENT: the evidence
stays in the bundle (it is what the user said about the subject); the status says it does
not answer the question.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from functools import lru_cache

from memory_service.domain.language import detect_language
from memory_service.modules.memory.replacement import (
    _NOUN_CAPITALISING,
    _function_words,
    _marked,
    _vocab_json,
    anchors,
)


class Asked(StrEnum):
    PHONE = "phone number"
    EMAIL = "e-mail address"
    SECRET = "code or password"  # noqa: S105 - the kind of value, not a password


#: The words that ask for each kind of value, in every pack language. Latin words match as
#: whole words (a typo's common forms included); Arabic words with their clitic prefixes;
#: Devanagari words not inside another word ("हेडफ़ोन" is headphones), nor as the verb "to
#: phone/e-mail someone" ("फ़ोन करूँ", "ईमेल किए").
_LATIN_ASKS: dict[Asked, tuple[str, ...]] = {
    Asked.PHONE: (
        r"phone",
        r"cell(?:phone)?",
        r"cel",
        r"mobile",
        r"telephone",
        r"handy(?:nummer)?",
        r"telefon(?:nummer)?",
        r"mobilnummer",
        r"m[oó]vil",
        r"tel[eé]fono",
        r"celular",
    ),
    Asked.EMAIL: (r"e-?mail", r"emial", r"correo", r"e-?mail-?adresse"),
    Asked.SECRET: (
        r"code",
        r"c[oó]digo",
        r"pass\s?code",
        r"pass\s?word",
        r"pasw(?:or)?d",
        r"pin",
        r"[^\W\d_]*(?:code|passwort|kennwort)",
        r"contrase[nñ]a",
        r"clave",
    ),
}
_ARABIC_ASKS: dict[Asked, tuple[str, ...]] = {
    Asked.PHONE: ("هاتف", "جوال", "موبايل", "محمول"),
    Asked.EMAIL: ("بريد إلكتروني", "بريد الإلكتروني", "بريده", "بريدها", "إيميل", "ايميل"),
    Asked.SECRET: ("رمز", "كلمة المرور", "كلمة مرور", "كلمة السر", "كود"),
}
_DEVANAGARI_ASKS: dict[Asked, tuple[str, ...]] = {
    Asked.PHONE: ("फ़ोन", "फोन", "मोबाइल"),
    Asked.EMAIL: ("ईमेल", "ई-मेल"),
    Asked.SECRET: ("पासवर्ड", "कोड", "पिन"),
}
#: A Devanagari ask word used as a verb ("फ़ोन करूँ", "ईमेल किया", "फ़ोन आया")
_DEVANAGARI_VERB = r"(?!\s*(?:कर|किय|किए|की\b|आय|आई))"


def _asks(kind: Asked) -> re.Pattern[str]:
    latin = "|".join(_LATIN_ASKS[kind])
    arabic = "|".join(map(re.escape, _ARABIC_ASKS[kind]))
    devanagari = "|".join(map(re.escape, _DEVANAGARI_ASKS[kind]))
    return re.compile(
        rf"(?<![^\W\d_])(?:{latin})(?![^\W\d_])"
        rf"|(?<![؀-ۿ])(?:[وفبلك]?(?:ال|لل)?)(?:{arabic})"
        rf"|(?<![ऀ-ॿ])(?:{devanagari})(?![ऀ-ॿ]){_DEVANAGARI_VERB}",
        re.IGNORECASE | re.UNICODE,
    )


_ASKS: dict[Asked, re.Pattern[str]] = {kind: _asks(kind) for kind in Asked}

_PHONE_NUMBER = re.compile(r"\+?\d(?:[\s().-]*\d){6,}")
_EXTENSION = re.compile(
    r"(?<![^\W\d_])(?:ext\.?|extension|durchwahl|nebenstelle|extensi[oó]n|anexo)\s*:?\s*\d{2,}"
    r"|(?:تحويلة|داخلي|एक्सटेंशन)\s*:?\s*\d{2,}",
    re.IGNORECASE,
)
_EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
#: a code: three or more digits, or a token of letters and digits ("PO-9", "4471", "7720#")
_CODE_VALUE = re.compile(
    r"(?<![\d.,])\d{3,}(?![\d.,]*%)|(?<![\w-])(?=[\w-]*\d)(?=[\w-]*[^\W\d_])[\w-]{3,}"
)


@lru_cache(maxsize=1)
def _month_patterns() -> tuple[re.Pattern[str], ...]:
    """Each month's names in every pack language, as whole words; abbreviations that are
    also words ("mar", "may", "jun") are left out."""
    out = []
    for names in _vocab_json().get("months", []):
        full = [n.lower() for n in names if len(n) > 3 or not n.isascii()]
        full = [n for n in full if n != "may"]
        out.append(
            re.compile(
                rf"(?<![^\W\d_])(?:{'|'.join(map(re.escape, full))})(?![^\W\d_])",
                re.IGNORECASE | re.UNICODE,
            )
        )
    return tuple(out)


def months_named(text: str) -> frozenset[int]:
    """The months ``text`` names (0 = January)."""
    return frozenset(i for i, p in enumerate(_month_patterns()) if p.search(text))


#: Writing a message is not asking for an address ("Draft a reorder email to our supplier",
#: "Escribe el correo de pedido", "ईमेल ड्राफ़्ट करो"): a verb of writing next to the word.
_COMPOSE = re.compile(
    r"(?<![^\W\d_])(?:draft\w*|writ\w*|wrote|compos\w*|schreib\w*|entw[iu]rf\w*|verfass\w*|"
    r"escrib\w*|redact\w*)(?![^\W\d_])|ड्राफ़्ट|ड्राफ्ट|लिख|اكتب|صغ|حرر",
    re.IGNORECASE | re.UNICODE,
)
_COMPOSE_REACH = 24


def _asks_for(kind: Asked, text: str) -> bool:
    for match in _ASKS[kind].finditer(text):
        if kind is Asked.EMAIL and _COMPOSE.search(
            text, max(0, match.start() - _COMPOSE_REACH), match.end() + _COMPOSE_REACH
        ):
            continue
        return True
    return False


def asked(question: str) -> frozenset[Asked]:
    """The kinds of value ``question`` asks for."""
    text = unicodedata.normalize("NFKC", question)
    return frozenset(kind for kind in Asked if _asks_for(kind, text))


def _holds(kind: Asked, text: str) -> bool:
    """``text`` states a value of ``kind``."""
    if kind is Asked.PHONE:
        return bool(_PHONE_NUMBER.search(text) or _EXTENSION.search(text))
    if kind is Asked.EMAIL:
        return bool(_EMAIL_ADDRESS.search(text))
    # a code: the item says what it is ("the alarm code ... is 4471")
    return bool(_ASKS[Asked.SECRET].search(text) and _CODE_VALUE.search(text))


def _unstop(text: str) -> str:
    """A full stop after a number set apart, so the number is read as an identifier ("Store
    33." is store 33, not a decimal)."""
    return re.sub(r"(\d)([.?!؟।])(?=\s|$)", r"\1 \2", text)


def _flat(text: str) -> str:
    """Lower case, with hyphens and inverted marks as spaces ("Sunny-Crunch", "¿Cuál")."""
    return re.sub(r"[\-\u2010-\u2015¿¡]", " ", unicodedata.normalize("NFKC", text)).lower()


#: Question words that open a capitalised run ("Welchen Code", "Not Hilltop's").
_INTERROGATIVES = frozenset(
    [
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "how",
        "not",
        "no",
        "give",
        "tell",
        "send",
        "show",
        "welche",
        "welcher",
        "welches",
        "welchen",
        "wer",
        "wen",
        "wem",
        "wessen",
        "wo",
        "wann",
        "wie",
        "nicht",
        "kein",
        "gib",
        "sag",
        "nenn",
        "qué",
        "que",
        "cuál",
        "cuáles",
        "quién",
        "quiénes",
        "dónde",
        "cuándo",
        "cómo",
        "no",
        "dame",
        "dime",
    ]
)


def _asked_word(word: str) -> bool:
    return any(p.fullmatch(word) for p in _ASKS.values())


def _name(raw: str) -> str:
    """A name as the question writes it, without the question words a capitalised run may
    open with, the possessive, and the words that ask for a value ("Welchen Code" -> "")."""
    words = [re.sub(r"['\u2019]s$", "", w) for w in raw.replace("-", " ").split()]
    skip = _function_words() | _INTERROGATIVES
    return " ".join(w for w in words if w and w not in skip and not _asked_word(w))


def question_anchors(question: str, lang: str) -> tuple[frozenset[str], frozenset[str]]:
    """What the question names: its names, and the values of its labelled identifiers. A
    bare quantity ("for 500 pallets") is no anchor, nor a value the question asks about by
    its kind ("not code 77 any more")."""
    found = anchors(_unstop(re.sub(r"[¿¡]", "", question)), lang)
    ids = frozenset(value for label, value in found.ids if label and not _asked_word(label))
    names = frozenset(n for n in (_name(raw) for raw in found.names) if n)
    return names, ids


def _about(names: frozenset[str], ids: frozenset[str], lang: str, text: str) -> bool:
    """``text`` names what the question names: every identifier, and one of its names."""
    if ids and not ids <= {value for _, value in anchors(_unstop(text), lang).ids}:
        return False
    return not names or any(name in _flat(text) for name in names)


def _known(names: frozenset[str], question: str, evidence: Sequence[str]) -> frozenset[str]:
    """A run of capitalised German words is as often nouns ("Kanister Abflussreiniger") as a
    name: it is one when a word of it is marked as the question writes it ("OSHA", "QX-410";
    the names are lower-cased) or the evidence uses it."""
    marked = {
        part.lower()
        for word in re.findall(r"[\w-]+", question)
        for part in word.split("-")
        if _marked(part)
    }
    flat = [_flat(t) for t in evidence]
    return frozenset(n for n in names if marked & set(n.split()) or any(n in f for f in flat))


def _value_gap(question: str, evidence: Sequence[str], lang: str) -> str | None:
    """The kind of value asked for that no evidence about what the question names states."""
    kinds = asked(question)
    if not kinds:
        return None
    names, ids = question_anchors(question, lang)
    if lang in _NOUN_CAPITALISING:
        names = _known(names, question, evidence)
    for kind in sorted(kinds):
        if not any(_holds(kind, t) and _about(names, ids, lang, t) for t in evidence):
            return f"no evidence states the {kind.value} asked for"
    return None


def _month_gap(
    question: str, evidence: Sequence[str], dated: Sequence[frozenset[int]] = ()
) -> str | None:
    """A month asked about that the evidence, naming others, never names. An item whose
    text names no month stands for the months it is dated in (``dated``, by position)."""
    wanted = months_named(question)
    if not wanted:
        return None
    named = [months_named(text) for text in evidence]
    if not any(named):
        return None
    about = [m or (dated[i] if i < len(dated) else frozenset()) for i, m in enumerate(named)]
    if not any(m & wanted for m in about):
        return "the evidence names other months than the one asked about"
    return None


def unanswered(
    question: str, evidence: Sequence[str], dated: Sequence[frozenset[int]] = ()
) -> str | None:
    """Why ``evidence`` does not answer ``question``, or None when it may. ``dated``: the
    months each item of ``evidence`` is dated in (0 = January), by position; optional."""
    if not evidence:
        return None
    return _value_gap(question, evidence, detect_language(question)) or _month_gap(
        question, evidence, dated
    )
