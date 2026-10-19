"""What a question about now asks for, and the values the evidence holds for it (ADR 0041).

Memory is never cleared: a value said in many conversations over many days stays, each
statement current in its own right unless a later one was tied to it at write (ADR 0038).
A question about the current state ("What's the reorder point for MK-220?", "Who drives
van #7?") then finds three statements of the reorder point, 40, 60 and 75, and served all
three as current. The question itself says what the slot is: the things it names (its
identifiers and names) and the words it asks with ("reorder point"). Of the statements that
name every one of those things and share an asked word, and that hold a value of their own
(a number, an identifier or a name the question does not carry), the newest is the answer and
every older one with another value is an earlier value of it: still in the bundle, in the
history section, never among the current evidence.

The other half of the question's anchors serve the second hop (``bridge_query``): evidence
that names what the question names but not what it asks ("Store 4779 is in Region X29" for
"Who is the buyer for store 4779?") names the thing to ask about next.

No model. Function words and question words are never asked words; a question that names
nothing, or asks with no word of its own, is left alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from memory_service.domain.language import detect_language
from memory_service.modules.memory.replacement import _function_words, anchors


def content_terms(text: str) -> set[str]:
    # the context modules import the engine, which imports this one: read them when called
    from memory_service.modules.context.evidence import content_terms as terms

    return terms(text)


def stemmed_terms(terms: set[str], lang: str) -> set[str]:
    from memory_service.modules.context.evidence import stemmed_terms as stems

    return stems(terms, lang)


def question_anchors(question: str, lang: str) -> tuple[frozenset[str], frozenset[str]]:
    from memory_service.modules.context.coverage import question_anchors as found

    return found(question, lang)


#: Question words and verbs of asking, in the pack languages: never what a question asks
#: about ("what", "tell", "welche", "dime", "ما", "क्या").
_ASKING = frozenset(
    [
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "how",
        "why",
        "tell",
        "give",
        "show",
        "list",
        "find",
        "get",
        "need",
        "please",
        "current",
        "currently",
        "now",
        "still",
        "latest",
        "today",
        "right",
        "moment",
        "these",
        "days",
        "welche",
        "welcher",
        "welches",
        "welchen",
        "wer",
        "wen",
        "wem",
        "wo",
        "wann",
        "wie",
        "warum",
        "gib",
        "sag",
        "zeig",
        "aktuell",
        "jetzt",
        "noch",
        "derzeit",
        "gerade",
        "momentan",
        "qué",
        "que",
        "cuál",
        "cuáles",
        "quién",
        "quiénes",
        "dónde",
        "cuándo",
        "cómo",
        "dame",
        "dime",
        "muestra",
        "actual",
        "ahora",
        "todavía",
        "sigue",
        "ما",
        "ماذا",
        "من",
        "أين",
        "متى",
        "كيف",
        "لماذا",
        "هل",
        "الآن",
        "حاليا",
        "حاليًا",
        "क्या",
        "कौन",
        "कहाँ",
        "कब",
        "कैसे",
        "क्यों",
        "अभी",
        "इस",
        "समय",
    ]
)
#: A question asked in the past tense is about an event, not about the current state ("What
#: did Caroline research?", "Was hatte ich bestellt?"): its older statements are no earlier
#: values of anything.
_PAST = re.compile(
    r"\b(?:did|was|were|had|used to|went|happened)\b"
    r"|\b(?:war|waren|hatte|hatten|wurde|wurden)\b"
    r"|\b(?:fue|fueron|era|eran|tuvo|hizo|estuvo)\b"
    r"|كان|كانت|كانوا"
    r"|(?<![\u0900-\u097f])(?:था|थी|थे)(?![\u0900-\u097f])",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?(?![\w])")


@dataclass(frozen=True)
class Slot:
    """What a question names (identifier values, names) and the stems it asks with."""

    ids: frozenset[str]
    names: frozenset[str]
    asked: frozenset[str]
    lang: str

    def __bool__(self) -> bool:
        return bool((self.ids or self.names) and self.asked)


@lru_cache(maxsize=1024)
def question_slot(question: str) -> Slot:
    lang = detect_language(question)
    names, ids = question_anchors(question, lang)
    words = _asked_words(question, lang, names)
    asked = frozenset(stemmed_terms(words, lang)) if words else frozenset()
    return Slot(ids=frozenset(ids), names=frozenset(names), asked=asked, lang=lang)


def _asked_words(question: str, lang: str, names: frozenset[str]) -> set[str]:
    """The question's content words but its question words, the words of what it names and
    the labels of its identifiers ("store" of "store 4779", "mk-220")."""
    unstopped = re.sub(r"(\d)([.?!؟।])(?=\s|$)", r"\1 \2", question)
    labels = {label.lstrip("#") for label, _ in anchors(unstopped, lang).ids if label}
    named = {w for n in names for w in n.split()} | labels
    return {
        w
        for w in content_terms(question)
        if w not in _ASKING
        and w not in _function_words()
        and w not in named
        and not any(ch.isdigit() for ch in w)
    }


@dataclass(frozen=True)
class Reading:
    """What one item of evidence says about a slot."""

    names_all: bool  # it names every thing the question names
    asks: bool  # it shares a word the question asks with
    values: frozenset[str]  # what it holds that the question does not name
    others: frozenset[str]  # the things it names that the question does not (ids, names)
    other_ids: frozenset[str] = frozenset()  # of those, the identifiers


def read(slot: Slot, text: str) -> Reading:
    """``text`` read against ``slot``."""
    unstopped = re.sub(r"(\d)([.?!؟।])(?=\s|$)", r"\1 \2", text)
    found = anchors(unstopped, slot.lang)
    ids = {value for _, value in found.ids}
    lowered = text.lower()
    names_all = slot.ids <= ids and all(n in lowered for n in slot.names)
    stems = stemmed_terms(content_terms(text), slot.lang)
    others = frozenset((ids - slot.ids) | {n for n in found.names if n not in slot.names})
    numbers = {
        n.replace(",", ".")
        for n in _NUMBER.findall(text)
        if not any(n in value for value in slot.ids)
    }
    return Reading(
        names_all=names_all,
        asks=bool(slot.asked & stems),
        values=frozenset(others | numbers),
        others=others,
        other_ids=frozenset(ids - slot.ids),
    )


def latest_values(question: str, candidates: Sequence[Any], *, newest_of: Any) -> list[Any]:
    """``candidates`` with every current memory that holds an older value of the slot the
    question asks about marked an earlier value of the newest one (see the module doc).
    ``newest_of(candidate)`` is the instant a candidate was said. Order is kept."""
    slot = question_slot(question)
    if not slot or _PAST.search(question):
        return list(candidates)
    holders = _holders(slot, candidates)
    if len(holders) < 2:
        return list(candidates)
    latest, latest_reading = max(holders, key=lambda h: (str(newest_of(h[0])), h[0].record_id))
    replaced = {
        c.record_id
        for c, reading in holders
        if c is not latest
        and str(newest_of(c)) < str(newest_of(latest))
        and not reading.values <= latest_reading.values
    }
    if not replaced:
        return list(candidates)
    return [
        replace(
            c,
            payload={
                **c.payload,
                "earlier_value": True,
                "replaced_by": latest.record_id,
                "replaced_in_slot": True,
            },
        )
        if c.record_id in replaced
        else c
        for c in candidates
    ]


def _holders(slot: Slot, candidates: Sequence[Any]) -> list[tuple[Any, Reading]]:
    """The current memories that hold a value of ``slot``: they name every thing it names,
    share a word it asks with and hold something the question does not name."""
    out: list[tuple[Any, Reading]] = []
    for c in candidates:
        if c.kind != "memory" or c.payload.get("earlier_value") or not c.payload.get("observed_at"):
            continue
        reading = read(slot, c.text)
        if reading.names_all and reading.asks and reading.values:
            out.append((c, reading))
    return out


def bridge_query(question: str, text: str) -> str | None:
    """The question to ask next when ``text`` names what ``question`` names but not what it
    asks about, and names something else: that thing and the asked words ("Store 4779 is in
    Region X29" -> "Region X29 buyer"). None otherwise."""
    slot = question_slot(question)
    if not slot:
        return None
    reading = read(slot, text)
    if not reading.names_all or reading.asks or not reading.others:
        return None
    asked = " ".join(sorted(_asked_words(question, slot.lang, slot.names)))
    words = " ".join(_surfaces(text, reading.others)).split()
    return " ".join(dict.fromkeys([*words, asked])).strip()


def _surfaces(text: str, things: frozenset[str]) -> list[str]:
    """How ``text`` writes each of ``things``, with the word before it ("Region X29")."""
    out: list[str] = []
    for thing in sorted(things):
        spelled = r"[-\s]?".join(map(re.escape, thing))
        m = re.search(rf"(?:[^\W\d_]+\s+)?(?<!\w){spelled}(?!\w)", text, re.IGNORECASE)
        out.append(m.group(0) if m else thing)
    return out


def bridged_things(question: str, text: str) -> frozenset[str]:
    """The things ``text`` names that ``question`` does not (the bridge's far end)."""
    slot = question_slot(question)
    if not slot:
        return frozenset()
    reading = read(slot, text)
    # an identifier says which thing ("X29"); a name beside it ("Region") only what kind
    return reading.other_ids or reading.others


def names_any(text: str, things: frozenset[str]) -> bool:
    """Whether ``text`` names one of ``things`` (an identifier however it is spelled, or a
    name)."""
    flat = re.sub(r"[-\s]", "", text.lower())
    return any(re.sub(r"[-\s]", "", thing) in flat for thing in things)
