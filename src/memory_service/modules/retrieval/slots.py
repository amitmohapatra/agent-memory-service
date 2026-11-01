"""What a question about now asks for, and the values the evidence holds for it (ADR 0041).

Memory is never cleared: a value said in many conversations over many days stays, each
statement current in its own right unless a later one was tied to it at write (ADR 0038).
A question about the current state ("What's the reorder point for PX-90?", "Who drives
van #7?") then finds three statements of the reorder point, 40, 60 and 75, and served all
three as current. The question itself says what the slot is: the things it names (its
identifiers and names) and the words it asks with ("reorder point"). Of the statements that
name every one of those things and share an asked word, and that hold a value of their own
(a number, an identifier or a name the question does not carry), the newest is the answer and
every older one with another value is an earlier value of it: still in the bundle, in the
history section, never among the current evidence.

The other half of the question's anchors serve the second hop (``bridge_query``): evidence
that names what the question names but not what it asks ("Branch 512 is in District D7" for
"Who is the dispatcher for branch 512?") names the thing to ask about next.

No model. Function words and question words are never asked words; a question that names
nothing, or asks with no word of its own, is left alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from memory_service.domain.language import detect_language, function_words
from memory_service.modules.memory.replacement import (
    MAX_CHARS,
    Anchors,
    _function_words,
    _marked,
    anchors,
    numbers,
)


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
        "किस",
        "किसे",
        "किसका",
        "किसकी",
        "किसके",
        "कौनसा",
        "कौनसी",
        "सा",
        "सी",
        "أي",
        "اي",
        "أيّ",
        "इस",
        "समय",
    ]
)
#: A question asked in the past tense is about an event, not about the current state ("What
#: did Maya research?", "Was hatte ich bestellt?"): its older statements are no earlier
#: values of anything.
_PAST = re.compile(
    r"\b(?:did|was|were|had|used to|went|happened)\b"
    r"|\b(?:war|waren|hatte|hatten|wurde|wurden)\b"
    r"|\b(?:fue|fueron|era|eran|tuvo|hizo|estuvo)\b"
    r"|كان|كانت|كانوا"
    r"|(?<![\u0900-\u097f])(?:था|थी|थे)(?![\u0900-\u097f])",
    re.IGNORECASE,
)
#: ... unless it asks for the latest one ("What was the last PO I raised?")
_LATEST = re.compile(
    r"\b(?:last|latest|most\s+recent|newest|current(?:ly)?|zuletzt|letzte[nrs]?|neueste[nrs]?"
    r"|aktuell\w*|[uú]ltim[oa]s?|reciente|actual)\b|آخر|الأحدث|आख़िरी|आखिरी|सबसे\s+नया|पिछल",
    re.IGNORECASE,
)
#: a question about many things: a plural copula ("Which vans are available?", "sind", "son",
#: "هم", "हैं") or a plural noun asked for ("What styles ...", "Which vans can I use?")
_MANY = re.compile(
    r"\b(?:are|were|sind|waren|son|eran|están|estaban)\b|(?<!\w)هم(?!\w)"
    r"|(?<![\u0900-\u097f])(?:हैं|थे)(?![\u0900-\u097f])"
    r"|^\s*¿?\s*(?:which|what)\s+[^\W\d_]+s\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?(?![\w])")


@dataclass(frozen=True)
class Slot:
    """What a question names (identifier values, names) and the words it asks with, each
    with its stems."""

    #: each identifier as the spellings that name it (``_id_forms``)
    ids: tuple[frozenset[str], ...]
    names: frozenset[str]
    asked: tuple[frozenset[str], ...]
    lang: str
    #: the question asks for the latest one ("What was the last PO I raised?")
    latest: bool = False

    @property
    def anchored(self) -> bool:
        return bool(self.ids or self.names)

    @property
    def needed(self) -> int:
        """How many asked words an item must share: one beside what the question names; of a
        slot that names nothing ("Who is my regional merchandise manager?"), every one - a
        word left out may be the very thing asked about ("Riverside" in a Hindi question)."""
        return 1 if self.anchored else len(self.asked)

    def __bool__(self) -> bool:
        return bool(self.asked) and (self.anchored or len(self.asked) >= 2 or self.latest)


@lru_cache(maxsize=1024)
def question_slot(question: str) -> Slot:
    lang = detect_language(question)
    names, _ = question_anchors(question, lang)
    if lang in _NOUNS_CAPITALISED:
        # a language that capitalises its nouns: a capitalised word names a thing only when
        # it is marked as a name or a code ("QX-410", "OSHA"); "Filiale" is a word asked with
        names = frozenset(n for n in names if any(_marked(w) for w in n.split()))
    ids = tuple(dict.fromkeys(_ids_of(question, lang)))
    words = sorted(_asked_words(question, lang, names))
    asked = tuple(frozenset(stemmed_terms({w}, lang)) for w in words)
    return Slot(
        ids=ids,
        names=frozenset(names),
        asked=asked,
        lang=lang,
        latest=bool(_LATEST.search(question)),
    )


def _shares(forms: frozenset[str], stems: set[str]) -> bool:
    """A word of the question is in an item: one of its forms is a stem of the item, or
    begins or ends one ("count" in "recount")."""
    return bool(forms & stems) or any(
        len(f) >= 4 and (w.endswith(f) or w.startswith(f)) for f in forms for w in stems
    )


_NOUNS_CAPITALISED = frozenset({"de"})
#: determiners and auxiliaries the packs' function words leave out: never an asked word
_MORE_FUNCTION = frozenset(
    {
        "dies",
        "diese",
        "dieser",
        "dieses",
        "diesen",
        "hat",
        "hatte",
        "haben",
        "wird",
        "werden",
        "soll",
        "sollte",
        "kann",
        "este",
        "esta",
        "estos",
        "estas",
        "ha",
        "han",
        "tiene",
        "does",
        "should",
        "could",
        "would",
        "can",
    }
)


def _asked_words(question: str, lang: str, names: frozenset[str]) -> set[str]:
    """The question's content words but its question words, the words of what it names and
    the labels of its identifiers ("store" of "branch 512", "px-90")."""
    unstopped = re.sub(r"(\d)([.?!؟।])(?=\s|$)", r"\1 \2", question)
    labels = {label.lstrip("#") for label, _ in anchors(unstopped, lang).ids if label}
    named = {w for n in names for w in n.split()} | labels
    return {
        w
        for w in content_terms(question)
        if w not in _ASKING
        and w not in _function_words()
        and w not in function_words(lang)
        and w not in _MORE_FUNCTION
        and not _LATEST.fullmatch(w)
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


def _id_forms(label: str, value: str) -> frozenset[str]:
    """The spellings of one identifier that name the same thing: its value, and its label with
    it ("van 4", "Van #4" and "van-4" are all ``van4``; "Branch 512" and
    "Filiale 512" share ``512``)."""
    label = label.lstrip("#").lower()
    if not label or value.startswith(label):
        return frozenset({value})
    return frozenset({value, f"{label}{value}"})


def _anchors(text: str, lang: str) -> Any:
    """``anchors`` of the whole of ``text``: a long message is read in windows of what one
    reading takes (``replacement.MAX_CHARS``), so a name past its first window still counts."""
    unstopped = re.sub(r"(\d)([.?!؟।])(?=\s|$)", r"\1 \2", text)
    if len(unstopped) <= MAX_CHARS:
        return anchors(unstopped, lang)
    step = MAX_CHARS // 2
    parts = [anchors(unstopped[i : i + MAX_CHARS], lang) for i in range(0, len(unstopped), step)]
    return Anchors(
        frozenset().union(*(p.names for p in parts)),
        frozenset().union(*(p.ids for p in parts)),
        frozenset().union(*(p.bare for p in parts)),
    )


def _ids_of(text: str, lang: str) -> list[frozenset[str]]:
    return [_id_forms(label, value) for label, value in _anchors(text, lang).ids]


#: an item holding at least this many values of its own (codes, numbers, names) is a list
LIST_VALUES = 3


def answers_as_list(slot: Slot, text: str) -> bool:
    """``text`` is a list that answers the question: it names everything ``slot`` names, says
    every word the question asks with (as a word or a stem, ``_shares``) and holds at least
    ``LIST_VALUES`` values ("Las sucursales piloto son BR-700, ..., BR-709"). The encoder puts
    a long list far from its question however plainly it answers it."""
    if not slot.asked:
        return False
    stems = stemmed_terms(content_terms(text), slot.lang)
    if not all(_shares(forms, stems) for forms in slot.asked):
        return False
    if not _names_in(slot, _flat(text)) or not read(slot, text).names_all:
        return False
    found = _anchors(text, slot.lang)
    held = {value for _, value in found.ids} | (found.names - slot.names)
    return max(len(held), len(numbers(text))) >= LIST_VALUES


def _flat(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def _names_in(slot: Slot, flat: str) -> bool:
    """``slot``'s names are all in ``flat`` (``_flat`` of an item's text)."""
    return all(" ".join(n.split()) in flat for n in slot.names)


def _asked_in(slot: Slot, text: str) -> bool:
    """``text`` shares the words ``slot`` asks with that it must (``Slot.needed``)."""
    stems = stemmed_terms(content_terms(text), slot.lang)
    return sum(_shares(forms, stems) for forms in slot.asked) >= slot.needed


def read(slot: Slot, text: str) -> Reading:
    """``text`` read against ``slot``."""
    found = _anchors(text, slot.lang)
    said = [_id_forms(label, value) for label, value in found.ids]
    held = frozenset().union(*said) if said else frozenset()
    names_all = all(forms & held for forms in slot.ids) and _names_in(slot, _flat(text))
    asked_ids = frozenset().union(*slot.ids) if slot.ids else frozenset()
    other_ids = frozenset(max(forms, key=len) for forms in said if not forms & asked_ids)
    others = other_ids | {n for n in found.names if n not in slot.names}
    numbers = {
        n.replace(",", ".")
        for n in _NUMBER.findall(text)
        if not any(n in value for value in asked_ids)
    }
    return Reading(
        names_all=names_all,
        asks=_asked_in(slot, text),
        values=frozenset(others | numbers),
        others=frozenset(others),
        other_ids=other_ids,
    )


def _may_hold(slot: Slot, text: str) -> bool:
    """Whether ``read(slot, text)`` can name every thing ``slot`` names and share the words it
    asks with: False only when it cannot. The two cheap parts of a reading - its names, its
    words - decide most items, so the identifiers and names of a long message (``_anchors``,
    the costly part) are read only for an item that passes both."""
    return _names_in(slot, _flat(text)) and _asked_in(slot, text)


def latest_values(question: str, candidates: Sequence[Any], *, newest_of: Any) -> list[Any]:
    """``candidates`` with every current memory that holds an older value of the slot the
    question asks about marked an earlier value of the newest one (see the module doc).
    ``newest_of(candidate)`` is the instant a candidate was said. Order is kept."""
    slot = question_slot(question)
    latest = bool(_LATEST.search(question))
    if not slot or (_PAST.search(question) and not latest):
        return list(candidates)
    if not slot.anchored and not latest and _MANY.search(question):
        # naming nothing, a question about many things ("Which vans are available?")
        # has many current values
        return list(candidates)
    holders = _holders(question, slot, candidates)
    if len(holders) < 2:
        return list(candidates)
    latest, latest_reading = max(holders, key=lambda h: (str(newest_of(h[0])), h[0].record_id))
    frame = _frame(latest.text, slot.lang, latest_reading.values)
    replaced = {
        c.record_id
        for c, reading in holders
        if c is not latest
        and str(newest_of(c)) < str(newest_of(latest))
        and reading.values != latest_reading.values
        and _same_frame(_frame(c.text, slot.lang, reading.values), frame)
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


#: two statements of one slot say it the same way: this share of the shorter one's words
#: (values aside) is in the other ("The buyer for District D7 is A" / "Update: B has replaced
#: A as the buyer for District D7"); two remarks that merely mention the thing do not
SAME_FRAME = 0.5


def _frame(text: str, lang: str, values: frozenset[str]) -> set[str]:
    """A statement's content words without its values (numbers, codes, names): what it says
    about the slot."""
    held = {part for value in values for part in value.split()}
    words = {w for w in content_terms(text) if w not in held and not any(c.isdigit() for c in w)}
    return stemmed_terms(words, lang)


def _same_frame(a: set[str], b: set[str]) -> bool:
    shorter = min(len(a), len(b))
    return bool(shorter) and len(a & b) / shorter >= SAME_FRAME


def _holders(question: str, slot: Slot, candidates: Sequence[Any]) -> list[tuple[Any, Reading]]:
    """The current memories that hold a value of ``slot``: they name every thing it names,
    share a word it asks with, hold something the question does not name, and do not qualify
    it otherwise ("last month" for a question about this month)."""
    out: list[tuple[Any, Reading]] = []
    for c in candidates:
        if c.kind != "memory" or c.payload.get("earlier_value") or not c.payload.get("observed_at"):
            continue
        if not _may_hold(slot, c.text):
            continue  # ``read`` would say it does not name the slot's things or ask its words
        reading = read(slot, c.text)
        if (
            reading.names_all
            and reading.asks
            and reading.values
            and _qualified(question, c.text) is not None
        ):
            out.append((c, reading))
    return out


#: "this month" / "last month" / "next week": a qualifier that keeps two subjects of one name
#: apart ("the featured item this week" is not last month's), in the pack languages
_UNITS = {
    "week": r"week|woche|semana|الأسبوع|हफ़्ते|हफ्ते|सप्ताह",
    "month": r"month|monat|mes|الشهر|महीने",
    "quarter": r"quarter|quartal|trimestre|الربع|तिमाही",
    "year": r"year|jahr|año|العام|السنة|साल",
}
_KINDS = {
    "this": r"this|current|diese[nmrs]?|dieses|este|esta|هذا|هذه|इस",
    "last": r"last|previous|past|letzte[nmrs]?|vorige[nmrs]?|pasad[oa]|anterior"
    r"|الماضي|الماضية|पिछले",
    "next": r"next|nächste[nmrs]?|próxim[oa]|القادم|القادمة|अगले",
}


def time_qualifiers(text: str) -> frozenset[tuple[str, str]]:
    """The (kind, unit) qualifiers ``text`` names: ("this", "month") for "this month",
    "diesen Monat", "este mes", "هذا الشهر", "इस महीने"; "el mes pasado" and "الشهر الماضي"
    are read with the unit first."""
    out: set[tuple[str, str]] = set()
    lowered = text.lower()
    for kind, kinds in _KINDS.items():
        for unit, units in _UNITS.items():
            before = rf"(?<!\w)(?:{kinds})\s+(?:{units})(?!\w)"
            after = rf"(?<!\w)(?:el\s+)?(?:{units})\s+(?:{kinds})(?!\w)"
            if re.search(f"{before}|{after}", lowered):
                out.add((kind, unit))
    return frozenset(out)


def _qualified(question: str, text: str) -> int | None:
    """How ``text``'s time qualifiers agree with the question's: None when one names the same
    unit with another kind ("last month" for "this month"), 1 when one names the question's,
    0 otherwise."""
    asked, said = time_qualifiers(question), time_qualifiers(text)
    if any(u == unit and k != kind for k, u in said for kind, unit in asked):
        return None
    return 1 if asked & said else 0


@dataclass(frozen=True)
class Bridge:
    """A memory that leads one hop further: how strongly, what to ask, and what it names. An
    object hop (``topic``) names no thing: what it reaches shares two of the object's words
    (stemmed, in ``lang``) and names someone."""

    strength: int
    query: str
    things: frozenset[str]
    topic: frozenset[str] = frozenset()
    lang: str = "en"


def bridge(question: str, text: str) -> Bridge | None:
    """The hop ``text`` leads to for ``question``, or None.

    * A question that names things: ``text`` names them but not what is asked, and names
      something else - that thing and the asked words ("Branch 512 is in District D7" ->
      "District D7 dispatcher").
    * A question that names nothing: ``text`` defines two or more of its words, not all of
      them, with an identifier - that identifier and the words it leaves ("The featured item this
      week is LMP-140" for "Which store has the most stock of the featured item this
      week?" -> "LMP-140 store most stock"). Its qualifiers keep subjects apart: "the
      clearance item is TBL-220" shares one word and leads nowhere, and of two that do the
      one sharing more is followed (``strength``)."""
    slot = question_slot(question)
    if not slot:
        return None
    reading = read(slot, text)
    words = sorted(_asked_words(question, slot.lang, slot.names))
    if slot.anchored:
        if not reading.names_all or reading.asks:
            return None
        if not reading.others:
            return _object_hop(question, text, slot.lang, words)
        things = reading.other_ids or reading.others
        rest = words
        strength = 1
    else:
        stems = stemmed_terms(content_terms(text), slot.lang)
        shared = [_shares(forms, stems) for forms in slot.asked]
        if sum(shared) < 2 or all(shared) or not reading.other_ids:
            return None
        qualified = _qualified(question, text)
        if qualified is None:
            return None
        things = reading.other_ids
        rest = [w for w, hit in zip(words, shared, strict=True) if not hit]
        strength = sum(shared) + qualified
    said = " ".join(_surfaces(text, things)).split()
    query = " ".join(dict.fromkeys([*said, *rest])).strip()
    return Bridge(strength=strength, query=query, things=frozenset(things))


#: A question asking for someone to turn to, in the five pack languages.
_ASKS_WHOM = re.compile(
    r"(?<![^\W\d_])(?:who|whom|contact|chase|wer|wen|wem|kontakt\w*|anmahnen|qui[eé]n|"
    r"contact\w*|reclam\w*)(?![^\W\d_])|(?:^|\s)من(?:\s|$)|بمن|कौन|किस(?:से|को|े)",
    re.IGNORECASE,
)
#: The least an object hop's record shares with the object's words.
_OBJECT_SHARED = 2


def _object_hop(question: str, text: str, lang: str, asked: list[str]) -> Bridge | None:
    """A record that names nothing but the question's identifier and its object ("I raised
    PO-77812 for 2,000 corrugated boxes"), asked who to turn to: the hop is the object's words
    ("corrugated boxes"), and what it reaches names someone."""
    if not _ASKS_WHOM.search(question):
        return None
    asked_stems = stemmed_terms(content_terms(question), lang)
    words = [
        w
        for w in content_terms(text)
        if not any(ch.isdigit() for ch in w) and not stemmed_terms({w}, lang) & asked_stems
    ]
    topic = frozenset(stemmed_terms(set(words), lang))
    if len(words) < _OBJECT_SHARED:
        return None
    query = " ".join(dict.fromkeys([*sorted(words), *asked]))
    return Bridge(strength=0, query=query, things=frozenset(), topic=topic, lang=lang)


def reaches(text: str, way: Bridge) -> bool:
    """Whether ``text`` is what ``way`` leads to: it names the bridged thing, or for an object
    hop shares two of the object's words and names someone."""
    if not way.topic:
        return names_any(text, way.things)
    shared = way.topic & stemmed_terms(content_terms(text), way.lang)
    return len(shared) >= _OBJECT_SHARED and bool(_anchors(text, way.lang).names)


def bridge_query(question: str, text: str) -> str | None:
    """``bridge``'s question to ask next, or None."""
    found = bridge(question, text)
    return found.query if found else None


def _surfaces(text: str, things: frozenset[str]) -> list[str]:
    """How ``text`` writes each of ``things``, with the word before it ("District D7")."""
    out: list[str] = []
    for thing in sorted(things):
        spelled = r"[-\s]?".join(map(re.escape, thing))
        m = re.search(rf"(?:[^\W\d_]+\s+)?(?<!\w){spelled}(?!\w)", text, re.IGNORECASE)
        out.append(m.group(0) if m else thing)
    return out


def bridged_things(question: str, text: str) -> frozenset[str]:
    """The things ``text`` names that ``question`` does not (the bridge's far end)."""
    found = bridge(question, text)
    return found.things if found else frozenset()


def names_any(text: str, things: frozenset[str]) -> bool:
    """Whether ``text`` names one of ``things`` (an identifier however it is spelled, or a
    name)."""
    flat = re.sub(r"[-\s]", "", text.lower())
    return any(re.sub(r"[-\s]", "", thing) in flat for thing in things)
