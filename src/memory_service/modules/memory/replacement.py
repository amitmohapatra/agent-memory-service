"""A user's later statement replaces an earlier one of theirs, in any of their conversations.

ADR 0036 made what a user says follow that user (``USER`` visibility) while consolidation still
compared a new statement only with the memories of its own conversation. A value corrected in a
later conversation therefore stayed current beside its correction: "Our main supplier of
shipping crates is Brightway" in March and "we ended the Brightway agreement ... crates now
come from Corrimax" in May were both served as current, in every conversation of the
user (ADR 0038). This module closes that gap at write time.

For a user's new statement (the turn and what the rules read from it) it reads the user's
current statements from any of their conversations and asks of each one, cheapest first:

1. **Is it about the same thing?** The two share a name ("Brightway", "Ana Petrova") or an
   identifier ("QX-410", "bike #4"), or the multilingual encoder puts them at a cosine of
   at least ``RELATED_COSINE``. Two statements that label the same kind of thing with
   different identifiers ("bike #3" / "bike #4", "Halle 2" / "Halle 6") and share no name
   are about two things and are never compared further.
2. **Does the new one say the old one no longer holds?** The NLI head the grounding cascade
   already loads scores the contradiction both ways (the new turn and each of its sentences
   against the old statement, and the old one against the new turn). A statement the labeller
   read as a change (``CORRECTION``, ``LIFECYCLE``, ``STATUS``: "Ana Petrova is back from
   sabbatical", "Priya handed the depot over to Tom") about a shared name or identifier needs
   less: the head reads "is back from sabbatical" against "is on sabbatical until further
   notice" as barely a contradiction. One subject's single-valued slot read from both
   statements with two values ("lives_in") relates them like a shared name. When the tenant allows
   ``conflict_adjudication``, a related pair the head leaves in between goes to the conflict
   adjudicator, as consolidation does.

What follows from a decision is graded, because a wrong one hides a true statement:

* **earlier value** (``EARLIER``): the old statement stays current and retrievable, carries
  ``replaced_by`` (the newer memory, the score, the reason), ranks below every current item,
  and is labelled as an earlier value wherever it is served;
* **superseded** (``SUPERSEDED``): above the stricter ``SUPERSEDE_CONTRADICTION``, chosen on
  the capability set's dev split for precision, the old statement is superseded by the new
  one exactly as consolidation supersedes: closed, kept for the temporal view, out of
  current retrieval.

Either way every form of the old statement goes with it: the turn as said, what the rules read
from it (a fact, an event) and its twins (``MemoryRepository.twins``). And either way it is
reversible: when the newer statement is forgotten or retracted, what it replaced is reinstated
(``reinstate``).

Nothing here runs without a trained NLI head: a stand-in (``LexicalNLI``) cannot read a
contradiction, and the pipeline then leaves cross-conversation replacement off.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    MemoryType,
    StatementKind,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.memory import (
    CanonicalMemory,
    replaced_by,
    statement_kind_of,
    unverified_representation,
)
from memory_service.domain.predicates import is_single_valued
from memory_service.domain.script import Script, detect_script, script_of
from memory_service.modules.llm.assist import LLMAssist
from memory_service.ports.models import EmbeddingProvider, NLIProvider
from memory_service.ports.uow import UnitOfWork

#: The thresholds were chosen on the capability set's dev split (ADR 0038): 468 pairs of one
#: user's statements in five languages and code-switched text, 80 of them a later statement
#: replacing an earlier one. Held out: 32 of 35 replacements found, none of the 388 other
#: pairs (dev and held-out) marked.
#:
#: Least cosine (multilingual encoder) at which two statements with no shared name or
#: identifier are compared at all: replacements sit at 0.32-0.94, and below 0.5 the head's
#: contradiction is mostly two unrelated sentences ("Depot 7 stays in the coastal zone" /
#: "the two-for-one offer runs only in the inland zone" reads as a 1.00 contradiction).
RELATED_COSINE = 0.5
#: Least contradiction (either way, best sentence) that marks the old statement as an earlier
#: value.
EARLIER_CONTRADICTION = 0.85
#: The same for a statement that says something changed (the labeller's CORRECTION,
#: LIFECYCLE or STATUS, or the lexicon suspecting one, or a number that changed) when the
#: two share a name or an identifier.
CHANGE_CONTRADICTION = 0.4
#: Least contradiction that supersedes outright: 100% precise on the dev split (22 of 22),
#: and on the held-out split (32 of 32).
SUPERSEDE_CONTRADICTION = 0.9
#: With the tenant's model (``conflict_adjudication``) a related pair the head leaves in its
#: unsure band - a contradiction of at least this much, below the marking thresholds - is put
#: to the conflict adjudicator, as consolidation does. Measured on the llm_core set with
#: Granite 3.3 2B, asking it also about pairs the head reads as no contradiction ("is back from
#: sabbatical") or that only the encoder relates ("I live in Leeds" / "I moved to Bristol")
#: marked four statements, all wrongly (a bridge between two facts - "staff of the Lyon
#: office follow the French calendar" by "Ines Moreau joined the Lyon office"): those stay
#: unmarked.
ADJUDICATE_FROM = 0.5
#: The most pairs one observation asks the model about: a bound on the write path's model
#: calls, the strongest contradictions first.
MAX_ADJUDICATIONS = 3
#: Statement kinds that say something changed.
CHANGE_KINDS = frozenset({StatementKind.CORRECTION, StatementKind.LIFECYCLE, StatementKind.STATUS})
#: How many of the user's current statements one new statement is compared with (most recent
#: first). A bound on the write path's work, not a window on what can be replaced: what a
#: user says about a thing is usually among their recent statements.
MAX_EARLIER = 64
#: How many of the messages a user said after a statement - stored before it, out of order -
#: it is compared with (ADR 0040): a bound on the write path's work.
MAX_LATER_MESSAGES = 8
#: The upper end of the range of later statements.
_END_OF_TIME = datetime.max.replace(tzinfo=UTC)
#: Longest text read for names and identifiers (a pasted corpus is not a statement).
MAX_CHARS = 2000


class Grade(StrEnum):
    EARLIER = "EARLIER"
    SUPERSEDED = "SUPERSEDED"


# --------------------------------------------------------------------------- anchors

_WORD = re.compile(r"[^\W_][\w'\u2019\-./#]*", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?;:।؟—\n]\s*$")
#: Languages that capitalise every noun: a single capitalised word there is not a name.
_NOUN_CAPITALISING = frozenset({"de"})


def _latin(word: str) -> bool:
    return any(c.isalpha() for c in word) and all(
        script_of(c) is Script.LATIN for c in word if c.isalpha()
    )


def _capitalised(word: str) -> bool:
    return word[:1].isupper() and _latin(word)


def _marked(word: str) -> bool:
    """A word that is a name or a code however it is placed: an inner capital or a digit
    ("BrightWay", "CorriMax", "CFO", "QX-410")."""
    letters = [c for c in word if c.isalpha()]
    return (any(c.isdigit() for c in word) and bool(letters)) or sum(
        c.isupper() for c in letters
    ) >= 2


@dataclass(frozen=True)
class Anchors:
    """What a statement names: names (lower-cased runs of capitalised Latin words, or any
    Latin run in another script), identifiers ((label, value) pairs whose value has a digit)
    and the labels those identifiers carry."""

    names: frozenset[str] = frozenset()
    ids: frozenset[tuple[str, str]] = frozenset()
    #: in a language that capitalises its nouns, single capitalised words used bare (no
    #: determiner, not opening a sentence, not a label or a unit by a number): a name when
    #: the other statement uses the same word bare too ("Ich lese gerade Dune" / "Ich habe
    #: Dune zu Ende gelesen"), never on its own
    bare: frozenset[str] = frozenset()

    @property
    def labels(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for label, value in self.ids:
            if label:
                out.setdefault(label, set()).add(value)
        return out


#: A number with the word that labels it ("bike #4", "Halle 6", "aisle 12", "Hubwagen
#: Nr. 3", "العربة رقم 4"): which thing is meant. A number marker between the two is
#: skipped; a percentage, an amount, a decimal, a time or a date is a quantity and is not read.
_LABELLED = re.compile(
    r"(?P<label>(?:[^\W\d_]|[\u0300-\u036f\u064b-\u065f\u0670\u0900-\u0963\u0971-\u097f])+"
    r"(?:['\u2019][^\W\d_]+)?)\s+"
    r"(?:(?:#|nr\.?|no\.?|n\.?\s?[ºo]\.?|núm\.?|num\.?|رقم|नंबर)\s*)?#?"
    r"(?<![$€£₹])(?P<num>\d+)(?![\d.,:%/]|\s?%)",
    re.UNICODE | re.IGNORECASE,
)
_DOTTED = re.compile(r"\b\d+(?:\.\d+){2,}\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
#: Words a German run of capitals opens with that are not part of a name.
_LEADING = frozenset({"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "nr", "no"})


def _short_caps(word: str) -> bool:
    """ "SKU", "UPC", "IT", "CFO": a short all-capital word is a label or a common abbreviation,
    not what tells two statements' things apart."""
    letters = [c for c in word if c.isalpha()]
    return (
        0 < len(letters) <= 4
        and all(c.isupper() for c in letters)
        and not any(c.isdigit() for c in word)
    )


@lru_cache(maxsize=8192)
def anchors(text: str, lang: str | None = None) -> Anchors:
    """The names and identifiers ``text`` carries (see ``Anchors``)."""
    text = unicodedata.normalize("NFKC", text[:MAX_CHARS])
    reading = _NameReader(
        text,
        foreign=detect_script(text) is not Script.LATIN,
        nouns=(lang or "")[:2].lower() in _NOUN_CAPITALISING,
    )
    return Anchors(
        frozenset(reading.names()),
        frozenset(_identifiers(text)),
        frozenset(_bare_words(text)) if reading.nouns and not reading.foreign else frozenset(),
    )


#: German determiners and preposition-article contractions: a capitalised word after one is a
#: noun.
_DETERMINERS = frozenset(
    [
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einen",
        "einem",
        "einer",
        "eines",
        "kein",
        "keine",
        "keinen",
        "keinem",
        "keiner",
        "mein",
        "meine",
        "meinen",
        "meinem",
        "meiner",
        "dein",
        "deine",
        "deinen",
        "deinem",
        "deiner",
        "unser",
        "unsere",
        "unseren",
        "unserem",
        "unserer",
        "euer",
        "eure",
        "euren",
        "eurem",
        "eurer",
        "ihr",
        "ihre",
        "ihren",
        "ihrem",
        "ihrer",
        "sein",
        "seine",
        "seinen",
        "seinem",
        "seiner",
        "jede",
        "jeder",
        "jeden",
        "jedem",
        "jedes",
        "diese",
        "dieser",
        "diesen",
        "diesem",
        "dieses",
        "alle",
        "allen",
        "im",
        "am",
        "zum",
        "zur",
        "beim",
        "vom",
        "ins",
        "ans",
        "aufs",
    ]
)
_BARE_WORD = re.compile(r"[^\W\d_][\w\-]*")


def _bare_words(text: str) -> set[str]:
    """Single capitalised words used bare (see ``Anchors.bare``)."""
    words = list(_BARE_WORD.finditer(text))
    return {m.group(0).lower() for i, m in enumerate(words) if i > 0 and _is_bare(text, words, i)}


def _is_bare(text: str, words: list[re.Match[str]], i: int) -> bool:
    word, m, previous = words[i].group(0), words[i], words[i - 1]
    if "-" in word or not (word[:1].isupper() and word[1:].islower()):
        return False
    if _SENTENCE_END.search(text[: m.start()]) or re.search(
        r"\d", text[previous.end() : m.start()]
    ):
        return False  # opens a sentence; a unit after a number ("um 3 Uhr")
    if previous.group(0).lower() in _DETERMINERS or not previous.group(0).islower():
        return False  # a noun after a determiner
    if _LABEL_NEXT.match(text, m.end()) or _in_run(text, words, i):
        return False  # a label before a number ("Tor 9"); part of "Region South"
    return word.lower() not in _calendar() and word.lower() not in _function_words()


def _in_run(text: str, words: list[re.Match[str]], i: int) -> bool:
    after = words[i + 1] if i + 1 < len(words) else None
    return (
        after is not None
        and after.group(0)[:1].isupper()
        and not _RUN_BREAK.search(text[words[i].end() : after.start()])
    )


class _NameReader:
    """Runs of capitalised Latin words (any Latin run in another script) read as names."""

    def __init__(self, text: str, *, foreign: bool, nouns: bool) -> None:
        self.text, self.foreign, self.nouns = text, foreign, nouns
        self.found: set[str] = set()
        self.run: list[str] = []
        self.run_start = False

    def names(self) -> set[str]:
        previous_end = 0
        for match in _WORD.finditer(self.text):
            word = match.group(0).rstrip(".'\u2019-/")
            if not word:
                continue
            gap = self.text[previous_end : match.start()]
            if self._name_like(word) and self.run and not _RUN_BREAK.search(gap):
                self.run.append(word)
            else:
                self._close(previous_end)
                if self._name_like(word):
                    self.run_start = previous_end == 0 or bool(
                        _SENTENCE_END.search(self.text[: match.start()])
                    )
                    self.run.append(word)
            previous_end = match.start() + len(word)
        self._close(previous_end)
        return self.found

    def _name_like(self, word: str) -> bool:
        return (
            _latin(word)
            and not any(c.isdigit() for c in word)
            and (self.foreign or _capitalised(word) or _marked(word))
        )

    def _close(self, after: int) -> None:
        words = self._words(self.run, after)
        self.run = []
        if not words or not any(len(w) > 1 for w in words):
            return
        single = len(words) == 1
        if self.foreign or not single or _marked(words[0]) or not (self.run_start or self.nouns):
            self.found.add(" ".join(w.lower() for w in words))

    def _words(self, run: list[str], after: int) -> list[str]:
        """The run without what is not part of a name: a label before a number ("Store 112"),
        abbreviations, month and day names, a German article."""
        if run and _LABEL_NEXT.match(self.text, after):
            run = run[:-1]
        words = [w for w in run if not _short_caps(w) and w.lower() not in _calendar()]
        while self.nouns and words and words[0].lower() in _LEADING:
            words = words[1:]
        return words


_RUN_BREAK = re.compile(r"[,;:()\"“”.!?।؟]")


def _identifiers(text: str) -> set[tuple[str, str]]:
    """(label, value) for every labelled number, ("#family", code) for every code."""
    ids: set[tuple[str, str]] = set()
    for match in _LABELLED.finditer(text):
        label = match.group("label").lower().rstrip("'\u2019")
        if label.endswith(("'s", "\u2019s")):
            label = label[:-2]
        if label in _function_words() or len(label) < 2:
            label = ""
        ids.add((label, match.group("num")))
    for match in _WORD.finditer(text):
        word = match.group(0).strip(".'\u2019-/#")
        if re.match(r"[^\W\d_]", word) and any(c.isdigit() for c in word):
            code = re.sub(r"[^\w]", "", word).lower()
            ids.add((_family(code), code))
    for dotted in _DOTTED.findall(text):
        ids.add(("#dotted", dotted))
    return ids


def _family(code: str) -> str:
    """A code's family: the letters before its first digit ("l2291" -> "#l"), so two lot
    numbers are two of one kind of thing."""
    head = re.match(r"[^\W\d_]*", code, re.UNICODE)
    return "#" + (head.group(0) if head else "")


_LABEL_NEXT = re.compile(r"[^\S\n]*(?:#|nr\.?|no\.?|n\.?\s?[ºo]\.?)?[^\S\n]*#?\d", re.IGNORECASE)


@lru_cache(maxsize=1)
def _calendar() -> frozenset[str]:
    """Month and weekday names in every pack language: dates, not names."""
    data = _vocab_json()
    out: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for w in value.lower().split():
                out.update((w, w + "s"))

    for key in ("months", "weekdays"):
        walk(data.get(key, []))
    return frozenset(out)


@lru_cache(maxsize=1)
def _function_words() -> frozenset[str]:
    """Words that never label a thing ("is 40 units", "bis 14:00", "के 3")."""
    from memory_service.modules.memory.native import _STOP

    generic = _vocab_json().get("stopwords", [])
    return frozenset({*_STOP, *generic, *_EXTRA_FUNCTION})


#: Copulas, prepositions and pronouns the packs' stopword lists leave out.
_EXTRA_FUNCTION = frozenset(
    {
        "is",
        "ist",
        "sind",
        "war",
        "es",
        "son",
        "está",
        "fue",
        "हैं",
        "है",
        "था",
        "थी",
        "up",
        "to",
        "bis",
        "hasta",
        "until",
        "unter",
        "um",
        "at",
        "about",
        "around",
        "over",
        "under",
        "than",
        "more",
        "less",
        "only",
        "just",
        "und",
        "y",
        "o",
        "or",
        "oder",
        "von",
        "from",
        "de",
        "del",
    }
)


@lru_cache(maxsize=1)
def _vocab_json() -> dict[str, Any]:
    import json
    from pathlib import Path

    from memory_service import domain

    path = Path(domain.__file__).parent / "vocabulary" / "generic.json"
    return json.loads(path.read_text(encoding="utf-8"))


def numbers(text: str) -> frozenset[str]:
    """Every number ``text`` states (an amount, a count, a percentage)."""
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(text))


@dataclass(frozen=True)
class Relation:
    """How two statements' anchors relate."""

    shared_name: bool
    shared_id: bool
    #: the same kind of thing labelled with different identifiers, and no name besides the
    #: label in common: two things ("bike #3" / "bike #4", "Halle 2" / "Halle 6")
    apart: bool
    #: the same named thing with a number that changed ("QX-410 ... 25 cases" / "40 cases"),
    #: or the same slot of the same subject with another value (``shared_slot``)
    value_changed: bool = False
    #: what the extraction read from the two statements names one subject's single-valued
    #: slot with two values ("user lives_in Leeds" / "user lives_in Bristol"), whatever names
    #: the sentences share
    shared_slot: bool = False

    @property
    def anchored(self) -> bool:
        return self.shared_name or self.shared_id or self.shared_slot


def relate(new: Anchors, old: Anchors, new_text: str = "", old_text: str = "") -> Relation:
    shared_values = {v for _, v in new.ids} & {v for _, v in old.ids}
    clashing = _clashing(new, old)
    label_words = {w for label in clashing for w in label.lstrip("#").split()}
    common = _common_names(new.names, old.names) | (new.bare & old.bare)
    rescuing = {n for n in common if not (set(n.split()) & label_words)}
    named = bool(common) and (not clashing or bool(rescuing))
    anchored = named or bool(shared_values)
    return Relation(
        shared_name=named,
        shared_id=bool(shared_values),
        apart=bool(clashing) and not shared_values and not rescuing,
        value_changed=anchored
        and bool(new_text and old_text)
        and _changed(new_text, old_text, new.ids | old.ids),
    )


def _clashing(new: Anchors, old: Anchors) -> set[str]:
    """The labels both use for different things ("forklift": 3 / 4), and "#" when each names
    its own codes and none is the other's ("KV-3301" / "LM-8812")."""
    mine, theirs = new.labels, old.labels
    out = {
        label
        for label, values in mine.items()
        if label in theirs and not (values & theirs[label]) and not label.startswith("#")
    }
    codes_new = {v for label, v in new.ids if label.startswith("#")}
    codes_old = {v for label, v in old.ids if label.startswith("#")}
    if codes_new and codes_old and not (codes_new & codes_old):
        out.add("#")
    return out


def _common_names(a: frozenset[str], b: frozenset[str]) -> set[str]:
    """Names both carry: equal, or one a run of the other's words ("owen pike" in "lager
    owen pike", where German capitalises the noun before the name)."""
    out = set(a & b)
    for x in a:
        for y in b:
            short, long_ = (x, y) if len(x) <= len(y) else (y, x)
            if short != long_ and f" {short} " in f" {long_} " and " " in short:
                out.add(short)
    return out


def _changed(new_text: str, old_text: str, ids: frozenset[tuple[str, str]] = frozenset()) -> bool:
    """Each statement states a number the other does not: a value that changed. The digits of
    a code either names (letters and digits: "G10", "DC-Foxhollow-E910", "PX-90") are no
    value: "Store 6474 belongs to Region G10" and "Region G10 is replenished from
    DC-Foxhollow-E910" share G10 and say two things about it, while "closes at 9" / "at 10"
    and "80 units" / "120 units" are changed amounts."""
    named = {
        run
        for _, value in ids
        if any(ch.isalpha() for ch in value)
        for run in re.findall(r"\d+", value)
    }
    a, b = numbers(new_text) - named, numbers(old_text) - named
    return bool(a - b) and bool(b - a)


# --------------------------------------------------------------------------- the decision


@dataclass(frozen=True)
class PairScores:
    """What the models said about one (new, old) pair."""

    cosine: float | None = None
    #: the highest contradiction: new turn -> old, each new sentence -> old, old -> new turn
    contradiction: float = 0.0


@dataclass(frozen=True)
class Decision:
    grade: Grade | None
    score: float
    reason: str


def _said_of(text: str, lang: str | None) -> set[str]:
    """What ``text`` says, without what it names: its content words but its identifiers'
    labels and values, its names, and any word with a digit, as stems in its language."""
    from memory_service.domain.language import detect_language, function_words
    from memory_service.modules.context.evidence import content_terms, stemmed_terms

    lang = lang or detect_language(text)
    found = anchors(text, lang)
    named = {w for label, value in found.ids for w in (*label.lstrip("#").split(), value)}
    named |= {w for name in found.names for w in name.split()}
    # the word right before a code or a number labels it ("المنطقة G10", "रीजन G10"), and a
    # code's own parts are its name ("DC-Foxhollow-E910")
    named |= {m.group(1).lower() for m in re.finditer(r"(\S+)\s+(?=[\w-]*\d)", text)}
    named |= {
        part.lower() for code in re.findall(r"[\w-]*\d[\w-]*", text) for part in code.split("-")
    }
    named |= _function_words() | function_words(lang)
    words = {
        w
        for w in content_terms(text)
        if w not in named and not any(ch.isdigit() for ch in w) and len(w) > 1
    }
    return stemmed_terms(words, lang) - named


def _slot_relation(
    relation: Relation,
    unit: CanonicalMemory,
    old: CanonicalMemory,
    new_slots: Slots | None,
    old_slots: dict[Any, Slots],
) -> Relation:
    """``relation``, or one subject's single-valued slot with another value when the readings
    or the words say so: the same thing, whatever else the anchors say."""
    if (
        new_slots and any(slot_changed(new_slots, old_slots.get(src, {})) for src in _sources(old))
    ) or slot_changed(spoken_slots(unit.content), spoken_slots(old.content)):
        return Relation(
            shared_name=relation.shared_name,
            shared_id=relation.shared_id,
            apart=False,
            value_changed=True,
            shared_slot=True,
        )
    return relation


def _links(new: CanonicalMemory, old: CanonicalMemory) -> bool:
    """Two links of a chain, not two values of one thing: they say nothing alike of what they
    share ("Store 6474 belongs to Region G10" / "Region G10 is replenished from
    DC-Foxhollow-E910") - unless a finished state follows an ongoing one ("I'm reading Dune"
    / "I finished Dune"), or the newer one says the subject both lead with changed."""
    if says_the_same_of_it(new.content, old.content, new.lang):
        return False
    if finished_after_ongoing(new.content, old.content):
        return False
    # one subject, and the newer statement says it changed ("Forklift #4 is in the shop" /
    # "... is back in service"); two facts about one subject are two links ("Region K41 is
    # replenished from DC-Harbor-T318" / "The buyer for Region K41 is Lena Ortiz")
    return not (says_a_change(new) and leads_with_shared(new, old) and leads_with_shared(old, new))


_CHANGE_CUES = (
    "correction",
    "correction_weak",
    "replacement",
    "lifecycle",
    "lifecycle_weak",
    "status",
    "status_weak",
)


@lru_cache(maxsize=1)
def _lexicon() -> Any:
    from memory_service.modules.memory.statements import Lexicon

    return Lexicon.default()


def says_a_change(memory: CanonicalMemory) -> bool:
    """The statement says something changed: its label (CORRECTION, LIFECYCLE, STATUS), no
    label at all, or a word the lexicon reads as one ("ended", "is back", "now")."""
    kind = statement_kind_of(memory.system_metadata)
    if kind is None or kind in CHANGE_KINDS:
        return True
    reader = _lexicon().reader(memory.content)
    return any(
        cue in reader.cues and reader.cues[cue].search(memory.content) for cue in _CHANGE_CUES
    )


def says_the_same_of_it(new_text: str, old_text: str, lang: str | None = None) -> bool:
    """Two statements about one named thing say the same of it: they share a word besides
    what they name ("The buyer for Region L60 is Thea" / "Amara has replaced Thea as the
    buyer for Region L60"), or one of them says nothing besides it."""
    # each in its own language: a stored language may be another statement's misreading
    a, b = _said_of(new_text, None), _said_of(old_text, None)
    return not a or not b or bool(a & b)


def related(relation: Relation, cosine: float | None) -> bool:
    """Worth asking the NLI head about."""
    if relation.apart:
        return False
    return relation.anchored or (cosine is not None and cosine >= RELATED_COSINE)


def decide(
    relation: Relation,
    scores: PairScores,
    kinds: Iterable[StatementKind | None] = (),
    *,
    thresholds: Thresholds | None = None,
) -> Decision:
    """The grade a new statement gives an old one, or None."""
    t = thresholds or Thresholds()
    if not related(relation, scores.cosine):
        return Decision(None, 0.0, "not about the same thing")
    c = scores.contradiction
    if c >= t.supersede:
        return Decision(Grade.SUPERSEDED, c, f"contradiction {c:.2f}")
    if c >= t.earlier:
        return Decision(Grade.EARLIER, c, f"contradiction {c:.2f}")
    change = sorted({k.value for k in kinds if k is not None and k in CHANGE_KINDS})
    if relation.value_changed:
        change.append("a changed value")
    if change and relation.anchored and c >= t.change:
        return Decision(Grade.EARLIER, c, f"{'/'.join(change)} about the same thing ({c:.2f})")
    return Decision(None, c, f"contradiction {c:.2f}")


@dataclass(frozen=True)
class Thresholds:
    earlier: float = EARLIER_CONTRADICTION
    change: float = CHANGE_CONTRADICTION
    supersede: float = SUPERSEDE_CONTRADICTION


def marker(new_id: str, decision: Decision) -> dict[str, Any]:
    """``system_metadata["replaced_by"]`` of a replaced memory."""
    return {
        "memory_id": new_id,
        "grade": decision.grade.value if decision.grade else None,
        "score": round(decision.score, 3),
        "reason": decision.reason,
    }


def same_statement(a: str, b: str) -> bool:
    """One text is the other or contains it: a turn and what a rule read from it."""
    na, nb = _norm(a), _norm(b)
    return bool(na and nb) and (na in nb or nb in na)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


# --------------------------------------------------------------------------- the write path

#: The category of a standing rule. A rule is replaced by the rule that revises it
#: (consolidation), and a preference by its own slot rules - but a statement stored as a rule
#: only because a model read it as one ("Our main supplier of shipping crates is Brightway"
#: confirmed as a RULE by the head) is a fact, and is compared like one (``misread_rule``).
_RULE_CATEGORY = "rule"


def statement_of(memory: CanonicalMemory, principal: str) -> bool:
    """A current statement a user made in their own words, readable in all their
    conversations: what this module compares and replaces."""
    return (
        memory.deleted_at is None
        and memory.temporal.status is TemporalStatus.CURRENT
        and memory.visibility is Visibility.USER
        and memory.owner_principal == principal
        and (memory.system_metadata.get("category") != _RULE_CATEGORY or misread_rule(memory))
        and (memory.memory_type is not MemoryType.PREFERENCE or is_rule(memory))
        and not memory.system_metadata.get("source_revisions")
        and not unverified_representation(memory.system_metadata)
    )


def is_rule(memory: CanonicalMemory) -> bool:
    return memory.system_metadata.get("category") == _RULE_CATEGORY


def misread_rule(memory: CanonicalMemory) -> bool:
    """Stored as a standing rule, although no sentence of it reads as one by the lexicon
    alone: a model (the head, or the tenant's) made it a rule. Such a statement is compared
    and replaced like a fact, but only about a shared name, identifier or slot - never on
    the encoder's cosine alone (``_related``)."""
    return not _lexically_rule(memory.content)


@lru_cache(maxsize=4096)
def _lexically_rule(text: str) -> bool:
    from memory_service.modules.memory.native import split_sentences
    from memory_service.modules.memory.statements import RULE_KINDS, Lexicon

    lexicon = Lexicon.default()
    for sentence in split_sentences(text) or [text]:
        label = lexicon.label(sentence)
        if label.decided and label.kind in RULE_KINDS:
            return True
    return False


#: ``(subject, predicate)`` -> the values one statement's readings give a single-valued slot
Slots = dict[tuple[str, str], frozenset[str]]


def slots_of(memories: Iterable[CanonicalMemory]) -> Slots:
    """The single-valued slots the extraction read (``user:u lives_in Bristol``): one subject
    holds one value of such a slot at a time, so another value of it later is a change."""
    out: dict[tuple[str, str], set[str]] = {}
    for m in memories:
        subject, value = _norm(m.subject or ""), _norm(m.object or "")
        if subject and value and is_single_valued(m.predicate):
            out.setdefault((subject, m.predicate or ""), set()).add(value)
    return {k: frozenset(v) for k, v in out.items()}


#: Words that open the place after a residence verb without being it (articles).
_PLACE_SKIP = (
    r"(?:(?:der|dem|den|die|das|einer|einem|einen|eine|ein|el|la|los|las|un|una|unos|unas)\s+)?"
)
#: The place: a run of capitalised Latin words ("New York"), or one word of any script.
_PLACE = r"(?P<place>(?-i:[A-Z])[\w\-]*(?:\s+(?-i:[A-Z])[\w\-]*)*|[^\s,.;:!?،؛()\"“”।]+)"
#: Where the speaker says they live, or that they moved to, in the pack languages other than
#: English, whose rules already read it (``native._LIVES_IN``, ``_MOVED_TO``): a first-person
#: verb of living or moving, then the place. Nothing else reads these sentences without a
#: model: "Ich wohne in Leeds." / "Ich bin nach Bristol gezogen." share no name and sit below
#: the encoder's 0.5. Read only to relate two statements (one slot, two places); the NLI head
#: still decides whether the later one replaces the earlier.
_RESIDENCE = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # de: "ich wohne (jetzt) in X", "ich lebe in X", "ich bin (letzten Monat) nach X gezogen"
        rf"\bich\s+(?:wohne|lebe)\s+(?:\S+\s+){{0,2}}?(?:in|im)\s+{_PLACE_SKIP}{_PLACE}",
        rf"\bich\s+bin\s+(?:\S+\s+){{0,4}}?nach\s+{_PLACE}\s+(?:\S+\s+){{0,2}}?(?:um)?gezogen",
        rf"\bich\s+(?:ziehe|zog)\s+(?:\S+\s+){{0,3}}?nach\s+{_PLACE}",
        # es: "vivo (ahora) en X", "me mudé (el mes pasado) a X", "me he mudado a X"
        rf"\b(?:vivo|resido)\s+(?:\S+\s+){{0,2}}?en\s+{_PLACE_SKIP}{_PLACE}",
        rf"\bme\s+(?:he\s+)?(?:mudé|mude|mudado|trasladé|trasladado|fui\s+a\s+vivir)\s+"
        rf"(?:\S+\s+){{0,3}}?a\s+{_PLACE_SKIP}{_PLACE}",
        # ar: "أعيش في X", "أسكن (الآن) في X", "انتقلت إلى X" (nothing Latin between: not "she")
        rf"(?<![ء-ي])و?(?:أعيش|أسكن|أقيم)\s+(?:[^\sA-Za-z]+\s+){{0,2}}?في\s+{_PLACE}",
        rf"(?<![ء-ي])و?(?:انتقلت|رحلت)\s+(?:[^\sA-Za-z]+\s+){{0,2}}?"
        rf"(?:إلى|الى|للعيش\s+في|للسكن\s+في)\s+{_PLACE}",
        # hi: "X में (...) रहता हूँ" (the first person is in "हूँ"), "मैं X शिफ्ट हो गया"
        rf"{_PLACE}\s+में[,]?\s+(?:\S+\s+){{0,4}}?(?:रहता|रहती)\s+(?:हूँ|हूं)",
        rf"(?:मैं|मै)\s+(?:\S+\s+){{0,2}}?{_PLACE}\s+(?:शिफ़्ट|शिफ्ट)\s+(?:हो\s+)?(?:गया|गई)",
    )
)
#: The speaker's own slot, read from the words (``spoken_slots``); never a stored subject.
SPEAKER = "~speaker"


@lru_cache(maxsize=4096)
def spoken_slots(text: str) -> Slots:
    """``{("~speaker", "lives_in"): {place}}`` when ``text`` says where its speaker lives or
    moved to (``_RESIDENCE``), else nothing."""
    text = unicodedata.normalize("NFKC", text[:MAX_CHARS])
    places = {
        _norm(m.group("place"))
        for pattern in _RESIDENCE
        for m in pattern.finditer(text)
        if _norm(m.group("place"))
    }
    return {(SPEAKER, "lives_in"): frozenset(places)} if places else {}


def slot_changed(new: Slots, old: Slots) -> bool:
    """A slot both name, with no value in common."""
    return any(key in old and not (values & old[key]) for key, values in new.items())


def _units(statements: Sequence[CanonicalMemory]) -> list[CanonicalMemory]:
    """What replaces: the turn as said (each piece of a long one), or what the rules read
    when the turn itself was not kept."""
    turns = [m for m in statements if m.system_metadata.get("category") == "verbatim_turn"]
    return turns or list(statements)


def _sources(memory: CanonicalMemory) -> set[tuple[str, str]]:
    return {(str(ev.source_type), ev.source_id) for ev in memory.evidence}


@dataclass
class _Pair:
    unit: CanonicalMemory
    old: CanonicalMemory
    relation: Relation
    cosine: float | None = None
    contradiction: float = 0.0


@dataclass
class Outcome:
    """What one observation replaced: old memory id -> (the newer memory, the decision)."""

    decisions: dict[str, tuple[CanonicalMemory, Decision]] = field(default_factory=dict)
    #: every memory written (marked or superseded), for the index and the revisions
    changed: list[CanonicalMemory] = field(default_factory=list)


class CrossConversationReplacement:
    """Marks or supersedes a user's earlier statements that their new one replaces, in any of
    their conversations (see the module docstring). Off without a trained NLI head."""

    def __init__(
        self,
        nli: NLIProvider | None,
        embedding: EmbeddingProvider | None = None,
        *,
        assist: LLMAssist | None = None,
        thresholds: Thresholds | None = None,
        max_earlier: int = MAX_EARLIER,
    ) -> None:
        self.nli = nli if nli is not None and getattr(nli, "representative", False) else None
        usable = embedding is not None and not embedding.fingerprint().startswith("hash-")
        self.embedding = embedding if usable else None
        self.assist = assist or LLMAssist.disabled()
        self.thresholds = thresholds or Thresholds()
        self.max_earlier = max_earlier
        self._lexicon: Any = None

    @property
    def enabled(self) -> bool:
        return self.nli is not None

    async def run(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        written: Sequence[CanonicalMemory],
        *,
        now: datetime,
        closed: Sequence[CanonicalMemory] = (),
    ) -> Outcome:
        """Compare the statements of one observation (``written``: what it created) with the
        user's earlier statements and apply what follows. Unchanged rows are not touched.
        ``closed``: what consolidation superseded for the same observation (an old value of a
        single-valued slot), read only for the slot it held."""
        new = [m for m in written if statement_of(m, ctx.principal_id)]
        if not self.enabled or not new:
            return Outcome()
        # the statement as said is what replaces: the turn (each piece of a long one), or what
        # the rules read when the turn itself was not kept
        units = _units(new)
        earlier = await self._earlier(uow, ctx, new)
        pairs = (
            await self._related(
                units, earlier, slots_of(new), closed=_closed_by(closed, new, ctx.principal_id)
            )
            if earlier
            else []
        )
        outcome = Outcome()
        if pairs:
            await self._contradiction(pairs)
            best = await self._decide(pairs, self._kinds(new, units), ctx)
            outcome = await self._apply(uow, ctx, best, earlier, now=now)
        later = await self._replaced_by_later(uow, ctx, new, now=now)
        outcome.changed += [m for m in later.changed if m.memory_id not in outcome.decisions]
        outcome.decisions.update(later.decisions)
        return outcome

    async def _replaced_by_later(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        new: Sequence[CanonicalMemory],
        *,
        now: datetime,
    ) -> Outcome:
        """What the user said after these statements, in a message stored before them,
        compared with them as it would have been had it come in order (ADR 0040). Messages
        are read in parallel - with a model, one can take a minute and the next a few seconds
        - so a later message can be stored first; it then found nothing to replace, and the
        older statement, stored after it, compared itself only with what was said before it.
        Event time decides which statement is the newer one, never the order of storing."""
        oldest = min(m.temporal.observed_at for m in new)
        mine = set().union(*(_sources(m) for m in new))
        rows = await uow.memories.statements_of(
            ctx.tenant_id,
            ctx.principal_id,
            before=_END_OF_TIME,
            after=oldest,
            limit=self.max_earlier,
        )
        by_message: dict[frozenset[tuple[str, str]], list[CanonicalMemory]] = {}
        for m in rows:
            if statement_of(m, ctx.principal_id) and not (_sources(m) & mine):
                by_message.setdefault(frozenset(_sources(m)), []).append(m)
        best: dict[str, tuple[CanonicalMemory, Decision]] = {}
        for said in list(by_message.values())[:MAX_LATER_MESSAGES]:
            units = _units(said)
            pairs = await self._related(units, new, slots_of(said))
            if not pairs:
                continue
            await self._contradiction(pairs)
            for old_id, (unit, decision) in (
                await self._decide(pairs, self._kinds(said, units), ctx)
            ).items():
                held = best.get(old_id)
                if held is None or _rank(decision) > _rank(held[1]):
                    best[old_id] = (unit, decision)
        if not best:
            return Outcome()
        return await self._apply(uow, ctx, best, new, now=now)

    async def _earlier(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, new: Sequence[CanonicalMemory]
    ) -> list[CanonicalMemory]:
        """The user's statements said before these, from other messages."""
        mine = set().union(*(_sources(m) for m in new))
        newest = max(m.temporal.observed_at for m in new)
        rows = await uow.memories.statements_of(
            ctx.tenant_id, ctx.principal_id, before=newest, limit=self.max_earlier
        )
        return [m for m in rows if statement_of(m, ctx.principal_id) and not (_sources(m) & mine)]

    async def _decide(
        self,
        pairs: Sequence[_Pair],
        kinds: Sequence[StatementKind | None],
        ctx: MemoryExecutionContext,
    ) -> dict[str, tuple[CanonicalMemory, Decision]]:
        """Old memory id -> (the newer memory, the strongest decision about it)."""
        decided: list[tuple[_Pair, Decision]] = []
        undecided: list[tuple[_Pair, Decision]] = []
        for pair in pairs:
            decision = decide(
                pair.relation,
                PairScores(pair.cosine, pair.contradiction),
                kinds,
                thresholds=self.thresholds,
            )
            (decided if decision.grade is not None else undecided).append((pair, decision))
        still: list[tuple[_Pair, Decision]] = []
        for pair, native in undecided:
            if moved_on(pair):
                reason = f"a finished state after an ongoing one ({pair.contradiction:.2f})"
                decided.append((pair, Decision(Grade.EARLIER, pair.contradiction, reason)))
            else:
                still.append((pair, native))
        undecided = still
        change = any(k in CHANGE_KINDS for k in kinds)
        if change and undecided:
            undecided = await self._stated_changes(undecided, decided)
        if self._adjudicating:
            asked = sorted(
                (x for x in undecided if worth_asking(x[0], change)),
                key=lambda x: x[0].contradiction,
                reverse=True,
            )[:MAX_ADJUDICATIONS]
            for pair, native in asked:
                decision = await self._adjudicate(pair, ctx, native)
                if decision.grade is not None:
                    decided.append((pair, decision))
        best: dict[str, tuple[CanonicalMemory, Decision]] = {}
        for pair, decision in decided:
            held = best.get(pair.old.memory_id)
            if held is None or _rank(decision) > _rank(held[1]):
                best[pair.old.memory_id] = (pair.unit, decision)
        return best

    async def _apply(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        best: dict[str, tuple[CanonicalMemory, Decision]],
        earlier: Sequence[CanonicalMemory],
        *,
        now: datetime,
    ) -> Outcome:
        outcome = Outcome()
        by_id = {m.memory_id: m for m in earlier}
        done: set[str] = set()
        for old_id, (unit, decision) in best.items():
            for form in await self._forms(uow, ctx, by_id[old_id], earlier):
                if form.memory_id in done:
                    continue
                done.add(form.memory_id)
                apply(form, unit, decision, now=now)
                await uow.memories.update(form)
                outcome.changed.append(form)
                outcome.decisions[form.memory_id] = (unit, decision)
        return outcome

    async def _related(
        self,
        units: Sequence[CanonicalMemory],
        earlier: Sequence[CanonicalMemory],
        new_slots: Slots | None = None,
        *,
        closed: Sequence[CanonicalMemory] = (),
    ) -> list[_Pair]:
        pairs: list[_Pair] = []
        pending: list[_Pair] = []
        old_slots = _slots_by_source([*earlier, *closed]) if new_slots else {}
        for unit in units:
            mine = anchors(unit.content, unit.lang or None)
            for old in earlier:
                if old.temporal.observed_at > unit.temporal.observed_at:
                    continue  # a statement replaces only what was said before it
                relation = _slot_relation(
                    relate(mine, anchors(old.content, old.lang or None), unit.content, old.content),
                    unit,
                    old,
                    new_slots,
                    old_slots,
                )
                if relation.apart:
                    continue
                if relation.anchored and not relation.shared_slot and _links(unit, old):
                    continue
                pair = _Pair(unit, old, relation)
                if relation.anchored:
                    pairs.append(pair)
                elif not is_rule(old) and not is_rule(unit):
                    pending.append(pair)  # a misread rule is compared only when anchored
        return pairs + await self._close_enough(pending)

    async def _stated_changes(
        self,
        undecided: list[tuple[_Pair, Decision]],
        decided: list[tuple[_Pair, Decision]],
    ) -> list[tuple[_Pair, Decision]]:
        """A stated change about the thing the older statement is about (``stated_change``)
        that the encoder also puts at ``RELATED_COSINE`` or closer marks an earlier value,
        whatever the head read: in Arabic and Hindi the head reads "Maria Lopez is back from
        leave" against "Maria Lopez is on leave until further notice" at 0.02. Never a
        supersession. Returns what is still undecided."""
        candidates = [pair for pair, _ in undecided if stated_change(pair)]
        if not candidates or self.embedding is None:
            return undecided
        await self._cosines(candidates)
        asked = {id(pair) for pair in candidates}
        still: list[tuple[_Pair, Decision]] = []
        for pair, native in undecided:
            cos = pair.cosine
            if id(pair) in asked and cos is not None and cos >= RELATED_COSINE:
                reason = f"a stated change about the same thing (cosine {cos:.2f})"
                decided.append((pair, Decision(Grade.EARLIER, pair.contradiction, reason)))
            else:
                still.append((pair, native))
        return still

    async def _cosines(self, pairs: Sequence[_Pair]) -> None:
        """The encoder's cosine of each pair, one batch."""
        assert self.embedding is not None
        texts = sorted({p.unit.content for p in pairs} | {p.old.content for p in pairs})
        vectors = dict(zip(texts, await self.embedding.embed_documents(texts), strict=True))
        for pair in pairs:
            pair.cosine = _cosine(vectors[pair.unit.content], vectors[pair.old.content])

    async def _close_enough(self, pending: Sequence[_Pair]) -> list[_Pair]:
        """The unanchored pairs the encoder puts close enough to compare."""
        if not pending or self.embedding is None:
            return []
        await self._cosines(pending)
        return [p for p in pending if p.cosine is not None and p.cosine >= RELATED_COSINE]

    @property
    def _adjudicating(self) -> bool:
        return self.assist.wants("conflict_adjudication")

    async def _contradiction(self, pairs: Sequence[_Pair]) -> None:
        """One NLI batch: each new turn and each of its sentences against the old statement,
        and the old statement against the new turn."""
        from memory_service.modules.memory.native import split_sentences

        assert self.nli is not None
        groups: list[tuple[Sequence[str], str]] = []
        for pair in pairs:
            parts = split_sentences(pair.unit.content)
            premises = [pair.unit.content, *(parts if len(parts) > 1 else [])]
            groups.append((premises, pair.old.content))
            groups.append(([pair.old.content], pair.unit.content))
        scores = await self.nli.entail_groups(groups)
        for n, pair in enumerate(pairs):
            forward, backward = scores[2 * n], scores[2 * n + 1]
            pair.contradiction = max(s.contradiction for s in (*forward, *backward))

    def _kinds(
        self, new: Sequence[CanonicalMemory], units: Sequence[CanonicalMemory]
    ) -> list[StatementKind | None]:
        """What the labeller stored on the new statement, and what the lexicon suspects of
        each of its sentences (a weak cue - "finished", "moved to", "repaired" - is left to
        the NLI head by the labeller and usually stored as a fact)."""
        from memory_service.modules.memory.native import split_sentences
        from memory_service.modules.memory.statements import Lexicon

        kinds: list[StatementKind | None] = [statement_kind_of(m.system_metadata) for m in new]
        if self._lexicon is None:
            self._lexicon = Lexicon.default()
        for unit in units:
            for sentence in split_sentences(unit.content) or [unit.content]:
                label = self._lexicon.label(sentence)
                kinds += [label.kind, label.maybe]
        return kinds

    async def _adjudicate(
        self, pair: _Pair, ctx: MemoryExecutionContext, native: Decision
    ) -> Decision:
        """The tenant's model on a related pair the head left in between, as consolidation
        asks it (``conflict_adjudication``): 'update' or 'contradict' marks an earlier value."""
        if not self._adjudicating:
            return native
        from memory_service.modules.memory.native import (
            _ADJUDICATION_SCHEMA,
            _ADJUDICATION_SYSTEM,
            _ASSIST_MAX_CHARS,
        )

        out = await self.assist.structured(
            "conflict_adjudication",
            system=_ADJUDICATION_SYSTEM,
            user=(
                f"NEW (by {ctx.principal_id}, observed "
                f"{pair.unit.temporal.observed_at.isoformat()}):\n"
                f"{pair.unit.content[:_ASSIST_MAX_CHARS]}\n\n"
                f"EXISTING (by {pair.old.owner_principal}, observed "
                f"{pair.old.temporal.observed_at.isoformat()}):\n"
                f"{pair.old.content[:_ASSIST_MAX_CHARS]}"
            ),
            schema=_ADJUDICATION_SCHEMA,
            max_tokens=50,
        )
        verdict = out.get("verdict") if out else None
        if verdict in ("update", "contradict"):
            return Decision(Grade.EARLIER, native.score, f"model: {verdict} ({native.reason})")
        return native

    @staticmethod
    async def _forms(
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        old: CanonicalMemory,
        earlier: Sequence[CanonicalMemory],
    ) -> list[CanonicalMemory]:
        """Every form of one statement: itself, its twins, and - for a turn - what the rules
        read from it (a fact, an event of the same message, inside the turn's words). A
        reading of one sentence never takes its whole turn with it: the turn may say more."""
        out = [old, *await uow.memories.twins(ctx.tenant_id, old)]
        if old.system_metadata.get("category") == "verbatim_turn":
            sources = _sources(old)
            out += [
                m
                for m in earlier
                if m.memory_id != old.memory_id
                and _sources(m) & sources
                and same_statement(m.content, old.content)
            ]
        forms: dict[str, CanonicalMemory] = {}
        for m in out:
            # only the same person's words: a twin is the owner's by definition, and a turn of
            # the same message is too, but nothing else is ever touched
            if (
                m.temporal.status is TemporalStatus.CURRENT
                and m.owner_principal == old.owner_principal
            ):
                forms.setdefault(m.memory_id, m)
        return list(forms.values())


def _closed_by(
    closed: Sequence[CanonicalMemory], new: Sequence[CanonicalMemory], principal: str
) -> list[CanonicalMemory]:
    """The user's own memories a new one of this observation superseded."""
    ids = {m.memory_id for m in new}
    return [
        m
        for m in closed
        if m.owner_principal == principal
        and m.temporal.status is TemporalStatus.SUPERSEDED
        and m.temporal.superseded_by in ids
    ]


def _slots_by_source(memories: Sequence[CanonicalMemory]) -> dict[tuple[str, str], Slots]:
    """Each message's slots, from what the rules read from it."""
    by_source: dict[tuple[str, str], list[CanonicalMemory]] = {}
    for m in memories:
        for src in _sources(m):
            by_source.setdefault(src, []).append(m)
    return {src: slots_of(ms) for src, ms in by_source.items()}


def worth_asking(pair: _Pair, change: bool = False) -> bool:
    """What the tenant's model is asked: a related pair the head left in its unsure band, or a
    stated change (``change``: a CORRECTION, LIFECYCLE or STATUS cue) whose first named thing
    is one the earlier statement names too ("Maria Lopez is back from leave" / "Maria Lopez is
    on leave"), whatever the head read: in Arabic and Hindi it reads such pairs at 0.02-0.2."""
    if not related(pair.relation, pair.cosine):
        return False
    if pair.contradiction >= ADJUDICATE_FROM:
        return True
    return change and stated_change(pair)


def stated_change(pair: _Pair) -> bool:
    """The pair is tied by a name or an identifier and the newer statement leads with one the
    older names (``leads_with_shared``): with a change cue, a change of that thing."""
    return (
        pair.relation.anchored
        and not pair.relation.apart
        and leads_with_shared(pair.unit, pair.old)
    )


#: A state that is going on: what the older statement of a finished-after-ongoing pair says
#: ("I'm reading Dune at the moment", "on leave until further notice", "investigating").
_ONGOING = re.compile(
    r"(?<![^\W\d_])(?:currently|at the moment|right now|these days|for now|so far|"
    r"in progress|until further notice|still|gerade|zurzeit|derzeit|momentan|im moment|"
    r"bis auf weiteres|noch|ahora mismo|actualmente|en este momento|por ahora|todav[ií]a|"
    r"hasta nuevo aviso|est(?:oy|á|amos|án) \w+(?:ando|iendo))(?![^\W\d_])"
    r"|(?:I'm|I am|we're|we are|is|are) \w+ing\b"
    r"|حاليًا|حاليا|في الوقت الحالي|حتى إشعار آخر|لا يزال|لا تزال|ما زال|ما زالت"
    r"|इन दिनों|फ़िलहाल|फिलहाल|अभी|अगली सूचना तक|रहा हूँ|रही हूँ|रहे हैं|रहा है|रही है",
    re.IGNORECASE | re.UNICODE,
)
#: A state that has ended or been reached: what the newer statement says ("I finished Dune",
#: "is back from leave", "resolved", "zu Ende gelesen", "ha vuelto", "أنهيت", "ख़त्म की").
_FINISHED = re.compile(
    r"(?<![^\W\d_])(?:finish(?:ed|es)|completed?|done with|is back|are back|came back|"
    r"returned|resolved|repaired|closed|ended|over now|zu ende|fertig|abgeschlossen|beendet|"
    r"zur[üu]ck|termin(?:é|ó|amos|aron|ado|ada)|acab(?:é|ó|ado|ada)|finaliz\w+|"
    r"(?:ha|han|he) vuelto|volvi(?:ó|eron)|regres(?:ó|aron|ado)|cerrad[oa])(?![^\W\d_])"
    r"|أنهيت|أنهى|أنهت|انتهى|انتهت|عاد|عادت|رجع|رجعت|أُغلق|أغلق"
    r"|ख़त्म|खत्म|पूरा हो|पूरी हो|पूरी कर|पूरा कर|लौट|वापस आ",
    re.IGNORECASE | re.UNICODE,
)


def finished_after_ongoing(new_text: str, old_text: str) -> bool:
    """The newer statement says a state has ended or been reached, the older one that it was
    going on: with a name or identifier both use, the newer says the thing moved on."""
    return bool(_FINISHED.search(new_text) and _ONGOING.search(old_text))


def moved_on(pair: _Pair) -> bool:
    """A finished state about the thing an ongoing one was about (``finished_after_ongoing``),
    the two tied by a name or an identifier (not by a slot alone, nor by the encoder)."""
    rel = pair.relation
    return (
        (rel.shared_name or rel.shared_id)
        and not rel.apart
        and finished_after_ongoing(pair.unit.content, pair.old.content)
    )


def leads_with_shared(new: CanonicalMemory, old: CanonicalMemory) -> bool:
    """The first name or identifier ``new`` names is one ``old`` names: the thing whose state
    changed is the one both are about, not another thing a shared place or team ties them to
    ("Lena Vogel joined the Berlin office" / "staff of the Berlin office follow the German
    calendar" lead with Lena Vogel, whom the older statement does not name)."""
    mine = anchors(new.content, new.lang or None)
    theirs = anchors(old.content, old.lang or None)
    shared_names = _common_names(mine.names, theirs.names) | (mine.bare & theirs.bare)
    shared_values = {v for _, v in mine.ids} & {v for _, v in theirs.ids}
    first = _first_anchor(new.content, Anchors(mine.names | mine.bare, mine.ids))
    return first is not None and (first in shared_names or first in shared_values)


def _first_anchor(text: str, found: Anchors) -> str | None:
    """The name (lower-cased) or identifier value ``text`` names first."""
    words = [
        re.sub(r"[^\w]", "", m.group(0)).casefold()
        for m in _WORD.finditer(unicodedata.normalize("NFKC", text[:MAX_CHARS]))
    ]
    values = {v for _, v in found.ids}
    starts = {name.split()[0]: name for name in sorted(found.names, key=len, reverse=True)}
    for n, word in enumerate(words):
        if not word:
            continue
        if word in starts:
            name = starts[word]
            if " ".join(words[n : n + len(name.split())]) == re.sub(r"[^\w ]", "", name):
                return name
        if word in values:
            return word
        digits = re.sub(r"\D", "", word)
        if digits and digits in values and not word.isdigit():
            return digits
    return None


def _rank(decision: Decision) -> tuple[int, float]:
    return (2 if decision.grade is Grade.SUPERSEDED else 1, decision.score)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def apply(
    form: CanonicalMemory, unit: CanonicalMemory, decision: Decision, *, now: datetime
) -> None:
    """Mark ``form`` as replaced by ``unit``: an earlier value (still current) or superseded
    (closed at ``unit``'s time, kept for the temporal view). The newer memory is not touched,
    so one statement may replace several."""
    form.system_metadata = {**form.system_metadata, "replaced_by": marker(unit.memory_id, decision)}
    if decision.grade is Grade.SUPERSEDED:
        form.temporal = form.temporal.model_copy(
            update={
                "status": TemporalStatus.SUPERSEDED,
                "superseded_by": unit.memory_id,
                "valid_to": form.temporal.valid_to or unit.temporal.valid_from or now,
            }
        )
        form.system_metadata["supersede_reason"] = "replaced in a later statement"
    form.updated_at = now


async def reinstate(
    uow: UnitOfWork, tenant_id: str, gone: Iterable[str], *, now: datetime
) -> list[CanonicalMemory]:
    """Undo what the statements ``gone`` (forgotten or retracted) replaced: an earlier value
    is current again, a superseded statement is reopened. The memories written."""
    ids = sorted(set(gone))
    if not ids:
        return []
    out: list[CanonicalMemory] = []
    for memory in await uow.memories.replaced_by_any(tenant_id, ids):
        if memory.deleted_at is not None or replaced_by(memory.system_metadata) not in ids:
            continue
        metadata = dict(memory.system_metadata)
        mark = metadata.pop("replaced_by", None) or {}
        if (
            mark.get("grade") == Grade.SUPERSEDED.value
            and memory.temporal.status is TemporalStatus.SUPERSEDED
            and memory.temporal.superseded_by in ids
        ):
            metadata.pop("supersede_reason", None)
            memory.temporal = memory.temporal.model_copy(
                update={"status": TemporalStatus.CURRENT, "superseded_by": None, "valid_to": None}
            )
        memory.system_metadata = metadata
        memory.updated_at = now
        await uow.memories.update(memory)
        out.append(memory)
    return out
