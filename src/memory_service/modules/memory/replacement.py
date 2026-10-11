"""A user's later statement replaces an earlier one of theirs, in any of their conversations.

ADR 0036 made what a user says follow that user (``USER`` visibility) while consolidation still
compared a new statement only with the memories of its own conversation. A value corrected in a
later conversation therefore stayed current beside its correction: "Our primary supplier for
heavy-duty pallets is Uline" in March and "we terminated our contract with Uline ... our new
supplier is PackagingCorp" in May were both served as current, in every conversation of the
user (ADR 0038). This module closes that gap at write time.

For a user's new statement (the turn and what the rules read from it) it reads the user's
current statements from any of their conversations and asks of each one, cheapest first:

1. **Is it about the same thing?** The two share a name ("Uline", "Maria Lopez") or an
   identifier ("MK-220", "Forklift #4"), or the multilingual encoder puts them at a cosine of
   at least ``RELATED_COSINE``. Two statements that label the same kind of thing with
   different identifiers ("Forklift #3" / "Forklift #4", "Tor 9" / "Tor 14") and share no name
   are about two things and are never compared further.
2. **Does the new one say the old one no longer holds?** The NLI head the grounding cascade
   already loads scores the contradiction both ways (the new turn and each of its sentences
   against the old statement, and the old one against the new turn). A statement the labeller
   read as a change (``CORRECTION``, ``LIFECYCLE``, ``STATUS``: "Maria Lopez is back from
   parental leave", "I finished Dune") about a shared name or identifier needs less: the
   head reads "is back from leave" against "is on leave until further notice" as barely a
   contradiction. When the tenant allows ``conflict_adjudication``, a related pair the head
   leaves in between goes to the conflict adjudicator, as consolidation does.

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
from datetime import datetime
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
#: contradiction is mostly two unrelated sentences ("Store 61 stays in Region South" /
#: "the BOGO promotion runs only in Region North" reads as a 1.00 contradiction).
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
#: leave") or that only the encoder relates ("I live in Seattle" / "I moved to Austin") marked
#: four statements, all wrongly (a multi-hop bridge "Employees in the Berlin office follow
#: the German calendar" by "Lena Vogel joined the Berlin office"): those stay unmarked.
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
    ("FreshWay", "PackagingCorp", "CFO", "MK-220")."""
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

    @property
    def labels(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for label, value in self.ids:
            if label:
                out.setdefault(label, set()).add(value)
        return out


#: A number with the word that labels it ("Forklift #4", "Tor 14", "aisle 12", "Gabelstapler
#: Nr. 3", "الرافعة الشوكية رقم 4"): which thing is meant. A number marker between the two is
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
    return Anchors(frozenset(reading.names()), frozenset(_identifiers(text)))


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
    #: label in common: two things ("Forklift #3" / "Forklift #4", "Tor 9" / "Tor 14")
    apart: bool
    #: the same named thing with a number that changed ("MK-220 ... 40 units" / "60 units")
    value_changed: bool = False

    @property
    def anchored(self) -> bool:
        return self.shared_name or self.shared_id


def relate(new: Anchors, old: Anchors, new_text: str = "", old_text: str = "") -> Relation:
    shared_values = {v for _, v in new.ids} & {v for _, v in old.ids}
    clashing = _clashing(new, old)
    label_words = {w for label in clashing for w in label.lstrip("#").split()}
    common = _common_names(new.names, old.names)
    rescuing = {n for n in common if not (set(n.split()) & label_words)}
    named = bool(common) and (not clashing or bool(rescuing))
    anchored = named or bool(shared_values)
    return Relation(
        shared_name=named,
        shared_id=bool(shared_values),
        apart=bool(clashing) and not shared_values and not rescuing,
        value_changed=anchored and bool(new_text and old_text) and _changed(new_text, old_text),
    )


def _clashing(new: Anchors, old: Anchors) -> set[str]:
    """The labels both use for different things ("forklift": 3 / 4), and "#" when each names
    its own codes and none is the other's ("FT-1180" / "PT-2207")."""
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
    """Names both carry: equal, or one a run of the other's words ("dave morris" in "lager
    dave morris", where German capitalises the noun before the name)."""
    out = set(a & b)
    for x in a:
        for y in b:
            short, long_ = (x, y) if len(x) <= len(y) else (y, x)
            if short != long_ and f" {short} " in f" {long_} " and " " in short:
                out.add(short)
    return out


def _changed(new_text: str, old_text: str) -> bool:
    a, b = numbers(new_text), numbers(old_text)
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
    change = sorted({k.value for k in kinds if k in CHANGE_KINDS})
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

#: Categories never replaced here: a standing rule is replaced by the rule that revises it
#: (consolidation), and a preference by its own slot rules.
_KEPT_CATEGORIES = frozenset({"rule"})


def statement_of(memory: CanonicalMemory, principal: str) -> bool:
    """A current statement a user made in their own words, readable in all their
    conversations: what this module compares and replaces."""
    return (
        memory.deleted_at is None
        and memory.temporal.status is TemporalStatus.CURRENT
        and memory.visibility is Visibility.USER
        and memory.owner_principal == principal
        and memory.memory_type is not MemoryType.PREFERENCE
        and memory.system_metadata.get("category") not in _KEPT_CATEGORIES
        and not memory.system_metadata.get("source_revisions")
        and not unverified_representation(memory.system_metadata)
    )


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
    ) -> Outcome:
        """Compare the statements of one observation (``written``: what it created) with the
        user's earlier statements and apply what follows. Unchanged rows are not touched."""
        new = [m for m in written if statement_of(m, ctx.principal_id)]
        if not self.enabled or not new:
            return Outcome()
        # the statement as said is what replaces: the turn (each piece of a long one), or what
        # the rules read when the turn itself was not kept
        units = [m for m in new if m.system_metadata.get("category") == "verbatim_turn"] or new
        earlier = await self._earlier(uow, ctx, new)
        pairs = await self._related(units, earlier) if earlier else []
        if not pairs:
            return Outcome()
        await self._contradiction(pairs)
        best = await self._decide(pairs, self._kinds(new, units), ctx)
        return await self._apply(uow, ctx, best, earlier, now=now)

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
        if self._adjudicating:
            asked = sorted(
                (x for x in undecided if worth_asking(x[0])),
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
        self, units: Sequence[CanonicalMemory], earlier: Sequence[CanonicalMemory]
    ) -> list[_Pair]:
        pairs: list[_Pair] = []
        pending: list[_Pair] = []
        for unit in units:
            mine = anchors(unit.content, unit.lang or None)
            for old in earlier:
                if old.temporal.observed_at > unit.temporal.observed_at:
                    continue  # a statement replaces only what was said before it
                relation = relate(
                    mine, anchors(old.content, old.lang or None), unit.content, old.content
                )
                if relation.apart:
                    continue
                pair = _Pair(unit, old, relation)
                (pairs if relation.anchored else pending).append(pair)
        return pairs + await self._close_enough(pending)

    async def _close_enough(self, pending: Sequence[_Pair]) -> list[_Pair]:
        """The unanchored pairs the encoder puts close enough to compare."""
        if not pending or self.embedding is None:
            return []
        texts = sorted({p.unit.content for p in pending} | {p.old.content for p in pending})
        vectors = dict(zip(texts, await self.embedding.embed_documents(texts), strict=True))
        out = []
        for pair in pending:
            pair.cosine = _cosine(vectors[pair.unit.content], vectors[pair.old.content])
            if pair.cosine >= RELATED_COSINE:
                out.append(pair)
        return out

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


def worth_asking(pair: _Pair) -> bool:
    """A related pair the head left in its unsure band: what the tenant's model is asked."""
    return related(pair.relation, pair.cosine) and pair.contradiction >= ADJUDICATE_FROM


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
