"""The statement labeller: what each sentence a user says *does*, decided at write (ADR 0036).

A sentence is a FACT, a standing RULE, a CONDITIONAL_RULE (a rule with an exception or a
trigger about the world), a STATUS of a thing, a CORRECTION of something said before, or a
LIFECYCLE change (a thing or a relationship began or ended) - or none of these: a question,
a greeting, a one-off request. The kind is stored on the memory so that later stages can act
on it; the labeller itself changes nothing about retrieval.

Three tiers, cheapest first, and each only where the one before is unsure:

1. **Lexicon** (every language, microseconds). Cue words live in data packs under
   ``lexicon/`` - ``generic.json`` (domain-free words: always, unless, actually, ...) and a
   domain pack (``retail.json``: out of stock, recalled, delisted, ...), both loaded by
   default. The code here holds only the *structure*: where in a clause a cue must sit to
   count, which cue wins, how a rule's trigger and exception are cut out of the sentence.
2. **NLI** (the frozen NLI head the grounding cascade already loads). For what the lexicon
   left open - a standing word or a condition before a clause that may or may not be an
   instruction, or a word that only suggests a kind ("stopped", "started", "actually") - the
   one hypothesis of the kind it suspects is scored against the sentence in whatever language
   it is written: one pair a sentence, all sentences of an observation in one batch, with
   per-kind thresholds (``StatementLabellerSettings``). A sentence no cue marks costs nothing.
3. **LLM** (when the tenant's policy allows ``contextual_extraction``). The model proposes a
   kind for the sentences still open; a proposal is accepted only when the NLI head confirms
   that the sentence entails that kind's hypothesis. The model never decides alone.

Precedence when two kinds apply to one sentence: CORRECTION > CONDITIONAL_RULE > RULE >
LIFECYCLE > STATUS > FACT ("Actually, we terminated our contract with Uline" corrects).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal

from memory_service.config.constants import STATEMENT_LABELLER, StatementLabellerSettings
from memory_service.domain.enums import StatementKind
from memory_service.domain.language import ENGLISH, detect_language, english_evidence
from memory_service.modules.llm.assist import LLMAssist
from memory_service.ports.models import NLIProvider

LEXICON_DIR: Final = Path(__file__).with_name("lexicon")
RULE_KINDS: Final = frozenset({StatementKind.RULE, StatementKind.CONDITIONAL_RULE})
#: Pack fields whose words only suggest a kind, in the order they are tried.
_WEAK: Final = (
    (StatementKind.CORRECTION, "correction_weak"),
    (StatementKind.LIFECYCLE, "lifecycle_weak"),
    (StatementKind.STATUS, "status_weak"),
)
#: The kind that wins when two apply to one sentence (and to a turn of several sentences).
PRECEDENCE: Final = (
    StatementKind.CORRECTION,
    StatementKind.CONDITIONAL_RULE,
    StatementKind.RULE,
    StatementKind.LIFECYCLE,
    StatementKind.STATUS,
    StatementKind.FACT,
)
Source = Literal["lexicon", "nli", "llm"]

#: The fields a pack may carry (see ``lexicon/generic.json`` for what each one means).
_FIELDS: Final = (
    "standing",
    "not_standing",
    "obligation",
    "universal",
    "policy_subject",
    "condition",
    "exception",
    "requester",
    "correction_start",
    "correction",
    "replacement",
    "lifecycle",
    "status",
    "status_weak",
    "lifecycle_weak",
    "correction_weak",
    "filler",
    "only",
    "addressee",
    "generic",
    "consequent",
    "subject_start",
    "imperative",
    "imperative_inside",
    "request",
    "question_start",
    "auxiliary",
    "chitchat",
)
#: A letter of any script the packs cover, combining marks included: Python's ``\w`` does not
#: match a Devanagari vowel sign or an Arabic diacritic, so ``\b`` would cut words apart. The
#: scripts' own punctuation (the Arabic comma and question mark, the danda) is not a letter.
_LETTER: Final = (
    r"\w\u0300-\u036f\u0610-\u061a\u0620-\u065f\u0670-\u06d3\u06d5-\u06ff"
    r"\u0750-\u077f\u0900-\u0963\u0966-\u097f"
)
_BEFORE: Final = rf"(?<![{_LETTER}])"
_AFTER: Final = rf"(?![{_LETTER}])"
_AR_MARKS: Final = "[\u064b-\u0652\u0670\u0640]*"
_AR_LETTER: Final = re.compile("([\u0621-\u064a])")
_AR_STRIP: Final = re.compile("[\u064b-\u0652\u0670\u0640]")
#: A clause ends at punctuation - but not at the comma inside "1,000" or the colon in "9:30".
_CLAUSE_BREAK: Final = re.compile(r"(?<!\d)[,:]|[,:](?!\d)|[;\u2014\u2013()\u060c\u061b]|\s-\s")
_QUESTION_MARK: Final = re.compile("[?\uff1f\u061f][\"'\u201d\u2019)\\]]*\\s*$")
_WORD: Final = re.compile(rf"[{_LETTER}'\u2019#/.-]+")
_LEAD_NOISE: Final = re.compile("^[\\s\"'\u201c\u2018\u00a1\u00bf(\\[*\u2022#-]+")


def _compile_term(
    term: str, lang: str, clitics: Sequence[str], enclitics: Sequence[str] = ()
) -> str:
    """One pack entry as a pattern: a space matches any whitespace; in Arabic every letter
    may carry a diacritic and the word may carry a clitic the pack names before it (wa-,
    al-, bi-) and an enclitic after it (-ni "me", -i of the feminine imperative, -u of the
    plural)."""
    term = term.replace(" ", r"\s+")
    if lang == "ar":
        term = _AR_LETTER.sub(lambda m: m.group(1) + _AR_MARKS, _AR_STRIP.sub("", term))
    prefix = f"(?:{'|'.join(sorted(clitics, key=len, reverse=True))})?" if clitics else ""
    suffix = f"(?:{'|'.join(sorted(enclitics, key=len, reverse=True))})?" if enclitics else ""
    return f"{prefix}(?:{term}){suffix}"


@dataclass(frozen=True)
class _Cue:
    """One field of the loaded packs: a pattern over every language, and per language."""

    any: re.Pattern[str] | None
    by_lang: dict[str, re.Pattern[str]]

    def search(self, text: str, pos: int = 0, endpos: int | None = None) -> re.Match[str] | None:
        if self.any is None:
            return None
        return self.any.search(text, pos, len(text) if endpos is None else endpos)

    def finditer(self, text: str) -> Iterable[re.Match[str]]:
        return () if self.any is None else self.any.finditer(text)

    def at(self, text: str, pos: int) -> re.Match[str] | None:
        """A match that starts exactly at ``pos``."""
        return None if self.any is None else self.any.match(text, pos)


def _cue(
    terms: dict[str, list[str]],
    clitics: dict[str, list[str]],
    enclitics: dict[str, list[str]] | None = None,
) -> _Cue:
    after = enclitics or {}
    by_lang = {
        lang: re.compile(
            _BEFORE
            + "(?:"
            + "|".join(
                _compile_term(t, lang, clitics.get(lang, ()), after.get(lang, ())) for t in words
            )
            + ")"
            + _AFTER,
            re.IGNORECASE,
        )
        for lang, words in terms.items()
        if words
    }
    if not by_lang:
        return _Cue(None, {})
    union = "|".join(f"(?:{p.pattern})" for p in by_lang.values())
    return _Cue(re.compile(union, re.IGNORECASE), by_lang)


#: Pack fields that are not cue patterns: word endings and clitics (see generic.json).
_MORPHOLOGY: Final = ("clitic", "enclitic", "imperative_ending", "declarative_ending")


def load_packs(names: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    """``{field: {lang: [term, ...]}}`` merged over the named packs, in order."""
    merged: dict[str, dict[str, list[str]]] = {name: {} for name in (*_FIELDS, *_MORPHOLOGY)}
    for name in names:
        pack = json.loads((LEXICON_DIR / f"{name}.json").read_text(encoding="utf-8"))
        for lang, fields in pack["languages"].items():
            for key, terms in fields.items():
                if key not in merged:
                    raise ValueError(f"lexicon pack {name!r}: unknown field {key!r}")
                merged[key].setdefault(lang, []).extend(terms)
    return merged


@dataclass(frozen=True)
class LexicalLabel:
    """The lexicon's verdict on one sentence.

    ``decided`` False means the lexicon is not sure: ``kind`` is its fallback - a fact, the
    precise answer when no model can say more - ``maybe`` the rule it may be (a standing
    word, or a condition, before what may or may not be an instruction), and the model tier
    decides.
    """

    kind: StatementKind | None
    decided: bool = True
    maybe: StatementKind | None = None
    #: the clause a rule depends on ("whenever I ask for a stock audit") and the one it does
    #: not apply under ("unless I specifically type 'include out of stock'")
    trigger: str | None = None
    exception: str | None = None
    #: the instruction without its opening "please" / "from now on"
    instruction: str | None = None


@dataclass(frozen=True)
class StatementLabel:
    """A sentence's kind, where it came from and, for a rule, its trigger and exception."""

    kind: StatementKind | None
    source: Source = "lexicon"
    confidence: float = 1.0
    trigger: str | None = None
    exception: str | None = None
    instruction: str | None = None

    @classmethod
    def of(cls, lexical: LexicalLabel) -> StatementLabel:
        return cls(
            kind=lexical.kind,
            confidence=1.0 if lexical.decided else 0.5,
            trigger=lexical.trigger,
            exception=lexical.exception,
            instruction=lexical.instruction,
        )


def most_salient(kinds: Iterable[StatementKind | None]) -> StatementKind | None:
    """The kind a turn of several sentences carries: the highest in ``PRECEDENCE``."""
    present = {k for k in kinds if k is not None}
    return next((k for k in PRECEDENCE if k in present), None)


class Lexicon:
    """The cue packs, compiled once per language.

    A sentence is read with its own language's cues (``domain.language``) and English's,
    or with every language's when its language is undetermined: a short cue of one language
    is a different word in another (Spanish "no," opens a correction; English "No, I
    didn't" answers a question).
    """

    def __init__(self, packs: Sequence[str]) -> None:
        terms = load_packs(packs)
        self.packs = tuple(packs)
        languages = sorted({lang for by_lang in terms.values() for lang in by_lang})

        def reader(langs: Sequence[str]) -> _Reader:
            return _Reader(
                {
                    name: _cue(
                        {k: v for k, v in terms[name].items() if k in langs},
                        terms["clitic"],
                        terms["enclitic"],
                    )
                    for name in _FIELDS
                },
                tuple(e for k in langs for e in terms["imperative_ending"].get(k, ())),
                tuple(e for k in langs for e in terms["declarative_ending"].get(k, ())),
            )

        self._any = reader(languages)
        self._readers = {lang: reader(sorted({lang, ENGLISH})) for lang in languages}

    @classmethod
    @cache
    def default(cls) -> Lexicon:
        return cls(STATEMENT_LABELLER.packs)

    def reader(self, sentence: str) -> _Reader:
        lang = detect_language(sentence)
        if lang == ENGLISH and not english_evidence(sentence):
            return self._any  # plain ASCII with no English word in it: "Guten Morgen!"
        return self._readers.get(lang, self._any)

    def is_question(self, sentence: str) -> bool:
        return self.reader(sentence).is_question(sentence)

    def label(self, sentence: str) -> LexicalLabel:
        return self.reader(sentence).label(sentence)


class _Reader:
    """The sentence structure the cues of one set of languages are read in."""

    def __init__(
        self,
        cues: dict[str, _Cue],
        imperative_endings: tuple[str, ...],
        declarative_endings: tuple[str, ...],
    ) -> None:
        self.cues = cues
        self.imperative_endings = imperative_endings
        self.declarative_endings = declarative_endings

    # -- public ---------------------------------------------------------------------
    def is_question(self, sentence: str) -> bool:
        """A question: a question mark at the end, or (English) a question word first.

        "When"/"whenever"/"if" open a condition as often as a question: "When I ask for a
        stock audit, always use a table" is an instruction, and reading it as a question
        stored nothing at all. A question word that also opens a condition is a question
        only when an auxiliary follows it ("when is the delivery").
        """
        s = _LEAD_NOISE.sub("", sentence.strip())
        if _QUESTION_MARK.search(s):
            return True
        lead = self.cues["question_start"].at(s, 0)
        if lead is None:
            return False
        if self.cues["condition"].at(s, 0) is None:
            return True
        return self.cues["auxiliary"].at(s, _skip_space(s, lead.end())) is not None

    def label(self, sentence: str) -> LexicalLabel:
        s = _LEAD_NOISE.sub("", sentence.strip())
        if not any(ch.isalpha() for ch in s) or self.is_question(s) or self._chitchat(s):
            return LexicalLabel(None)
        work = self._mask(s)
        if self._correction(s):
            return LexicalLabel(StatementKind.CORRECTION)
        if rule := self._rule(s, work):
            return rule
        if self.cues["replacement"].search(s):
            return LexicalLabel(StatementKind.CORRECTION)
        if self.cues["lifecycle"].search(s):
            return LexicalLabel(StatementKind.LIFECYCLE)
        if self.cues["status"].search(s):
            return LexicalLabel(StatementKind.STATUS)
        if self._request(s):
            return LexicalLabel(None)
        for kind, cue in _WEAK:
            if self.cues[cue].search(s):
                # a word that only suggests the kind ("closed", "new", "instead of"): the
                # head confirms it with one pair, or the statement stays a fact
                return LexicalLabel(StatementKind.FACT, decided=False, maybe=kind)
        # no cue at all: a fact, with no model pass (the most common sentence there is)
        return LexicalLabel(StatementKind.FACT)

    # -- structure ------------------------------------------------------------------
    def _mask(self, s: str) -> str:
        """``s`` with every not-standing phrase ("for good", "hamesha ke liye") blanked out,
        so a standing word inside one is not read as an instruction."""
        return (
            self.cues["not_standing"].any.sub(lambda m: " " * len(m.group()), s)
            if (self.cues["not_standing"].any is not None)
            else s
        )

    def _chitchat(self, s: str) -> bool:
        m = self.cues["chitchat"].at(s, 0)
        if m is None:
            return False
        rest = _WORD.findall(s[m.end() :])
        return len(rest) <= 5 or all(self.cues["chitchat"].at(w, 0) for w in rest)

    def _correction(self, s: str) -> bool:
        return bool(self.cues["correction_start"].at(s, 0) or self.cues["correction"].search(s))

    def _request(self, s: str) -> bool:
        """A one-off request: it opens with a request verb or a please/don't (a verb-final
        language: it ends with an imperative). "Store 41 is ..." opens with a noun."""
        pos = self._past_fillers(s, 0, len(s))
        if self.cues["imperative"].at(s, 0) or self.cues["imperative"].at(s, pos):
            return True
        verb = self.cues["request"].at(s, pos)
        if verb is not None:
            after = _skip_space(s, verb.end())
            return not (s[after : after + 1].isdigit() or s[after : after + 1] == "#") and (
                self.cues["auxiliary"].at(s, after) is None
            )
        return self._imperative_end(s)

    def _imperative_end(self, clause: str) -> bool:
        """A verb-final clause (Hindi) that ends in an imperative form ("bhejo", "dena")."""
        words = _WORD.findall(clause)
        return bool(
            words
            and self.imperative_endings
            and words[-1].endswith(self.imperative_endings)
            and not words[-1].endswith(self.declarative_endings)
        )

    def _past_fillers(self, s: str, start: int, end: int) -> int:
        pos = _skip_space(s, start)
        while pos < end and (m := self.cues["filler"].at(s, pos)) is not None:
            pos = _skip_space(s, m.end())
        return pos

    def _imperative(self, s: str, start: int, end: int, at: int | None = None) -> str | None:
        """How surely the clause ``s[start:end]`` is an instruction: "strong" (a standing
        word or a request verb opens it, a please/don't marks it, a Hindi imperative ends
        it), "weak" (it does not open with a subject, so it may be a verb) or None."""
        pos = self._past_fillers(s, start, end)
        if pos >= end:
            return None
        only = self.cues["only"].at(s, pos)
        if only is not None:
            pos = self._past_fillers(s, only.end(), end)
        if at is not None and at <= pos:
            return "strong"
        if self.cues["request"].at(s, pos) or self.cues["imperative"].at(s, pos):
            return "strong"
        if self.cues["imperative_inside"].search(s, start, end):
            return "strong"
        if self._imperative_end(s[start:end]):
            return "strong"
        if self.cues["subject_start"].at(s, pos) or self.cues["policy_subject"].at(s, pos):
            return None
        if s[pos : pos + 1].isdigit():
            return None
        return "weak"

    def _rule(self, s: str, work: str) -> LexicalLabel | None:
        """A rule, when the sentence gives a standing instruction, with its trigger and its
        exception cut out of it; None when it does not."""
        shape = self._shape(s, work)
        decided = self._rule_certainty(s, work, shape)
        if decided is None:
            return None
        condition, exception = shape.condition, shape.exception
        # "whenever I ask for a stock audit" scopes a rule to a request; "if a delivery is
        # late" makes it depend on the world, which is what a conditional rule is
        conditional = exception is not None or (
            condition is not None and not self._requested(work, condition)
        )
        kind = StatementKind.CONDITIONAL_RULE if conditional else StatementKind.RULE
        # unsure, the sentence keeps the reading it has without the rule: a request asks for
        # nothing to be stored, anything else is a fact
        fallback = None if self._request(s) else StatementKind.FACT
        return LexicalLabel(
            kind if decided else fallback,
            decided=decided,
            maybe=None if decided else kind,
            trigger=_clean_clause(s[condition[0] : condition[1]]) if condition else None,
            exception=_clean_clause(s[exception[0] : exception[1]]) if exception else None,
            instruction=self._instruction(s),
        )

    def _requested(self, work: str, condition: Span | None) -> bool:
        """The condition is about the user's own requests or the assistant's work ("when I
        ask for a stock audit", "every time you draft an email"), not about the world."""
        return condition is not None and bool(
            self.cues["requester"].search(work, condition[0], condition[1])
        )

    def _obligation(self, s: str, work: str, shape: _Shape) -> bool:
        """A universal obligation: "All seasonal merchandise must be routed to Facility B",
        "Every invoice must be approved by two people" - the quantifier (or, verb-first, the
        obligation) opens the clause. "It must be exciting to see it all" does not."""
        start, end = shape.main
        opening = self._past_fillers(s, start, end)
        universal, obligation = self.cues["universal"], self.cues["obligation"]
        return bool(
            (universal.at(s, opening) and obligation.search(work, opening, end))
            or (obligation.at(s, opening) and universal.search(work, opening, end))
        )

    def _clauses(self, s: str) -> list[Span]:
        """Clauses end at punctuation and before a consequent ("if X then Y", Hindi "to")."""
        cuts = sorted(
            [(m.start(), m.end()) for m in _CLAUSE_BREAK.finditer(s)]
            + [(m.start(), m.start()) for m in self.cues["consequent"].finditer(s)]
        )
        out, start = [], 0
        for cut, resume in [*cuts, (len(s), len(s))]:
            begin = start + len(s[start:cut]) - len(s[start:cut].lstrip())
            if cut > begin:
                out.append((begin, cut))
            start = max(start, resume)
        return out or [(0, len(s))]

    def _shape(self, s: str, work: str) -> _Shape:
        clauses = self._clauses(s)
        exception = _span(self.cues["exception"].search(work), clauses, s)
        condition = next(
            (
                _span(m, clauses, s)
                for m in self.cues["condition"].finditer(work)
                if not _within(m.start(), exception)
            ),
            None,
        )
        markers = list(self.cues["standing"].finditer(work))
        free = [
            m
            for m in markers
            if not _within(m.start(), exception) and not _within(m.start(), condition)
        ]
        marker = free[0].start() if free else None
        leads = False
        if free:
            main = _clause_of(clauses, free[0].start(), s)
            following = [c for c in clauses if c[0] >= main[1]]
            if not s[free[0].end() : main[1]].strip() and following:
                # a standing phrase set off by itself ("From now on, reply in German") marks
                # the clause after it, which must still read as an instruction on its own
                main, marker, leads = following[0], None, True
        else:
            # the first clause with words of its own: not the condition, the exception, or a
            # vocative ("AI, ...") or consequent ("... to AI, ...") on its own
            main = next(
                (
                    c
                    for c in clauses
                    if not _within(c[0], exception)
                    and not _within(c[0], condition)
                    and self._past_fillers(s, c[0], c[1]) < c[1]
                ),
                clauses[0],
            )
        return _Shape(
            main,
            condition,
            exception,
            marker,
            bool(markers),
            self._past_fillers(s, 0, len(s)),
            leads,
        )

    def _rule_certainty(self, s: str, work: str, shape: _Shape) -> bool | None:
        """True: surely a rule. False: perhaps - the model tier decides. None: not a rule."""
        start, end = shape.main
        evidence = self._imperative(s, start, end, at=shape.marker)
        marked = shape.marker is not None or shape.leads
        if (marked and evidence == "strong") or self._obligation(s, work, shape):
            return True
        if shape.exception and evidence == "strong":
            return True  # "Don't reorder seasonal items unless the buyer approves it"
        if shape.condition is not None and evidence:
            return self._conditional_certainty(s, work, shape, evidence)
        if marked and self.cues["policy_subject"].at(s, self._past_fillers(s, start, end)):
            return False  # "We never ship hazardous goods on Fridays" / "We always look forward"
        if marked and evidence:
            # "We never ship hazardous goods on Fridays" / "We always look forward to ...";
            # "Formatiere Berichte immer als Tabelle" after a verb the packs cannot list
            return False
        return None

    def _conditional_certainty(
        self, s: str, work: str, shape: _Shape, evidence: str
    ) -> bool | None:
        """An instruction under a condition. Sure when it is plainly addressed to the
        assistant on the user's behalf - the condition is the user's own request ("each time
        I upload a log"), the instruction names its recipient ("please alert the manager",
        "notify me") after a leading condition, or the condition is any occurrence ("if
        there's any change", "if a delivery is late") - or it is restricted to the condition
        ("only escalate if"). Otherwise - "If SF is your thing, check out The Expanse",
        "Tell me when it arrives" - perhaps.

        A clause the packs cannot read as an instruction is perhaps one only when the shape
        of the sentence says so: the condition comes first ("Si el pedido llega tarde,
        ..."), or a standing word or a recipient marks it. With the condition trailing a
        clause like that, the sentence describes ("Life is better when we're together",
        "Started when I was young"): not a rule."""
        start, end = shape.main
        condition = shape.condition
        assert condition is not None
        first = condition[0] == shape.opening
        # the instruction's own words: not the condition when it trails in the same clause
        own = condition[0] if start < condition[0] < end else end
        addressed = bool(
            self.cues["addressee"].search(work, start, own)
            or self.cues["imperative_inside"].search(work, start, own)
        )
        if evidence != "strong":
            marked = shape.marker is not None or shape.leads
            return False if (first or marked or addressed) else None
        if self.cues["only"].search(work, start, end):
            return True
        lead = self.cues["condition"].at(work, condition[0])
        after_lead = lead.end() if lead is not None else condition[0]
        if self.cues["generic"].search(work, after_lead, condition[1]):
            return True
        return first and (self._requested(work, condition) or addressed)

    def _instruction(self, s: str) -> str:
        """The sentence without what opens it: "please", and a standing phrase set off by a
        comma ("from now on, reply in German" -> "reply in German")."""
        pos = self._past_fillers(s, 0, len(s))
        m = self.cues["standing"].at(s, pos)
        if m is not None and re.match(r"\s*[,:\u060c]", s[m.end() :]):
            pos = self._past_fillers(s, m.end(), len(s))
        return s[pos:].strip(" ,;:.!\u0964")


def _skip_space(s: str, pos: int) -> int:
    while pos < len(s) and (s[pos].isspace() or s[pos] in ",:;\u060c"):
        pos += 1
    return pos


Span = tuple[int, int]


@dataclass(frozen=True)
class _Shape:
    """Where a possible rule's parts sit in its sentence (character spans)."""

    main: Span  # the clause that carries the instruction
    condition: Span | None  # "whenever I ask for ...", "if a delivery is late"
    exception: Span | None  # "unless I type ..."
    marker: int | None  # the first standing word outside both
    marked: bool  # a standing word anywhere
    opening: int  # where the sentence's first word after "please" / "also" is
    leads: bool = False  # a standing phrase set off before the main clause


def _within(pos: int, span: Span | None) -> bool:
    return span is not None and span[0] <= pos < span[1]


def _span(m: re.Match[str] | None, clauses: list[Span], s: str) -> Span | None:
    """From a cue to the end of its clause."""
    return None if m is None else (m.start(), _clause_of(clauses, m.start(), s)[1])


def _clause_of(clauses: list[tuple[int, int]], pos: int, s: str) -> tuple[int, int]:
    return next(((a, b) for a, b in clauses if a <= pos < b), (pos, len(s)))


def _clean_clause(text: str) -> str:
    return text.strip(" ,;:.!\u060c\u0964")


# --------------------------------------------------------------------------- the labeller

#: The JSON the model answers with: one kind (or "none") per open sentence, by index.
_LLM_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "kind"],
                "properties": {
                    "index": {"type": "integer"},
                    "kind": {
                        "type": "string",
                        "enum": [k.value.lower() for k in StatementKind] + ["none"],
                    },
                },
            },
        }
    },
}
_LLM_SYSTEM: Final = (
    "Label what each numbered sentence a user said does. Answer one kind per sentence: "
    "'rule' - a standing instruction for future answers (always/never/from now on, "
    "'whenever I ask for X, do Y'); 'conditional_rule' - a standing instruction with an "
    "exception or a condition about the world ('unless ...', 'if stock drops below ...'); "
    "'status' - the current state of a thing (broken, repaired, out of stock, delayed); "
    "'lifecycle' - a thing or a relationship began or ended (terminated, dismantled, "
    "launched, a new supplier); 'correction' - revises something said earlier; 'fact' - any "
    "other statement; 'none' - a question, a greeting or a one-off request."
)


@dataclass
class _Open:
    index: int
    sentence: str
    lexical: LexicalLabel
    candidates: tuple[StatementKind, ...] = field(default_factory=tuple)


class StatementLabeller:
    """Labels the sentences of one observation: the lexicon for all, the NLI head for the
    open ones (one batch), and the tenant's model, NLI-confirmed, for what is still open."""

    def __init__(
        self,
        lexicon: Lexicon | None = None,
        *,
        nli: NLIProvider | None = None,
        assist: LLMAssist | None = None,
        settings: StatementLabellerSettings = STATEMENT_LABELLER,
    ) -> None:
        self.lexicon = lexicon or Lexicon.default()
        # A stand-in classifier (token overlap) cannot read a hypothesis: the model tiers are
        # used only with a trained head, never silently with an arithmetic substitute.
        self.nli = nli if nli is not None and getattr(nli, "representative", False) else None
        self.assist = assist or LLMAssist.disabled()
        self.cfg = settings

    def lexical(self, sentence: str) -> StatementLabel:
        return StatementLabel.of(self.lexicon.label(sentence))

    async def label(self, sentences: Sequence[str], *, models: bool = True) -> list[StatementLabel]:
        """One label per sentence. ``models`` False keeps to the lexicon (agent chatter)."""
        lexical = [self.lexicon.label(s) for s in sentences]
        labels = [StatementLabel.of(x) for x in lexical]
        open_ = self._open(sentences, lexical) if models and self.nli is not None else []
        if not open_:
            return labels
        # one pair per open sentence - the kind it may be - in one batch for the observation
        seen = await self._entail(
            open_, [(n, k) for n, o in enumerate(open_) for k in o.candidates]
        )
        proposals = await self._ask_model(open_, seen)
        for n, o in enumerate(open_):
            labels[o.index] = self._resolve(o, proposals.get(n), seen.get(n, {}), labels[o.index])
        return labels

    def _open(self, sentences: Sequence[str], lexical: list[LexicalLabel]) -> list[_Open]:
        """The sentences the lexicon left open, as many as one observation may score."""
        return [
            _Open(i, s, x, self._candidates(x))
            for i, (s, x) in enumerate(zip(sentences, lexical, strict=True))
            if not x.decided and len(s) <= self.cfg.nli_max_chars
        ][: self.cfg.nli_max_sentences]

    async def _ask_model(
        self, open_: list[_Open], seen: dict[int, dict[StatementKind, float]]
    ) -> dict[int, StatementKind]:
        """The tenant's model's kinds for what the head was unsure of - a score in the band
        between confirming a proposal and deciding alone - so a model call is the exception
        on the write path, not one per message. ``seen`` gains the confirming scores."""
        unsure = [n for n, o in enumerate(open_) if self._unsure(o, seen.get(n, {}))]
        proposals = await self._proposals(open_, unsure)
        extra = [
            (n, k) for n, p in proposals.items() for k in _checked(p) if k not in seen.get(n, {})
        ]
        if extra:
            for n, scores in (await self._entail(open_, extra)).items():
                seen.setdefault(n, {}).update(scores)
        return proposals

    def _candidates(self, x: LexicalLabel) -> tuple[StatementKind, ...]:
        """What the NLI head checks for an open sentence: the kind the lexicon suspects."""
        return (x.maybe,) if x.maybe is not None else ()

    def _unsure(self, o: _Open, seen: dict[StatementKind, float]) -> bool:
        decided = any(seen[k] >= self.cfg.thresholds[k] for k in seen)
        return not decided and any(seen[k] >= self.cfg.llm_confirm_min for k in seen)

    async def _proposals(self, open_: list[_Open], unsure: list[int]) -> dict[int, StatementKind]:
        """The model's kind for each unsure sentence (by its index in ``open_``)."""
        if not unsure or not self.assist.wants("contextual_extraction"):
            return {}
        numbered = "\n".join(f"{k}. {open_[n].sentence}" for k, n in enumerate(unsure))
        out = await self.assist.structured(
            "contextual_extraction",
            system=_LLM_SYSTEM,
            user=numbered,
            schema=_LLM_SCHEMA,
            max_tokens=40 + 16 * len(unsure),
        )
        proposals: dict[int, StatementKind] = {}
        for item in out.get("labels", []) if isinstance(out, dict) else []:
            try:
                k, kind = int(item["index"]), str(item["kind"]).upper()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= k < len(unsure) and kind in StatementKind.__members__:
                proposals[unsure[k]] = StatementKind(kind)
        return proposals

    async def _entail(
        self, open_: list[_Open], checks: Sequence[tuple[int, StatementKind]]
    ) -> dict[int, dict[StatementKind, float]]:
        """Entailment of each (sentence, kind) check: one NLI batch for the observation, one
        group per hypothesis (both rule kinds share theirs)."""
        by_hypothesis: dict[str, list[tuple[int, StatementKind]]] = {}
        for n, kind in checks:
            by_hypothesis.setdefault(self.cfg.hypotheses[kind], []).append((n, kind))
        order = list(by_hypothesis)
        assert self.nli is not None
        scores = await self.nli.entail_groups(
            [([open_[n].sentence for n, _ in by_hypothesis[h]], h) for h in order]
        )
        seen: dict[int, dict[StatementKind, float]] = {}
        for hypothesis, rows in zip(order, scores, strict=True):
            for (n, kind), score in zip(by_hypothesis[hypothesis], rows, strict=True):
                seen.setdefault(n, {})[kind] = score.entailment
        return seen

    def _resolve(
        self,
        o: _Open,
        proposed: StatementKind | None,
        seen: dict[StatementKind, float],
        label: StatementLabel,
    ) -> StatementLabel:
        for checked in _checked(proposed):
            if seen.get(checked, 0.0) >= self.cfg.llm_confirm_min:
                # a proposed rule keeps the lexicon's reading of its trigger and exception
                kind = (
                    o.lexical.maybe
                    if proposed in RULE_KINDS and o.lexical.maybe in RULE_KINDS
                    else proposed
                )
                return replace(label, kind=kind, source="llm", confidence=seen[checked])
        maybe = o.lexical.maybe
        score = seen.get(maybe, 0.0) if maybe is not None else 0.0
        if maybe is not None and score >= self.cfg.thresholds[maybe]:
            return replace(label, kind=maybe, source="nli", confidence=score)
        # not confirmed: the reading the sentence has without it (a fact, or a request)
        return replace(label, source="nli", confidence=1.0 - score)


def _checked(proposed: StatementKind | None) -> tuple[StatementKind, ...]:
    """The hypothesis a model proposal is confirmed by; nothing to confirm for no proposal
    or a fact, the fallback anyway."""
    if proposed is None or proposed is StatementKind.FACT:
        return ()
    return (proposed,)
