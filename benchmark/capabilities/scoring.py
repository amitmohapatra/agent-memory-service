"""Scoring one answer against a query's assertions. Generic: it knows items, anchors and
assertion kinds, never a domain.

What is scored is what the service hands an agent for the question, as a list of items:

* ``context`` - the pushed context (``context(format="full")``): profile blocks, the thread
  summary, the window, learned skills, memories, document passages, graph facts, summaries;
* ``search`` - the agent's pull (``search(kinds=[memory, episode, message])``), ranked items;
* ``answer`` - the final answer an agent produced from it (harness mode), one item.

The assertions (all optional, all language-aware):

``contains``      groups; each group is an anchor or a list of alternatives, and is met when
                  any item matches any alternative. Every group must be met.
``absent``        stale or wrong values, ``{"text": anchor, "unless": [anchors]}``: a leak is a
                  current (not historical) item that matches the anchor and none of ``unless``
                  - "Uline" inside "we terminated Uline, PackagingCorp is the supplier now" is
                  not a leak; "our pallet supplier is Uline" served as current is.
``abstain``       the right answer is "I do not know": the context must say INSUFFICIENT (and
                  an answer must refuse), and nothing in ``absent`` may appear.
``table_columns`` (answers) a markdown table whose header names every column anchor.
``answer``        overrides any of the above for the ``answer`` surface.

An anchor is an invariant string ("8492"), ``{"re": pattern}``, or a language map
``{"en": "repaired", "de": "repariert", ...}``. Matching is on normalised text: NFKC,
case-folded, whitespace collapsed, every script's digits as ASCII, thousands separators
dropped, Arabic diacritics and letter variants folded.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from benchmark.capabilities.dataset import CODE_SWITCHED

# --------------------------------------------------------------------------- normalisation

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = dict.fromkeys(map(ord, "\u2018\u2019\u201a\u201b\u2032"), "'") | dict.fromkeys(
    map(ord, "\u201c\u201d\u201e\u201f\u2033"), '"'
)
#: alef with hamza/madda/wasla -> alef, alef maqsura -> ya, tatweel dropped
_ARABIC_FOLD = dict.fromkeys(map(ord, "\u0623\u0625\u0622\u0671"), "\u0627") | {
    ord("\u0649"): "\u064a",
    ord("\u0640"): None,
}
#: a thousands separator between digit groups ("5,000", "5.000", "5\u202f000"); a plain space
#: is not one - "SKU 10442 412 units" must not read as one number
_THOUSANDS = re.compile("(?<=\\d)[,.\u00a0\u202f\u066c'](?=\\d{3}(?!\\d))")


def _ascii_digits(s: str) -> str:
    return "".join(
        str(unicodedata.digit(ch)) if not ch.isascii() and ch.isdigit() else ch for ch in s
    )


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _ascii_digits(s).translate(_DASHES).translate(_QUOTES).translate(_ARABIC_FOLD)
    # Arabic short vowels and marks (harakat); Devanagari marks are letters and stay
    s = "".join(ch for ch in s if not ("\u064b" <= ch <= "\u065f" or ch == "\u0670"))
    s = _THOUSANDS.sub("", s)
    return " ".join(s.casefold().split())


# --------------------------------------------------------------------------- anchors


@dataclass(frozen=True)
class Pattern:
    label: str
    literal: str | None = None
    regex: re.Pattern[str] | None = None

    def matches(self, normalized_text: str) -> bool:
        if self.literal is not None:
            return self._literal_in(self.literal, normalized_text)
        assert self.regex is not None
        return self.regex.search(normalized_text) is not None

    @staticmethod
    def _literal_in(literal: str, text: str) -> bool:
        """A substring, except that a number does not match inside a longer number: "8492"
        is not in "18492". Words keep plain substring matching - a German compound or an
        Arabic word with its attached article still contains the anchor."""
        if not literal:
            return False
        start = 0
        while (at := text.find(literal, start)) >= 0:
            before = text[at - 1] if at > 0 else ""
            after = text[at + len(literal)] if at + len(literal) < len(text) else ""
            if not (literal[0].isdigit() and before.isdigit()) and not (
                literal[-1].isdigit() and after.isdigit()
            ):
                return True
            start = at + 1
        return False


def resolve(anchor: Any, lang: str) -> Pattern:
    """An anchor in ``lang``. A language map without the code-switched variant is read in
    English (a mixed run's setup is the user's English)."""
    if isinstance(anchor, str):
        return Pattern(anchor, literal=normalize(anchor))
    if isinstance(anchor, Mapping):
        if "re" in anchor:
            return Pattern(f"re:{anchor['re']}", regex=re.compile(anchor["re"], re.IGNORECASE))
        key = lang if lang in anchor else ("en" if lang == CODE_SWITCHED else lang)
        value = anchor[key]
        if isinstance(value, Mapping | list):
            raise TypeError(f"nested anchor {anchor!r}")
        return Pattern(str(value), literal=normalize(str(value)))
    raise TypeError(f"not an anchor: {anchor!r}")


def group(raw: Any) -> list[Any]:
    """A ``contains`` group: one anchor, or a list of alternatives."""
    return list(raw) if isinstance(raw, list) else [raw]


# --------------------------------------------------------------------------- items


@dataclass(frozen=True)
class Item:
    kind: str
    text: str
    rank: int
    #: marked as no longer current by the service (superseded, or a fact with an end)
    historical: bool = False

    @property
    def norm(self) -> str:
        return normalize(self.text)


def items_from_bundle(bundle: Mapping[str, Any]) -> list[Item]:
    """Every part of a full context an agent's model would read, in the order it renders
    them (pinned sections first, then ranked evidence)."""
    out: list[Item] = []

    def add(kind: str, text: str | None, historical: bool = False) -> None:
        if text:
            out.append(Item(kind, text, len(out), historical))

    for block in bundle.get("profile") or ():
        add("profile", block.get("text"))
    add("thread_summary", bundle.get("thread_summary"))
    for skill in bundle.get("skills") or ():
        add("skill", " ".join([skill.get("name", ""), *skill.get("steps", [])]))
    for tool in bundle.get("tools") or ():
        add("tool", tool.get("name"))
    conversation = bundle.get("conversation") or {}
    for m in conversation.get("messages") or ():
        add("message", m.get("text"))
    for m in bundle.get("memories") or ():
        add("memory", m.get("text"), bool(m.get("superseded")))
    for k in bundle.get("knowledge") or ():
        add("knowledge", k.get("text"))
    for f in bundle.get("graph_facts") or ():
        add(
            "fact",
            " ".join(str(f.get(x, "")) for x in ("subject", "predicate", "object")),
            bool(f.get("valid_to")),
        )
    for s in bundle.get("summaries") or ():
        add("summary", s.get("text"))
    # what the assistant answered in earlier chats (ADR 0044): a corrected one is not current
    for c in bundle.get("earlier_chats") or ():
        add("earlier_chat", c.get("text"), bool(c.get("corrected")))
    return out


def items_from_search(results: Sequence[Mapping[str, Any]]) -> list[Item]:
    return [
        Item(str(r.get("kind", "item")), str(r.get("text", "")), i, bool(r.get("superseded")))
        for i, r in enumerate(results)
    ]


def items_from_answer(answer: str) -> list[Item]:
    return [Item("answer", answer, 0)] if answer else []


# --------------------------------------------------------------------------- verdicts

#: Failure reasons, most specific first; a verdict carries the first that applies.
#: every failure reason, in precedence order: a verdict's reason is the first that applies
REASONS: tuple[str, ...] = (
    "hallucinated",  # the answer was unknowable and something it must not invent appeared
    "no_abstain",  # the answer was unknowable and the service did not say so
    "false_abstain",  # answerable, and the service said INSUFFICIENT
    "not_retrieved",  # answerable, and nothing came back at all
    "missing_evidence",  # things came back, the evidence the answer needs did not
    "stale_leak",  # a replaced or obsolete value served as current
    "format",  # an answer without the table the standing instruction asks for
    "error",  # the request itself failed (recorded by the runner, never dropped)
)


@dataclass
class Verdict:
    passed: bool
    reason: str | None = None
    #: contains-groups met / required (partial credit, informational)
    groups_met: int = 0
    groups_required: int = 0
    missing: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    #: None when the surface has no abstention signal
    abstained: bool | None = None
    items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "groups_met": self.groups_met,
            "groups_required": self.groups_required,
            "missing": self.missing,
            "leaks": self.leaks,
            "abstained": self.abstained,
            "items": self.items,
        }


def expectations(expect: Mapping[str, Any], surface: str) -> dict[str, Any]:
    """The assertions for a surface: the answer surface takes ``answer`` overrides."""
    out = {k: v for k, v in expect.items() if k != "answer"}
    if surface == "answer":
        out.update(expect.get("answer") or {})
    return out


def scorable(expect: Mapping[str, Any], surface: str) -> bool:
    """Ranked search results carry no abstention signal: an unknowable question is scored on
    the surfaces that can say so (context, answer)."""
    return not (surface == "search" and expectations(expect, surface).get("abstain"))


def _leaks(absent: Iterable[Any], items: Sequence[Item], lang: str) -> list[str]:
    found: list[str] = []
    for raw in absent:
        spec = raw if isinstance(raw, Mapping) and "text" in raw else {"text": raw}
        stale = resolve(spec["text"], lang)
        unless = [resolve(a, lang) for a in spec.get("unless", ())]
        for item in items:
            if item.historical:
                continue
            norm = item.norm
            if stale.matches(norm) and not any(u.matches(norm) for u in unless):
                found.append(stale.label)
                break
    return found


def _table_ok(columns: Sequence[Any], items: Sequence[Item], lang: str) -> bool:
    wanted = [resolve(c, lang) for c in columns]
    for item in items:
        for line in item.text.splitlines():
            if line.count("|") >= 2 and all(w.matches(normalize(line)) for w in wanted):
                return True
    return False


def refused(answer: str, markers: Sequence[str]) -> bool:
    norm = normalize(answer)
    return any(normalize(m) in norm for m in markers)


def score(
    expect: Mapping[str, Any],
    items: Sequence[Item],
    lang: str,
    *,
    surface: str,
    evidence_status: str | None = None,
    refusal_markers: Sequence[str] = (),
) -> Verdict:
    """The verdict on one surface's items for one phrasing."""
    e = expectations(expect, surface)
    groups = [group(g) for g in e.get("contains", ())]
    missing: list[str] = []
    met = 0
    for alternatives in groups:
        patterns = [resolve(a, lang) for a in alternatives]
        if any(p.matches(i.norm) for i in items for p in patterns):
            met += 1
        else:
            missing.append(" | ".join(p.label for p in patterns))
    leaks = _leaks(e.get("absent", ()), items, lang)

    abstained: bool | None = None
    if surface == "context":
        abstained = evidence_status == "INSUFFICIENT"
    elif surface == "answer" and items:
        abstained = refused(items[0].text, refusal_markers)
    elif surface == "answer":
        abstained = False  # an empty answer said nothing - not "I do not know" either

    v = Verdict(
        passed=False,
        groups_met=met,
        groups_required=len(groups),
        missing=missing,
        leaks=leaks,
        abstained=abstained,
        items=len(items),
    )
    unknowable = bool(e.get("abstain"))
    applies = {
        "hallucinated": unknowable and bool(leaks),
        "no_abstain": unknowable and abstained is False,
        "false_abstain": not unknowable and bool(missing) and bool(abstained),
        "not_retrieved": not unknowable and bool(missing) and not items,
        "missing_evidence": not unknowable and bool(missing),
        "stale_leak": not unknowable and bool(leaks),
        "format": not unknowable
        and bool(e.get("table_columns"))
        and not _table_ok(e["table_columns"], items, lang),
    }
    v.reason = next((r for r in REASONS if applies.get(r)), None)
    v.passed = v.reason is None
    return v


def preview(items: Sequence[Item], n: int = 3, width: int = 140) -> list[str]:
    """What came back, for a failure example."""
    return [f"{i.kind}: {' '.join(i.text.split())[:width]}" for i in items[:n]]
