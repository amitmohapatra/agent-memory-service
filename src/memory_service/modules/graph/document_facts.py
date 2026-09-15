"""Deterministic information extraction for business documents (no LLM).

Turns a parsed document into *typed* entities with resolved aliases and *factual*
relations with attributes and evidence — the difference between "Adjusted EBITDA is
mentioned on page 11" and "Adjusted EBITDA (FY26) = EUR 98 million, +21% vs FY25
(EUR 81 million), driven by restructuring savings; footnote 3 excludes a EUR 7 million
litigation settlement".

Two passes:

1. **Lexicon** (document level): named things worth linking — defined terms, metrics
   from a financial lexicon and from numeric tables, organisations, people and roles,
   locations (gazetteer), programmes/events from headings, section titles — each with
   aliases (``ARR`` for ``Recurring Revenue``, ``ACME`` for ``ACME Corporation``, the
   lowercase prose form of a heading such as "the restructuring programme").
2. **Linking + facts** (chunk level): mentions are matched case-insensitively, longest
   first, so "FY26 Adjusted EBITDA" and "adjusted EBITDA" resolve to one entity; sentences
   are then mined with a small grammar of business statements: metric values and changes
   (with period, previous value, direction), table cells (row metric × column period),
   causes ("driven by", "reflects", "due to"), definitions and exclusions, organisation
   facts (provides / serves / operates in / approved by / acquired / closed on / employs),
   people and roles, cross-references, and counterfactuals ("would have been" is kept
   apart as ``would_have_value`` so a hypothetical never masquerades as the actual figure).

Precision over recall: every rule is anchored on an explicit cue, every fact keeps the
sentence it came from, and nothing is inferred across sentences except "the increase /
the decline" referring to the metric of the previous sentence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import Representation
from memory_service.modules.ingestion.context_graph import (
    canonical_entity,
    extract_definitions,
    extract_entities,
)

# --------------------------------------------------------------------------- lexicons

_METRIC_LEXICON = """
revenue|total revenue|net revenue|recurring revenue|annual recurring revenue|arr|mrr|
subscription revenue|services revenue|bookings|billings|backlog|gross profit|gross margin|
operating profit|operating income|operating margin|adjusted operating profit|
adjusted operating margin|ebitda|adjusted ebitda|ebitda margin|net income|net profit|
net loss|profit before tax|earnings per share|eps|free cash flow|operating cash flow|
cash flow|capital expenditure|capex|operating expenses|opex|headcount|employees|
churn|net revenue retention|nrr|arpu|customer count|customers|net debt|total debt|
leverage|dividend|restructuring charges|restructuring costs|restructuring savings|
integration costs|impairment charges|impairment|acquisition costs|litigation costs|
litigation settlement|warranty provision|share-based compensation|
depreciation and amortisation|depreciation and amortization|
annualised savings|annualized savings|cost savings|synergies|expected synergies|
input prices|guidance|market share|order intake|working capital|inventory
"""
METRICS: frozenset[str] = frozenset(
    t.strip() for t in _METRIC_LEXICON.replace("\n", "").split("|") if t.strip()
)

_LOCATIONS = """
europe|north america|south america|latin america|asia|asia-pacific|asia pacific|apac|emea|
middle east|africa|oceania|nordics|benelux|dach|united states|u.s.|us|usa|united kingdom|uk|
germany|france|italy|spain|netherlands|belgium|switzerland|austria|sweden|norway|denmark|
finland|ireland|poland|portugal|canada|mexico|brazil|argentina|chile|india|china|japan|
south korea|korea|singapore|australia|new zealand|indonesia|vietnam|thailand|malaysia|
philippines|israel|uae|united arab emirates|saudi arabia|south africa|nigeria|egypt|turkey|
russia|ukraine|czech republic|hungary|romania|greece|london|new york|paris|berlin|munich|
frankfurt|zurich|amsterdam|dublin|madrid|milan|stockholm|singapore|tokyo|sydney|toronto|
san francisco|boston|chicago|austin|seattle|bangalore|bengaluru|mumbai|delhi|hyderabad
"""
LOCATIONS: frozenset[str] = frozenset(
    t.strip() for t in _LOCATIONS.replace("\n", "").split("|") if t.strip()
)

_ORG_SUFFIX = (
    r"(?:Corporation|Corp\.?|Incorporated|Inc\.?|Limited|Ltd\.?|GmbH|AG|SE|SA|S\.A\.|N\.V\.|"
    r"plc|PLC|LLC|LLP|Group|Holdings|Company|Co\.|Bank|Partners|Industries|Technologies|"
    r"Systems|Software|Labs|Ventures|Capital|Networks|Solutions|Services|Energy|Motors|"
    r"Pharmaceuticals|Foundation|Institute|University|Authority|Agency)"
)
_ORG_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9&\-]*\s+){0,4}[A-Z][A-Za-z0-9&\-]*\s+"
    + _ORG_SUFFIX
    + r")(?=[\s,.;:)]|$)"
)
_BOARD_RE = re.compile(r"\b(?:the\s+)?(Board(?:\s+of\s+Directors)?)\b")
_ROLE_WORDS = (
    r"(?:Chief\s+[A-Z][a-z]+\s+Officer|Chief\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+Officer|CEO|CFO|COO|"
    r"CTO|CIO|CMO|CRO|Chairman|Chairwoman|Chair|Vice\s+Chair|President|Vice\s+President|"
    r"Managing\s+Director|Executive\s+Director|Non-Executive\s+Director|Director|"
    r"General\s+Counsel|Company\s+Secretary|Treasurer|Head\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|"
    r"Founder|Co-Founder|Partner|General\s+Manager)"
)
_NAME = r"(?:(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,2}"
_PERSON_ROLE_RES = [
    # "Jane Doe, Chief Financial Officer[ of ACME]"
    re.compile(
        r"\b("
        + _NAME
        + r"),\s+(?:the\s+)?("
        + _ROLE_WORDS
        + r")(?:\s+of\s+([A-Z][A-Za-z0-9&\-\s]{2,40}?))?(?=[\s,.;:)]|$)"
    ),
    # "Chief Financial Officer Jane Doe" / "CEO Jane Doe"
    re.compile(r"\b(" + _ROLE_WORDS + r")\s+(" + _NAME + r")\b"),
    # "appointed Jane Doe as Chief Financial Officer"
    re.compile(r"\bappointed\s+(" + _NAME + r")\s+(?:as|to)\s+(?:the\s+)?(" + _ROLE_WORDS + r")\b"),
    # "Jane Doe (CEO)"
    re.compile(r"\b(" + _NAME + r")\s+\((" + _ROLE_WORDS + r")\)"),
]
_EVENT_HEADING_RE = re.compile(
    r"^(?:(?:[A-Z][A-Za-z\-]+\s+){0,3}(?:Programme|Program|Plan|Initiative|Project|Transaction|"
    r"Merger|Offering|IPO|Divestment|Divestiture|Spin-off|Restructuring|Reorganisation|"
    r"Reorganization)|(?:Acquisition|Disposal|Sale|Purchase|Merger)\s+of\s+[A-Z][A-Za-z0-9&\-\s]{1,40})$"
)
_SECTION_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+")
_PERIOD_RE = re.compile(
    r"\b(FY\s?\d{2,4}|Q[1-4](?:\s?FY\s?\d{2,4}|\s?\d{4})?|H[12](?:\s?FY\s?\d{2,4}|\s?\d{4})?|"
    r"(?:19|20)\d{2})\b"
)
_MONEY_RE = re.compile(
    r"(?P<cur>EUR|USD|GBP|CHF|JPY|INR|CAD|AUD|SEK|NOK|DKK|€|\$|£)\s?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"\s?(?P<scale>trillion|billion|million|thousand|tn|bn|mn|m|k)?\b"
)
_PERCENT_RE = re.compile(r"(?P<sign>[+\-−]?)(?P<num>\d+(?:\.\d+)?)\s?%")
_COUNT_RE = re.compile(
    r"\b(?P<num>\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|"
    r"twenty|thirty|fifty|hundred)\s+(?P<unit>data\s+cent(?:re|er)s|employees|people|staff|"
    r"customers|sites|plants|offices|stores|countries|branches|facilities|warehouses|"
    r"factories|subsidiaries|patents)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)(?:\s+\d{4})?|(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?|(?:January|February|"
    r"March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?)\b"
)
_CHANGE_VERB_RE = re.compile(
    r"\b(increased|rose|grew|climbed|improved|expanded|jumped|surged|decreased|fell|declined|"
    r"dropped|contracted|shrank|slipped|reduced|narrowed|widened)\b",
    re.IGNORECASE,
)
_UP = {
    "increased",
    "rose",
    "grew",
    "climbed",
    "improved",
    "expanded",
    "jumped",
    "surged",
    "widened",
}
_CHANGE_NOUN_RE = re.compile(
    r"\b(?:an?\s+)?(increase|decrease|improvement|decline|growth|reduction|rise|fall|drop)\s+of\s+"
    r"([+\-−]?\d+(?:\.\d+)?\s?%)",
    re.IGNORECASE,
)
_CAUSE_RE = re.compile(
    r"\b(?:driven by|reflects|reflecting|due to|because of|as a result of|owing to|"
    r"attributable to|primarily (?:from|due to)|mainly (?:from|due to)|on the back of|"
    r"supported by|offset by|impacted by|helped by|hurt by)\s+(.+?)(?:[.;]|,\s+(?:and|which|while)\b|$)",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(
    r"\b(would have been|had\s+\w+(?:\s+\w+)*\s+been|if\s+.+?\s+had)\b", re.IGNORECASE
)
_EXCLUDES_RE = re.compile(
    r"\b(?:adjusted\s+to\s+exclude|excludes?|excluding|net\s+of)\s+(.+?)(?:[.;]|\band\s+(?:is|are)\b|$)",
    re.IGNORECASE,
)
_EXCLUDED_FROM_RE = re.compile(
    r"\b(?:is|are|were|was)\s+excluded\s+from\s+(.+?)(?:\s+per\b|[.;,]|$)", re.IGNORECASE
)
_PROVIDES_RE = re.compile(
    r"\b(provides|offers|sells|manufactures|develops|produces|delivers|supplies|operates|builds|"
    r"designs)\s+(.+?)(?:\s+(?:to|for)\s+(.+?))?(?:\s+in\s+([A-Z][A-Za-z\-]+(?:(?:,\s*|\s+and\s+)"
    r"[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)*(?:\s+[A-Z][A-Za-z\-]+)?))?(?=[.;,]|$)"
)
_APPROVED_RE = re.compile(
    r"\bapproved\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+in\s+("
    + _PERIOD_RE.pattern[2:-2]
    + r"))?(?=\s*\(see|[.;,]|$)",
    re.IGNORECASE,
)
_REDUCED_RE = re.compile(
    r"\b(reduced|cut|lowered|increased|raised|grew|expanded)\s+(?:its\s+|the\s+)?([a-z][a-z\s\-]{2,40}?)\s+by\s+([+\-−]?\d+(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?(?:\s+\w+)?)",
    re.IGNORECASE,
)
_CONSOLIDATED_RE = re.compile(
    r"\b(consolidated|closed|opened|added|divested|sold|acquired)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z][a-z\s\-]{2,40}?)(?=[.;,]|\s+(?:in|during|across)\b|$)",
    re.IGNORECASE,
)
_CLOSED_ON_RE = re.compile(
    r"\bclosed\s+on\s+(.+?)(?:\s+for\s+(" + _MONEY_RE.pattern + r"))?(?=[.;,]|$)", re.IGNORECASE
)
_EMPLOYS_RE = re.compile(
    r"\bemploys\s+(?:approximately\s+|about\s+|around\s+|over\s+)?(\d[\d,]*)\s+(?:people|employees|staff)\b",
    re.IGNORECASE,
)
_FOUNDED_RE = re.compile(
    r"\b(?:founded|established|incorporated)\s+in\s+((?:19|20)\d{2})\b", re.IGNORECASE
)
_HQ_RE = re.compile(
    r"\b(?:headquartered|based)\s+in\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)", re.IGNORECASE
)
_ACQUIRED_RE = re.compile(
    r"\b(acquired|acquisition of|merged with|merger with|partnered with|partnership with|invested in|divested|sold)\s+([A-Z][A-Za-z0-9&\-]+(?:\s+[A-Z][A-Za-z0-9&\-]+){0,3})"
)
_XREF_RE = re.compile(
    r"\b(?:see\s+)?(Section|Note|Table|Appendix|Chapter)\s+(\d+(?:\.\d+)*|[A-Z])\b"
)
_ADOPTED_RE = re.compile(
    r"\bchange\s+in\s+(?:the\s+)?(.+?)\s+adopted\s+in\s+(" + _PERIOD_RE.pattern[2:-2] + r")",
    re.IGNORECASE,
)
_TABLE_TITLE_RE = re.compile(r"^(Table|Figure|Exhibit)\s+(\d+)\s*[:.\-–]\s*(.+)$", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"“])|\n{2,}")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-'&]*")
_ANAPHORA = (
    "the increase",
    "the decrease",
    "the decline",
    "the improvement",
    "the growth",
    "the reduction",
    "the rise",
    "the fall",
    "the drop",
    "this increase",
    "this decrease",
    "this decline",
)

_SCALE = {
    "trillion": 1e12,
    "tn": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}
_CUR = {"€": "EUR", "$": "USD", "£": "GBP"}
_SMALL_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "fifty": 50,
    "hundred": 100,
}
_GENERIC_WORDS = frozenset(
    "table figure section page note appendix chapter total item change segment fy eur usd "
    "million billion m bn k percent".split()
)
_CURRENCIES = frozenset({"eur", "usd", "gbp", "chf", "jpy", "inr", "cad", "aud"})
_GENERIC_HEADINGS = {
    "definitions",
    "business overview",
    "overview",
    "introduction",
    "financial results",
    "results",
    "notes to the financial statements",
    "notes",
    "summary",
    "contents",
    "highlights",
    "outlook",
    "risk factors",
    "governance",
}


# --------------------------------------------------------------------------- types


@dataclass
class LexEntity:
    name: str
    type: str
    canonical: str
    aliases: set[str] = field(default_factory=set)
    definition: str | None = None
    definition_node_id: str | None = None
    node_id: str | None = None
    page: int | None = None

    def all_forms(self) -> set[str]:
        return {self.canonical} | {a for a in self.aliases if a}


@dataclass
class Mention:
    entity: LexEntity
    start: int
    end: int
    surface: str


@dataclass
class Value:
    kind: str  # MONEY | PERCENT | COUNT | DATE
    display: str
    canonical: str
    start: int
    end: int
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    subject: LexEntity
    predicate: str
    object: LexEntity | Value
    text: str
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    chunk: Chunk | None = None


# --------------------------------------------------------------------------- helpers


def strip_number(title: str) -> str:
    return _SECTION_NUMBER.sub("", title or "").strip()


def acronym_of(name: str) -> str | None:
    words = [w for w in _WORD.findall(name) if w[0].isupper()]
    if len(words) < 2:
        return None
    return "".join(w[0] for w in words).upper()


def money_value(m: re.Match[str]) -> Value:
    cur = _CUR.get(m.group("cur"), m.group("cur"))
    num = float(m.group("num").replace(",", ""))
    scale = (m.group("scale") or "").lower()
    amount = num * _SCALE.get(scale, 1)
    word = {
        "tn": "trillion",
        "bn": "billion",
        "mn": "million",
        "m": "million",
        "k": "thousand",
    }.get(scale, scale)
    shown = m.group("num")
    display = f"{cur} {shown}{(' ' + word) if word else ''}"
    return Value(
        "MONEY",
        display,
        canonical_entity(display),
        m.start(),
        m.end(),
        {"currency": cur, "amount": amount},
    )


def percent_value(m: re.Match[str]) -> Value:
    sign = "-" if m.group("sign") in ("-", "−") else ("+" if m.group("sign") == "+" else "")
    num = m.group("num")
    display = f"{sign}{num}%"
    return Value(
        "PERCENT", display, display.lower(), m.start(), m.end(), {"percent": float(f"{sign}{num}")}
    )


def values_in(sentence: str) -> list[Value]:
    out: list[Value] = [money_value(m) for m in _MONEY_RE.finditer(sentence)]
    taken = [(v.start, v.end) for v in out]

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for m in _PERCENT_RE.finditer(sentence):
        if free(m.start(), m.end()):
            out.append(percent_value(m))
            taken.append((m.start(), m.end()))
    for m in _COUNT_RE.finditer(sentence):
        if free(m.start(), m.end()):
            raw = m.group("num").lower()
            n = _SMALL_NUMBERS.get(raw) or int(raw.replace(",", ""))
            unit = re.sub(r"\s+", " ", m.group("unit").lower())
            display = f"{n} {unit}"
            out.append(
                Value("COUNT", display, display, m.start(), m.end(), {"count": n, "unit": unit})
            )
            taken.append((m.start(), m.end()))
    for m in _DATE_RE.finditer(sentence):
        if free(m.start(), m.end()):
            out.append(Value("DATE", m.group(1), m.group(1).lower(), m.start(), m.end(), {}))
            taken.append((m.start(), m.end()))
    return sorted(out, key=lambda v: v.start)


def period_in(text: str) -> str | None:
    m = _PERIOD_RE.search(text)
    return m.group(1).replace(" ", "").upper() if m else None


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        p = re.sub(r"^\[\^\d+\]:\s*", "", p)  # footnote marker
        p = re.sub(r"\[\^\d+\]", "", p)
        p = p.strip("*#> ").strip()
        if len(_WORD.findall(p)) >= 3:
            out.append(p)
    return out


def clean_phrase(text: str, *, max_words: int = 8) -> str:
    t = re.sub(r"\s+", " ", text).strip(" ,.;:()\"'")
    t = re.sub(
        r"^(?:the|a|an|its|their|our|higher|lower|strong|weak)\s+", "", t, flags=re.IGNORECASE
    )
    t = (
        re.sub(
            r"\s+(?:unit|division|segment|business|programme|program)$", "", t, flags=re.IGNORECASE
        )
        if len(t.split()) > 2
        else t
    )
    words = t.split()
    return " ".join(words[:max_words])


def parse_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    rows = [r.strip() for r in text.strip().splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return None
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    header = cells[0]
    body = [r for r in cells[1:] if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
    return header, body


def is_numeric_cell(c: str) -> bool:
    return bool(re.fullmatch(r"[+\-−]?\(?\d[\d,]*(?:\.\d+)?\)?\s?%?", c.strip()))


# --------------------------------------------------------------------------- extractor


class DocumentIE:
    def __init__(
        self,
        nodes: Sequence[DocumentNode],
        chunks: Sequence[Chunk],
        *,
        document_title: str = "",
    ) -> None:
        self.nodes = list(nodes)
        self.chunks = list(chunks)
        self.title = document_title or ""
        self.lexicon: dict[str, LexEntity] = {}  # canonical -> entity
        self.forms: dict[str, LexEntity] = {}  # every alias form -> entity
        self.sections: dict[str, LexEntity] = {}  # section number -> entity
        self.doc_period = period_in(self.title)
        seen_text: set[str] = set()
        pieces: list[str] = []
        for t in [n.text for n in self.nodes if n.text] + [c.text for c in self.chunks if c.text]:
            if t not in seen_text:
                seen_text.add(t)
                pieces.append(t)
        self.full_text = "\n".join(pieces)
        self.lower_words = set(w.lower() for w in _WORD.findall(self.full_text) if w[0].islower())
        self.facts: list[Fact] = []
        self.mentions: dict[str, list[tuple[Chunk, Mention]]] = {}
        self.primary_org: LexEntity | None = None
        self._build_lexicon()

    # -- lexicon --------------------------------------------------------------------
    def add(
        self,
        name: str,
        type_: str,
        *,
        aliases: Iterable[str] = (),
        definition: str | None = None,
        node_id: str | None = None,
        page: int | None = None,
    ) -> LexEntity:
        name = re.sub(r"\s+", " ", name).strip(" *\"'“”")
        canon = canonical_entity(name)
        if not canon:
            raise ValueError("empty entity name")
        existing = self.forms.get(canon)
        if existing is None:
            existing = LexEntity(name=name, type=type_, canonical=canon, node_id=node_id, page=page)
            self.lexicon[canon] = existing
            self.forms[canon] = existing
        elif existing.type in ("THING", "SECTION") and type_ not in ("THING", "SECTION"):
            existing.type = type_
        if definition and not existing.definition:
            existing.definition = definition
            existing.definition_node_id = node_id
        for a in aliases:
            ac = canonical_entity(a)
            if ac and ac != existing.canonical and ac not in self.forms:
                existing.aliases.add(ac)
                self.forms[ac] = existing
        return existing

    def _build_lexicon(self) -> None:
        text = self.full_text
        # 1. section headings -> SECTION (and PROGRAMME/EVENT when the heading names one)
        for n in self.nodes:
            if n.representation in (Representation.SECTION, Representation.SUBSECTION) and n.title:
                title = strip_number(n.title)
                number = _SECTION_NUMBER.match(n.title)
                if not title:
                    continue
                if _EVENT_HEADING_RE.match(title):
                    ent = self.add(title, "EVENT", node_id=n.node_id, page=n.page_start)
                    if title.lower().startswith(
                        ("acquisition of", "disposal of", "sale of", "purchase of", "merger of")
                    ):
                        ent.aliases.add(canonical_entity("the " + title))
                        self.forms.setdefault(canonical_entity("the " + title), ent)
                        target = title.split(" of ", 1)[1].strip()
                        if target and target[0].isupper():
                            org = self.add(target, "ORG")
                            self.facts.append(
                                Fact(ent, "involves", org, title, 0.9, {"role": "target"})
                            )
                else:
                    ent = self.add(title, "SECTION", node_id=n.node_id, page=n.page_start)
                if number:
                    self.sections[number.group(0).strip().rstrip(".")] = ent
        # 2. definitions -> METRIC (financial) or TERM, with "(ABBR)" aliases
        for n in self.nodes:
            if not n.text or n.representation is Representation.DOCUMENT:
                continue
            for term in extract_definitions(n.text):
                canon = canonical_entity(term)
                if (
                    len(canon) < 2
                    or _PERIOD_RE.fullmatch(term)
                    or re.fullmatch(
                        r"(?:eur|usd|gbp|chf|jpy|inr)\s*(?:m|mn|bn|k|million|billion|thousand)?",
                        canon,
                    )
                ):
                    continue
                type_ = (
                    "METRIC"
                    if (
                        canon in METRICS
                        or any(
                            canon.endswith(" " + m) or canon.startswith(m + " ") for m in METRICS
                        )
                    )
                    else "TERM"
                )
                first = n.text.strip().split("\n", 1)[0].strip("* ")
                ent = self.add(
                    term,
                    type_ if not term.isupper() else "TERM",
                    definition=first[:400],
                    node_id=n.node_id,
                    page=n.page_start,
                )
                # "Recurring Revenue ("ARR")" -> alias; an ALL-CAPS defined term is the alias
                # of the long form defined in the same line
                m = re.search(r"\(\"?([A-Z][A-Za-z0-9 ]{1,30}?)\"?\)", first)
                if m and canonical_entity(m.group(1)) != canon:
                    if term.isupper():
                        continue
                    ent.aliases.add(canonical_entity(m.group(1)))
                    self.forms[canonical_entity(m.group(1))] = ent
        for canon, ent in list(self.lexicon.items()):
            if ent.name.isupper() and ent.type == "TERM":
                # an ALL-CAPS term defined alongside a long form is folded into it
                for other in self.lexicon.values():
                    if other is not ent and canon in other.aliases:
                        self.lexicon.pop(canon, None)
                        break
        # 3. organisations
        for m in _ORG_RE.finditer(text):
            name = m.group(1).strip()
            ent = self.add(name, "ORG")
            acr = acronym_of(name)
            head = name.split()[0]
            if head.isupper() and len(head) >= 3:
                ent.aliases.add(head.lower())
                self.forms.setdefault(head.lower(), ent)
            if acr and len(acr) >= 3 and re.search(r"\b" + re.escape(acr) + r"\b", text):
                ent.aliases.add(acr.lower())
                self.forms.setdefault(acr.lower(), ent)
        for m in _BOARD_RE.finditer(text):
            self.add(
                "The Board",
                "ORG",
                aliases=["board", "board of directors", "the board of directors"],
            )
        for m in _ACQUIRED_RE.finditer(text):
            target = m.group(2).strip()
            if canonical_entity(target) not in self.forms and target.split()[0] not in (
                "The",
                "Section",
                "Note",
            ):
                self.add(target, "ORG")
        self.primary_org = self._primary_org()
        # 4. people + roles
        for pat in _PERSON_ROLE_RES:
            for m in pat.finditer(text):
                g = m.groups()
                if pat is _PERSON_ROLE_RES[1]:
                    role, person = g[0], g[1]
                    org = None
                else:
                    person, role = g[0], g[1]
                    org = g[2] if len(g) > 2 else None
                person = re.sub(r"^(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+", "", person)
                p = self.add(person, "PERSON")
                r = self.add(re.sub(r"\s+", " ", role), "ROLE")
                self.facts.append(
                    Fact(p, "has_role", r, m.group(0), 0.85, {"org": org.strip() if org else None})
                )
                if org:
                    o = self.add(org.strip(), "ORG")
                    self.facts.append(Fact(p, "works_at", o, m.group(0), 0.85))
        # 5. locations (gazetteer, longest first)
        low = text.lower()
        for loc in sorted(LOCATIONS, key=len, reverse=True):
            if re.search(r"(?<![a-z\-])" + re.escape(loc) + r"(?![a-z\-])", low):
                name = " ".join(w.capitalize() if len(w) > 2 else w.upper() for w in loc.split())
                name = name.replace("Asia-pacific", "Asia-Pacific").replace("Of", "of")
                self.add(name, "LOCATION")
        # 6. metrics from the financial lexicon (present in the text) and numeric tables
        for metric in sorted(METRICS, key=len, reverse=True):
            m = re.search(r"(?<![a-z\-])" + re.escape(metric) + r"(?![a-z\-])", low)
            if m:
                surface = text[m.start() : m.end()]
                if surface.islower():
                    surface = surface[0].upper() + surface[1:]
                self.add(surface, "METRIC")
        # "total revenue" is the same metric as "revenue" when both occur
        for canon in list(self.lexicon):
            if (
                canon.startswith("total ")
                and canon[6:] in self.lexicon
                and self.lexicon[canon].type == "METRIC"
            ):
                whole = self.lexicon.pop(canon)
                base = self.lexicon[canon[6:]]
                base.aliases.add(canon)
                self.forms[canon] = base
                for a in whole.aliases:
                    base.aliases.add(a)
                    self.forms[a] = base
        for n in self.nodes:
            if n.representation is Representation.TABLE and n.text:
                self._table_lexicon(n)
        # 7. generic capitalised multi-word names not typed above ("Legacy Services unit",
        #    "Restructuring Programme" without a heading) -> THING / EVENT; single common
        #    words that also appear in lower case elsewhere are not names
        candidates: list[str] = list(extract_entities(text, max_entities=60))
        for c in self.chunks:
            candidates.extend(c.entities or [])
        for name in candidates:
            canon = canonical_entity(name)
            words = canon.split()
            if canon in self.forms or not words:
                continue
            if _PERIOD_RE.fullmatch(name) or _MONEY_RE.fullmatch(name) or canon in _GENERIC_WORDS:
                continue
            if _PERIOD_RE.search(name) or any(
                re.search(r"(?<![a-z0-9\-])" + re.escape(f) + r"(?![a-z0-9\-])", canon)
                for f in self.forms
                if len(f) >= 4
            ):
                continue  # "FY26 Adjusted EBITDA", "Europe and North America": already covered
            if len(words) == 1:
                if not (name.isupper() and 3 <= len(name) <= 6) or canon in _CURRENCIES:
                    continue
            elif any(w in _GENERIC_WORDS for w in words) or words[0] in ("the", "a", "an"):
                continue
            elif (
                name.split()[0].lower() in self.lower_words
                and name.split()[-1].lower() in self.lower_words
                and len(words) == 2
            ):
                continue  # "Total revenue"-style common words are not names
            type_ = "EVENT" if _EVENT_HEADING_RE.match(name) else "THING"
            self.add(name, type_)

    def _primary_org(self) -> LexEntity | None:
        """The organisation the document is about: named in the title, else the first one."""
        orgs = [e for e in self.lexicon.values() if e.type == "ORG" and e.canonical != "the board"]
        title = self.title.lower()
        for e in orgs:
            if any(form in title for form in e.all_forms() if len(form) >= 3):
                return e
        return orgs[0] if orgs else None

    def _table_lexicon(self, node: DocumentNode) -> None:
        parsed = parse_table(node.text)
        if not parsed:
            return
        header, body = parsed
        caption = strip_number(node.title or "")
        m = _TABLE_TITLE_RE.match(caption)
        subject = m.group(3).strip() if m else caption
        by_segment = bool(
            re.search(
                r"\bby\s+(segment|region|geography|business|product|unit|category)\b",
                subject,
                re.IGNORECASE,
            )
        )
        for row in body:
            if not row or not row[0]:
                continue
            label = row[0].strip(" *")
            if (
                not label
                or canonical_entity(label) == "total"
                or not any(is_numeric_cell(c) for c in row[1:])
            ):
                continue
            type_ = "SEGMENT" if by_segment else "METRIC"
            ent = self.add(label, type_)
            if by_segment and ent.type == "ORG":
                ent.type = "SEGMENT"

    # -- linking --------------------------------------------------------------------
    def link(self, text: str) -> list[Mention]:
        """Longest-first, case-insensitive dictionary matching with word boundaries."""
        forms = sorted(self.forms.keys(), key=len, reverse=True)
        low = text.lower()
        taken: list[tuple[int, int]] = []
        out: list[Mention] = []
        for form in forms:
            if len(form) < 2:
                continue
            stem = form[:-1] if form.endswith("s") and len(form) > 4 else form
            for m in re.finditer(
                r"(?<![a-z0-9\-])" + re.escape(stem) + r"(?:s|es)?(?![a-z0-9\-])", low
            ):
                if any(m.start() < e and m.end() > s for s, e in taken):
                    continue
                ent = self.forms[form]
                if ent.type == "SECTION" and ent.canonical in _GENERIC_HEADINGS:
                    continue
                taken.append((m.start(), m.end()))
                out.append(Mention(ent, m.start(), m.end(), text[m.start() : m.end()]))
        return sorted(out, key=lambda x: x.start)

    # -- facts ----------------------------------------------------------------------
    def run(self) -> None:
        for c in self.chunks:
            node = next((n for n in self.nodes if n.node_id == c.node_id), None)
            if node is not None and node.representation is Representation.TABLE:
                self._table_facts(c, node)
                for men in self.link(c.text):
                    self.mentions.setdefault(men.entity.canonical, []).append((c, men))
                continue
            section = self._section_of(c)
            last_metric: LexEntity | None = None
            for sent in sentences(c.text):
                mentions = self.link(sent)
                for men in mentions:
                    self.mentions.setdefault(men.entity.canonical, []).append((c, men))
                last_metric = self._sentence_facts(sent, mentions, c, section, last_metric)
        used = {f.subject.canonical for f in self.facts} | {
            f.object.canonical for f in self.facts if isinstance(f.object, LexEntity)
        }
        for canon, ent in list(self.lexicon.items()):
            if ent.type in ("MONEY", "PERCENT", "COUNT", "DATE", "NUMBER", "PERIOD"):
                continue
            if canon not in self.mentions and canon not in used:
                self.lexicon.pop(canon)

    def _section_of(self, c: Chunk) -> LexEntity | None:
        parts = [p.strip() for p in (c.section_path or "").split(">") if p.strip()]
        for p in reversed(parts):
            ent = self.forms.get(canonical_entity(strip_number(p)))
            if ent is not None and (ent.node_id is not None or ent.type in ("SECTION", "EVENT")):
                return ent
        return None

    def _period(self, sent: str, c: Chunk) -> str | None:
        return period_in(sent) or period_in(c.section_path or "") or self.doc_period

    def _value_entity(self, v: Value) -> LexEntity:
        canon = v.canonical
        ent = self.lexicon.get(canon)
        if ent is None:
            ent = LexEntity(name=v.display, type=v.kind, canonical=canon)
            self.lexicon[canon] = ent
        return ent

    def _factor(self, phrase: str, mentions_in_phrase: list[Mention]) -> LexEntity | None:
        if mentions_in_phrase:
            # prefer a named thing inside the phrase over a free-text factor
            best = max(mentions_in_phrase, key=lambda m: m.end - m.start)
            if best.entity.type not in ("SECTION",):
                return best.entity
        cleaned = clean_phrase(phrase)
        if len(cleaned) < 4 or len(cleaned.split()) > 8:
            return None
        canon = canonical_entity(cleaned)
        if canon in self.forms:
            return self.forms[canon]
        return self.add(cleaned, "FACTOR")

    def _sentence_facts(
        self,
        sent: str,
        mentions: list[Mention],
        c: Chunk,
        section: LexEntity | None,
        last_metric: LexEntity | None,
    ) -> LexEntity | None:
        vals = values_in(sent)
        period = self._period(sent, c)
        hypothetical = bool(_HYPOTHETICAL_RE.search(sent))
        metrics = [m for m in mentions if m.entity.type in ("METRIC", "SEGMENT")]
        low = sent.lower()
        subject_metric: LexEntity | None = metrics[0].entity if metrics else None
        if any(low.startswith(a) for a in _ANAPHORA) and last_metric is not None:
            subject_metric = last_metric
            metrics = []  # the sentence talks about the previous metric
        elif (
            not metrics
            and last_metric is not None
            and re.search(r"\bthe (?:margin|figure|measure|metric|ratio|total|number)\b", low)
        ):
            subject_metric = last_metric
        # -- metric values and changes ------------------------------------------
        if metrics or (subject_metric is not None and vals):
            self._value_facts(sent, metrics, vals, c, period, hypothetical, subject_metric)
        # -- causes ------------------------------------------------------------------
        for m in _CAUSE_RE.finditer(sent):
            if subject_metric is None:
                break
            phrase = m.group(1)
            offset = m.start(1)
            parts = re.split(
                r"\s+and\s+(?=(?:the|a|an|higher|lower)\b|[a-z]+\s+(?:costs?|prices?|savings?|unit|demand|growth))",
                phrase,
            )
            for part in parts:
                pstart = phrase.find(part) + offset
                inside = [
                    mm
                    for mm in mentions
                    if mm.start >= pstart
                    and mm.end <= pstart + len(part)
                    and mm.entity is not subject_metric
                ]
                factor = self._factor(part, inside)
                if factor is None or factor is subject_metric:
                    continue
                direction = self._direction(sent)
                self.facts.append(
                    Fact(
                        subject_metric,
                        "driven_by",
                        factor,
                        sent,
                        0.75,
                        {"period": period, "direction": direction, "hypothetical": hypothetical},
                        c,
                    )
                )
        # -- definitions / exclusions ------------------------------------------------
        for men in mentions:
            ent = men.entity
            if (
                ent.definition
                and ent.definition_node_id == c.node_id
                and low.startswith(ent.canonical)
            ):
                for m in _EXCLUDES_RE.finditer(sent):
                    for item in re.split(r",\s*|\s+and\s+", m.group(1)):
                        item_c = clean_phrase(item, max_words=5)
                        if not item_c or len(item_c) < 4:
                            continue
                        target = self.forms.get(canonical_entity(item_c)) or self.add(
                            item_c, "METRIC" if canonical_entity(item_c) in METRICS else "TERM"
                        )
                        self.facts.append(
                            Fact(ent, "excludes", target, sent, 0.85, {"source": "definition"}, c)
                        )
                break
        m = _EXCLUDED_FROM_RE.search(sent)
        if m:
            target_mentions = [mm for mm in self.link(m.group(1))]
            excluded_from = target_mentions[0].entity if target_mentions else None
            what = (
                metrics[0].entity
                if metrics and (excluded_from is None or metrics[0].entity is not excluded_from)
                else None
            )
            if excluded_from is not None and what is not None:
                value = next((v for v in vals if v.kind == "MONEY"), None)
                self.facts.append(
                    Fact(
                        excluded_from,
                        "excludes",
                        what,
                        sent,
                        0.85,
                        {"period": period, "value": value.display if value else None},
                        c,
                    )
                )
        m = re.search(
            r"\bexcludes\s+an?\s+(" + _MONEY_RE.pattern + r")\s+([a-z][a-z\s\-]{2,40}?)(?=[.;,]|$)",
            sent,
            re.IGNORECASE,
        )
        if m and metrics:
            item = clean_phrase(m.group(m.re.groups), max_words=5)
            target = self.forms.get(canonical_entity(item)) or self.add(
                item, "METRIC" if canonical_entity(item) in METRICS else "TERM"
            )
            mv = money_value(_MONEY_RE.search(m.group(1)))  # type: ignore[arg-type]
            self.facts.append(
                Fact(
                    metrics[0].entity,
                    "excludes",
                    target,
                    sent,
                    0.85,
                    {"period": period, "value": mv.display},
                    c,
                )
            )
            self.facts.append(
                Fact(
                    target,
                    "has_value",
                    self._value_entity(mv),
                    sent,
                    0.8,
                    {"period": period, **mv.attributes},
                    c,
                )
            )
        # -- organisation / programme facts --------------------------------------------
        orgs = [mm for mm in mentions if mm.entity.type == "ORG"]
        events = [mm for mm in mentions if mm.entity.type == "EVENT"]
        subject_org = orgs[0].entity if orgs and orgs[0].start < 40 else None
        subject_event = events[0].entity if events and events[0].start < 40 else None
        actor = subject_org or subject_event
        if subject_org is not None:
            m = _PROVIDES_RE.search(sent)
            if m and m.start() > orgs[0].end - 1:
                offering = clean_phrase(m.group(2), max_words=6)
                if offering:
                    self.facts.append(
                        Fact(
                            subject_org,
                            "provides",
                            self.add(offering, "OFFERING"),
                            sent,
                            0.8,
                            {},
                            c,
                        )
                    )
                if m.group(3):
                    customers = clean_phrase(m.group(3).split(" in ")[0], max_words=6)
                    if customers and not customers[0].isupper():
                        self.facts.append(
                            Fact(
                                subject_org,
                                "serves",
                                self.add(customers, "CUSTOMER_SEGMENT"),
                                sent,
                                0.75,
                                {},
                                c,
                            )
                        )
                for loc in [mm for mm in mentions if mm.entity.type == "LOCATION"]:
                    self.facts.append(
                        Fact(subject_org, "operates_in", loc.entity, sent, 0.8, {}, c)
                    )
            m = _EMPLOYS_RE.search(sent)
            if m:
                head = self.add("Headcount", "METRIC")
                v = Value(
                    "COUNT",
                    f"{m.group(1)} employees",
                    f"{m.group(1)} employees",
                    0,
                    0,
                    {"count": int(m.group(1).replace(",", ""))},
                )
                self.facts.append(
                    Fact(
                        head,
                        "has_value",
                        self._value_entity(v),
                        sent,
                        0.85,
                        {"period": period, "org": subject_org.name},
                        c,
                    )
                )
            m = _FOUNDED_RE.search(sent)
            if m:
                self.facts.append(
                    Fact(
                        subject_org,
                        "founded_in",
                        self._value_entity(Value("DATE", m.group(1), m.group(1), 0, 0)),
                        sent,
                        0.85,
                        {},
                        c,
                    )
                )
            m = _HQ_RE.search(sent)
            if m:
                self.facts.append(
                    Fact(
                        subject_org,
                        "headquartered_in",
                        self.add(m.group(1), "LOCATION"),
                        sent,
                        0.85,
                        {},
                        c,
                    )
                )
        for m in _ACQUIRED_RE.finditer(sent):
            target = self.forms.get(canonical_entity(m.group(2)))
            verb = m.group(1).lower()
            noun_form = verb.endswith(" of") or verb.endswith(" with")
            # "X acquired Y" -> X; "the acquisition of Y" -> the document's own organisation
            acquirer = (
                subject_org
                if (
                    subject_org is not None
                    and not noun_form
                    and subject_org.canonical != "the board"
                )
                else self.primary_org
            )
            if target is None or acquirer is None or target is acquirer:
                continue
            if noun_form and any(
                f.predicate == "acquired" and f.object is target for f in self.facts
            ):
                continue
            pred = verb.split()[0]
            pred = {
                "acquisition": "acquired",
                "merger": "merged_with",
                "merged": "merged_with",
                "partnered": "partnered_with",
                "partnership": "partnered_with",
                "invested": "invested_in",
            }.get(pred, pred)
            self.facts.append(
                Fact(
                    acquirer,
                    pred,
                    target,
                    sent,
                    0.8 if not noun_form else 0.7,
                    {"period": period},
                    c,
                )
            )
        m = _APPROVED_RE.search(sent)
        if m and actor is not None:
            approved_mentions = self.link(m.group(1))
            target = approved_mentions[0].entity if approved_mentions else None
            if target is None:
                phrase = clean_phrase(m.group(1), max_words=6)
                target = self.forms.get(canonical_entity(phrase)) or (
                    self.add(phrase, "EVENT") if phrase else None
                )
            if target is not None and target is not actor:
                self.facts.append(
                    Fact(
                        target,
                        "approved_by",
                        actor,
                        sent,
                        0.85,
                        {"period": (m.group(2) or "").replace(" ", "").upper() or period},
                        c,
                    )
                )
        if actor is not None or subject_metric is None:
            subj = (
                subject_event or subject_org or section
                if (section is not None and section.type == "EVENT")
                else actor
            )
            if subj is not None:
                for m in _REDUCED_RE.finditer(sent):
                    what = self.forms.get(canonical_entity(m.group(2))) or self.add(
                        m.group(2).strip(),
                        "METRIC" if canonical_entity(m.group(2)) in METRICS else "TERM",
                    )
                    self.facts.append(
                        Fact(
                            subj,
                            m.group(1).lower(),
                            what,
                            sent,
                            0.85,
                            {"by": m.group(3).strip(), "period": period},
                            c,
                        )
                    )
                for m in _CONSOLIDATED_RE.finditer(sent):
                    raw = m.group(2).lower()
                    n = _SMALL_NUMBERS.get(raw) or int(raw)
                    what = self.add(m.group(3).strip(), "TERM")
                    self.facts.append(
                        Fact(
                            subj,
                            m.group(1).lower(),
                            what,
                            sent,
                            0.85,
                            {"count": n, "period": period},
                            c,
                        )
                    )
        if subject_event is not None:
            m = _CLOSED_ON_RE.search(sent)
            if m:
                date = m.group(1).strip()
                self.facts.append(
                    Fact(
                        subject_event,
                        "closed_on",
                        self._value_entity(Value("DATE", date, date.lower(), 0, 0)),
                        sent,
                        0.85,
                        {"period": period},
                        c,
                    )
                )
                money = next((v for v in vals if v.kind == "MONEY"), None)
                if money is not None:
                    self.facts.append(
                        Fact(
                            subject_event,
                            "consideration",
                            self._value_entity(money),
                            sent,
                            0.85,
                            {"period": period, **money.attributes},
                            c,
                        )
                    )
        m = _ADOPTED_RE.search(sent)
        if m:
            what = self.add(clean_phrase(m.group(1), max_words=8), "TERM")
            self.facts.append(
                Fact(
                    what,
                    "adopted_in",
                    self._value_entity(
                        Value("PERIOD", m.group(2).upper(), m.group(2).lower(), 0, 0)
                    ),
                    sent,
                    0.8,
                    {},
                    c,
                )
            )
        # -- cross references ---------------------------------------------------------
        for m in _XREF_RE.finditer(sent):
            if m.group(1) == "Section":
                target = self.sections.get(m.group(2))
                src = section
                if target is not None and src is not None and target is not src:
                    self.facts.append(
                        Fact(
                            src,
                            "refers_to",
                            target,
                            sent,
                            0.9,
                            {"reference": f"Section {m.group(2)}"},
                            c,
                        )
                    )
        return subject_metric or last_metric

    def _direction(self, sent: str) -> str | None:
        m = _CHANGE_VERB_RE.search(sent)
        if m:
            return "up" if m.group(1).lower() in _UP else "down"
        m = _CHANGE_NOUN_RE.search(sent)
        if m:
            return (
                "up"
                if m.group(1).lower() in ("increase", "improvement", "growth", "rise")
                else "down"
            )
        low = sent.lower()
        if "the increase" in low or "the improvement" in low or "the growth" in low:
            return "up"
        if "the decrease" in low or "the decline" in low or "the reduction" in low:
            return "down"
        return None

    def _value_facts(
        self,
        sent: str,
        metrics: list[Mention],
        vals: list[Value],
        c: Chunk,
        period: str | None,
        hypothetical: bool,
        subject_metric: LexEntity | None,
    ) -> None:
        if not vals:
            return
        change = _CHANGE_NOUN_RE.search(sent)
        change_pct = change.group(2).replace(" ", "") if change else None
        direction = self._direction(sent)
        if change_pct and direction == "down" and not change_pct.startswith("-"):
            change_pct = "-" + change_pct.lstrip("+")
        if change_pct and direction == "up" and not change_pct.startswith(("+", "-")):
            change_pct = "+" + change_pct
        # pair each metric mention with the values that follow it (before the next metric)
        anchors = [(m.end, m.entity) for m in metrics] or (
            [(0, subject_metric)] if subject_metric else []
        )
        for i, (pos, metric) in enumerate(anchors):
            if metric is None:
                continue
            nxt = anchors[i + 1][0] if i + 1 < len(anchors) else len(sent) + 1
            window = [
                v
                for v in vals
                if v.start >= pos and v.start < nxt and v.kind in ("MONEY", "PERCENT", "COUNT")
            ]
            if not window:
                continue
            seg = sent[pos:nxt].lower()
            if seg.lstrip().startswith("by ") or re.match(
                r"\s*(?:excludes?|excluding|net of|includes?|including)\s+an?\s", seg
            ):
                continue  # "reduced X by 12%" is a change; "X excludes a EUR 7m item" is not X's value
            if change and change.start() >= pos and change.start() < nxt:
                window = [
                    v
                    for v in window
                    if not (
                        v.kind == "PERCENT"
                        and v.start >= change.start(1) - 1
                        and v.end <= change.end()
                    )
                ]
            if not window:
                continue
            main = window[0]
            # "grew 9% to EUR 301 million": the percent is the change, the money the value
            pct_change = None
            if (
                main.kind == "PERCENT"
                and len(window) > 1
                and window[1].kind == "MONEY"
                and re.search(
                    r"\b(?:grew|increased|rose|fell|declined|decreased|dropped|improved|climbed)\s+"
                    + re.escape(main.display.lstrip("+-")),
                    seg,
                )
            ):
                pct_change = main.display
                main = window[1]
                window = window[1:]
            prev = None
            m_from = re.search(r"\bfrom\s+", seg)
            for v in window[1:]:
                if m_from and v.start >= pos + m_from.start() and v.kind == main.kind:
                    prev = v
                    break
            m_between = sent[pos : main.start].lower()
            if re.search(r"\b(would have been|had\b.*\bbeen)", m_between) or (
                hypothetical and "would" in m_between
            ):
                predicate = "would_have_value"
            elif re.search(
                r"\b(?:estimated at|estimate|expected|forecast|guidance|targets?|anticipated|projected)\b",
                m_between + " " + sent[:pos].lower(),
            ):
                predicate = "has_value"
            else:
                predicate = "has_value"
            if (
                hypothetical
                and predicate == "has_value"
                and re.search(r"\b(would|had)\b", m_between)
            ):
                predicate = "would_have_value"
            attrs: dict[str, Any] = {"period": period, **main.attributes}
            if pct_change:
                attrs["change"] = (
                    "+" if direction != "down" and not pct_change.startswith(("+", "-")) else ""
                ) + pct_change
            elif change_pct:
                attrs["change"] = change_pct
            if prev is not None:
                attrs["previous_value"] = prev.display
            if direction:
                attrs["direction"] = direction
            if re.search(
                r"\b(?:estimated|expected|forecast|guidance|anticipated|projected)\b",
                sent[: main.end].lower(),
            ):
                attrs["estimate"] = True
            if re.search(r"\bper\s+(?:year|annum)\b", sent[main.end : main.end + 20].lower()):
                attrs["per"] = "year"
            fm = re.search(r"\bfrom\s+(" + _PERIOD_RE.pattern[2:-2] + r")\b", sent[main.end :])
            if (
                fm
                and predicate == "has_value"
                and re.search(r"\bper\b", sent[main.end : main.end + 20].lower())
            ):
                attrs["period"] = fm.group(1).replace(" ", "").upper() + "+"
            if predicate == "would_have_value":
                attrs["hypothetical"] = True
                cond = re.search(r"\bhad\s+(.+?)\s+been\s+(\w+)", sent, re.IGNORECASE)
                if cond:
                    attrs["condition"] = f"{cond.group(1)} {cond.group(2)}"
            self.facts.append(
                Fact(
                    metric,
                    predicate,
                    self._value_entity(main),
                    sent,
                    0.85 if predicate == "has_value" else 0.8,
                    attrs,
                    c,
                )
            )
            if prev is not None:
                prev_period = None
                if period and re.fullmatch(r"FY\d{2,4}", period):
                    n = int(period[2:])
                    prev_period = f"FY{n - 1:0{len(period) - 2}d}"
                self.facts.append(
                    Fact(
                        metric,
                        "has_value",
                        self._value_entity(prev),
                        sent,
                        0.8,
                        {"period": prev_period, **prev.attributes, "previous_of": period},
                        c,
                    )
                )

    def _table_facts(self, c: Chunk, node: DocumentNode) -> None:
        parsed = parse_table(node.text)
        if not parsed:
            return
        header, body = parsed
        caption = strip_number(node.title or "")
        m = _TABLE_TITLE_RE.match(caption)
        table_subject = m.group(3).strip() if m else caption
        by_segment = bool(
            re.search(
                r"\bby\s+(segment|region|geography|business|product|unit|category)\b",
                table_subject,
                re.IGNORECASE,
            )
        )
        parent_metric: LexEntity | None = None
        if by_segment:
            head = re.split(r"\s+by\s+", table_subject, flags=re.IGNORECASE)[0]
            parent_metric = self.forms.get(canonical_entity(head)) or self.add(head, "METRIC")
        unit_hint = None
        um = re.search(
            r"\((EUR|USD|GBP|CHF|JPY|INR)\s*(m|mn|million|bn|billion|k|thousand)?\)",
            " ".join(header),
        )
        if um:
            unit_hint = (um.group(1), um.group(2) or "")
        columns: list[tuple[int, str | None, str]] = []
        for i, h in enumerate(header[1:], start=1):
            p = period_in(h)
            columns.append((i, p, h))
        for row in body:
            if not row or not row[0]:
                continue
            label = row[0].strip(" *")
            canon = canonical_entity(label)
            is_total = canon == "total"
            row_entity: LexEntity | None
            if is_total:
                row_entity = parent_metric or self.forms.get(canonical_entity(table_subject))
                if row_entity is None:
                    continue
            else:
                row_entity = self.forms.get(canon)
                if row_entity is None:
                    continue
            if parent_metric is not None and not is_total and row_entity is not parent_metric:
                self.facts.append(
                    Fact(
                        row_entity,
                        "segment_of",
                        parent_metric,
                        f"{caption}: {label}",
                        0.9,
                        {"table": caption},
                        c,
                    )
                )
            for i, p, h in columns:
                if i >= len(row):
                    continue
                cell = row[i].strip()
                if not is_numeric_cell(cell):
                    continue
                if p is not None:
                    if unit_hint:
                        cur, scale = unit_hint
                        scale_word = {
                            "m": "million",
                            "mn": "million",
                            "bn": "billion",
                            "k": "thousand",
                            "": "",
                        }.get(scale, scale)
                        display = f"{cur} {cell}{(' ' + scale_word) if scale_word else ''}"
                        v = Value(
                            "MONEY",
                            display,
                            canonical_entity(display),
                            0,
                            0,
                            {
                                "currency": cur,
                                "amount": float(cell.replace(",", "")) * _SCALE.get(scale_word, 1),
                            },
                        )
                    elif cell.endswith("%"):
                        v = percent_value(_PERCENT_RE.search(cell))  # type: ignore[arg-type]
                    else:
                        v = Value(
                            "NUMBER",
                            cell,
                            cell.lower(),
                            0,
                            0,
                            {"number": float(cell.replace(",", ""))},
                        )
                    text = f"{row_entity.name} ({p}) = {v.display} [{caption}]"
                    attrs = {"period": p, "table": caption, "column": h, **v.attributes}
                    if parent_metric is not None and not is_total:
                        attrs["metric"] = parent_metric.name
                    self.facts.append(
                        Fact(row_entity, "has_value", self._value_entity(v), text, 0.9, attrs, c)
                    )
                elif re.search(
                    r"\bchange\b|\bvs\b|\byoy\b|\bgrowth\b", h, re.IGNORECASE
                ) and cell.endswith("%"):
                    # attach the change to the latest period's value fact for this row
                    for f in reversed(self.facts):
                        if (
                            f.subject is row_entity
                            and f.predicate == "has_value"
                            and f.attributes.get("table") == caption
                        ):
                            f.attributes["change"] = cell.replace("−", "-")
                            f.text += f" ({cell} vs {columns[0][1] or 'prior'})"
                            break
