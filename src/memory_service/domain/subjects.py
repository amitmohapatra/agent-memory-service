"""Whether two statements name the same subject: "Forklift 4" and "forklift #4" do,
"Forklift #3" and "Forklift #4" never do.

Consolidation needs this before it can replace a value or link two facts: a new battery
level for Forklift 4 must not overwrite Forklift 3's. Equality of the raw strings is too
strict (case, "#", "No.", spacing, "hazmat" for "hazardous materials", "PO" for "purchase
order") and similarity is too loose (Warehouse 3 and Warehouse 13 are near-identical
strings and near-identical vectors). So a subject is *parsed* into the parts that decide
identity, and two parsed subjects are compared under rules that can only block or allow:

* **identifiers** - every number, code ("A12", "Q3", "SKU-1001"), number with its unit
  ("5 kg", "$5"), date and month name. Two subjects that both carry identifiers and differ in
  any of them are DIFFERENT, whatever else they share. This is the hard constraint.
* **words** - the rest, normalised: NFKC, case-folded, digits of any script read as ASCII,
  "#", "No.", "Nr.", "رقم", "नंबर" before a number dropped, articles, connectives and titles
  dropped, a light plural fold, and every alias of the vocabulary replaced by its canonical
  phrase ("OOS" -> "out of stock", "DC" -> "distribution center", "centre" -> "center").
* **legal form** - "GmbH", "Inc", "Ltd" is set aside: "Acme Logistics GmbH" is "Acme
  Logistics", but "Acme Inc" and "Acme Ltd" are two companies.

The vocabulary is data (``domain/vocabulary/*.json``): a generic pack and a retail pack, both
always on. A tenant's own shorthand is learned from its own text where it is defined
("hazardous materials (hazmat)", "OOS stands for out of stock") - see ``defined_aliases``.

The verdicts are SAME (safe to treat as one subject without asking anyone), POSSIBLE (worth
asking a model: a spelling variant, a short form, a missing identifier, a vector that says
so) and DIFFERENT. Nothing here is a guess dressed as SAME: a vector, a typo or a short form
can at most make a pair POSSIBLE.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, lru_cache
from importlib import resources
from typing import Any, Literal

from memory_service.domain.text import normalise_number

# --------------------------------------------------------------------------- verdicts


class SubjectVerdict(StrEnum):
    SAME = "same"
    POSSIBLE = "possible"
    DIFFERENT = "different"


@dataclass(frozen=True, slots=True)
class SubjectMatch:
    verdict: SubjectVerdict
    #: how strongly the pair matches, for ranking several POSSIBLE pairs (0..1)
    score: float
    reason: str


def _same(score: float, reason: str) -> SubjectMatch:
    return SubjectMatch(SubjectVerdict.SAME, score, reason)


def _possible(score: float, reason: str) -> SubjectMatch:
    return SubjectMatch(SubjectVerdict.POSSIBLE, score, reason)


def _different(reason: str) -> SubjectMatch:
    return SubjectMatch(SubjectVerdict.DIFFERENT, 0.0, reason)


#: The least cosine (the multilingual encoder, subject against subject) that makes a pair
#: whose words differ POSSIBLE: the best separation (Youden's J) of the dev half of the
#: labelled pair set (``tests/eval/golden/subject_pairs.json``; ``benchmark/subjects.py``).
#: It decides only what the adjudicator is asked about; it never merges anything.
DENSE_POSSIBLE = 0.77

# --------------------------------------------------------------------------- vocabulary

TermKind = Literal["alias", "legal", "month", "weekday"]


@dataclass(frozen=True, slots=True)
class Term:
    kind: TermKind
    #: the canonical words of an alias; the code of a unit, legal form, month or weekday
    value: tuple[str, ...]
    #: an acronym that is also an ordinary word: it counts only written in capitals or
    #: directly before a number ("PO 4471", "OH" but not "oh")
    cased: bool = False
    #: an abbreviated month or weekday ("Oct", "Sat"): it counts only next to a number
    short: bool = False


@dataclass(frozen=True, eq=False)
class Vocabulary:
    """The phrase table of the packs (plus any learned aliases), folded as the input is."""

    phrases: Mapping[tuple[str, ...], Term]
    #: unit forms -> unit code; a unit counts only directly after a number ("5 kg")
    units: Mapping[tuple[str, ...], str]
    #: units that count things ("4 pallets"): after a number that labels a noun
    #: ("Warehouse 3 pallets") they are an ordinary word
    count_units: frozenset[str]
    longest: int
    markers: frozenset[str]
    stopwords: frozenset[str]
    #: the articles a subject may start with ("the Berlin office", "la tienda 12")
    articles: frozenset[str]
    #: prepositions that give a direction ("to" / "from", "a" / "de", "إلى" / "من"): words
    #: of the subject, never dropped
    directions: frozenset[str]
    honorifics: frozenset[str]
    #: two-letter words that read as a word before a number, not as a code prefix
    #: ("PO4471" is purchase order 4471; "A12" is bin A12)
    word_prefixes: frozenset[str]
    learned: tuple[tuple[str, str], ...] = ()
    #: alias forms by canonical phrase, as written in the packs (for ``spellings``)
    surface: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def with_aliases(self, pairs: Iterable[tuple[str, str]]) -> Vocabulary:
        """This vocabulary with ``(short, long)`` aliases learned from a tenant's text. A
        short form defined as two different things ("Berlin Hub (BH)", "Bonn Hub (BH)") is
        evidence of neither and is not learned."""
        meanings: dict[str, set[str]] = {}
        for short, long in pairs:
            meanings.setdefault(short.casefold(), set()).add(" ".join(long.casefold().split()))
        extra = tuple(
            sorted(
                {
                    (short, long)
                    for short, long in pairs
                    if len(meanings[short.casefold()]) == 1 and (short, long) not in self.learned
                }
            )
        )
        return _extended(self, extra) if extra else self


PACK_NAMES: tuple[str, ...] = ("generic", "retail")


@cache
def _pack(name: str) -> dict[str, Any]:
    raw = resources.files("memory_service.domain.vocabulary").joinpath(f"{name}.json")
    return json.loads(raw.read_text(encoding="utf-8"))


def _strings(pack: dict[str, Any], key: str) -> list[str]:
    return [unicodedata.normalize("NFKC", str(v)) for v in pack.get(key, [])]


def _table(pack: dict[str, Any], key: str) -> dict[str, list[str]]:
    return {str(k): [str(f) for f in v] for k, v in pack.get(key, {}).items()}


def pack_aliases(name: str) -> dict[str, list[str]]:
    """``canonical phrase -> forms`` of one pack, in the pack's order."""
    return _table(_pack(name), "aliases")


@cache
def vocabulary() -> Vocabulary:
    """The generic and the retail pack together: what every tenant starts from."""
    builder = _Builder()
    for name in PACK_NAMES:
        builder.read(_pack(name), name)
    return builder.build()


class _Builder:
    """Reads the packs into one phrase table, each form folded as the parser folds input."""

    def __init__(self) -> None:
        self.phrases: dict[tuple[str, ...], Term] = {}
        self.units: dict[tuple[str, ...], str] = {}
        self.surface: dict[str, tuple[str, ...]] = {}
        self.markers: set[str] = set()
        self.stop: set[str] = set()
        self.articles: set[str] = set()
        self.directions: set[str] = set()
        self.honorifics: set[str] = set()
        self.count_units: set[str] = set()

    def put(self, form: str, term: Term) -> None:
        key = _form_key(form)
        if key and key not in self.phrases:
            self.phrases[key] = term

    def read(self, pack: dict[str, Any], name: str) -> None:
        self.markers |= {m.casefold() for m in _strings(pack, "number_markers")}
        self.stop |= {_fold(w) for w in _strings(pack, "stopwords")}
        self.articles |= {_fold(w) for w in _strings(pack, "articles")}
        self.directions |= {_fold(w) for w in _strings(pack, "directions")}
        self.honorifics |= {_fold(w) for w in _strings(pack, "honorifics")}
        self.count_units |= set(_strings(pack, "count_units"))
        self._aliases(pack_aliases(name), {c.casefold() for c in _strings(pack, "cased")})
        for code, forms in _table(pack, "legal_forms").items():
            for form in [code, *forms]:
                self.put(form, Term("legal", (code,)))
        for code, forms in _table(pack, "units").items():
            for form in [code, *forms]:
                self.units.setdefault(_form_key(form), code)
        for kind in ("month", "weekday"):
            for index, forms in enumerate(pack.get(f"{kind}s", []), start=1):
                for form in forms:
                    short = len(form) <= 4 and form.isascii()
                    self.put(str(form), Term(kind, (f"{kind[0]}{index}",), short=short))

    def _aliases(self, aliases: dict[str, list[str]], cased: set[str]) -> None:
        for canonical, forms in aliases.items():
            value = _form_key(canonical)
            self.surface[canonical.casefold()] = tuple(f.casefold() for f in forms)
            self.put(canonical, Term("alias", value))
            for form in forms:
                acronym = form.isupper() and form.isalnum()
                is_cased = acronym and (len(form) <= 2 or form.casefold() in cased)
                self.put(form, Term("alias", value, cased=is_cased))

    def _canonical_values(self) -> None:
        """A canonical phrase is itself canonical: "return to vendor" reads "return to
        supplier"."""
        single = {
            k[0]: t.value[0]
            for k, t in self.phrases.items()
            if t.kind == "alias" and len(k) == 1 and len(t.value) == 1
        }
        for key, term in list(self.phrases.items()):
            value = tuple(single.get(w, w) for w in term.value)
            if term.kind == "alias" and value != term.value:
                self.phrases[key] = Term("alias", value, cased=term.cased, short=term.short)
                self.phrases.setdefault(value, Term("alias", value))

    def build(self) -> Vocabulary:
        self.units.pop((), None)
        self._canonical_values()
        short_words = {
            k[0]
            for k, t in self.phrases.items()
            if len(k) == 1 and len(k[0]) <= 2 and t.kind == "alias"
        }
        return Vocabulary(
            phrases=self.phrases,
            units=self.units,
            count_units=frozenset(self.count_units),
            longest=max(len(k) for k in [*self.phrases, *self.units]),
            markers=frozenset(self.markers),
            stopwords=frozenset(self.stop),
            articles=frozenset(self.articles),
            directions=frozenset(self.directions),
            honorifics=frozenset(self.honorifics),
            word_prefixes=frozenset(short_words | {m for m in self.markers if len(m) <= 2}),
            surface=self.surface,
        )


@lru_cache(maxsize=256)
def _extended(base: Vocabulary, extra: tuple[tuple[str, str], ...]) -> Vocabulary:
    phrases = dict(base.phrases)
    surface = dict(base.surface)
    for short, long in extra:
        value, key = _form_key(long), _form_key(short)
        if key and value:
            # the tenant's own definition wins over a pack's ("Data Center (DC)"); a learned
            # form of two or three letters counts in capitals or before a number ("BH",
            # "BH 2"), not as the word "bh"
            phrases[key] = Term("alias", value, cased=len(short) <= 3)
            # and the long form reads as the same canonical words ("funds" as "fund")
            phrases.setdefault(value, Term("alias", value))
            surface[long.casefold()] = (*surface.get(long.casefold(), ()), short.casefold())
    return Vocabulary(
        phrases=phrases,
        units=base.units,
        count_units=base.count_units,
        longest=max(base.longest, *(len(k) for k in phrases)),
        markers=base.markers,
        stopwords=base.stopwords,
        articles=base.articles,
        directions=base.directions,
        honorifics=base.honorifics,
        word_prefixes=base.word_prefixes,
        learned=(*base.learned, *extra),
        surface=surface,
    )


# --------------------------------------------------------------------------- folding


def _is_latin(word: str) -> bool:
    return all(c < "ɐ" for c in word)


#: alef with hamza (above, below) or madda -> alef; ta marbuta -> heh; alef maqsura -> yeh
_ARABIC_LETTERS = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0629": "\u0647",
        "\u0649": "\u064a",
    }
)
_ARABIC = re.compile("[\u064b-\u0652\u0640]")


def _orth(word: str) -> str:
    """One word as written, compared without its case and its script's interchangeable
    spellings: for Arabic, hamza on alef, ta marbuta, alef maqsura, short-vowel marks,
    tatweel and the article."""
    w = word.casefold()
    if any("\u0600" <= c <= "\u06ff" for c in w):
        w = _ARABIC.sub("", w.translate(_ARABIC_LETTERS))
        return w[2:] if w.startswith("\u0627\u0644") and len(w) > 3 else w
    return w


def _fold(word: str) -> str:
    """``_orth`` and an English-shaped plural folded ("batteries" -> "battery", "pallets" ->
    "pallet"). Deliberately not a stemmer: Porter maps "organization" and "organ" to one
    stem. Still lossy - "Roberts" is not "Robert" - so a match that needs it is only
    POSSIBLE."""
    w = _orth(word)
    if len(w) >= 4 and _is_latin(w) and w.isalpha():
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith(("xes", "ches", "shes", "sses")):
            return w[:-2]
        if w.endswith("s") and not w.endswith(("ss", "us", "is", "ís", "ys")):
            return w[:-1]
    return w


def _form_key(form: str) -> tuple[str, ...]:
    """A pack form as the parser will meet it: folded words, a number glued to its letters
    kept as one code ("3PL" -> "3pl", as ``parse`` reads it)."""
    out: list[str] = []
    for tok in _scan(form):
        if tok.kind == "sym":
            continue
        if out and tok.gap == "none" and (tok.kind == "num" or out[-1][-1:].isdigit()):
            out[-1] += tok.text
        else:
            out.append(tok.text if tok.kind == "num" else _fold(tok.text))
    return tuple(out)


# --------------------------------------------------------------------------- scanning


@dataclass(frozen=True, slots=True)
class _Tok:
    kind: Literal["word", "num", "sym"]
    text: str
    #: what separated it from the previous token: nothing, a hyphen, or space/punctuation
    gap: Literal["none", "hyphen", "space"]
    upper: bool = False
    #: starts with a capital ("El Salvador", "April Jones")
    capital: bool = False


_DOTTED = re.compile(r"\b(?:[^\W\d_]\.){2,}")
_POSSESSIVE = re.compile("(?<=\\w)['\u2019]s\\b")
_SYMBOLS = "#№%$€£₹"
_CURRENCY = {"$": "usd", "€": "eur", "£": "gbp", "₹": "inr"}


def _letter(c: str) -> bool:
    return c.isalpha() or unicodedata.category(c).startswith("M")


Gap = Literal["none", "hyphen", "space"]


def _number_end(text: str, i: int) -> int:
    """Where the number starting at ``text[i]`` ends: digits, and a separator only between
    two digits ("4.99", "2026-10-01", "1,000")."""
    j = i + 1
    while j < len(text) and (
        text[j].isdecimal()
        or (text[j] in ".,:/-" and j + 1 < len(text) and text[j + 1].isdecimal())
    ):
        j += 1
    return j


def _run_end(text: str, i: int) -> int:
    j = i + 1
    while j < len(text) and _letter(text[j]):
        j += 1
    return j


_DECIMAL_COMMA = re.compile(r"^[\d.]*\d,\d{1,2}$")
#: "1,250": a thousand and a quarter, or one and a quarter - kept as written, so it never
#: matches "1250" or "1.25"
_AMBIGUOUS_COMMA = re.compile(r"^\d{1,3},\d{3}$")


def _ascii_digits(raw: str) -> str:
    """ASCII digits, and the separators read the way they are written: a comma followed by
    one or two final digits is a decimal comma ("1,5 kg", "4,99 €", "1.234,56"); any other
    comma groups thousands ("1,000,000"), except a single group of three ("1,250"), which is
    ambiguous and kept as written. A leading minus (or U+2212) is kept ("Level -1")."""
    ascii_ = normalise_number(
        "".join(
            str(unicodedata.decimal(ch)) if ch.isdecimal() else ch
            for ch in raw.replace(",", "\x00").replace("\u2212", "-")
        )
    ).replace("\x00", ",")
    sign, body = ("-", ascii_[1:]) if ascii_.startswith("-") else ("", ascii_)
    if _DECIMAL_COMMA.match(body):
        body = body.replace(".", "").replace(",", ".")
    elif not _AMBIGUOUS_COMMA.match(body):
        body = body.replace(",", "")
    return sign + body


def _minus(text: str, i: int, gap: str) -> bool:
    """A minus sign (or U+2212) starting a number: "-1", not the hyphen of "SKU-1001"."""
    return text[i] in "-\u2212" and gap == "space" and i + 1 < len(text) and text[i + 1].isdecimal()


_RANGE_DASH = re.compile(r"(?<=\d)[\u2013\u2012](?=\d)")


def _attached(text: str, i: int, j: int) -> int:
    """Where a short code ends that has signs written onto it: "C#", "C++", "A+", "A-"."""
    if j - i > 3:
        return j
    k = j
    while k < len(text) and text[k] in "#+":
        k += 1
    if (
        k == j
        and k < len(text)
        and text[k] == "-"
        and (k + 1 == len(text) or not text[k + 1].isalnum())
    ):
        k += 1
    return k


def _scan(text: str) -> tuple[_Tok, ...]:
    text = _RANGE_DASH.sub("-", unicodedata.normalize("NFKC", text))
    text = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), text)
    text = _POSSESSIVE.sub("", text)
    out: list[_Tok] = []
    gap: Gap = "space"
    i = 0
    while i < len(text):
        c = text[i]
        if c.isdecimal() or _minus(text, i, gap):
            j = _number_end(text, i + 1 if not c.isdecimal() else i)
            out.append(_Tok("num", _ascii_digits(text[i:j]), gap))
        elif _letter(c):
            j = _attached(text, i, _run_end(text, i))
            word = text[i:j]
            out.append(
                _Tok("word", word.casefold(), gap, upper=word.isupper(), capital=c.isupper())
            )
        elif c in _SYMBOLS:
            j = i + 1
            out.append(_Tok("sym", c, gap))
        else:
            # a hyphen right after a token glues the next one to it; anything else separates
            gap = "hyphen" if c == "-" and gap in ("none", "hyphen") else "space"
            i += 1
            continue
        i, gap = j, "none"
    return tuple(out)


# --------------------------------------------------------------------------- parsing

_ORDINAL = frozenset({"st", "nd", "rd", "th", "er", "º", "ª", "वां", "वीं"})


@dataclass(frozen=True, slots=True)
class Subject:
    """A subject reduced to what decides its identity (see the module docstring)."""

    text: str
    #: a namespaced identity ("user:u1", "thread:thr_1"): compared by equality only
    identity: str | None = None
    #: the words as written, compared without case: aliases read as what they stand for, a
    #: leading article dropped, connectives, titles and plurals kept ("Bank of China",
    #: "Mrs Patel", "Roberts")
    words: tuple[str, ...] = ()
    #: words and identifiers in their order: what SAME compares
    sequence: tuple[str, ...] = ()
    #: every identifier in order, with the word it labels ("Aisle 3 Bay 4" ->
    #: ("aisle", "3"), ("bay", "4")); "" when no word stands right before it
    labelled: tuple[tuple[str, str], ...] = ()
    #: the words with plurals folded, connectives and titles dropped: a match on these alone
    #: is POSSIBLE
    loose: tuple[str, ...] = ()
    #: the titles it carries ("mrs", "herr", "श्री"): two different ones name two people
    titles: frozenset[str] = frozenset()
    #: its directions ("to", "from", "إلى"): two different ones name two movements
    directions: frozenset[str] = frozenset()
    legal: str | None = None

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(i for _, i in self.labelled)

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(self.loose)

    @property
    def empty(self) -> bool:
        return self.identity is None and not self.words and not self.labelled


_IDENTITY = re.compile(r"^[a-z][a-z_]*:\S+$")


def is_identity(text: str) -> bool:
    """A namespaced principal or anchor ("user:u1", "thread:thr_1", "work:w9")."""
    return bool(_IDENTITY.match(text))


#: The longest text a subject is read from. A subject is a few words; a topic is cut here
#: too, so no caller can make parsing (or its cache) grow with its input.
MAX_SUBJECT_CHARS = 300


#: Subjects up to this long are cached (names and topics recur): at most
#: ``_PARSE_CACHE`` of them, each the few tuples of words a subject reduces to.
CACHED_SUBJECT_CHARS = MAX_SUBJECT_CHARS
_PARSE_CACHE = 2048


def parse(text: str, vocab: Vocabulary | None = None) -> Subject:
    """``text`` reduced to identifiers, canonical words and legal form."""
    learned = vocab.learned if vocab is not None else ()
    text = text.strip()[:MAX_SUBJECT_CHARS]
    if len(text) <= CACHED_SUBJECT_CHARS:
        return _parse(text, learned)
    return _parse.__wrapped__(text, learned)


#: (kind, value, source token): "word", "sym" or "id"
_Item = tuple[str, str, "_Tok | None"]


@lru_cache(maxsize=_PARSE_CACHE)
def _parse(text: str, learned: tuple[tuple[str, str], ...]) -> Subject:
    if is_identity(text):
        return Subject(text=text, identity=text)
    vocab = vocabulary().with_aliases(learned)
    items = _identifiers(_scan(text), vocab)
    return _subject(text, _entries(items, vocab), vocab)


def _subject(text: str, entries: list[_Entry], vocab: Vocabulary) -> Subject:
    words: list[str] = []
    sequence: list[str] = []
    loose: list[str] = []
    labelled: list[tuple[str, str]] = []
    titles: list[str] = []
    directions: list[str] = []
    legal: str | None = None
    previous = ""
    for kind, strict, folded in entries:
        if kind == "legal":
            legal = strict
            continue
        sequence.append(strict)
        if kind == "id":
            labelled.append((previous, strict))
            previous = ""
            continue
        words.append(strict)
        if kind == "title":
            titles.append(folded)
        elif kind in ("word", "direction"):
            loose.append(folded)
            previous = folded
            if kind == "direction":
                directions.append(folded)
    return Subject(
        text=text,
        words=tuple(words),
        sequence=tuple(sequence),
        labelled=tuple(labelled),
        loose=tuple(loose),
        titles=frozenset(titles),
        directions=frozenset(directions),
        legal=legal,
    )


def _identifiers(toks: Sequence[_Tok], vocab: Vocabulary) -> list[_Item]:
    """The tokens with every number turned into one identifier item, together with the
    letters, marker, currency and unit around it."""
    items: list[_Item] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind != "num":
            items.append(("word" if t.kind == "word" else "sym", t.text, t))
            i += 1
            continue
        value, unit = _before_number(items, t, vocab)
        suffix, after, i = _after_number(toks, i + 1, items, vocab, unit=bool(unit))
        value, unit = value + suffix, unit or after
        term = vocab.phrases.get((value,)) if not unit else None
        if term is not None and term.kind == "alias":
            items.append(("word", value, t))  # a code that is shorthand: "3PL"
        else:
            items.append(("id", value + unit, None))
    return items


def _before_number(items: list[_Item], t: _Tok, vocab: Vocabulary) -> tuple[str, str]:
    """Take from ``items`` what belongs to the number ``t``: the letters of a code ("A12",
    "Q3", "FY26"), a currency sign ("$5") and a number marker ("#4", "No. 4", "رقم 4")."""
    value, unit = t.text, ""
    prev = items[-1] if items else None
    if (
        prev is not None
        and prev[0] == "word"
        and t.gap in ("none", "hyphen")
        and len(prev[1]) <= 2
        and _is_latin(prev[1])
        and prev[1] not in vocab.word_prefixes
    ):
        value = items.pop()[1] + value
        prev = items[-1] if items else None
    if prev is not None and prev[0] == "sym" and prev[1] in _CURRENCY:
        unit = _CURRENCY[items.pop()[1]]
        prev = items[-1] if items else None
    if prev is not None and prev[1] in vocab.markers:
        items.pop()
    return value, unit


def _after_number(
    toks: Sequence[_Tok], j: int, items: list[_Item], vocab: Vocabulary, *, unit: bool
) -> tuple[str, str, int]:
    """What follows a number and belongs to it: a glued suffix ("3B") or ordinal ("4th"),
    a unit ("5kg", "5 kg", "10%"). Returns (suffix, unit, next index)."""
    nxt = toks[j] if j < len(toks) else None
    if nxt is not None and nxt.kind == "sym":
        return _symbol_after(nxt, j, unit=unit)
    if nxt is None or nxt.kind != "word":
        return "", "", j
    if nxt.gap == "none":
        return _glued(nxt, vocab, j)
    matched = None if unit else _unit_at(toks, j, vocab)
    # "Warehouse 3 pallets": the 3 names the warehouse, it does not count pallets
    prev = items[-1] if items else None
    labels = prev is not None and prev[0] == "word" and _fold(prev[1]) not in vocab.stopwords
    if matched is None or (labels and matched[0] in vocab.count_units):
        return "", "", j
    return "", matched[0], j + matched[1]


def _symbol_after(nxt: _Tok, j: int, *, unit: bool) -> tuple[str, str, int]:
    """A sign after a number: "10%", a trailing currency ("5 €", "4,99€")."""
    if nxt.text == "%":
        return "", "%", j + 1
    if nxt.text in _CURRENCY and not unit:
        return "", _CURRENCY[nxt.text], j + 1
    return "", "", j


def _glued(nxt: _Tok, vocab: Vocabulary, j: int) -> tuple[str, str, int]:
    """Letters written onto a number: a unit ("5kg"), an ordinal ("4th"), a code's suffix
    ("3B")."""
    code = vocab.units.get((_fold(nxt.text),))
    if code is not None:
        return "", code, j + 1
    if nxt.text in _ORDINAL:
        return "", "", j + 1
    if len(nxt.text) <= 2 and _is_latin(nxt.text):
        return nxt.text, "", j + 1
    return "", "", j


#: (kind, strict, folded): "word", "title", "id" or "legal"
_Entry = tuple[str, str, str]


def _entries(items: list[_Item], vocab: Vocabulary) -> list[_Entry]:
    """``items`` in order as entries: vocabulary phrases replaced (an alias by its canonical
    words, a month or weekday by an identifier, a trailing legal form set aside),
    short codes made identifiers ("Store LA", "Block A", "Sales IN": never a connective),
    connectives dropped, titles marked, every other word kept as written and folded."""
    out: list[_Entry] = []
    k = 0
    while k < len(items):
        kind, value, tok = items[k]
        if kind == "id":
            out.append(("id", value, value))
            k += 1
            continue
        term, width = _term_at(items, k, vocab) if kind == "word" and tok is not None else (None, 1)
        if term is not None and _applies(term, items, k, width, bool(out)):
            out.extend(_from_term(term, vocab))
            k += width
            continue
        if kind == "word":
            out.extend(_word_entry(items, k, vocab, first=not out))
        k += 1
    return out


def _applies(term: Term, items: list[_Item], k: int, width: int, after_words: bool) -> bool:
    """Whether a matched phrase stands for its term here. A legal form only at the end of a
    name, and of three letters or more: "SE", "SA", "CO", "AG" are as likely a code ("Sales
    SE", "Store CO") and are read as one. A month or weekday spelled out in full and followed
    by a capitalised word is a name ("April Jones", "Friday Smith")."""
    if term.kind == "legal":
        return after_words and _at_end(items, k + width) and len(items[k][1]) > 2
    if term.kind in ("month", "weekday") and not term.short:
        tok, nxt = items[k][2], items[k + width] if k + width < len(items) else None
        named = nxt is not None and nxt[2] is not None and nxt[2].capital and nxt[0] == "word"
        return not (tok is not None and tok.capital and named)
    return True


def _from_term(term: Term, vocab: Vocabulary) -> list[_Entry]:
    if term.kind == "alias":
        return [("word", w, w) for w in term.value if w not in vocab.stopwords]
    if term.kind == "legal":
        return [("legal", term.value[0], term.value[0])]
    return [("id", term.value[0], term.value[0])]  # a month or weekday


def _word_entry(items: list[_Item], k: int, vocab: Vocabulary, *, first: bool) -> list[_Entry]:
    _, value, tok = items[k]
    strict, folded = _orth(value), _fold(value)
    if _code(items, k, tok):
        return [("id", strict, strict)]
    nxt = items[k + 1] if k + 1 < len(items) else None
    if first and folded in vocab.articles and nxt is not None and nxt[0] == "word":
        # "the warehouse", "la tienda 12" - but not the article of a name ("El Salvador")
        return [] if not _capitalised_span(tok, nxt[2]) else [("word", strict, folded)]
    if folded in vocab.directions:
        return [("direction", strict, folded)]  # "to" / "from" Berlin: never one shipment
    if folded in vocab.stopwords:
        return [("connective", strict, folded)]
    return [("title" if folded in vocab.honorifics else "word", strict, folded)]


def _capitalised_span(tok: _Tok | None, nxt: _Tok | None) -> bool:
    return tok is not None and nxt is not None and tok.capital and nxt.capital


def _code(items: list[_Item], k: int, tok: _Tok | None) -> bool:
    """A word that is a short code - an identifier, never a connective: three capitals at
    most ("LA", "IN", "DE", "API"), a code with signs written onto it ("C#", "A+"), a single
    Latin letter after a word ("block a north"), or two letters at most and not followed by
    another word ("Block a", "Store la", "Sales in" as stored in lower case)."""
    value = items[k][1]
    if any(c in "#+-" for c in value):
        return True
    if tok is not None and tok.upper and value.isalpha() and len(value) <= 3:
        return True
    if len(value) == 1 and _is_latin(value) and k > 0:
        return True
    following = items[k + 1][0] if k + 1 < len(items) else None
    return len(value) <= 2 and _is_latin(value) and following != "word"


def _term_at(items: list[_Item], k: int, vocab: Vocabulary) -> tuple[Term | None, int]:
    matched = _match_items(items, k, vocab)
    return matched if matched is not None else (None, 1)


def _at_end(items: Sequence[tuple[str, str, _Tok | None]], k: int) -> bool:
    return all(kind == "sym" for kind, _, _ in items[k:])


def _beside_number(items: Sequence[tuple[str, str, _Tok | None]], start: int, end: int) -> bool:
    before = items[start - 1][0] if start > 0 else None
    after = items[end][0] if end < len(items) else None
    return before == "id" or after == "id"


def _match_items(
    items: Sequence[tuple[str, str, _Tok | None]], start: int, vocab: Vocabulary
) -> tuple[Term, int] | None:
    """The longest vocabulary phrase starting at ``items[start]`` (words only)."""
    run: list[str] = []
    for kind, value, _ in items[start : start + vocab.longest]:
        if kind != "word":
            break
        run.append(_fold(value))
    for width in range(len(run), 0, -1):
        term = vocab.phrases.get(tuple(run[:width]))
        if term is None:
            continue
        sources = [tok for _, _, tok in items[start : start + width]]
        beside = _beside_number(items, start, start + width)
        if term.cased and not (all(t is not None and t.upper for t in sources) or beside):
            continue
        if term.short and not beside:
            continue
        return term, width
    return None


def _unit_at(toks: Sequence[_Tok], start: int, vocab: Vocabulary) -> tuple[str, int] | None:
    """The unit written at ``toks[start]`` (after a number), and how many words it takes."""
    run: list[str] = []
    for t in toks[start : start + vocab.longest]:
        if t.kind != "word":
            break
        run.append(_fold(t.text))
    for width in range(len(run), 0, -1):
        code = vocab.units.get(tuple(run[:width]))
        if code is not None:
            return code, width
    return None


# --------------------------------------------------------------------------- comparison


def compare(
    a: Subject,
    b: Subject,
    *,
    cosine: float | None = None,
    names: Sequence[Subject] = (),
    topic: bool = False,
) -> SubjectMatch:
    """Whether ``a`` and ``b`` name the same subject.

    ``cosine`` is the two subjects' similarity under the multilingual encoder, when the
    caller has it: it can make a pair POSSIBLE, never SAME. ``names`` are other subjects
    the tenant's data mentions: a short form that more than one of them extends
    ("John" with "John Smith" and "John Miller" known) is no evidence of either.
    ``topic`` compares two topics rather than two names: a shared word is reason enough
    to ask ("tea in the morning", "coffee in the morning").
    """
    blocked = _blocked(a, b)
    if blocked is not None:
        return blocked
    overlap = _overlap(a, b, names)
    if overlap is not None:
        return overlap
    only_a, only_b = a.tokens - b.tokens, b.tokens - a.tokens
    if _spelling_variants(only_a, only_b):
        return _possible(0.7, "spelling variant")
    if _initialism(only_a, a.words, only_b, b.words):
        return _possible(0.6, "initialism")
    if cosine is not None and cosine >= DENSE_POSSIBLE:
        return _possible(round(cosine * 0.6, 3), f"dense similarity {cosine:.2f}")
    if topic and a.tokens & b.tokens:
        shared = len(a.tokens & b.tokens) / len(a.tokens | b.tokens)
        return _possible(round(0.3 + shared / 2, 3), "shared topic words")
    return _different(f"names differ: {' '.join(sorted(only_a))} vs {' '.join(sorted(only_b))}")


def _blocked(a: Subject, b: Subject) -> SubjectMatch | None:
    """The verdicts nothing can override: identities, then ``_conflict``."""
    if a.identity is not None or b.identity is not None:
        if a.identity == b.identity:
            return _same(1.0, "same identity")
        return _different("different identities")
    if a.empty or b.empty:
        return _different("nothing to compare")
    conflict = _conflict(a, b)
    return _different(conflict) if conflict else None


def _conflict(a: Subject, b: Subject) -> str | None:
    """What makes two subjects two: identifiers (their values, order and count, and the
    word each one labels), legal forms, titles."""
    if a.ids and b.ids and a.ids != b.ids:
        return f"identifiers differ: {' '.join(a.ids)} vs {' '.join(b.ids)}"
    labels_a = {word: i for word, i in a.labelled if word}
    for word, i in b.labelled:
        if word and labels_a.get(word, i) != i:
            return f"{word} {labels_a[word]} vs {word} {i}"
    if a.legal and b.legal and a.legal != b.legal:
        return f"legal forms differ: {a.legal} vs {b.legal}"
    for name, x, y in (("titles", a.titles, b.titles), ("directions", a.directions, b.directions)):
        if x and y and x != y:
            return f"{name} differ: {' '.join(sorted(x ^ y))}"
    return None


def _overlap(a: Subject, b: Subject, names: Sequence[Subject]) -> SubjectMatch | None:
    """The same words - as written, or only once folded - or one side's inside the other's."""
    same_ids = a.ids == b.ids
    if a.sequence == b.sequence:
        return _same(1.0, "same subject after normalisation")
    if same_ids and a.labelled == b.labelled and _joined(a, b):
        return _same(0.95, "same words, joined differently")
    if a.tokens == b.tokens:
        if same_ids:
            return _possible(0.7, "the same only once folded, reordered or without a title")
        return _possible(0.6, "one side names no identifier")
    code_only = _code_against_name(a, b) or _code_against_name(b, a)
    if code_only is not None:
        return code_only
    if a.tokens < b.tokens or b.tokens < a.tokens:
        if _ambiguous(a if a.tokens < b.tokens else b, names):
            return _different("short form shared by several known subjects")
        if not same_ids:
            return _possible(0.4, "one names more than the other and no identifier")
        return _possible(0.5, "one names more than the other")
    return None


def _code_against_name(code: Subject, name: Subject) -> SubjectMatch | None:
    """A subject that is only a code ("LA", "Q3", "4471") against a name with no identifier
    ("Berlin Hub"): different, unless the code is the name's initialism ("BH")."""
    if code.tokens or not code.ids or not name.tokens or name.ids:
        return None
    letters = code.ids[0] if len(code.ids) == 1 and code.ids[0].isalpha() else ""
    if letters and abbreviates(letters, name.loose):
        return _possible(0.5, "initialism")
    return _different("a code is not a name")


def _ambiguous(short: Subject, names: Sequence[Subject]) -> bool:
    extended = [
        n for n in names if short.tokens < n.tokens and (not short.ids or short.ids == n.ids)
    ]
    return any(
        (x.tokens - y.tokens) and (y.tokens - x.tokens)
        for i, x in enumerate(extended)
        for y in extended[i + 1 :]
    )


def _joined(a: Subject, b: Subject) -> bool:
    """ "Wal-Mart" and "Walmart", "on-boarding" and "onboarding": the same letters, split
    differently (identifiers have been compared already)."""
    return _letters(a.text) == _letters(b.text) or "".join(a.words) == "".join(b.words)


def _letters(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if _letter(c))


def _edit_distance_at_most_one(x: str, y: str) -> bool:
    if x == y:
        return True
    if abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        diff = [i for i in range(len(x)) if x[i] != y[i]]
        if len(diff) == 1:
            return True
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and x[diff[0]] == y[diff[1]]
            and x[diff[1]] == y[diff[0]]
        )
    if len(x) > len(y):
        x, y = y, x
    i = 0
    while i < len(x) and x[i] == y[i]:
        i += 1
    return x[i:] == y[i + 1 :]


def _spelling_variants(only_a: frozenset[str], only_b: frozenset[str]) -> bool:
    """Every word one side alone has is one edit from a word the other side alone has.
    Words under five letters are never typos of each other ("Jon" and "Joan" are people)."""
    if len(only_a) != len(only_b) or len(only_a) > 2:
        return False
    rest = set(only_b)
    for x in only_a:
        match = next(
            (y for y in rest if min(len(x), len(y)) >= 5 and _edit_distance_at_most_one(x, y)),
            None,
        )
        if match is None:
            return False
        rest.discard(match)
    return True


def _initialism(
    only_a: frozenset[str], a: Sequence[str], only_b: frozenset[str], b: Sequence[str]
) -> bool:
    for short, words in ((only_a, b), (only_b, a)):
        if len(short) == 1:
            (token,) = short
            rest = [w for w in words if w not in short]
            if 2 <= len(token) <= 8 and len(rest) >= 2 and abbreviates(token, rest):
                return True
    return False


# --------------------------------------------------------------------------- learning


def abbreviates(short: str, words: Sequence[str]) -> bool:
    """True when ``short`` is made of prefixes of ``words`` in order - every word that is
    not a connective gives at least its first letter: "hazmat" (HAZardous MATerials), "OOS"
    (Out Of Stock), "SKU" (Stock Keeping Unit) - or, for one long compound, its capitals in
    order ("ZL", Zentrallager). "Berlin (Germany)" is not one."""
    s = "".join(c for c in short.casefold() if c.isalnum())
    ws = tuple(part for w in words for part in w.casefold().split("-") if part)
    if len(s) < 2 or not ws or s == "".join(ws) or s[0] != ws[0][0]:
        return False
    if len(ws) == 1 and short.isupper() and len(s) <= 4 and len(ws[0]) >= 8:
        # a compound's initials: "Zentrallager (ZL)", "Wareneingang (WE)"
        rest = iter(ws[0][1:])
        return all(c in rest for c in s[1:])
    return _covers(s, ws, 0, 0)


@lru_cache(maxsize=4096)
def _covers(s: str, ws: tuple[str, ...], i: int, k: int) -> bool:
    """``s[i:]`` is made of prefixes of ``ws[k:]``, one per word (a connective may give
    none)."""
    if k == len(ws):
        return i == len(s)
    w = ws[k]
    if _fold(w) in vocabulary().stopwords and _covers(s, ws, i, k + 1):
        return True
    n = 1
    while n <= len(w) and i + n <= len(s) and s[i : i + n] == w[:n]:
        if _covers(s, ws, i + n, k + 1):
            return True
        n += 1
    return False


#: A word of a definition. Anchored at a word start and bounded, so a scan of a long text
#: is linear: unanchored and unbounded, a 20,000-letter word cost seconds.
_WORD = r"(?<![\w-])[^\W\d_][\w\-]{0,30}(?![\w-])"
_SHORT = r"(?<![\w-])[^\W_][\w\-]{1,11}"
#: the longest text definitions are looked for in
MAX_DEFINITION_CHARS = 2000
_LONG_THEN_SHORT = re.compile(
    rf"((?:{_WORD}\s+){{0,5}}{_WORD})\s*\(\s*[\"'“]?([^\W_][\w\-]{{1,11}})[\"'”]?\s*\)"
)
_SHORT_THEN_LONG = re.compile(rf"({_SHORT})\s*\(\s*((?:{_WORD}\s+){{0,5}}{_WORD})\s*\)")
_SHORT_MEANS_LONG = re.compile(
    rf"({_SHORT})\s+(?:stands for|is short for|means|steht für|bedeutet|"
    rf"significa|quiere decir|=)\s+((?:{_WORD}\s+){{0,5}}{_WORD})",
    re.IGNORECASE,
)


def defined_aliases(text: str) -> list[tuple[str, str]]:
    """``(short, long)`` pairs a text defines, checked by ``abbreviates`` so a parenthesis
    that is not an abbreviation ("Berlin (Germany)", "the meeting (Monday)") teaches
    nothing: "hazardous materials (hazmat)", "OOS (out of stock)", "OOS stands for out of
    stock items" -> ("hazmat", "hazardous materials"), ("OOS", "out of stock") twice. Only
    the first ``MAX_DEFINITION_CHARS`` characters are read."""
    text = text[:MAX_DEFINITION_CHARS]
    out: list[tuple[str, str]] = []
    for m in _LONG_THEN_SHORT.finditer(text):
        before, short = m.group(1).split(), m.group(2)
        for width in range(1, len(before) + 1):
            long = before[-width:]
            if abbreviates(short, long):
                out.append((short, " ".join(long)))
                break
    for pattern in (_SHORT_THEN_LONG, _SHORT_MEANS_LONG):
        for m in pattern.finditer(text):
            short, after = m.group(1), m.group(2).split()
            for width in range(1, len(after) + 1):
                long = after[:width]
                if abbreviates(short, long):
                    out.append((short, " ".join(long)))
                    break
    seen: set[tuple[str, str]] = set()
    unique = []
    for short, long in out:
        key = (short.casefold(), long.casefold())
        if key not in seen and short.casefold() != long.casefold():
            seen.add(key)
            unique.append((short, long))
    return unique


# --------------------------------------------------------------------------- spellings


_ID_SPLIT = re.compile(r"\b([^\W\d_]+)[\s\-]*(\d[\w.\-]*)")


def _alias_variants(base: str, vocab: Vocabulary) -> list[str]:
    """``base`` with each alias it contains swapped for the alias's other forms."""
    out: list[str] = []
    for canonical, forms in vocab.surface.items():
        group = [canonical, *forms]
        found = next((f for f in group if f in base and _form_pattern(f).search(base)), None)
        if found is not None:
            pattern = _form_pattern(found)
            out.extend(pattern.sub(other, base) for other in group if other != found)
    return out


@lru_cache(maxsize=1024)
def _form_pattern(form: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(form)}(?![\w-])")


def spellings(text: str, vocab: Vocabulary | None = None, *, limit: int = 12) -> list[str]:
    """Ways the same subject may have been stored, for an exact-match lookup: ``text`` as
    written first (an identity such as "user:Alice" only as written - ids are
    case-sensitive), then lower-cased, the first identifier joined by nothing, a space, a
    hyphen, "#" or "no." ("sku-1001", "sku 1001", "sku1001"), and every alias swapped for
    its other forms ("hazmat storage", "hazardous material storage"); at most ``limit``.
    A lookup aid only: the comparison itself is ``compare``."""
    vocab = vocab or vocabulary()
    written = " ".join(text.strip()[:MAX_SUBJECT_CHARS].split())
    if not written or is_identity(written):
        return [written] if written else []
    base = written.casefold()
    out = [written]

    def add(s: str) -> None:
        s = " ".join(s.split())
        if s and s not in out:
            out.append(s)

    add(base)

    for variant in _alias_variants(base, vocab):
        add(variant)
    for variant in list(out[1:]):
        m = _ID_SPLIT.search(variant)
        if m:
            for joiner in (" ", "", "-", " #", " no. "):
                add(variant[: m.start()] + m.group(1) + joiner + m.group(2) + variant[m.end() :])
    return out[:limit]
