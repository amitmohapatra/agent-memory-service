"""Standing rules and preferences that apply to a request, as profile sections (ADR 0041).

A user's standing rule ("Never schedule a delivery on a public holiday", "Always answer weekly
reports as a table with columns item, branch, count; no prose", "Nie ...", "Nunca
...") and their stated preferences ("I care about fast-moving items with low stock", "Mir ist
eine kurze Lieferzeit wichtig") are lasting memories (ADR 0036/0037), and they reached a new
conversation only through the ``user`` profile block: one bounded block of all of them,
newest first, so under memory that is never cleared a rule said a month ago fell off its
end - and the ranked memories carried it only when it happened to rank among the answers.

Here every request reads the user's rules and preferences that apply to it, from the
memories collection, in one round trip beside the ranking (a dense search in the primary
space and a keyword search over those categories only, with the query's own vectors): a rule
applies when the encoder puts it close to the request (``RULE_SIMILARITY``) or it shares a
content word or code with it (``SHARED_STEMS``), in any language; a preference the same way at
``PREFERENCE_SIMILARITY``. A later rule on the same subject supersedes an earlier one
(``_superseded``), what was said with an applying one comes with it; an earlier value (ADR
0038) never applies. They come back as two profile sections, ``standing_rules`` and
``preferences``, in new conversations as in old ones, bounded (``RULES_MAX``,
``PREFERENCES_MAX``).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any, Final

from memory_service.domain.context_bundle import ProfileBlockView
from memory_service.domain.language import detect_language
from memory_service.modules.context.evidence import content_terms, stemmed_terms
from memory_service.modules.memory.replacement import _function_words
from memory_service.ports.search import SearchHit

RULES_BLOCK: Final = "standing_rules"
PREFERENCES_BLOCK: Final = "preferences"
#: the sections this module writes (``sections.within_budget`` keeps them before the rest)
BLOCKS: Final = frozenset({RULES_BLOCK, PREFERENCES_BLOCK})
RULE_CATEGORIES: Final = ("rule", "instruction")
#: a statement the labeller read as a rule (ADR 0037) is one whatever rule extracted it; a
#: correction of a fact ("Update: B replaced A as the buyer") is not, and reaches the bundle as
#: the memory it is; and so is a statement that gives an instruction it says is for every
#: time, which the labeller leaves to a model a no-model deployment does not have ("Formatiere
#: Berichte immer als Tabelle": ``standing_instruction``)
RULE_KINDS: Final = ("RULE", "CONDITIONAL_RULE")
PREFERENCE_CATEGORIES: Final = ("preference", "attribute")
#: how many of each a search returns, and how many are shown
SEARCH_K: Final = 24
RULES_MAX: Final = 6
PREFERENCES_MAX: Final = 4
#: Calibrated on the capability eval's dev split (the rules and preferences its scenarios
#: state, against their own dev questions and every other scenario's, five languages): the
#: primary space's query-to-statement cosines run low (own p50 0.29, others' p99 0.35); at 0.4
#: (with two shared words) a statement applies to 52% of its own questions and 4% of others',
#: most of either by the shared words. Higher floors add nothing but misses.
RULE_SIMILARITY: Final = 0.4
PREFERENCE_SIMILARITY: Final = 0.4
#: shared content words that make a rule apply whatever the encoder says. On the dev split
#: (as for the floors above): one shared word applies a statement to 69.5% of its own
#: questions and 14.9% of other scenarios', two to 49% and 3.9%. Missing a rule that applies
#: costs more than listing one that does not (the reader judges a rule's condition), and the
#: sections are bounded and ordered by shared words, then similarity.
SHARED_STEMS: Final = 1
#: two rules are about the same subject when this share of the shorter one's stems is shared
SAME_SUBJECT: Final = 0.5
#: the words that make a rule standing are not its subject
_STANDING_WORDS: Final = frozenset(
    {
        "always",
        "never",
        "from",
        "now",
        "permanently",
        "every",
        "time",
        "whenever",
        "immer",
        "nie",
        "niemals",
        "jedes",
        "mal",
        "siempre",
        "nunca",
        "cada",
        "vez",
        "دائما",
        "دائمًا",
        "أبدا",
        "أبدًا",
        "हमेशा",
        "कभी",
        "नहीं",
    }
)


#: a number or a code ("512", "PX-90") is a shared word too: the thing it names
_CODED = re.compile(r"(?<![\w-])[\w-]*\d[\w-]*")


def _frame(text: str, lang: str) -> set[str]:
    """What a statement says, without what it says it about: its content words but its codes,
    numbers and capitalised names ("I manage store JN4786 Lakeshore Arcade" and "I manage store
    ZB1871 Willow Creek Mall" say one thing; so do two "never before 06:50 / 06:20" rules)."""
    words = {
        w
        for w in content_terms(text)
        if w not in _function_words() and not any(c.isdigit() for c in w)
    } - _STANDING_WORDS
    named = {m.lower() for m in re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][\w-]*", text)}
    return stemmed_terms(words - named, lang) - _STANDING_WORDS


def _stems(text: str, lang: str) -> set[str]:
    words = {w for w in content_terms(text) if w not in _function_words()} - _STANDING_WORDS
    coded = {c.lower().strip("-") for c in _CODED.findall(text) if len(c) > 1}
    return (stemmed_terms(words, lang) - _STANDING_WORDS) | coded


def applies(query_stems: set[str], hit_stems: set[str], similarity: float, floor: float) -> bool:
    return similarity >= floor or len(query_stems & hit_stems) >= SHARED_STEMS


def restated_rules(memories: Sequence[Any]) -> set[str]:
    """The ids of the rules (``RULE_CATEGORIES``) of ``memories`` that a later rule on the
    same subject supersedes (``SAME_SUBJECT`` of the shorter one's words, codes, numbers and
    names aside): a rule restated with new values keeps only its newest form."""
    rules = [
        (m, _frame(m.content, detect_language(m.content)))
        for m in memories
        if (m.system_metadata or {}).get("category") in RULE_CATEGORIES
    ]
    out: set[str] = set()
    for i, (a, fa) in enumerate(rules):
        for b, fb in rules[i + 1 :]:
            shorter = min(len(fa), len(fb))
            if not shorter or len(fa & fb) / shorter < SAME_SUBJECT:
                continue
            when_a, when_b = a.temporal.observed_at, b.temporal.observed_at
            if when_a == when_b:
                continue  # said at one instant: no order
            out.add((a if when_a < when_b else b).memory_id)
    return out


def _superseded(chosen: Sequence[tuple[SearchHit, set[str]]]) -> set[str]:
    """The rules a later one on the same subject supersedes: of two that share
    ``SAME_SUBJECT`` of the shorter one's stems, the earlier said."""
    out: set[str] = set()
    for i, (a, sa) in enumerate(chosen):
        for b, sb in chosen[i + 1 :]:
            shorter = min(len(sa), len(sb))
            if not shorter or len(sa & sb) / shorter < SAME_SUBJECT:
                continue
            said_a, said_b = _said(a), _said(b)
            if said_a == said_b or said_a[1] == said_b[1]:
                continue  # one statement's forms, or two said at one instant: no order
            out.add((a if said_a[1] < said_b[1] else b).record_id)
    return out


async def standing_blocks(
    engine: Any, query: str, visibility: Any, vectors: Any
) -> list[ProfileBlockView]:
    """The ``standing_rules`` and ``preferences`` sections for ``query`` (see the module
    doc). A rule the ``user`` block also lists is still listed here: this section is what
    applies to the request, the block everything the user keeps."""
    if vectors is None:
        return []
    lang = detect_language(query)
    query_stems = _stems(query, lang)
    rules, preferences = await asyncio.gather(
        _search(
            engine,
            visibility,
            vectors,
            should_any={
                "category": list(RULE_CATEGORIES),
                "statement_kind": list(RULE_KINDS),
                "standing_instruction": ["yes"],
            },
        ),
        _search(engine, visibility, vectors, must_any={"category": list(PREFERENCE_CATEGORIES)}),
    )
    out: list[ProfileBlockView] = []
    shown: set[tuple] = set()
    for block, found, floor, most in (
        (RULES_BLOCK, rules, RULE_SIMILARITY, RULES_MAX),
        (PREFERENCES_BLOCK, preferences, PREFERENCE_SIMILARITY, PREFERENCES_MAX),
    ):
        chosen = [c for c in _choose(found, query_stems, lang, floor) if _said(c[0]) not in shown]
        lines = _lines(chosen)[:most]
        shown |= {_said(hit) for hit, _, _ in chosen}
        if lines:
            out.append(ProfileBlockView(block=block, text="\n".join(lines), version=0))
    return out


def _choose(
    found: Sequence[tuple[SearchHit, float]], query_stems: set[str], lang: str, floor: float
) -> list[tuple[SearchHit, set[str], float]]:
    """The current, not superseded statements that apply, best first, with what was said in
    the same turn as each ("I manage branch 512 in the north district. I care about
    fast-moving items.")."""
    current = [
        (hit, _stems(_text(hit), lang), similarity)
        for hit, similarity in found
        if not hit.payload.get("earlier_value") and _text(hit)
    ]
    gone = _superseded([(hit, _frame(_text(hit), lang)) for hit, _, _ in current])
    live = [c for c in current if c[0].record_id not in gone]
    chosen = sorted(
        (c for c in live if applies(query_stems, c[1], c[2], floor)),
        key=lambda c: (-len(query_stems & c[1]), -c[2]),
    )
    said = {_said(hit) for hit, _, _ in chosen}
    return chosen + [c for c in live if c not in chosen and _said(c[0]) in said]


def _lines(chosen: Sequence[tuple[SearchHit, set[str], float]]) -> list[str]:
    """One line per statement, in the order chosen, each in its fullest form: a sentence read
    from a turn that the turn itself also brings ("... with columns: Store, SKU, Style. No
    prose before the table.") is not listed apart, so a list or a format comes back whole."""
    texts: dict[tuple, list[str]] = {}
    for hit, _, _ in chosen:
        texts.setdefault(_said(hit), []).append(_text(hit))
    lines: list[str] = []
    for hit, _, _ in chosen:
        text, same = _text(hit), texts[_said(hit)]
        if not any(text != other and text in other for other in same):
            lines.append(f"- {text}")
    return list(dict.fromkeys(lines))


def _said(hit: SearchHit) -> tuple:
    """Where and when a statement was said: its source turns and its instant."""
    refs = hit.payload.get("source_refs") or []
    return (
        tuple(sorted(str(r.get("source_id")) for r in refs if isinstance(r, dict))),
        str(hit.payload.get("observed_at")),
    )


def _text(hit: SearchHit) -> str:
    return " ".join(str(hit.payload.get("text") or "").split())


async def _search(
    engine: Any,
    visibility: Any,
    vectors: Any,
    *,
    must_any: dict[str, list[str]] | None = None,
    should_any: dict[str, list[str]] | None = None,
) -> list[tuple[SearchHit, float]]:
    """The user's current memories the filter admits, each with its dense similarity to the
    query (0 for one only the keyword search found), best first."""
    collection, flt = engine.memory_scope(visibility)
    flt = flt.model_copy(
        update={"must_any": {**flt.must_any, **(must_any or {})}, "should_any": should_any or {}}
    )
    primary = engine.indexer.spaces.primary_space.name
    dense = (
        (primary, vectors.dense[primary])
        if primary in vectors.dense
        else next(iter(vectors.dense.items()), None)
    )
    searches = []
    if dense is not None:
        searches.append(
            engine.store.search_dense(collection, dense[0], dense[1], flt, limit=SEARCH_K)
        )
    if vectors.sparse is not None and vectors.sparse.indices:
        searches.append(engine.store.search_sparse(collection, vectors.sparse, flt, limit=SEARCH_K))
    results = await asyncio.gather(*searches)
    scored: dict[str, tuple[SearchHit, float]] = {}
    for n, hits in enumerate(results):
        for hit in hits:
            similarity = hit.score if n == 0 and dense is not None else 0.0
            if hit.record_id not in scored or similarity > scored[hit.record_id][1]:
                scored[hit.record_id] = (hit, similarity)
    return sorted(scored.values(), key=lambda pair: -pair[1])
