"""Document Context Graph: deterministic structural links (no LLM).

Edges (see ``ContextGraphEdge``): PARENT/CHILD, PREVIOUS/NEXT (siblings in reading order),
ON_PAGE (node -> ``page:<document>:<n>``), IN_TABLE (chunk -> table node), FOOTNOTE
(paragraph citing ``[^n]`` or a superscript marker -> footnote node), CROSS_REFERENCE
("see Section 3.2", "Table 4", "Appendix B", "Note 12" -> target node), MENTIONS
(node -> ``entity:<canonical>``) and DEFINED_BY / DEFINES (a node that uses a defined term ->
the node that defines it).

This graph is separate from the semantic Knowledge Graph (M8) and is what makes
"definition on page 1, value on page 11, footnote on page 20" recoverable at retrieval time.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from memory_service.domain.documents import ContextEdge, DocumentNode
from memory_service.domain.enums import ContextGraphEdge, Representation

# --------------------------------------------------------------------------- entities

_STOP = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "There",
    "Here",
    "When",
    "Where",
    "While",
    "With",
    "Without",
    "For",
    "From",
    "During",
    "After",
    "Before",
    "Under",
    "Over",
    "Our",
    "Its",
    "Their",
    "Your",
    "His",
    "Her",
    "And",
    "But",
    "However",
    "Therefore",
    "Thus",
    "Also",
    "Note",
    "Notes",
    "See",
    "Section",
    "Table",
    "Figure",
    "Page",
    "Appendix",
    "Chapter",
    "Part",
    "Item",
    "Total",
    "Other",
    "Net",
    "Gross",
    "Year",
    "Quarter",
    "In",
    "On",
    "At",
    "To",
    "Of",
    "As",
    "By",
    "Or",
    "If",
    "It",
    "We",
    "A",
    "An",
    "Is",
    "Are",
    "Was",
    "Were",
}
_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9&]{1,9})\b")
_CAP_PHRASE = re.compile(
    r"\b([A-Z][a-zA-Z0-9\-]+(?:\s+(?:of|and|for|the|de|du)?\s*[A-Z][a-zA-Z0-9\-]+){0,4})\b"
)
_QUOTED_TERM = re.compile(r"[\"“]([A-Za-z][A-Za-z0-9 \-]{2,40})[\"”]")


_CONNECTORS = {"of", "and", "for", "the", "de", "du"}
_CURRENCY = {"EUR", "USD", "GBP", "CHF", "JPY", "INR", "CAD", "AUD"}


def _phrase_variants(phrase: str) -> list[str]:
    """Split connector-joined phrases: 'Adjusted EBITDA of EUR' -> ['Adjusted EBITDA']."""
    words = phrase.split()
    idx = [i for i, w in enumerate(words) if w in _CONNECTORS]
    if not idx:
        return [phrase]
    first = idx[0]
    head = " ".join(words[:first])
    tail = words[first + 1 :]
    variants = [head] if head else []
    if tail and not (tail[0].isupper() and (tail[0] in _CURRENCY or len(tail[0]) <= 3)):
        variants.append(phrase)
    return variants or [phrase]


def canonical_entity(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def extract_entities(text: str, *, max_entities: int = 24) -> list[str]:
    """Deterministic entity candidates: acronyms, capitalised phrases, quoted defined terms."""
    found: dict[str, int] = {}
    for m in _ACRONYM.finditer(text):
        token = m.group(1)
        if token not in _STOP and not token.isdigit() and len(token) >= 2:
            found[token] = found.get(token, 0) + 2
    for m in _CAP_PHRASE.finditer(text):
        for phrase in _phrase_variants(m.group(1).strip()):
            words = phrase.split()
            if words[0] in _STOP and len(words) == 1:
                continue
            if all(w in _STOP for w in words):
                continue
            if len(words) == 1 and len(phrase) < 4:
                continue
            found[phrase] = found.get(phrase, 0) + 1
    for m in _QUOTED_TERM.finditer(text):
        found[m.group(1).strip()] = found.get(m.group(1).strip(), 0) + 2
    ranked = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[str] = []
    seen: set[str] = set()
    for name, _ in ranked:
        canon = canonical_entity(name)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(name)
        if len(out) >= max_entities:
            break
    return out


# --------------------------------------------------------------------------- definitions

_DEF_PATTERNS = [
    re.compile(
        r"^\**\"?([A-Z][A-Za-z0-9 \-/&]{1,60}?)\"?\**\s*(?:\(\"?[A-Za-z ]+\"?\))?\s*"
        r"(?:means|refers to|is defined as|denotes|shall mean)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\**([A-Z][A-Za-z0-9 \-/&]{1,60}?)\**\s*[:—-]\s+(?:[A-Z]|the\b|a\b|an\b)", re.MULTILINE
    ),
    re.compile(r"\bwe define\s+\"?([A-Z][A-Za-z0-9 \-/&]{1,60}?)\"?\s+as\b", re.IGNORECASE),
    re.compile(r"\"([A-Z][A-Za-z0-9 \-/&]{1,60}?)\"\s+(?:means|refers to|is defined as)\b"),
]
_PAREN_ALIAS = re.compile(r"([A-Z][A-Za-z0-9 \-/&]{2,60}?)\s+\(\"?([A-Z][A-Za-z0-9 ]{1,30}?)\"?\)")


def extract_definitions(text: str) -> list[str]:
    """Terms this text *defines* (deterministic patterns; glossary-style and prose)."""
    terms: list[str] = []
    first_line = text.strip().split("\n", 1)[0]
    for pat in _DEF_PATTERNS:
        for m in pat.finditer(
            text
            if pat.flags & re.MULTILINE or pat is _DEF_PATTERNS[2] or pat is _DEF_PATTERNS[3]
            else first_line
        ):
            term = m.group(1).strip(' *"')
            if 2 <= len(term) <= 60 and term.split()[0] not in _STOP:
                terms.append(term)
    for m in _PAREN_ALIAS.finditer(text):
        long, short = m.group(1).strip(), m.group(2).strip()
        if short.isupper() or short[0].isupper():
            terms.extend([long, short])
    seen: set[str] = set()
    out = []
    for t in terms:
        c = canonical_entity(t)
        if c in seen or _NOT_A_TERM.fullmatch(c):
            continue
        seen.add(c)
        out.append(t)
    return out


_NOT_A_TERM = re.compile(
    r"(?:fy\s?\d{2,4}|q[1-4](?:\s?\d{2,4})?|h[12]\s?\d{2,4}|(?:19|20)\d{2}|"
    r"(?:eur|usd|gbp|chf|jpy|inr|cad|aud)\s*(?:m|mn|bn|k|million|billion|thousand)?|"
    r"table|figure|note|section)"
)


# --------------------------------------------------------------------------- references

_XREF = re.compile(
    r"\b(?:see|refer to|in|per|under|as (?:described|discussed|noted) in)?\s*"
    r"(Section|Sec\.|Table|Figure|Fig\.|Appendix|Note|Chapter|Part|Item|Schedule|Exhibit|Annex)\s+"
    r"([A-Z]?\d+(?:\.\d+)*[a-z]?|[A-Z](?![a-z]))\b",
    re.IGNORECASE,
)
_FOOTNOTE_REF = re.compile(
    r"\[\^([^\]]+)\]|(?<=[A-Za-z0-9%)])\^(\d{1,2})\b|\(\s*(?:note|footnote)\s+(\d{1,3})\s*\)",
    re.IGNORECASE,
)


_KIND_ALIASES = {"sec": "Section", "fig": "Figure"}


def extract_references(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _XREF.finditer(text):
        kind = m.group(1).rstrip(".").lower()
        out.append((_KIND_ALIASES.get(kind, kind.title()), m.group(2)))
    return out


def extract_footnote_refs(text: str) -> list[str]:
    out = []
    for m in _FOOTNOTE_REF.finditer(text):
        label = m.group(1) or m.group(2) or m.group(3)
        if label:
            out.append(label)
    return out


# --------------------------------------------------------------------------- graph


def _edge(
    tenant_id: str,
    document_id: str,
    src: str,
    dst: str,
    kind: ContextGraphEdge,
    *,
    label: str | None = None,
    weight: float = 1.0,
) -> ContextEdge:
    return ContextEdge(
        tenant_id=tenant_id,
        document_id=document_id,
        source_id=src,
        target_id=dst,
        edge=kind,
        label=label,
        weight=weight,
    )


def build_context_graph(
    nodes: Iterable[DocumentNode], *, tenant_id: str, document_id: str
) -> list[ContextEdge]:
    nodes = list(nodes)
    by_id = {n.node_id: n for n in nodes}
    edges: list[ContextEdge] = []
    seen: set[tuple[str, str, str]] = set()

    def add(
        src: str, dst: str, kind: ContextGraphEdge, *, label: str | None = None, weight: float = 1.0
    ) -> None:
        key = (src, dst, kind.value)
        if src == dst or key in seen:
            return
        seen.add(key)
        edges.append(_edge(tenant_id, document_id, src, dst, kind, label=label, weight=weight))

    # PARENT / CHILD / siblings
    children: dict[str | None, list[DocumentNode]] = defaultdict(list)
    for n in nodes:
        children[n.parent_id].append(n)
    for parent_id, kids in children.items():
        kids.sort(key=lambda n: n.ordinal)
        for i, kid in enumerate(kids):
            if parent_id:
                add(kid.node_id, parent_id, ContextGraphEdge.PARENT)
                add(parent_id, kid.node_id, ContextGraphEdge.CHILD)
            if i > 0:
                add(kid.node_id, kids[i - 1].node_id, ContextGraphEdge.PREVIOUS)
                add(kids[i - 1].node_id, kid.node_id, ContextGraphEdge.NEXT)

    # Reading order across section boundaries.
    #
    # The sibling pass above links children of the *same* parent, so the last paragraph of one
    # section and the first of the next are never connected — and that is exactly where a
    # referent goes missing: "Acme is headquartered in Dortmund" under Overview, "the city
    # hosts its largest facility" under Operations. Neighbour expansion could not cross it, so
    # the answer reached the model without the thing it refers to.
    #
    # Only consecutive *leaves* are linked, and only where the sibling pass did not already
    # connect them, so this adds one edge pair per section transition rather than growing the
    # fan-out (already ~10 edges per chunk).
    leaves = sorted(
        (
            n
            for n in nodes
            if n.text
            and n.representation
            not in (Representation.DOCUMENT, Representation.SECTION, Representation.SUBSECTION)
        ),
        key=lambda n: (n.page_start or 0, n.ordinal),
    )
    for i in range(1, len(leaves)):
        previous, current = leaves[i - 1], leaves[i]
        if previous.parent_id == current.parent_id:
            continue  # already linked as siblings
        add(current.node_id, previous.node_id, ContextGraphEdge.PREVIOUS)
        add(previous.node_id, current.node_id, ContextGraphEdge.NEXT)

    # ON_PAGE
    for n in nodes:
        if n.page_start is not None and n.representation not in (
            Representation.DOCUMENT,
            Representation.SECTION,
            Representation.SUBSECTION,
        ):
            add(n.node_id, f"page:{document_id}:{n.page_start}", ContextGraphEdge.ON_PAGE)

    # index headings, tables, footnotes, definitions, entities
    section_by_number: dict[str, str] = {}
    section_by_title: dict[str, str] = {}
    tables: dict[str, str] = {}
    footnotes: dict[str, str] = {}
    definitions: dict[str, str] = {}
    entity_nodes: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        meta = n.system_metadata or {}
        if n.representation in (Representation.SECTION, Representation.SUBSECTION) and n.title:
            if meta.get("section_number"):
                section_by_number[str(meta["section_number"])] = n.node_id
            section_by_title[canonical_entity(n.title)] = n.node_id
            m = re.match(
                r"^(appendix|annex|exhibit|schedule|part|chapter)\s+([A-Z]|\d+)\b",
                n.title,
                re.IGNORECASE,
            )
            if m:
                section_by_number[f"{m.group(1).title()} {m.group(2)}"] = n.node_id
        if n.representation is Representation.TABLE:
            label = str(meta.get("label") or n.title or "")
            m = re.match(r"^table\s+([A-Z]?\d+(?:\.\d+)*)", label, re.IGNORECASE)
            if m:
                tables[m.group(1)] = n.node_id
        if meta.get("block_kind") == "footnote" and meta.get("label"):
            footnotes[str(meta["label"])] = n.node_id
        if n.text:
            for term in extract_definitions(n.text):
                definitions.setdefault(canonical_entity(term), n.node_id)
            for ent in n.entities or extract_entities(n.text):
                entity_nodes[canonical_entity(ent)].append(n.node_id)

    # MENTIONS, DEFINED_BY / DEFINES, FOOTNOTE, CROSS_REFERENCE, IN_TABLE
    for n in nodes:
        if not n.text:
            continue
        ents = n.entities or extract_entities(n.text)
        for ent in ents:
            canon = canonical_entity(ent)
            add(n.node_id, f"entity:{canon}", ContextGraphEdge.MENTIONS, label=ent)
            definer = definitions.get(canon)
            if definer and definer != n.node_id:
                add(n.node_id, definer, ContextGraphEdge.DEFINED_BY, label=ent, weight=2.0)
                add(definer, n.node_id, ContextGraphEdge.DEFINES, label=ent)
        lowered = n.text.casefold()
        for canon, definer in definitions.items():
            if definer != n.node_id and canon in lowered:
                add(n.node_id, definer, ContextGraphEdge.DEFINED_BY, label=canon, weight=2.0)
                add(definer, n.node_id, ContextGraphEdge.DEFINES, label=canon)
        for label in extract_footnote_refs(n.text):
            target = footnotes.get(label)
            if target:
                add(n.node_id, target, ContextGraphEdge.FOOTNOTE, label=label, weight=2.0)
        for kind, number in extract_references(n.text):
            target = None
            if kind == "Table":
                target = tables.get(number)
            elif kind in ("Section", "Chapter", "Part", "Item"):
                target = section_by_number.get(number)
            elif kind in ("Appendix", "Annex", "Exhibit", "Schedule"):
                target = section_by_number.get(f"{kind} {number}")
            elif kind == "Note":
                target = footnotes.get(number) or section_by_number.get(number)
            if target:
                add(n.node_id, target, ContextGraphEdge.CROSS_REFERENCE, label=f"{kind} {number}")
        parent = by_id.get(n.parent_id or "")
        if parent is not None and parent.representation is Representation.TABLE:
            add(n.node_id, parent.node_id, ContextGraphEdge.IN_TABLE)
    return edges
