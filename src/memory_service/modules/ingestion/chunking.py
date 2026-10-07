"""Natural-unit chunking + Contextual Retrieval representations.

Rules:
* a paragraph/table/code block that fits ``max_tokens`` becomes exactly one chunk;
* an oversized paragraph is split on sentence boundaries with token overlap;
* an oversized table is split by rows, repeating the header row in every part;
* an oversized code block is split on blank-line boundaries;
* a chunk's ``text`` is an exact slice of its node's text — the source whitespace kept,
  overlap included — except a split table's parts, which repeat the header row;
* an oversized multi-line node (a list, merged lines) is split between lines first: a line
  that fits the budget is never broken across two chunks;
* every chunk's ``contextual_text`` prepends deterministic context (document title, section
  path, page, entities, and for a footnote the sentence that cites it) so BM25 and dense
  embeddings see where the text sits;
* when the ``chunk_context`` LLM use is enabled, a bounded subset of chunks (parts of a split
  node, tables) additionally gets a model-written situating sentence after that header.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from itertools import accumulate
from typing import Any

from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import Representation
from memory_service.domain.ids import content_hash
from memory_service.domain.language import is_english
from memory_service.domain.text import SENTENCE_BREAK, token_units
from memory_service.modules.ingestion.context_graph import extract_entities, footnote_citations
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.llm.assist import LLMAssist

_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "context": {"type": "string"}},
                "required": ["index", "context"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["contexts"],
    "additionalProperties": False,
}
_CONTEXT_SYSTEM = (
    "You write a short situating context for passages of a document so search can find "
    "them. For every chunk given, write 1-2 sentences (at most 60 words) saying what the "
    "passage is about and how it fits its section: the subject, what its figures or table "
    "rows refer to, and any name or term the passage relies on but does not repeat. Use only "
    "the text provided. Return JSON only: "
    '{"contexts": [{"index": <chunk index>, "context": "..."}]}.'
)
_CONTEXT_MAX_CHARS = 400
_BLANK_LINE = re.compile(r"\n\s*\n")
_LINE_BREAK = re.compile(r"\n+")
_HEADER_MAX_TOKENS = 96


def _document_salience(nodes: Iterable[DocumentNode], *, keep: int = 8) -> list[str]:
    """The entities this document is *about*, by how often they are named.

    A first attempt walked the node tree upwards, which turned out to be a no-op: the parser
    gives DOCUMENT and SECTION nodes empty text, so ancestors carry no entities, and the
    subject of a document is almost always introduced in a *sibling* section — "Acme is
    headquartered in Dortmund" under Overview, "the city hosts its largest facility" under
    Operations. Ancestry cannot reach across that; document scope can.

    Frequency is the salience signal: a name repeated through a document is what the document
    is about, and a name mentioned once is not worth crowding every chunk header with.
    """
    counts: dict[str, int] = {}
    for node in nodes:
        for name in node.entities or extract_entities(node.text or ""):
            counts[name] = counts.get(name, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:keep]]


def _chunk_entities(
    node: DocumentNode, document_entities: list[str], *, limit: int = 12
) -> list[str]:
    """The node's own entities first, then the document's salient ones.

    This is the deterministic half of what late chunking does with attention — and unlike late
    chunking it puts the proper noun in the *text*, so BM25 benefits too. That matters more
    here than the dense side: retrieval on an external corpus held up even with a random
    embedding, so the lexical path is carrying the result. It also needs no token-level access
    to the model, so it survives moving inference to a served tier.
    """
    seen: dict[str, None] = dict.fromkeys(node.entities or extract_entities(node.text or ""))
    for name in document_entities:
        seen.setdefault(name, None)
    return list(seen)[:limit]


def contextual_header(
    *,
    document_title: str,
    section_path: str,
    page: int | None,
    entities: list[str],
    table_title: str | None = None,
    footnote_to: str | None = None,
) -> str:
    lines = [f"Document: {document_title}"]
    if section_path and section_path != document_title:
        lines.append(f"Section: {section_path}")
    if page is not None:
        lines.append(f"Page: {page}")
    if table_title:
        lines.append(f"Table: {table_title}")
    if footnote_to:
        # a footnote's subject is in the sentence that cites it (context_graph.footnote_citations)
        lines.append(f"Footnote to: {footnote_to}")
    if entities:
        lines.append("Entities: " + ", ".join(entities[:12]))
    header = "\n".join(lines)
    # Source bodies must not be displaced by unbounded titles or section paths. Full
    # metadata remains on the document/node; this prefix is an indexing representation.
    # This shares the conservative packing estimate, not an exact tokenizer guarantee.
    weight = 0
    for index, char in enumerate(header):
        weight += token_units(char)
        if weight > _HEADER_MAX_TOKENS * 4:
            return header[:index].rstrip()
    return header


Span = tuple[int, int]


def _trimmed(text: str, start: int, end: int) -> Span | None:
    """``(start, end)`` without its surrounding whitespace; ``None`` when nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _weights(text: str) -> list[int]:
    """Prefix sums of :func:`token_units`, so any slice's weight is one subtraction."""
    return list(accumulate(map(token_units, text), initial=0))


def _hard_split_spans(
    text: str, max_tokens: int, start: int = 0, end: int | None = None
) -> list[Span]:
    """Linear scan with a whitespace preference; unspaced text is bounded too.

    Repeatedly joining all words made long inputs quadratic and never split a single
    oversized word. Track its weighted length instead; source characters are never omitted.
    Returns spans of ``text`` (within ``start:end``), each trimmed of surrounding whitespace.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    end = len(text) if end is None else end
    spans: list[Span] = []
    weight = boundary_weight = 0
    boundary = -1
    for index in range(start, end):
        char = text[index]
        cost = token_units(char)
        if cost > max_tokens * 4:
            raise ValueError("max_tokens cannot hold one source character")
        while weight + cost > max_tokens * 4 and index > start:
            cut = boundary if boundary > start else index
            if span := _trimmed(text, start, cut):
                spans.append(span)
            weight = weight - boundary_weight if boundary > start else 0
            start, boundary, boundary_weight = cut, -1, 0
        weight += cost
        if char.isspace():
            boundary, boundary_weight = index + 1, weight
    if span := _trimmed(text, start, end):
        spans.append(span)
    return spans


def _hard_split(text: str, max_tokens: int) -> list[str]:
    return [text[a:b] for a, b in _hard_split_spans(text, max_tokens)]


def _bounded_spans(
    text: str, spans: Iterable[Span], weights: list[int], max_tokens: int
) -> list[Span]:
    """``spans`` with every one heavier than the whole budget hard-split into pieces."""
    out: list[Span] = []
    for a, b in spans:
        if weights[b] - weights[a] > max_tokens * 4:
            out.extend(_hard_split_spans(text, max_tokens, a, b))
        else:
            out.append((a, b))
    return out


def _pack_spans(
    spans: Sequence[Span],
    weights: list[int],
    max_tokens: int,
    overlap_tokens: int = 0,
    starts: Sequence[int] = (),
) -> list[Span]:
    """Greedily pack consecutive ``spans`` into parts of at most ``max_tokens``.

    A part is the *source slice* from its first unit's start to its last unit's end, so the
    whitespace between units is the original whitespace and is counted in the budget. With
    ``overlap_tokens``, a part begins with the previous part's tail from the earliest of
    ``starts`` (the sentence starts, ascending) that keeps that tail within the overlap
    and the next unit within the budget — so a part's overlap can begin inside a line
    that the previous part holds whole."""
    budget, overlap = max_tokens * 4, overlap_tokens * 4
    parts: list[Span] = []
    first = last = -1
    for a, b in spans:
        if first >= 0 and weights[b] - weights[first] > budget:
            parts.append((first, last))
            previous, first = first, -1
            if overlap:
                for point in starts[bisect_right(starts, previous) :]:
                    if point >= last:
                        break
                    if (
                        weights[last] - weights[point] <= overlap
                        and weights[b] - weights[point] <= budget
                    ):
                        first = point
                        break
        if first < 0:
            first = a
        last = b
    if first >= 0:
        parts.append((first, last))
    return parts


def _segments(
    text: str, separator: re.Pattern[str], start: int = 0, end: int | None = None
) -> list[Span]:
    """The pieces of ``text[start:end]`` between ``separator`` matches, as trimmed spans."""
    end = len(text) if end is None else end
    spans: list[Span] = []
    position = start
    for match in separator.finditer(text, start, end):
        if span := _trimmed(text, position, match.start()):
            spans.append(span)
        position = match.end()
    if span := _trimmed(text, position, end):
        spans.append(span)
    return spans


def _split_sentence_spans(text: str, max_tokens: int, overlap_tokens: int) -> list[Span]:
    """A line (a paragraph or list item) that fits the budget is one unit, so a part never
    breaks inside it; a longer line is split into its sentences, and a "sentence" longer
    than the whole budget is not a sentence, so it is hard-split. The overlap may begin at
    any sentence."""
    weights = _weights(text)
    units: list[Span] = []
    starts: list[int] = []
    for a, b in _segments(text, _LINE_BREAK):
        sentences = _bounded_spans(text, _segments(text, SENTENCE_BREAK, a, b), weights, max_tokens)
        starts.extend(start for start, _ in sentences)
        units.extend([(a, b)] if weights[b] - weights[a] <= max_tokens * 4 else sentences)
    return _pack_spans(units, weights, max_tokens, overlap_tokens, starts)


def _split_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Parts of ``text`` split on line, then sentence boundaries with overlap; each an exact
    slice of ``text``."""
    return [text[a:b] for a, b in _split_sentence_spans(text, max_tokens, overlap_tokens)]


def _split_code_spans(text: str, max_tokens: int) -> list[Span]:
    weights = _weights(text)
    blocks = _bounded_spans(text, _segments(text, _BLANK_LINE), weights, max_tokens)
    return _pack_spans(blocks, weights, max_tokens)


def _split_code(text: str, max_tokens: int) -> list[str]:
    return [text[a:b] for a, b in _split_code_spans(text, max_tokens)]


def _split_table(text: str, max_tokens: int) -> list[str]:
    rows = text.split("\n")
    if len(rows) < 3:
        return _hard_split(text, max_tokens)
    header = rows[:2] if re.match(r"^\s*\|?\s*:?-{2,}", rows[1]) else rows[:1]
    prefix = "\n".join(header) + "\n"
    # Repeating a header that consumes the budget leaves no room for source rows.
    # Preserve all text in bounded fragments; node lineage retains the full table.
    if sum(map(token_units, prefix)) + 16 > max_tokens * 4:
        return _hard_split(text, max_tokens)
    return _pack_units(rows[len(header) :], max_tokens, separator="\n", prefix=prefix)


def _pack_units(
    units: Iterable[str], max_tokens: int, *, separator: str, prefix: str = ""
) -> list[str]:
    """Pack natural units in linear time, splitting even a single oversized unit."""
    prefix_weight = sum(map(token_units, prefix))
    available = (max_tokens * 4 - prefix_weight) // 4
    separator_weight = sum(map(token_units, separator))
    parts: list[str] = []
    current: list[str] = []
    weight = prefix_weight
    for unit in units:
        for fragment in _hard_split(unit, available):
            fragment_weight = sum(map(token_units, fragment))
            added = fragment_weight + (separator_weight if current else 0)
            if current and weight + added > max_tokens * 4:
                parts.append(prefix + separator.join(current))
                current = []
                weight = prefix_weight
            weight += fragment_weight + (separator_weight if current else 0)
            current.append(fragment)
    if current:
        parts.append(prefix + separator.join(current))
    return parts


def _node_parts(
    node: DocumentNode, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, Span | None]]:
    """The node's parts, each with its span in ``node.text`` — every part but a split table's
    is an exact source slice (a table part repeats the header row, so it has no span)."""
    text = node.text
    # Stored estimates can predate a tokenizer/packing policy change.
    if estimate_tokens(text) <= max_tokens:
        spans = [span] if (span := _trimmed(text, 0, len(text))) else []
    elif node.representation is Representation.TABLE:
        return [(part, None) for part in _split_table(text, max_tokens) if part.strip()]
    elif node.representation is Representation.CODE_BLOCK:
        spans = _split_code_spans(text, max_tokens)
    else:
        spans = _split_sentence_spans(text, max_tokens, overlap_tokens)
    return [(text[a:b], (a, b)) for a, b in spans]


def _merge_tiny(
    node: DocumentNode,
    parts: Sequence[tuple[str, Span | None]],
    *,
    max_tokens: int,
    min_tokens: int,
) -> list[str]:
    """Merge a tiny part into the previous one when the two fit (keeps natural units). Two
    source slices merge into the slice that covers both, so an overlap is not repeated and
    the original whitespace between them is kept."""
    merged: list[tuple[str, Span | None]] = []
    for text, span in parts:
        if merged and estimate_tokens(text) < min_tokens:
            previous, previous_span = merged[-1]
            if previous_span is not None and span is not None:
                joined_span: Span | None = (previous_span[0], span[1])
                joined = node.text[previous_span[0] : span[1]]
            else:
                joined_span, joined = None, previous + "\n" + text
            if estimate_tokens(joined) <= max_tokens:
                merged[-1] = (joined, joined_span)
                continue
        merged.append((text, span))
    return [text for text, _ in merged]


def chunk_nodes(
    nodes: Iterable[DocumentNode],
    *,
    document_title: str,
    max_tokens: int = 400,
    min_tokens: int = 40,
    overlap_tokens: int = 40,
    contextual: bool = True,
) -> list[Chunk]:
    """Chunks of ``nodes``. A chunk's ``text`` is an exact slice of its node's text — the
    source whitespace kept, overlap included — except a split table's parts, which repeat
    the header row; ``contextual_text`` is the deterministic header plus that text."""
    chunks: list[Chunk] = []
    nodes = tuple(nodes)
    document_entities = _document_salience(nodes)
    citations = footnote_citations(nodes)
    for node in nodes:
        if (
            node.representation
            in (Representation.DOCUMENT, Representation.SECTION, Representation.SUBSECTION)
            or not node.text
        ):
            continue
        parts = _merge_tiny(
            node,
            _node_parts(node, max_tokens, overlap_tokens),
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        if not parts:
            continue
        entities = _chunk_entities(node, document_entities)
        table_title = node.title if node.representation is Representation.TABLE else None
        header = contextual_header(
            document_title=document_title,
            section_path=node.section_path,
            page=node.page_start,
            entities=entities,
            table_title=table_title,
            footnote_to=citations.get(node.node_id),
        )
        for ordinal, text in enumerate(parts):
            chunks.append(
                Chunk(
                    node_id=node.node_id,
                    document_id=node.document_id,
                    document_version_id=node.document_version_id,
                    tenant_id=node.tenant_id,
                    ordinal=ordinal,
                    text=text,
                    text_hash=content_hash(text),
                    contextual_text=f"{header}\n\n{text}" if contextual else text,
                    page=node.page_start,
                    section_path=node.section_path,
                    token_estimate=estimate_tokens(text),
                    entities=entities,
                )
            )
    return chunks


def _window(node_text: str, chunk_text: str, chars: int) -> str:
    half = chars // 2
    pos = node_text.find(chunk_text[:120])
    if pos < 0:
        return node_text[:chars]
    before = node_text[max(0, pos - half) : pos]
    after = node_text[pos + len(chunk_text) : pos + len(chunk_text) + half]
    return f"{before}[…the chunk…]{after}".strip()


def situated_candidates(
    chunks: Sequence[Chunk], nodes: Sequence[DocumentNode], *, max_chunks: int
) -> list[int]:
    """Indexes of the chunks whose deterministic header lost context: parts of a node that
    was split (first), then tables, then chunks not in English (the header's salient
    entities come from English-shaped rules, so it situates them poorly); bounded to
    ``max_chunks``."""
    by_node = {n.node_id: n for n in nodes}
    parts_per_node: dict[str, int] = {}
    for c in chunks:
        parts_per_node[c.node_id] = parts_per_node.get(c.node_id, 0) + 1
    split = [i for i, c in enumerate(chunks) if parts_per_node[c.node_id] > 1]
    tables = [
        i
        for i, c in enumerate(chunks)
        if parts_per_node[c.node_id] == 1
        and c.node_id in by_node
        and by_node[c.node_id].representation is Representation.TABLE
    ]
    chosen = {*split, *tables}
    foreign = [i for i, c in enumerate(chunks) if i not in chosen and not is_english(c.lang)]
    return [*split, *tables, *foreign][:max_chunks]


async def situate_chunks(
    assist: LLMAssist,
    chunks: Sequence[Chunk],
    nodes: Sequence[DocumentNode],
    *,
    document_title: str,
    max_chunks: int = 48,
    batch_size: int = 8,
    window_chars: int = 1500,
) -> list[Chunk]:
    """Prepend a model-written situating context (after the deterministic header) to a bounded
    subset of chunks. ``text`` never changes; any model failure leaves the chunk as it is."""
    out = list(chunks)
    if not assist.wants("chunk_context"):
        return out
    by_node = {n.node_id: n for n in nodes}
    wanted = situated_candidates(chunks, nodes, max_chunks=max_chunks)
    for start in range(0, len(wanted), batch_size):
        batch = wanted[start : start + batch_size]
        sections = [f"Document: {document_title}"]
        for slot, i in enumerate(batch):
            c = chunks[i]
            node = by_node.get(c.node_id)
            surrounding = _window(node.text, c.text, window_chars) if node else ""
            sections.append(
                f"### Chunk {slot}\nSection: {c.section_path or document_title}\n"
                f"Surrounding text: {surrounding}\nChunk text: {c.text[:800]}"
            )
        result = await assist.structured(
            "chunk_context",
            system=_CONTEXT_SYSTEM,
            user="\n\n".join(sections),
            schema=_CONTEXT_SCHEMA,
            max_tokens=1024,
        )
        if result is None:
            continue
        for item in result.get("contexts", []):
            slot = item.get("index")
            context = " ".join(str(item.get("context", "")).split())[:_CONTEXT_MAX_CHARS]
            if not isinstance(slot, int) or not 0 <= slot < len(batch) or not context:
                continue
            c = chunks[batch[slot]]
            head = (
                c.contextual_text[: len(c.contextual_text) - len(c.text)]
                if c.contextual_text.endswith(c.text)
                else ""
            )
            out[batch[slot]] = c.model_copy(
                update={"contextual_text": f"{head}Context: {context}\n\n{c.text}"}
            )
    return out
