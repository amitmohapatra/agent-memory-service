"""Natural-unit chunking + Contextual Retrieval representations.

Rules:
* a paragraph/table/code block that fits ``max_tokens`` becomes exactly one chunk;
* an oversized paragraph is split on sentence boundaries with token overlap;
* an oversized table is split by rows, repeating the header row in every part;
* an oversized code block is split on blank-line boundaries;
* every chunk's ``contextual_text`` prepends deterministic context (document title, section
  path, page, entities) so BM25 and dense embeddings see where the text sits;
* when the ``chunk_context`` LLM use is enabled, a bounded subset of chunks (parts of a split
  node, tables) additionally gets a model-written situating sentence after that header.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import Representation
from memory_service.domain.ids import content_hash
from memory_service.domain.text import SENTENCE_BREAK, token_units
from memory_service.modules.ingestion.context_graph import extract_entities
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
) -> str:
    lines = [f"Document: {document_title}"]
    if section_path and section_path != document_title:
        lines.append(f"Section: {section_path}")
    if page is not None:
        lines.append(f"Page: {page}")
    if table_title:
        lines.append(f"Table: {table_title}")
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


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Linear scan with a whitespace preference; unspaced text is bounded too.

    Repeatedly joining all words made long inputs quadratic and never split a single
    oversized word. Track its weighted length instead; source characters are never omitted.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    parts: list[str] = []
    start = weight = boundary_weight = 0
    boundary = -1
    for index, char in enumerate(text):
        cost = token_units(char)
        if cost > max_tokens * 4:
            raise ValueError("max_tokens cannot hold one source character")
        while weight + cost > max_tokens * 4 and index > start:
            cut = boundary if boundary > start else index
            if part := text[start:cut].strip():
                parts.append(part)
            weight = weight - boundary_weight if boundary > start else 0
            start, boundary, boundary_weight = cut, -1, 0
        weight += cost
        if char.isspace():
            boundary, boundary_weight = index + 1, weight
    if part := text[start:].strip():
        parts.append(part)
    return parts


def _split_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_BREAK.split(text) if s.strip()]
    if not sentences:
        return [text]
    # a "sentence" longer than the whole budget is not a sentence
    expanded: list[str] = []
    for sentence in sentences:
        if estimate_tokens(sentence) > max_tokens:
            expanded.extend(_hard_split(sentence, max_tokens))
        else:
            expanded.append(sentence)
    sentences = expanded
    parts: list[str] = []
    current: list[tuple[str, int]] = []
    current_units = 0
    budget = max_tokens * 4
    for sentence in sentences:
        units = sum(map(token_units, sentence))
        if current and current_units + 1 + units > budget:
            parts.append(" ".join(text for text, _ in current))
            # overlap: keep trailing sentences worth ~overlap_tokens
            kept: list[tuple[str, int]] = []
            kept_units = 0
            for retained, weight in reversed(current):
                added = weight + bool(kept)
                if (
                    kept_units + added > overlap_tokens * 4
                    or kept_units + added + 1 + units > budget
                ):
                    break
                kept.append((retained, weight))
                kept_units += added
            current, current_units = list(reversed(kept)), kept_units
        current_units += units + bool(current)
        current.append((sentence, units))
    if current:
        parts.append(" ".join(text for text, _ in current))
    return parts


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


def _split_code(text: str, max_tokens: int) -> list[str]:
    return _pack_units(re.split(r"\n\s*\n", text), max_tokens, separator="\n\n")


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


def chunk_nodes(
    nodes: Iterable[DocumentNode],
    *,
    document_title: str,
    max_tokens: int = 400,
    min_tokens: int = 40,
    overlap_tokens: int = 40,
    contextual: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    nodes = tuple(nodes)
    document_entities = _document_salience(nodes)
    for node in nodes:
        if (
            node.representation
            in (Representation.DOCUMENT, Representation.SECTION, Representation.SUBSECTION)
            or not node.text
        ):
            continue
        entities = _chunk_entities(node, document_entities)
        # Stored estimates can predate a tokenizer/packing policy change.
        tokens = estimate_tokens(node.text)
        if tokens <= max_tokens:
            parts = [node.text]
        elif node.representation is Representation.TABLE:
            parts = _split_table(node.text, max_tokens)
        elif node.representation is Representation.CODE_BLOCK:
            parts = _split_code(node.text, max_tokens)
        else:
            parts = _split_sentences(node.text, max_tokens, overlap_tokens)
        table_title = node.title if node.representation is Representation.TABLE else None
        header = contextual_header(
            document_title=document_title,
            section_path=node.section_path,
            page=node.page_start,
            entities=entities,
            table_title=table_title,
        )
        for ordinal, part in enumerate(parts):
            text = part.strip()
            if not text:
                continue
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
    # merge tiny trailing chunks of the same node into the previous one (keeps natural units)
    merged: list[Chunk] = []
    for c in chunks:
        if (
            merged
            and merged[-1].node_id == c.node_id
            and c.token_estimate < min_tokens
            and estimate_tokens(merged[-1].text + "\n" + c.text) <= max_tokens
        ):
            prev = merged[-1]
            text = prev.text + "\n" + c.text
            merged[-1] = prev.model_copy(
                update={
                    "text": text,
                    "text_hash": content_hash(text),
                    "contextual_text": prev.contextual_text + "\n" + c.text,
                    "token_estimate": estimate_tokens(text),
                }
            )
        else:
            merged.append(c)
    return merged


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
    was split (first), then tables; bounded to ``max_chunks``."""
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
    return [*split, *tables][:max_chunks]


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
