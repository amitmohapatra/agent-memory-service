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
from memory_service.modules.ingestion.context_graph import extract_entities
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.llm.assist import LLMAssist

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")

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
    return "\n".join(lines)


def _split_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        t = estimate_tokens(sentence)
        if t > max_tokens:  # a single huge "sentence": hard split by words
            words = sentence.split()
            step = max(1, max_tokens * 3)
            for i in range(0, len(words), step):
                parts.append(" ".join(words[i : i + step]))
            continue
        if current and current_tokens + t > max_tokens:
            parts.append(" ".join(current))
            # overlap: keep trailing sentences worth ~overlap_tokens
            kept: list[str] = []
            kept_tokens = 0
            for s in reversed(current):
                if kept_tokens + estimate_tokens(s) > overlap_tokens:
                    break
                kept.insert(0, s)
                kept_tokens += estimate_tokens(s)
            current, current_tokens = kept, kept_tokens
        current.append(sentence)
        current_tokens += t
    if current:
        parts.append(" ".join(current))
    return parts


def _split_table(text: str, max_tokens: int) -> list[str]:
    rows = text.split("\n")
    if len(rows) < 3:
        return [text]
    header = rows[:2] if re.match(r"^\s*\|?\s*:?-{2,}", rows[1]) else rows[:1]
    body = rows[len(header) :]
    parts: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = "\n".join([*header, *current, row])
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append("\n".join([*header, *current]))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join([*header, *current]))
    return parts


def _split_code(text: str, max_tokens: int) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    parts: list[str] = []
    current: list[str] = []
    for block in blocks:
        candidate = "\n\n".join([*current, block])
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_nodes(
    nodes: Iterable[DocumentNode],
    *,
    document_title: str,
    max_tokens: int = 400,
    min_tokens: int = 40,
    overlap_tokens: int = 40,
    keep_tables_intact: bool = True,
    keep_code_intact: bool = True,
    contextual: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for node in nodes:
        if (
            node.representation
            in (Representation.DOCUMENT, Representation.SECTION, Representation.SUBSECTION)
            or not node.text
        ):
            continue
        entities = node.entities or extract_entities(node.text)
        tokens = node.token_estimate or estimate_tokens(node.text)
        if tokens <= max_tokens:
            parts = [node.text]
        elif node.representation is Representation.TABLE:
            parts = (
                [node.text]
                if keep_tables_intact and tokens <= max_tokens * 4
                else _split_table(node.text, max_tokens)
            )
        elif node.representation is Representation.CODE_BLOCK:
            parts = (
                [node.text]
                if keep_code_intact and tokens <= max_tokens * 4
                else _split_code(node.text, max_tokens)
            )
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
            and merged[-1].token_estimate + c.token_estimate <= max_tokens
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
