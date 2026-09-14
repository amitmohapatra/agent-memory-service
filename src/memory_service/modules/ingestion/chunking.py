"""Natural-unit chunking + Contextual Retrieval representations.

Rules:
* a paragraph/table/code block that fits ``max_tokens`` becomes exactly one chunk;
* an oversized paragraph is split on sentence boundaries with token overlap;
* an oversized table is split by rows, repeating the header row in every part;
* an oversized code block is split on blank-line boundaries;
* every chunk's ``contextual_text`` prepends deterministic context (document title, section
  path, page, entities) so BM25 and dense embeddings see where the text sits.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import Representation
from memory_service.domain.ids import content_hash
from memory_service.modules.ingestion.context_graph import extract_entities
from memory_service.modules.ingestion.hierarchy import estimate_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


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
