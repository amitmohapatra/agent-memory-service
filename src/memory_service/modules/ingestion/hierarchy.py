"""Document hierarchy: an intermediate, parser-independent tree.

Parsers (Docling, builtin markdown/text/html) emit ``Block``s; the hierarchy builder turns
them into ``DocumentNode``s (document > section > subsection > paragraph/table/code) with
section paths and page ranges. Nothing here needs a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from memory_service.domain.documents import DocumentNode
from memory_service.domain.enums import Representation
from memory_service.domain.ids import content_hash

BlockKind = Literal["heading", "paragraph", "table", "code", "list", "footnote", "caption"]


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: int = 0  # heading level (1..6) for headings
    page: int | None = None
    label: str | None = None  # footnote label, table title, caption
    metadata: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Deterministic, tokenizer-free estimate (~4 chars/token for English, +1 per line)."""
    return max(1, len(text) // 4 + text.count("\n"))


_HEADING_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def build_hierarchy(
    blocks: list[Block], *, document_id: str, document_version_id: str, tenant_id: str, title: str
) -> list[DocumentNode]:
    """Turn a flat block list into a node tree (returned in reading order)."""
    root = DocumentNode(
        document_id=document_id,
        document_version_id=document_version_id,
        tenant_id=tenant_id,
        representation=Representation.DOCUMENT,
        ordinal=0,
        depth=0,
        title=title,
        section_path=title,
        text="",
        text_hash=content_hash(""),
    )
    nodes: list[DocumentNode] = [root]
    # stack of (level, node) for open sections; level 0 = document
    stack: list[tuple[int, DocumentNode]] = [(0, root)]
    ordinals: dict[str, int] = {root.node_id: 0}
    page_span: dict[str, list[int]] = {}

    def _child(
        parent: DocumentNode,
        rep: Representation,
        text: str,
        *,
        title: str | None,
        page: int | None,
        meta: dict,
    ) -> DocumentNode:
        ordinals[parent.node_id] = ordinals.get(parent.node_id, 0)
        ordinal = ordinals[parent.node_id]
        ordinals[parent.node_id] += 1
        path = (
            parent.section_path
            if parent.representation is not Representation.DOCUMENT
            else title_root
        )
        if title and rep in (Representation.SECTION, Representation.SUBSECTION):
            path = f"{path} > {title}" if path else title
        node = DocumentNode(
            document_id=document_id,
            document_version_id=document_version_id,
            tenant_id=tenant_id,
            representation=rep,
            parent_id=parent.node_id,
            ordinal=ordinal,
            depth=parent.depth + 1,
            title=title,
            section_path=path,
            page_start=page,
            page_end=page,
            text=text,
            text_hash=content_hash(text),
            token_estimate=estimate_tokens(text) if text else 0,
            system_metadata=meta,
        )
        nodes.append(node)
        ordinals[node.node_id] = 0
        if page is not None:
            # propagate page span up the chain
            for _, ancestor in stack:
                page_span.setdefault(ancestor.node_id, [page, page])
                span = page_span[ancestor.node_id]
                span[0], span[1] = min(span[0], page), max(span[1], page)
        return node

    title_root = title
    for block in blocks:
        if block.kind == "heading":
            level = max(1, min(block.level, 6))
            if level == 1 and len(nodes) == 1 and block.text.strip() == title:
                # the document's own title heading: the root already represents it
                stack = [(1, root)]
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1]
            rep = (
                Representation.SECTION
                if parent.representation is Representation.DOCUMENT
                else Representation.SUBSECTION
            )
            heading_text = block.text.strip()
            number = _HEADING_NUMBER.match(heading_text)
            meta: dict[str, Any] = {"heading_level": level}
            if number:
                meta["section_number"] = number.group(1)
            node = _child(parent, rep, "", title=heading_text, page=block.page, meta=meta)
            stack.append((level, node))
            continue
        parent = stack[-1][1]
        rep = {
            "paragraph": Representation.PARAGRAPH,
            "list": Representation.PARAGRAPH,
            "table": Representation.TABLE,
            "code": Representation.CODE_BLOCK,
            "footnote": Representation.PARAGRAPH,
            "caption": Representation.PARAGRAPH,
        }[block.kind]
        meta = {"block_kind": block.kind, **block.metadata}
        if block.label:
            meta["label"] = block.label
        text = block.text.strip()
        if not text:
            continue
        _child(
            parent,
            rep,
            text,
            title=block.label if block.kind == "table" else None,
            page=block.page,
            meta=meta,
        )

    # apply page spans to sections/document
    out: list[DocumentNode] = []
    for node in nodes:
        span = page_span.get(node.node_id)
        if span and node.representation in (
            Representation.DOCUMENT,
            Representation.SECTION,
            Representation.SUBSECTION,
        ):
            out.append(node.model_copy(update={"page_start": span[0], "page_end": span[1]}))
        else:
            out.append(node)
    return out
