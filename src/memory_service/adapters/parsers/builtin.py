"""Builtin DocumentParser: Markdown, plain text, HTML, CSV/JSON as text. No models.

Page markers understood: form feed (``\\f``), ``<!-- page: N -->`` and ``<<<PAGE N>>>``
lines (Docling-style exports and test fixtures use these). Footnotes ``[^n]: ...`` become
footnote blocks; pipe tables and fenced code are kept intact.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from memory_service.domain.documents import DocumentVersion
from memory_service.domain.enums import Representation
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.text import sanitise
from memory_service.modules.ingestion.context_graph import build_context_graph
from memory_service.modules.ingestion.hierarchy import Block, build_hierarchy
from memory_service.ports.intelligence import ParsedDocument
from memory_service.ports.models import ProviderInfo

_PAGE_RE = re.compile(r"^\s*(?:<!--\s*page:\s*(\d+)\s*-->|<<<PAGE\s+(\d+)>>>)\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:\s*(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


#: Control characters that carry no text: C0 except tab/newline/carriage return, and the C1
#: block. NUL is the dangerous one — PostgreSQL rejects it outright ("text fields cannot
#: contain NUL (0x00) bytes"), so a single stray byte failed the whole document.parse job and
#: the legible parts of the document were lost with it. Stripping is strictly better than
#: failing: the readable text is worth keeping, and a control character never was.
class BuiltinParser:
    info = ProviderInfo(name="builtin", license="Apache-2.0", origin="internal", locality="local")
    supported_media_types = frozenset(
        {
            "text/markdown",
            "text/plain",
            "text/html",
            "text/csv",
            "application/json",
            "text/x-markdown",
        }
    )

    async def parse(
        self, *, document_id: str, tenant_id: str, filename: str, media_type: str, data: bytes
    ) -> ParsedDocument:
        # `supported_media_types` above already said this parser handles text formats only,
        # but nothing enforced it: a PDF routed here — which is what happens when
        # `documents.parser=docling` falls back in an image built without docling — was
        # decoded as UTF-8 and indexed as document text. A 39 KB PDF produced 38,494
        # characters beginning "%PDF-1.7 %âãÏÓ 1 0 obj << /Producer (pypdf)". Embedded,
        # chunked and stored as knowledge, with one log line as the only sign.
        #
        # Refusing is the only safe answer: a document that cannot be parsed must fail
        # loudly, never enter the index as binary noise.
        if media_type not in self.supported_media_types:
            raise ValidationFailed(
                f"the builtin parser cannot read {media_type!r} ({filename!r}). Rich formats "
                "need documents.parser=docling in an image built with the docling extra."
            )
        text = sanitise(data.decode("utf-8", errors="replace"))
        if media_type == "text/html" or filename.lower().endswith((".html", ".htm")):
            text = html_to_markdown(text)
        title = _title_from(text, filename)
        blocks = markdown_blocks(text)
        version = DocumentVersion(
            document_id=document_id, tenant_id=tenant_id, parser="builtin", parser_version="1"
        )
        nodes = build_hierarchy(
            blocks,
            document_id=document_id,
            document_version_id=version.document_version_id,
            tenant_id=tenant_id,
            title=title,
        )
        pages = [n.page_end for n in nodes if n.page_end]
        edges = build_context_graph(nodes, tenant_id=tenant_id, document_id=document_id)
        return ParsedDocument(
            version=version,
            nodes=nodes,
            edges=edges,
            title=title,
            page_count=max(pages) if pages else None,
        )


def _title_from(text: str, filename: str) -> str:
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or filename


def markdown_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    page: int | None = None
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    para: list[str] = []
    list_items: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            blocks.append(Block("paragraph", "\n".join(para).strip(), page=page))
            para = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append(Block("list", "\n".join(list_items).strip(), page=page))
            list_items = []

    while i < len(lines):
        line = lines[i]
        if "\f" in line:
            flush_para()
            flush_list()
            page = (page or 1) + 1
            line = line.replace("\f", "")
            if not line.strip():
                i += 1
                continue
        pm = _PAGE_RE.match(line)
        if pm:
            flush_para()
            flush_list()
            page = int(pm.group(1) or pm.group(2))
            i += 1
            continue
        if _FENCE_RE.match(line):
            flush_para()
            flush_list()
            lang = line.strip().strip("`").strip() or None
            code: list[str] = []
            i += 1
            while i < len(lines) and not _FENCE_RE.match(lines[i]):
                code.append(lines[i])
                i += 1
            blocks.append(
                Block(
                    "code", "\n".join(code), page=page, metadata={"language": lang} if lang else {}
                )
            )
            i += 1
            continue
        hm = _HEADING_RE.match(line.strip())
        if hm:
            flush_para()
            flush_list()
            blocks.append(Block("heading", hm.group(2).strip(), level=len(hm.group(1)), page=page))
            i += 1
            continue
        fm = _FOOTNOTE_DEF_RE.match(line.strip())
        if fm:
            flush_para()
            flush_list()
            body = [fm.group(2)]
            i += 1
            while i < len(lines) and lines[i].startswith(("    ", "\t")):
                body.append(lines[i].strip())
                i += 1
            blocks.append(Block("footnote", " ".join(body).strip(), page=page, label=fm.group(1)))
            continue
        if _TABLE_ROW_RE.match(line):
            flush_para()
            flush_list()
            rows: list[str] = []
            title = None
            if (
                blocks
                and blocks[-1].kind == "paragraph"
                and re.match(r"^(Table\s+\S+[:.]?\s*.*)$", blocks[-1].text, re.IGNORECASE)
                and len(blocks[-1].text) < 160
            ):
                title = blocks.pop().text.strip()
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            blocks.append(Block("table", "\n".join(rows), page=page, label=title))
            continue
        if _LIST_RE.match(line):
            flush_para()
            list_items.append(line.rstrip())
            i += 1
            continue
        if not line.strip():
            flush_para()
            flush_list()
            i += 1
            continue
        flush_list()
        para.append(line.rstrip())
        i += 1
    flush_para()
    flush_list()
    return blocks


class _HTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self._skip = 0
        self._cell: list[str] = []
        self._row: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n" + "#" * int(tag[1]) + " ")
        elif tag in ("p", "div", "section", "article", "br", "li"):
            self.out.append("\n" + ("- " if tag == "li" else ""))
        elif tag == "pre":
            self._in_pre = True
            self.out.append("\n```\n")
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag == "pre":
            self._in_pre = False
            self.out.append("\n```\n")
        elif tag in ("td", "th"):
            self._row.append(" ".join("".join(self._cell).split()))
        elif tag == "tr":
            self.out.append("\n| " + " | ".join(self._row) + " |")
        elif tag == "table" or tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self.lasttag in ("td", "th"):
            self._cell.append(data)
        else:
            self.out.append(
                data if self._in_pre else " ".join(data.split()) + (" " if data.strip() else "")
            )


def html_to_markdown(text: str) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(text)
    return html.unescape("".join(parser.out))


__all__ = ["BuiltinParser", "Representation", "html_to_markdown", "markdown_blocks"]
