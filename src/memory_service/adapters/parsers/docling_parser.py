"""Docling DocumentParser (MIT). PDF/DOCX/PPTX/XLSX/HTML/images -> structured blocks.

Markdown/plain text is routed to the builtin parser (Docling splits inline formatting of
Markdown into separate items, which loses paragraph boundaries). PDF and image inputs need
Docling's layout/OCR models (downloaded from Hugging Face on first use); DOCX/PPTX/XLSX/HTML
work offline.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from memory_service.adapters.parsers.builtin import BuiltinParser
from memory_service.domain.documents import DocumentVersion
from memory_service.domain.errors import CorruptSource, DependencyUnavailable
from memory_service.modules.ingestion.context_graph import build_context_graph
from memory_service.modules.ingestion.hierarchy import Block, build_hierarchy
from memory_service.observability.logging import get_logger
from memory_service.ports.intelligence import ParsedDocument
from memory_service.ports.models import ProviderInfo

_TEXT_TYPES = frozenset(
    {"text/markdown", "text/x-markdown", "text/plain", "text/csv", "application/json"}
)
_DOCLING_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/html",
        "image/png",
        "image/jpeg",
        "image/tiff",
    }
)
_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
}


log = get_logger(__name__)


class DoclingParser:
    info = ProviderInfo(
        name="docling", license="MIT", origin="docling-project/docling", locality="local"
    )
    supported_media_types = _DOCLING_TYPES | _TEXT_TYPES

    def __init__(self) -> None:
        self._builtin = BuiltinParser()
        self._converter = None

    def _get_converter(self):  # type: ignore[no-untyped-def]
        """Build the converter against *local* weights.

        A bare ``DocumentConverter()`` resolves its layout and table models through the
        HuggingFace cache and downloads them on first use. That breaks two rules at once: the
        service does not download models at runtime (a missing model is a startup error, not
        a silent stall on the first document), and the runtime image runs as ``nobody`` with
        ``HOME=/nonexistent``, so the download failed with ``PermissionError`` and every PDF
        came back as ``CorruptSource``.

        ``MEMORY_DOCLING_ARTIFACTS`` points at the directory ``make models`` populates.
        """
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as exc:
                raise DependencyUnavailable("docling is not installed (install [docling])") from exc

            artifacts = os.environ.get("MEMORY_DOCLING_ARTIFACTS") or ""
            options = PdfPipelineOptions()
            # Layout and table structure are what this service actually consumes: the context
            # graph is built from section paths, page numbers and IN_TABLE edges.
            options.do_table_structure = True
            # OCR is deliberately off. It is a different capability (scanned images, not
            # digital PDFs), it pulls a third model family, and its runtime needs cv2 - which
            # needs libxcb, which a python:slim image does not have. Leaving the default True
            # meant the first scanned page died with an ImportError from inside a worker
            # instead of a clear "this build cannot OCR". If OCR is wanted, it needs an image
            # built for it, and `download_models` has to fetch the rapidocr weights to match.
            options.do_ocr = False
            if artifacts and Path(artifacts).is_dir():
                options.artifacts_path = artifacts
            else:
                log.warning(
                    "docling.artifacts_missing",
                    path=artifacts or "<unset>",
                    detail="docling will try to download its models on first use",
                    hint="run `make models` and set MEMORY_DOCLING_ARTIFACTS",
                )
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    async def parse(
        self, *, document_id: str, tenant_id: str, filename: str, media_type: str, data: bytes
    ) -> ParsedDocument:
        if media_type in _TEXT_TYPES:
            return await self._builtin.parse(
                document_id=document_id,
                tenant_id=tenant_id,
                filename=filename,
                media_type=media_type,
                data=data,
            )
        blocks, title, page_count = await asyncio.to_thread(
            self._convert, filename, media_type, data
        )
        version = DocumentVersion(
            document_id=document_id,
            tenant_id=tenant_id,
            parser="docling",
            parser_version=_docling_version(),
            page_count=page_count,
        )
        nodes = build_hierarchy(
            blocks,
            document_id=document_id,
            document_version_id=version.document_version_id,
            tenant_id=tenant_id,
            title=title or filename,
        )
        edges = build_context_graph(nodes, tenant_id=tenant_id, document_id=document_id)
        return ParsedDocument(
            version=version,
            nodes=nodes,
            edges=edges,
            title=title or filename,
            page_count=page_count,
        )

    def _convert(
        self, filename: str, media_type: str, data: bytes
    ) -> tuple[list[Block], str | None, int | None]:
        converter = self._get_converter()
        suffix = _EXT.get(media_type) or Path(filename).suffix or ".bin"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"input{suffix}"
            path.write_bytes(data)
            try:
                result = converter.convert(str(path))
            except Exception as exc:
                # The type alone is not actionable. "CorruptSource: ... ImportError" sent two
                # separate investigations looking at the PDF, when the cause was a missing
                # shared library inside docling both times. Carry the message.
                detail = str(exc).strip() or type(exc).__name__
                raise CorruptSource(
                    f"docling failed to parse {filename}: {type(exc).__name__}: {detail[:300]}"
                ) from exc
        doc = result.document
        blocks: list[Block] = []
        title: str | None = None
        page_count = len(doc.pages) if getattr(doc, "pages", None) else None
        for raw_item, _level in doc.iterate_items():
            item: Any = raw_item  # docling item types are dynamic; attributes checked by kind
            kind = type(item).__name__
            page = None
            prov = getattr(item, "prov", None) or []
            if prov:
                page = getattr(prov[0], "page_no", None)
            if kind == "TitleItem":
                title = title or (item.text or "").strip()
                blocks.append(Block("heading", item.text or "", level=1, page=page))
            elif kind == "SectionHeaderItem":
                level = int(getattr(item, "level", 1) or 1) + 1
                blocks.append(Block("heading", item.text or "", level=min(level, 6), page=page))
            elif kind == "TableItem":
                try:
                    md = item.export_to_markdown(doc)
                except Exception:
                    md = ""
                caption = None
                try:
                    caption = item.caption_text(doc) or None
                except Exception:
                    caption = None
                if md.strip():
                    blocks.append(Block("table", md.strip(), page=page, label=caption))
            elif kind == "CodeItem":
                blocks.append(Block("code", item.text or "", page=page))
            elif kind == "ListItem":
                marker = getattr(item, "marker", "-") or "-"
                blocks.append(Block("list", f"{marker} {item.text or ''}".strip(), page=page))
            elif kind == "PictureItem":
                try:
                    caption = item.caption_text(doc)
                except Exception:
                    caption = ""
                if caption:
                    blocks.append(Block("caption", caption, page=page))
            elif kind in ("TextItem", "FormulaItem"):
                text = (item.text or "").strip()
                label = str(getattr(item, "label", "") or "")
                if not text:
                    continue
                if label == "footnote" or text.startswith("[^"):
                    lab = None
                    if text.startswith("[^") and "]:" in text:
                        lab, _, text = text[2:].partition("]:")
                    blocks.append(Block("footnote", text.strip(), page=page, label=lab))
                elif label in ("caption",):
                    blocks.append(Block("caption", text, page=page))
                else:
                    blocks.append(Block("paragraph", text, page=page))
        # merge consecutive list items into one list block for natural chunking
        merged: list[Block] = []
        for b in blocks:
            if (
                b.kind == "list"
                and merged
                and merged[-1].kind == "list"
                and merged[-1].page == b.page
            ):
                merged[-1].text += "\n" + b.text
            else:
                merged.append(b)
        return merged, title, page_count


def _docling_version() -> str:
    try:
        from importlib.metadata import version

        return version("docling")
    except Exception:
        return "unknown"
