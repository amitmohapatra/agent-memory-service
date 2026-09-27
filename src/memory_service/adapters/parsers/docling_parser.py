"""Docling DocumentParser (MIT). PDF/DOCX/PPTX/XLSX/HTML/images -> structured blocks.

Markdown/plain text is routed to the builtin parser (Docling splits inline formatting of
Markdown into separate items, which loses paragraph boundaries). PDF and image inputs need
local Docling layout/table weights and Tesseract language/script packs. OCR selects a
script per page, stays on CPU and does not require a generative model. DOCX/PPTX/XLSX/HTML
do not load the PDF models.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from memory_service.adapters.models._runner import SerialRunner
from memory_service.adapters.parsers.builtin import BuiltinParser
from memory_service.config.constants import DOCLING_ARTIFACTS_DIR, MODEL_ROOTS, local_model_path
from memory_service.domain.documents import DocumentVersion
from memory_service.domain.errors import CorruptSource, DependencyUnavailable
from memory_service.modules.ingestion.context_graph import build_context_graph
from memory_service.modules.ingestion.hierarchy import Block, build_hierarchy
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


class DoclingParser:
    info = ProviderInfo(
        name="docling", license="MIT", origin="docling-project/docling", locality="local"
    )
    supported_media_types = _DOCLING_TYPES | _TEXT_TYPES

    def __init__(self) -> None:
        self._builtin = BuiltinParser()
        self._converter = None
        self._artifacts = Path(
            os.environ.get("MEMORY_DOCLING_ARTIFACTS")
            or local_model_path(DOCLING_ARTIFACTS_DIR)
            or MODEL_ROOTS[0] / DOCLING_ARTIFACTS_DIR
        )
        #: Docling's layout and table models are the heaviest thing the service loads, and
        #: they fan their work over every core. ``asyncio.to_thread`` hands them to the event
        #: loop's default executor - ``min(32, cpu_count + 4)``, eight threads on a four-core
        #: box - so eight documents could be inside those models at once, each competing with
        #: the others and with the encoder for the same cores. That is the oversubscription
        #: ``SerialRunner`` exists to prevent; it was applied to the encoder and the NLI head
        #: and never here, on the most expensive model of the three.
        self._runner = SerialRunner("docling")

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
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                    TesseractCliOcrOptions,
                )
                from docling.document_converter import (
                    DocumentConverter,
                    ImageFormatOption,
                    PdfFormatOption,
                )
            except ImportError as exc:
                raise DependencyUnavailable("docling is not installed (install [docling])") from exc

            options = PdfPipelineOptions()
            # Layout and table structure are what this service actually consumes: the context
            # graph is built from section paths, page numbers and IN_TABLE edges.
            options.do_table_structure = True
            options.do_ocr = True
            # Empty languages selects per-page orientation/script detection. The image
            # installs both language and script traineddata, including osd. Tesseract
            # cannot download missing weights or execute a generative model/tool call.
            options.ocr_options = TesseractCliOcrOptions(lang=[])
            options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU, num_threads=2
            )
            options.artifacts_path = self._artifacts
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
                }
            )
        return self._converter

    def _require_pdf_dependencies(self) -> None:
        if not self._artifacts.is_dir():
            raise DependencyUnavailable(
                f"Docling layout/table weights are missing at {self._artifacts}; "
                "run `make models` and set MEMORY_DOCLING_ARTIFACTS"
            )
        if shutil.which("tesseract") is None:
            raise DependencyUnavailable(
                "CPU OCR requires tesseract and its language/script packs; rebuild the "
                "image with the docling extra"
            )

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
        blocks, title, page_count = await self._runner.run(
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
        if media_type == "application/pdf" or media_type.startswith("image/"):
            self._require_pdf_dependencies()
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
