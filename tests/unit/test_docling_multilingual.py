"""PDF/image OCR uses one CPU policy; text formats need no OCR dependencies."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from memory_service.adapters.parsers.docling_parser import DoclingParser
from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.unit


def test_pdf_and_image_share_explicit_cpu_script_detection(monkeypatch, tmp_path):
    captured = {}

    def converter(**kwargs):
        captured.update(kwargs)
        return object()

    modules = {
        "docling.datamodel.accelerator_options": {
            "AcceleratorDevice": SimpleNamespace(CPU="cpu"),
            "AcceleratorOptions": SimpleNamespace,
        },
        "docling.datamodel.base_models": {
            "InputFormat": SimpleNamespace(PDF="pdf", IMAGE="image"),
        },
        "docling.datamodel.pipeline_options": {
            "PdfPipelineOptions": SimpleNamespace,
            "TesseractCliOcrOptions": SimpleNamespace,
        },
        "docling.document_converter": {
            "DocumentConverter": converter,
            "ImageFormatOption": SimpleNamespace,
            "PdfFormatOption": SimpleNamespace,
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("MEMORY_DOCLING_ARTIFACTS", str(tmp_path))
    parser = DoclingParser()
    first = parser._get_converter()
    assert parser._get_converter() is first
    formats = captured["format_options"]
    pdf = formats["pdf"].pipeline_options
    assert pdf is formats["image"].pipeline_options
    assert pdf.do_ocr is True and pdf.ocr_options.lang == []
    assert pdf.accelerator_options.device == "cpu"
    assert pdf.accelerator_options.num_threads == 2
    assert pdf.artifacts_path == tmp_path


def test_missing_pdf_dependencies_fail_without_runtime_download(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("MEMORY_DOCLING_ARTIFACTS", str(missing))
    parser = DoclingParser()
    with pytest.raises(DependencyUnavailable, match="layout/table weights"):
        parser._convert("scan.pdf", "application/pdf", b"unopened")
    missing.mkdir()
    monkeypatch.setattr(
        "memory_service.adapters.parsers.docling_parser.shutil.which", lambda _: None
    )
    with pytest.raises(DependencyUnavailable, match="language/script packs"):
        parser._convert("scan.png", "image/png", b"unopened")
    assert parser._converter is None


async def test_unicode_text_does_not_require_layout_or_ocr(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DOCLING_ARTIFACTS", str(tmp_path / "absent"))
    parser = DoclingParser()
    text = "北京办公室将于星期一开放。"
    parsed = await parser.parse(
        document_id="doc",
        tenant_id="tenant",
        filename="office.txt",
        media_type="text/plain",
        data=text.encode(),
    )
    assert any(text in node.text for node in parsed.nodes)
    assert parser._converter is None
