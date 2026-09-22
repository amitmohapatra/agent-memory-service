"""Ingestion fidelity on real-world PDFs parsed by Docling: every line of every parsed node is
preserved verbatim in exactly one chunk, the chunk carries the page the node came from,
tables stay whole in one chunk, prose chunks carry a section path, and known table rows,
footnotes and two-column prose land on the right page. Needs Docling + its models
(``models`` marker); runs in the validation container."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Representation
from memory_service.modules.jobs.registry import register_handlers

pytestmark = [pytest.mark.integration, pytest.mark.models]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")

# (filename, {page: [substrings that must appear in one chunk of that page]}, table page + cells)
EXPECTATIONS = {
    "cdc_mmwr_7301.pdf": {
        "prose": {1: ["6.5 million", "$231 million"], 2: ["butoconazole"], 5: ["four"]},
        "table": {3: ["2,364,169", "1,035.38", "188.0"], 4: ["13,106", "1,017,417"]},
    },
    "fed_fsr_2024_04.pdf": {
        "prose": {2: ["hedge fund leverage"], 3: ["March 11, 2024"]},
        "table": {6: ["57,175", "22,518", "3,420"]},
    },
    "nist_sp800_63_3.pdf": {
        "prose": {6: ["identity proofing process"], 7: ["national security"]},
        "table": {3: ["Clarified flowcharts"], 7: ["Normative", "Federation Considerations"]},
    },
    "fed_beige_book_2024_01.pdf": {
        "prose": {1: ["January 8, 2024"], 4: ["Tenth District declined moderately"]},
        "table": {},
    },
}


async def _ingest(container, uow_factory, filename: str):
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            CTX,
            filename=filename,
            media_type="application/pdf",
            data=(FIXTURES / filename).read_bytes(),
            title=filename,
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    async with uow_factory() as uow:
        doc = await container.services["ingestion"].get_document(uow, CTX, ack.document_id)
        version = await uow.documents.get_version(CTX.tenant_id, doc.current_version_id)
        nodes = await uow.documents.list_nodes(CTX.tenant_id, ack.document_id)
        chunks = await uow.documents.list_chunks(CTX.tenant_id, ack.document_id)
    return doc, version, nodes, chunks


@pytest.mark.parametrize("filename", sorted(EXPECTATIONS))
async def test_pdf_lines_tables_pages_and_sections(container, uow_factory, filename) -> None:
    pytest.importorskip("docling")
    if getattr(container.document_parser.info, "name", "") != "docling":
        pytest.skip("documents.parser=docling required (MEMORY_TEST_PROVIDERS=env)")
    register_handlers(container)
    doc, version, nodes, chunks = await _ingest(container, uow_factory, filename)
    assert doc.status.value == "READY", doc
    assert version is not None and version.parser == "docling", version
    assert version.page_count and version.page_count >= 4
    node_by_id = {n.node_id: n for n in nodes}
    problems: list[str] = []
    for n in nodes:
        if not n.text.strip():
            continue
        lines = [ln.strip() for ln in n.text.splitlines() if len(ln.strip()) >= 12]
        if n.representation is Representation.TABLE:
            table_chunks = [c for c in chunks if c.node_id == n.node_id]
            if len(table_chunks) != 1:
                problems.append(f"table {n.title!r} split across {len(table_chunks)} chunks")
            elif not all(ln in table_chunks[0].text for ln in lines):
                problems.append(f"table {n.title!r} rows missing from its chunk")
            continue
        for ln in lines:
            holders = [c for c in chunks if c.node_id == n.node_id and ln in c.text]
            if len(holders) != 1:
                problems.append(f"line in {len(holders)} chunks (expected 1): {ln[:60]!r}")
    for c in chunks:
        node = node_by_id[c.node_id]
        assert c.page is not None, c.chunk_id
        lo, hi = node.page_start or c.page, node.page_end or node.page_start or c.page
        if not lo <= c.page <= hi:
            problems.append(f"chunk page {c.page} outside node pages {lo}-{hi}")
        if node.representation is not Representation.TABLE and not c.section_path:
            problems.append(f"prose chunk without section path: {c.text[:50]!r}")
    assert not problems, problems[:25]
    expect = EXPECTATIONS[filename]
    for page, needles in expect["prose"].items():
        for needle in needles:
            hits = [c for c in chunks if needle in c.text and c.page == page]
            assert hits, (
                f"{needle!r} not found on page {page} (pages seen: {sorted({c.page for c in chunks if needle in c.text})})"
            )
    for page, cells in expect["table"].items():
        table_chunks = [
            c
            for c in chunks
            if c.page == page and node_by_id[c.node_id].representation is Representation.TABLE
        ]
        assert table_chunks, f"no table chunk on page {page}"
        assert any(all(cell in c.text for cell in cells) for c in table_chunks), (
            f"table cells {cells} not together in one chunk on page {page}"
        )
