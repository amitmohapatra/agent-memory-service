"""Ingestion fidelity on real-world PDFs parsed by Docling.

The fidelity contract is *no text lost or altered, in document order*:

* every prose chunk is a verbatim slice of its node's parsed text — no whitespace
  normalisation beyond what the parser itself produced — and a node's chunks run in document
  order, each starting inside the previous one (the overlap) or after nothing but whitespace;
* every line of a parsed (non-table) node is recoverable verbatim from that node's chunks:
  either inside one chunk, or — a line longer than one chunk — as consecutive pieces across
  consecutive chunks; a line may additionally repeat only in an overlap region, so the
  chunks that hold it whole are consecutive.

Beyond that, each chunk carries the page its node came from, table rows survive bounded
splitting (a split table repeats its header row, so its parts are not slices), prose chunks
carry a section path, and known table rows, footnotes and two-column prose land on the right
page. Needs Docling + its models (``models`` marker); runs in the validation container."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Representation
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.jobs.registry import register_handlers

# Docling lays out, OCRs and table-parses each PDF on CPU: minutes, not the suite default
pytestmark = [pytest.mark.integration, pytest.mark.models, pytest.mark.timeout(900)]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")

# (filename, {page: [substrings that must appear in one chunk of that page]}, table page + cells)
EXPECTATIONS = {
    "cdc_mmwr_7301.pdf": {
        "prose": {1: ["6.5 million", "$231 million"], 2: ["butoconazole"], 5: ["four"]},
        "table": {3: ["2,364,169", "1,035.38", "188.0"], 4: ["13,106", "1,017,417"]},
    },
    "fed_fsr_2024_04.pdf": {
        # the PDF's text layer sets this date with no-break spaces, and a chunk is the parsed
        # text verbatim: the needle is what the document says, not a retyped approximation
        "prose": {2: ["hedge fund leverage"], 3: ["March\u00a011,\u00a02024"]},
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


@pytest.fixture
def container_overrides() -> dict:
    """The configured parser (docling) in place of the hermetic builtin stand-in, so this runs
    wherever the ``models`` marker does, not only under ``MEMORY_TEST_PROVIDERS=env``."""
    return {"document_parser": None}


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


def _lost_or_altered(text: str, lines: list[str], chunks: list, budget: int) -> list[str]:
    """Where ``chunks`` (one node's, in ordinal order) break the fidelity contract for the
    node ``text`` and its ``lines``; empty when they keep it. See the module docstring."""
    if not chunks:
        return [f"node without chunks: {text[:60]!r}"]
    spans: list[tuple[int, int]] = []
    for c in chunks:
        covered = spans[-1][1] if spans else 0
        start = text.find(c.text, spans[-1][0] + 1 if spans else 0)
        if start < 0:
            return [f"chunk {c.ordinal} is not a verbatim slice of its node: {c.text[:60]!r}"]
        # the next chunk starts inside the previous one (overlap) or after whitespace only
        if text[covered:start].strip():
            return [f"text lost before chunk {c.ordinal}: {text[covered:start][:60]!r}"]
        spans.append((start, start + len(c.text)))
    if text[spans[-1][1] :].strip():
        return [f"text lost after the last chunk: {text[spans[-1][1] :][:60]!r}"]
    problems: list[str] = []
    position = 0
    for line in lines:
        begin = text.find(line, position)
        assert begin >= 0, line  # a stripped line of the node is in the node
        end = position = begin + len(line)
        holders = [i for i, (a, b) in enumerate(spans) if a <= begin and end <= b]
        if holders:
            # more than one chunk holds it whole only through the overlap between neighbours
            if holders != list(range(holders[0], holders[-1] + 1)):
                problems.append(f"line repeated outside an overlap: {line[:60]!r}")
            continue
        # not inside any chunk: it must be longer than a chunk, in consecutive pieces
        pieces = [i for i, (a, b) in enumerate(spans) if a < end and begin < b]
        if estimate_tokens(line) <= budget:
            problems.append(f"line fits one chunk but is split over {len(pieces)}: {line[:60]!r}")
        elif not pieces or pieces != list(range(pieces[0], pieces[-1] + 1)):
            problems.append(f"line not in consecutive chunks: {line[:60]!r}")
    return problems


@pytest.mark.parametrize("filename", sorted(EXPECTATIONS))
async def test_pdf_lines_tables_pages_and_sections(container, uow_factory, filename) -> None:
    pytest.importorskip("docling")
    if getattr(container.document_parser.info, "name", "") != "docling":
        pytest.skip("documents.parser=docling required (MEMORY_TEST_PROVIDERS=env)")
    register_handlers(container)
    doc, version, nodes, chunks = await _ingest(container, uow_factory, filename)
    assert doc.system_metadata.get("status") == "READY", doc
    assert version is not None and version.parser == "docling", version
    assert version.page_count and version.page_count >= 4
    node_by_id = {n.node_id: n for n in nodes}
    budget = container.tuning.documents.max_chunk_tokens
    problems: list[str] = []
    for n in nodes:
        if not n.text.strip():
            continue
        lines = [ln.strip() for ln in n.text.splitlines() if len(ln.strip()) >= 12]
        if n.representation is Representation.TABLE:
            table_chunks = [c for c in chunks if c.node_id == n.node_id]
            if not table_chunks or not all(
                any(line in chunk.text for chunk in table_chunks) for line in lines
            ):
                problems.append(f"table {n.title!r} rows missing from its chunks")
            if any(
                c.token_estimate > container.tuning.documents.max_chunk_tokens for c in table_chunks
            ):
                problems.append(f"table {n.title!r} exceeds the model input planning budget")
            continue
        own = sorted((c for c in chunks if c.node_id == n.node_id), key=lambda c: c.ordinal)
        problems.extend(_lost_or_altered(n.text, lines, own, budget))
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
        # Values must retain their common table identity even when that table is too
        # large for one encoder input. A table from another node cannot satisfy the group.
        table_nodes = {c.node_id for c in table_chunks}
        assert any(
            all(any(cell in c.text for c in table_chunks if c.node_id == node) for cell in cells)
            for node in table_nodes
        ), f"table cells {cells} not together in one table on page {page}"
