"""Ingestion fidelity: every line of source prose and every table row is preserved verbatim
in exactly one chunk, each chunk carries the page it came from, chunks never straddle a page
break, tables stay whole, and the section path matches the heading hierarchy."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Representation
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
_PAGE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def _source_lines(text: str) -> list[tuple[int, str, str]]:
    """(page, kind, line) for every content line: prose, table row, or heading."""
    page = 1
    out: list[tuple[int, str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = _PAGE.match(line)
        if m:
            page = int(m.group(1))
            continue
        if not line:
            continue
        if line.startswith("#"):
            out.append((page, "heading", line.lstrip("# ").strip()))
        elif line.startswith("|"):
            if re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", line):
                continue
            out.append((page, "row", line))
        elif line.startswith("Table ") and ":" in line[:12]:
            out.append((page, "caption", line))
        else:
            out.append((page, "prose", re.sub(r"^\[\^\d+\]:\s*", "", line)))
    return out


@pytest.mark.parametrize("filename", ["acme_fy26_annual_report.md", "globex_fy26_annual_report.md"])
async def test_every_line_lands_in_one_chunk_with_the_right_page(
    container, uow_factory, filename
) -> None:
    register_handlers(container)
    source = (FIXTURES / filename).read_text()
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow, CTX, filename=filename, media_type="text/markdown", data=source.encode(), title="t"
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    async with uow_factory() as uow:
        doc = await container.services["ingestion"].get_document(uow, CTX, ack.document_id)
        nodes = await uow.documents.list_nodes("acme", ack.document_id)
        chunks = await uow.documents.list_chunks("acme", ack.document_id)
    assert doc.current_version_id is not None
    node_by_id = {n.node_id: n for n in nodes}
    h1 = next(line for _, kind, line in _source_lines(source) if kind == "heading")
    problems: list[str] = []
    for page, kind, line in _source_lines(source):
        if kind == "heading":
            titles = {n.title for n in nodes if n.title}
            if line not in titles:
                problems.append(f"heading missing: {line!r}")
            continue
        if kind == "caption":
            if not any(
                (n.title or "") == line for n in nodes if n.representation is Representation.TABLE
            ):
                problems.append(f"table caption missing: {line!r}")
            continue
        holders = [c for c in chunks if line in c.text]
        if len(holders) != 1:
            problems.append(f"{kind} line in {len(holders)} chunks (expected 1): {line[:60]!r}")
            continue
        c = holders[0]
        if c.page != page:
            problems.append(f"page {c.page} != {page} for {line[:50]!r}")
        node = node_by_id[c.node_id]
        if kind == "row" and node.representation is not Representation.TABLE:
            problems.append(f"table row outside a TABLE node: {line[:50]!r}")
        if kind == "prose" and not c.section_path.startswith(h1):
            problems.append(f"section path {c.section_path!r} for {line[:50]!r}")
    # tables are whole: a table chunk holds every row of its table
    for n in nodes:
        if n.representation is Representation.TABLE:
            rows = [r for r in n.text.splitlines() if r.strip().startswith("|")]
            table_chunks = [c for c in chunks if c.node_id == n.node_id]
            if len(table_chunks) != 1 or not all(r.strip() in table_chunks[0].text for r in rows):
                problems.append(f"table {n.title!r} split across {len(table_chunks)} chunks")
    # no chunk straddles a page break and every chunk has a page
    assert all(c.page is not None for c in chunks)
    assert not problems, problems
    # hierarchy: subsection chunks carry Doc > Section > Subsection
    sub = next(
        c
        for c in chunks
        if "3.1 Revenue" in c.section_path
        or "3.1 Revenue" in (node_by_id[c.node_id].section_path or "")
    )
    assert sub.section_path.count(">") >= 2
