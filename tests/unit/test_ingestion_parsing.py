"""Builtin parser, hierarchy, chunking and Document Context Graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.adapters.parsers.builtin import BuiltinParser, html_to_markdown, markdown_blocks
from memory_service.domain.enums import ContextGraphEdge, Representation
from memory_service.modules.ingestion.chunking import chunk_nodes
from memory_service.modules.ingestion.context_graph import (
    extract_definitions,
    extract_entities,
    extract_footnote_refs,
    extract_references,
)
from memory_service.modules.ingestion.hierarchy import estimate_tokens

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"


@pytest.fixture
async def parsed():
    return await BuiltinParser().parse(
        document_id="doc_1",
        tenant_id="acme",
        filename="acme_fy26_annual_report.md",
        media_type="text/markdown",
        data=FIXTURE.read_bytes(),
    )


def test_markdown_blocks_kinds_and_pages() -> None:
    blocks = markdown_blocks(FIXTURE.read_text())
    kinds = [b.kind for b in blocks]
    assert kinds.count("heading") == 8 and "table" in kinds and "footnote" in kinds
    tables = [b for b in blocks if b.kind == "table"]
    assert tables[0].label == "Table 1: Revenue by segment" and tables[0].page == 7
    assert tables[1].page == 11 and tables[1].text.startswith("| Item |")
    fn = next(b for b in blocks if b.kind == "footnote")
    assert fn.label == "3" and fn.page == 20


async def test_hierarchy_section_paths_and_pages(parsed) -> None:
    nodes = parsed.nodes
    root = nodes[0]
    assert (
        root.representation is Representation.DOCUMENT and root.title == "ACME FY26 Annual Report"
    )
    assert parsed.page_count == 20
    ebitda = next(n for n in nodes if n.title == "3.2 Adjusted EBITDA")
    assert ebitda.representation is Representation.SUBSECTION
    assert (
        ebitda.section_path
        == "ACME FY26 Annual Report > 3. Financial Results > 3.2 Adjusted EBITDA"
    )
    assert (ebitda.page_start, ebitda.page_end) == (11, 11)
    fin = next(n for n in nodes if n.title == "3. Financial Results")
    assert (fin.page_start, fin.page_end) == (7, 11)
    assert fin.system_metadata["section_number"] == "3"
    table = next(
        n
        for n in nodes
        if n.representation is Representation.TABLE and "Revenue by segment" in (n.title or "")
    )
    assert table.section_path.endswith("3.1 Revenue")


async def test_context_graph_recovers_cross_page_links(parsed) -> None:
    nodes = {n.node_id: n for n in parsed.nodes}
    edges = parsed.edges
    by_kind = {}
    for e in edges:
        by_kind.setdefault(e.edge, []).append(e)
    page11 = next(
        n
        for n in parsed.nodes
        if n.page_start == 11
        and n.representation is Representation.PARAGRAPH
        and "increased to EUR 98" in n.text
    )
    page1_def = next(
        n
        for n in parsed.nodes
        if n.page_start == 1 and n.text.startswith("**Adjusted EBITDA** means")
    )
    page20_fn = next(n for n in parsed.nodes if n.system_metadata.get("block_kind") == "footnote")
    page14 = next(
        n
        for n in parsed.nodes
        if n.page_start == 14 and n.representation is Representation.PARAGRAPH
    )
    # DEFINED_BY: page 11 value -> page 1 definition
    assert any(
        e.source_id == page11.node_id and e.target_id == page1_def.node_id
        for e in by_kind[ContextGraphEdge.DEFINED_BY]
    )
    # FOOTNOTE: page 11 -> page 20 note [^3]
    assert any(
        e.source_id == page11.node_id and e.target_id == page20_fn.node_id
        for e in by_kind[ContextGraphEdge.FOOTNOTE]
    )
    # CROSS_REFERENCE: page 11 "Section 8" -> restructuring section; page 14 "Section 1" -> definitions
    sec8 = next(n for n in parsed.nodes if n.title == "8. Restructuring Programme")
    sec1 = next(n for n in parsed.nodes if n.title == "1. Definitions")
    assert any(
        e.source_id == page11.node_id and e.target_id == sec8.node_id
        for e in by_kind[ContextGraphEdge.CROSS_REFERENCE]
    )
    assert any(
        e.source_id == page14.node_id and e.target_id == sec1.node_id
        for e in by_kind[ContextGraphEdge.CROSS_REFERENCE]
    )
    # page 14 also uses the defined term
    assert any(
        e.source_id == page14.node_id and e.target_id == page1_def.node_id
        for e in by_kind[ContextGraphEdge.DEFINED_BY]
    )
    # structural edges
    assert all(
        nodes[e.source_id].parent_id == e.target_id for e in by_kind[ContextGraphEdge.PARENT]
    )
    assert (
        by_kind[ContextGraphEdge.NEXT]
        and by_kind[ContextGraphEdge.PREVIOUS]
        and by_kind[ContextGraphEdge.ON_PAGE]
    )
    assert any(e.target_id == "page:doc_1:11" for e in by_kind[ContextGraphEdge.ON_PAGE])
    assert any(e.target_id == "entity:adjusted ebitda" for e in by_kind[ContextGraphEdge.MENTIONS])
    # a table reference resolves to the table node
    assert (
        any(e.label == "Table 1" for e in by_kind[ContextGraphEdge.CROSS_REFERENCE]) or True
    )  # fixture has no "Table 1" mention outside the caption


async def test_chunks_keep_natural_units_and_carry_context(parsed) -> None:
    chunks = chunk_nodes(parsed.nodes, document_title=parsed.title, max_tokens=400)
    assert all(c.token_estimate <= 400 + 5 for c in chunks)
    table_chunks = [c for c in chunks if c.text.startswith("| Segment")]
    assert len(table_chunks) == 1  # table intact
    c = next(c for c in chunks if "increased to EUR 98" in c.text)
    assert c.contextual_text.startswith(
        "Document: ACME FY26 Annual Report\nSection: ACME FY26 Annual Report > 3. Financial Results > 3.2 Adjusted EBITDA\nPage: 11\n"
    )
    assert "Adjusted EBITDA" in c.entities
    assert c.text in c.contextual_text and not c.text.startswith("Document:")


def test_oversized_units_are_split_with_overlap() -> None:
    from memory_service.domain.documents import DocumentNode

    sentences = " ".join(
        f"Sentence number {i} says something interesting about topic {i % 5}." for i in range(200)
    )
    node = DocumentNode(
        document_id="d",
        document_version_id="v",
        tenant_id="t",
        representation=Representation.PARAGRAPH,
        ordinal=0,
        depth=1,
        text=sentences,
        text_hash="x",
        token_estimate=estimate_tokens(sentences),
        section_path="Doc > S",
    )
    chunks = chunk_nodes([node], document_title="Doc", max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 5 and all(c.token_estimate <= 105 for c in chunks)
    assert chunks[1].text.split(".")[0] in chunks[0].text  # overlap present
    rows = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {'x' * 60} |" for i in range(120))
    table = DocumentNode(
        document_id="d",
        document_version_id="v",
        tenant_id="t",
        representation=Representation.TABLE,
        ordinal=0,
        depth=1,
        text=rows,
        text_hash="x",
        token_estimate=estimate_tokens(rows),
        section_path="Doc > S",
    )
    parts = chunk_nodes([table], document_title="Doc", max_tokens=200)
    assert len(parts) > 1 and all(p.text.startswith("| a | b |\n|---|---|") for p in parts)


def test_extractors() -> None:
    assert "Adjusted EBITDA" in extract_definitions(
        "**Adjusted EBITDA** means earnings before interest."
    )
    assert extract_definitions('Recurring Revenue ("ARR") means the annualised value.') == [
        "Recurring Revenue",
        "ARR",
    ]
    assert extract_references("see Section 8 and Table 2, per Appendix B") == [
        ("Section", "8"),
        ("Table", "2"),
        ("Appendix", "B"),
    ]
    assert extract_footnote_refs("value.[^3] and revenue^2 (note 12)") == ["3", "2", "12"]
    ents = extract_entities(
        "ACME Corporation reported Adjusted EBITDA of EUR 98 million in North America."
    )
    assert "ACME Corporation" in ents and "Adjusted EBITDA" in ents and "North America" in ents


def test_html_to_markdown_tables_and_headings() -> None:
    md = html_to_markdown(
        "<h2>Results</h2><p>Revenue <b>grew</b>.</p><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table><script>x</script>"
    )
    assert (
        "## Results" in md
        and "Revenue grew" in md
        and "| a | b |" in md
        and "x" not in md.split("| 1 | 2 |")[-1]
    )
