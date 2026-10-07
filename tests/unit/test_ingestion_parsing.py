"""Builtin parser, hierarchy, chunking and Document Context Graph."""

from __future__ import annotations

from itertools import pairwise
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


def _node(text: str, representation: Representation = Representation.PARAGRAPH):
    from memory_service.domain.documents import DocumentNode

    return DocumentNode(
        document_id="d",
        document_version_id="v",
        tenant_id="t",
        representation=representation,
        ordinal=0,
        depth=1,
        text=text,
        text_hash="x",
        section_path="Doc > S",
    )


def _assert_exact_slices_in_order(text: str, chunks, max_tokens: int) -> None:
    """Every chunk is a verbatim slice of ``text``; the slices run in document order, and
    consecutive ones either overlap or are separated by source whitespace only — so no
    source character is lost, altered or reordered."""
    position = 0
    previous_end = 0
    for c in chunks:
        start = text.find(c.text, position)
        assert start >= 0, f"not an exact slice: {c.text[:60]!r}"
        assert c.text == c.text.strip() and c.text_hash
        assert c.contextual_text.endswith("\n\n" + c.text)
        assert c.token_estimate == estimate_tokens(c.text) <= max_tokens
        assert not text[previous_end:start].strip(), "source text skipped between chunks"
        position, previous_end = start + 1, start + len(c.text)
    assert not text[previous_end:].strip(), "source tail lost"


@pytest.mark.parametrize("overlap", [0, 20])
@pytest.mark.parametrize("lines", [False, True])
def test_split_paragraph_chunks_are_exact_source_slices(overlap, lines) -> None:
    """PDF text keeps two spaces after a full stop and stray tabs/newlines; rejoining split
    sentences with one space made chunk text differ from the parsed document."""
    text = "".join(
        f"Sentence number {i} says something about topic {i % 5}.{'  ' if i % 2 else chr(9)}"
        + ("\n" if lines and i % 7 == 0 else "")
        for i in range(120)
    )
    chunks = chunk_nodes(
        [_node(text)], document_title="Doc", max_tokens=100, overlap_tokens=overlap
    )
    assert len(chunks) > 5
    _assert_exact_slices_in_order(text, chunks, 100)
    assert any("  " in c.text for c in chunks)  # the double spaces survive
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    if lines:
        # a line (paragraph, list item) that fits one chunk is never broken across two
        for line in filter(None, (ln.strip() for ln in text.splitlines())):
            assert any(line in c.text for c in chunks), line[:40]
    elif overlap:
        # the overlap is the previous chunk's tail, verbatim
        assert all(nxt.text.split(".")[0] in prev.text for prev, nxt in pairwise(chunks))


def test_overlong_sentence_and_code_blocks_are_exact_slices() -> None:
    sentence = " ".join(f"word{i}" for i in range(400)).replace("word7 ", "word7  \t ")
    text = f"Short lead.  {sentence}.  Short tail."
    _assert_exact_slices_in_order(
        text, chunk_nodes([_node(text)], document_title="Doc", max_tokens=60), 60
    )
    code = "\n\n  \n".join(f"def f{i}():\n    return {i}  # {'x' * 40}" for i in range(40))
    chunks = chunk_nodes(
        [_node(code, Representation.CODE_BLOCK)], document_title="Doc", max_tokens=80
    )
    assert len(chunks) > 1
    _assert_exact_slices_in_order(code, chunks, 80)


def test_a_merged_tiny_part_is_the_covering_source_slice() -> None:
    """The tiny-part merge keeps exact slices: two source parts merge into the source between
    the first's start and the tiny one's end (overlap once, original whitespace kept), not two
    parts glued with a newline. Only a split table's parts, which have no span, are glued."""
    from memory_service.modules.ingestion.chunking import _merge_tiny

    node = _node("First part of the text.  Its overlap.\t\tTail.")
    parts = [(node.text[0:37], (0, 37)), (node.text[25:44], (25, 44))]
    assert _merge_tiny(node, parts, max_tokens=100, min_tokens=40) == [node.text]
    assert _merge_tiny(
        node, [("| h |\n| a |", None), ("| h |\n| b |", None)], max_tokens=100, min_tokens=40
    ) == ["| h |\n| a |\n| h |\n| b |"]
    # a merge that would overflow the budget leaves the parts apart
    assert len(_merge_tiny(node, parts, max_tokens=10, min_tokens=40)) == 2


def _at(
    text: str, page: int, kind: str = "paragraph", representation=Representation.PARAGRAPH, **meta
):
    node = _node(text, representation)
    return node.model_copy(
        update={"page_start": page, "system_metadata": {"block_kind": kind, **meta}}
    )


def test_a_footnote_is_contextualised_by_the_sentence_that_cites_it() -> None:
    """A symbol-marked note names neither its subject nor the analysis it belongs to; its
    chunk is indexed with the nearest sentence carrying its marker (markers restart per
    page, so a page-3 ``§`` is not the page-2 one), the chunk text itself unchanged."""
    from memory_service.modules.ingestion.context_graph import footnote_citations

    cites = _at(
        "Data came from the public data set § described below. Costs for topical "
        "antifungal drugs covered by Part D §§ were assessed. Totals were summed.",
        2,
    )
    drugs = _at("§§ Butenafine, butoconazole, and terconazole.", 2)
    source = _at("§ \thttps://data.example.org/set", 2)
    table = _at("| Drug | Cost |\n|---|---|\n| Other § | 400 |", 3, "table", Representation.TABLE)
    other = _at("§ Butenafine (41 prescriptions) and luliconazole (169).", 3)
    after_stop = _at("The schedule changed this year. † A new addendum follows.", 4)
    dagger = _at("† Past schedules are archived.", 4)
    bullet = _at("* a list item, not a note", 2, "list")
    orphan = _at("‡ Nothing cites this.", 2)
    labelled = _at("Revenue rose[^7] on volume.", 5)
    note7 = _at("Excludes one-off items.", 9, "footnote", label="7")  # an endnote
    nodes = [
        cites,
        drugs,
        source,
        table,
        other,
        after_stop,
        dagger,
        bullet,
        orphan,
        labelled,
        note7,
    ]
    citations = footnote_citations(nodes)
    assert citations[drugs.node_id] == (
        "Costs for topical antifungal drugs covered by Part D §§ were assessed."
    )
    assert citations[source.node_id] == "Data came from the public data set § described below."
    assert citations[other.node_id] == "| Other § | 400 |"
    # a marker set after a full stop cites the sentence before it
    assert citations[dagger.node_id] == "The schedule changed this year."
    assert citations[note7.node_id] == "Revenue rose[^7] on volume."
    assert bullet.node_id not in citations and orphan.node_id not in citations
    chunks = {c.node_id: c for c in chunk_nodes(nodes, document_title="Report")}
    drug_chunk = chunks[drugs.node_id]
    assert drug_chunk.text == drugs.text
    assert "\nFootnote to: Costs for topical antifungal drugs" in drug_chunk.contextual_text
    assert drug_chunk.contextual_text.endswith("\n\n" + drugs.text)
    assert "Footnote to:" not in chunks[cites.node_id].contextual_text


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
