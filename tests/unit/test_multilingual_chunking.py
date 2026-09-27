"""No source tail disappears merely because a language has no word spaces."""

# ruff: noqa: RUF001 — literal multilingual fixtures.

import pytest

from memory_service.domain.documents import DocumentNode
from memory_service.domain.enums import Representation
from memory_service.modules.ingestion.chunking import _hard_split, _split_sentences, chunk_nodes
from memory_service.modules.ingestion.hierarchy import estimate_tokens

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        "北京办公室将于星期一开放" * 80,
        "สำนักงานของฉันอยู่ในกรุงเทพ" * 80,
        "東京の事務所は月曜日に開きます" * 80,
        "कार्यालयसोमवारकोखुलताहै" * 80,
        "x" * 4000,
        "short " + "长" * 500 + " ending",
    ],
)
def test_long_unspaced_content_is_bounded_and_preserved(text):
    parts = _hard_split(text, 48)
    assert len(parts) > 1
    assert all(estimate_tokens(part) <= 48 for part in parts)
    assert "".join("".join(parts).split()) == "".join(text.split())
    assert all(part in text for part in parts)


def test_multilingual_estimate_does_not_apply_english_character_ratio():
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("北京" * 200) >= 400
    assert estimate_tokens("मेरा कार्यालय दिल्ली में है।") > len("मेरा कार्यालय दिल्ली में है।") // 4


def test_sentence_boundaries_work_without_spaces_and_ignore_stale_node_estimates():
    sentence = "北京办公室将于星期一开放。"
    node = DocumentNode(
        document_id="doc",
        document_version_id="version",
        tenant_id="tenant",
        representation=Representation.PARAGRAPH,
        ordinal=0,
        depth=1,
        text=sentence * 30,
        text_hash="fixture",
        token_estimate=1,
    )
    chunks = chunk_nodes([node], document_title="北京办公室", max_tokens=48, overlap_tokens=0)
    assert len(chunks) == 30
    assert all(chunk.text == sentence for chunk in chunks)
    assert all(chunk.token_estimate <= 48 for chunk in chunks)


def test_overlap_cannot_make_the_next_full_sentence_overflow():
    text = "A short sentence. " + "long " * 30 + "."
    node = DocumentNode(
        document_id="doc",
        document_version_id="v",
        tenant_id="t",
        representation=Representation.PARAGRAPH,
        ordinal=0,
        depth=1,
        text=text,
        text_hash="fixture",
    )
    chunks = chunk_nodes([node], document_title="Doc", max_tokens=40, overlap_tokens=20)
    assert all(chunk.token_estimate <= 40 for chunk in chunks)


@pytest.mark.parametrize("sentence", ["Go!", "東京。", "Офис открыт."])
@pytest.mark.parametrize("overlap", [0, 12, 24])
def test_sentence_joining_accounts_for_separators_and_rounding(sentence, overlap):
    text = " ".join([sentence] * 60)
    parts = _split_sentences(text, 24, overlap)
    assert len(parts) > 1
    assert all(estimate_tokens(part) <= 24 for part in parts)
    if overlap == 0:
        assert " ".join(parts) == text


@pytest.mark.parametrize("representation", [Representation.TABLE, Representation.CODE_BLOCK])
@pytest.mark.parametrize("text", ["北京办公室" * 100, "x" * 1200, "a\nb\n" + "東京" * 300])
def test_structural_blocks_cannot_bypass_the_chunk_budget(representation, text):
    node = DocumentNode(
        document_id="doc",
        document_version_id="v",
        tenant_id="t",
        representation=representation,
        ordinal=0,
        depth=1,
        text=text,
        text_hash="fixture",
    )
    chunks = chunk_nodes([node], document_title="Doc", max_tokens=48, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.token_estimate <= 48 for c in chunks)
    assert all(c.node_id == node.node_id for c in chunks)
    # Tables may repeat their header, but every source character remains represented.
    flattened = "".join("".join(c.text for c in chunks).split())
    source = iter(flattened)
    assert all(any(char == found for found in source) for char in "".join(text.split()))


def test_table_with_an_oversized_row_repeats_a_bounded_header():
    from memory_service.modules.ingestion.chunking import _split_table

    header = "| City |\n|---|\n"
    text = header + "| " + "北京" * 100 + " |"
    parts = _split_table(text, 48)
    assert len(parts) > 1
    assert all(p.startswith(header) and estimate_tokens(p) <= 48 for p in parts)
    assert "".join(p[len(header) :] for p in parts).replace(" ", "") == text[len(header) :].replace(
        " ", ""
    )


@pytest.mark.parametrize("title", ["Very long document title " * 200, "北京办公室" * 300])
def test_context_metadata_cannot_displace_the_whole_source_body(title):
    from memory_service.modules.ingestion.chunking import contextual_header

    header = contextual_header(
        document_title=title, section_path=title + " > Section", page=7, entities=[title]
    )
    assert estimate_tokens(header) <= 96
    node = DocumentNode(
        document_id="doc",
        document_version_id="v",
        tenant_id="t",
        representation=Representation.PARAGRAPH,
        ordinal=0,
        depth=1,
        text="The office opens Monday.",
        text_hash="fixture",
        section_path=title,
    )
    chunk = chunk_nodes(iter([node]), document_title=title)[0]
    assert chunk.text == node.text
    assert chunk.contextual_text.endswith("\n\n" + node.text)
    assert chunk.section_path == title
