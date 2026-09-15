"""Unit tests: extractive summaries, rolling conversation summary, abstention rule and
required-group derivation."""

from __future__ import annotations

import pytest

from memory_service.domain.conversation import Message
from memory_service.domain.documents import Chunk, ContextEdge, DocumentNode
from memory_service.domain.enums import ContextGraphEdge, MessageKind, MessageRole, Representation
from memory_service.domain.ids import content_hash, new_id
from memory_service.modules.context.builder import rolling_summary
from memory_service.modules.context.evidence import content_terms, overlaps
from memory_service.modules.context.expansion import edges_to_groups
from memory_service.modules.context.summaries import build_summaries, sentences, summarize
from memory_service.modules.retrieval.engine import Candidate

pytestmark = pytest.mark.unit

TEXT = (
    "Adjusted EBITDA increased to EUR 98 million from EUR 81 million, an improvement of 21%. "
    "The increase reflects the restructuring savings described in Section 8. "
    "Management considers the result satisfactory. "
    "A one-time litigation cost was excluded from the measure. "
    "| Item | FY25 | FY26 |\n| Operating profit | 44 | 39 |\n"
    "The Board thanked the team for their effort during the year."
)


def test_sentences_and_summary_are_extractive_and_bounded() -> None:
    sents = sentences(TEXT)
    assert all("|" not in s for s in sents) and len(sents) == 5
    summary = summarize(TEXT, max_sentences=2, max_chars=300)
    assert summary.count(".") >= 1 and len(summary) <= 300
    assert "Adjusted EBITDA increased" in summary  # lead + numbers win
    # every sentence in the summary exists verbatim in the source
    for s in sentences(summary):
        assert s in TEXT
    assert summarize("short.") == "" and summarize("") == ""
    assert summarize(TEXT) == summarize(TEXT)  # deterministic


def test_build_summaries_walks_the_hierarchy() -> None:
    doc = DocumentNode(
        document_id="doc_1",
        document_version_id="dv_1",
        tenant_id="acme",
        representation=Representation.DOCUMENT,
        ordinal=0,
        depth=0,
        title="ACME FY26",
    )
    section = DocumentNode(
        document_id="doc_1",
        document_version_id="dv_1",
        tenant_id="acme",
        representation=Representation.SECTION,
        ordinal=1,
        depth=1,
        title="3. Financial Results",
        parent_id=doc.node_id,
    )
    para = DocumentNode(
        document_id="doc_1",
        document_version_id="dv_1",
        tenant_id="acme",
        representation=Representation.PARAGRAPH,
        ordinal=2,
        depth=2,
        parent_id=section.node_id,
        text=TEXT,
    )
    chunk = Chunk(
        node_id=para.node_id,
        document_id="doc_1",
        document_version_id="dv_1",
        tenant_id="acme",
        ordinal=0,
        text=TEXT,
        text_hash=content_hash(TEXT),
        contextual_text=TEXT,
    )
    out = build_summaries([doc, section, para], [chunk], title="ACME FY26")
    assert set(out) == {doc.node_id, section.node_id}
    assert out[doc.node_id].startswith("ACME FY26: ") and out[section.node_id].startswith(
        "3. Financial Results: "
    )
    assert "EUR 98 million" in out[section.node_id]


def _msg(content: str, role=MessageRole.USER, kind=MessageKind.VISIBLE) -> Message:
    return Message(
        message_id=new_id("message"),
        tenant_id="t",
        thread_id="thr",
        session_id="s",
        turn_id="u",
        sequence=1,
        role=role,
        kind=kind,
        content=content,
        content_hash=content_hash(content),
        author_principal="user:u1",
    )


def test_rolling_summary_digests_older_turns() -> None:
    older = [
        _msg("Let's review the FY26 numbers. There is a lot to cover."),
        _msg("Sure, starting with revenue.", role=MessageRole.ASSISTANT),
        _msg("internal reasoning", kind=MessageKind.INTERNAL),
    ]
    s = rolling_summary(older)
    assert s.splitlines() == [
        "user: Let's review the FY26 numbers.",
        "assistant: Sure, starting with revenue.",
    ]
    long = rolling_summary([_msg("x" * 200)] * 10, max_chars=300)
    assert long.endswith("…") and len(long) <= 320


def test_abstention_rule_and_required_groups() -> None:
    c = Candidate(
        record_id="chk_1",
        kind="chunk",
        text="Adjusted EBITDA increased to EUR 98 million",
        score=0.5,
    )
    assert overlaps("why did adjusted ebitda increase?", [c])
    assert not overlaps("dividend policy of Initech", [c])
    assert content_terms("The dividend policy") == {"dividend", "policy"}
    edges = [
        ContextEdge(
            tenant_id="t",
            document_id="d",
            source_id="n1",
            target_id="n2",
            edge=ContextGraphEdge.DEFINED_BY,
            label="Adjusted EBITDA",
        ),
        ContextEdge(
            tenant_id="t",
            document_id="d",
            source_id="n1",
            target_id="n3",
            edge=ContextGraphEdge.FOOTNOTE,
            label="3",
        ),
        ContextEdge(
            tenant_id="t",
            document_id="d",
            source_id="n1",
            target_id="n4",
            edge=ContextGraphEdge.NEXT,
        ),
    ]
    assert edges_to_groups(edges) == {"defined_by:Adjusted EBITDA": "n2", "footnote:3": "n3"}
