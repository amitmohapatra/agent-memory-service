"""Document discovery should diversify without destroying focused evidence retrieval."""

from memory_service.modules.retrieval.engine import Candidate, diverse_head


def chunk(identifier, document, **kwargs):
    return Candidate(
        record_id=identifier,
        kind="chunk",
        text=identifier,
        score=1,
        payload={"document_id": document},
        **kwargs,
    )


def test_diversity_opens_slots_for_other_documents_before_primary_cut():
    candidates = [
        chunk("a1", "a"),
        chunk("a2", "a"),
        chunk("a3", "a"),
        chunk("b1", "b"),
        chunk("c1", "c"),
    ]
    assert [c.record_id for c in diverse_head(candidates, limit=4, per_document=2)] == [
        "a1",
        "a2",
        "b1",
        "c1",
    ]
    assert diverse_head(candidates, limit=4, per_document=0) == candidates[:4]


def test_overflow_fills_spare_slots_without_mutating_pool():
    candidates = [chunk(f"a{i}", "a") for i in range(5)]
    assert diverse_head(candidates, limit=4, per_document=1) == candidates[:4]
    assert len(candidates) == 5


def test_explicit_hits_companions_and_memories_are_not_collapsed():
    candidates = [
        chunk("a1", "a"),
        chunk("a2", "a", retrievers=["exact"]),
        chunk("note", "a", expansion_edge="FOOTNOTE"),
        Candidate(record_id="mem", kind="memory", text="fact", score=1),
        chunk("no_document1", None),
        chunk("no_document2", None),
    ]
    assert diverse_head(candidates, limit=6, per_document=1) == candidates
