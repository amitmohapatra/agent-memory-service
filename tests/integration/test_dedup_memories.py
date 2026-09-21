"""Identical memories must collapse, not compete.

Collapsing applied to chunks only — ``if c.kind == "chunk"`` — because memories carry no
``text_hash`` in their search payload, so there was nothing to group them by. The effect was
that identical memories never collapsed at all. Measured on a live bundle: eleven memory
items with **two** distinct texts, six copies of one sentence and five of another, crowding
out every other piece of evidence. With injection caps that becomes fatal — the top three
would be three copies of one fact.
"""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval.engine import Candidate, _dedup

pytestmark = pytest.mark.integration


def _candidate(record_id: str, text: str, score: float, *, kind: str = "memory", **payload):
    return Candidate(
        record_id=record_id,
        kind=kind,
        text=text,
        score=score,
        retrievers=["fusion"],
        payload=payload,
    )


def test_identical_memories_collapse_onto_the_best_ranked() -> None:
    out = _dedup(
        [
            _candidate("mem_a", "SKU-1 was reordered", 0.9),
            _candidate("mem_b", "SKU-1 was reordered", 0.7),
            _candidate("mem_c", "SKU-1 was reordered", 0.5),
            _candidate("mem_d", "SKU-1 ships from EU-1", 0.4),
        ]
    )
    assert [c.record_id for c in out] == ["mem_a", "mem_d"]
    assert out[0].payload["duplicates"] == ["mem_b", "mem_c"]


def test_no_twin_is_silently_lost() -> None:
    """A collapsed record is recorded, not discarded — a caller can still reach it."""
    out = _dedup(
        [
            _candidate("mem_a", "same", 0.9),
            _candidate("mem_b", "same", 0.8),
        ]
    )
    reachable = {c.record_id for c in out} | {
        d for c in out for d in (c.payload.get("duplicates") or [])
    }
    assert reachable == {"mem_a", "mem_b"}


def test_the_survivor_keeps_the_best_score_and_every_retriever() -> None:
    a = _candidate("mem_a", "same", 0.5)
    b = _candidate("mem_b", "same", 0.9)
    b.retrievers = ["exact"]
    out = _dedup([a, b])
    assert len(out) == 1 and out[0].score == 0.9
    assert out[0].retrievers == ["exact", "fusion"]


def test_a_payload_hash_is_used_when_present() -> None:
    """Chunks carry text_hash; that path must keep working without rehashing."""
    out = _dedup(
        [
            _candidate("chk_a", "page one", 0.9, kind="chunk", text_hash="h1"),
            _candidate("chk_b", "page one rendered differently", 0.8, kind="chunk", text_hash="h1"),
            _candidate("chk_c", "page two", 0.7, kind="chunk", text_hash="h2"),
        ]
    )
    assert [c.record_id for c in out] == ["chk_a", "chk_c"]


def test_distinct_memories_are_untouched() -> None:
    texts = ["fact one", "fact two", "fact three"]
    out = _dedup([_candidate(f"mem_{i}", t, 0.9 - i / 10) for i, t in enumerate(texts)])
    assert [c.text for c in out] == texts


def test_repeated_record_ids_still_merge() -> None:
    a = _candidate("mem_a", "same", 0.5)
    b = _candidate("mem_a", "same", 0.9)
    b.retrievers = ["exact"]
    out = _dedup([a, b])
    assert len(out) == 1 and out[0].score == 0.9
    assert out[0].retrievers == ["exact", "fusion"]
