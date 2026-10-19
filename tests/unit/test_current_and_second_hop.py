"""ADR 0041: a question about now reads the newest value of its slot, and evidence that names
what the question names but not what it asks is a bridge to one more hop."""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval import engine as engine_module
from memory_service.modules.retrieval.engine import Candidate, RetrievalEngine
from memory_service.modules.retrieval.slots import (
    bridge_query,
    latest_values,
    names_any,
    question_slot,
)
from memory_service.ports.search import Retriever, SearchHit, SparseVector

pytestmark = pytest.mark.unit


def _mem(rid: str, text: str, day: str, **extra) -> Candidate:
    return Candidate(
        record_id=rid,
        kind="memory",
        text=text,
        score=1.0,
        retrievers=[],
        payload={"observed_at": f"{day}T09:00:00+00:00", "text": text, **extra},
    )


def _said(c: Candidate) -> str:
    return c.payload["observed_at"]


def test_the_newest_value_of_the_asked_slot_is_current_and_the_rest_are_earlier() -> None:
    held = [
        _mem("a", "Reorder point for MK-220 is 40.", "2026-09-01"),
        _mem("c", "We set the MK-220 reorder point to 75 today.", "2026-10-09"),
        _mem("b", "MK-220 reorder point is now 60.", "2026-09-20"),
        _mem("d", "MK-220 ships in cases of 12.", "2026-10-10"),  # another slot
        _mem("e", "MK-110 reorder point is 30.", "2026-10-10"),  # another thing
    ]
    got = {
        c.record_id: c
        for c in latest_values("What's the reorder point for MK-220?", held, newest_of=_said)
    }
    assert [r for r, c in got.items() if c.payload.get("earlier_value")] == ["a", "b"]
    assert got["a"].payload["replaced_by"] == "c"
    assert not got["d"].payload.get("earlier_value") and not got["e"].payload.get("earlier_value")


@pytest.mark.parametrize(
    "question",
    [
        "What was the reorder point for MK-220 in May?",  # the past: an event, not the state
        "What's the weather like?",  # names nothing
    ],
)
def test_past_questions_and_questions_naming_nothing_are_left_alone(question: str) -> None:
    held = [
        _mem("a", "Reorder point for MK-220 is 40.", "2026-09-01"),
        _mem("b", "MK-220 reorder point is now 60.", "2026-09-20"),
    ]
    assert not any(
        c.payload.get("earlier_value") for c in latest_values(question, held, newest_of=_said)
    )


def test_the_same_value_said_again_is_no_earlier_value() -> None:
    held = [
        _mem("a", "Van #7 is driven by Peter Brandt.", "2026-09-01"),
        _mem("b", "Peter Brandt drives van #7 now.", "2026-10-01"),
    ]
    assert not any(
        c.payload.get("earlier_value")
        for c in latest_values("Who drives van #7?", held, newest_of=_said)
    )


@pytest.mark.parametrize(
    ("question", "evidence", "expected"),
    [
        ("Who is the buyer for store 4779?", "Store 4779 is in Region X29.", "Region X29 buyer"),
        (
            "Wer ist der Einkäufer für Filiale 4779?",
            "Filiale 4779 gehört zur Region X29.",
            "Region X29 einkäufer",
        ),
        # the evidence answers: no hop
        ("Who is the buyer for store 4779?", "The buyer for store 4779 is Ana Ruiz.", None),
        # the evidence is about something else: no hop
        ("Who is the buyer for store 4779?", "Region X29 had a good quarter.", None),
    ],
)
def test_a_bridge_names_the_next_thing_to_ask_about(question, evidence, expected) -> None:
    got = bridge_query(question, evidence)
    assert (got and got.split()[-2:]) == (expected and expected.split()[-2:])


def test_identifiers_match_however_they_are_spelled() -> None:
    assert names_any("The buyer for region X-29 is Priya", frozenset({"x29"}))
    assert not names_any("Region X30", frozenset({"x29"}))
    assert question_slot("Who is the buyer for store 4779?").asked == frozenset({"buyer"})


class _Store:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits, self.asked = hits, []

    async def search_sparse(self, collection, vector, flt, *, limit):
        self.asked.append((vector, limit))
        return self.hits[:limit]


class _Sparse:
    def encode_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


class _Indexer:
    sparse = _Sparse()

    def collection(self, name: str) -> str:
        return name


class _Visibility:
    def search_filter(self, **must):
        from memory_service.ports.search import SearchFilter

        return SearchFilter(tenant_id="t", must=dict(must))


def _hit(rid: str, text: str) -> SearchHit:
    return SearchHit(
        record_id=rid, score=1.0, retriever=Retriever.BM25, payload={"text": text, "kind": "memory"}
    )


@pytest.mark.asyncio
async def test_the_second_hop_adds_what_the_bridged_thing_holds_after_the_bridge() -> None:
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.store = _Store(
        [
            _hit("m1", "Store 4779 is in Region X29."),  # in hand already
            _hit("m3", "Region X30's buyer is Lee."),  # not the bridged thing
            _hit("m2", "The buyer for Region X29 is Priya Shah."),
        ]
    )
    engine.indexer = _Indexer()
    routed = type("R", (), {"query": "Who is the buyer for store 4779?"})()
    bridge = _mem("m1", "Store 4779 is in Region X29.", "2026-10-01")
    other = _mem("m9", "Store 4779 opens at 7 am.", "2026-10-02")
    diagnostics: dict = {}
    got = await engine._second_hop(routed, [bridge, other], _Visibility(), diagnostics)
    assert [c.record_id for c in got] == ["m1", "m2", "m9"]
    assert got[1].expansion_edge == "SECOND_HOP" and got[1].expanded_from == "m1"
    assert diagnostics["second_hop"]["added"] == 1
    assert engine.store.asked[0][1] == engine_module.SECOND_HOP_K


@pytest.mark.asyncio
async def test_no_bridge_no_search() -> None:
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.store = _Store([])
    engine.indexer = _Indexer()
    routed = type("R", (), {"query": "Who is the buyer for store 4779?"})()
    held = [_mem("m1", "The buyer for store 4779 is Ana Ruiz.", "2026-10-01")]
    assert await engine._second_hop(routed, held, _Visibility(), {}) == held
    assert engine.store.asked == []
