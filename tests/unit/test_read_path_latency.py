"""The read path's per-request costs that change nothing in what a request returns.

* ``slots.latest_values`` reads a long message's identifiers and names (``_anchors``, the
  costly part of ``slots.read``) only for an item that names the slot's things and shares the
  words it asks with: every other item is decided by those cheap parts, as ``read`` decides it.
* ``coverage._unknown_code`` answers as it did when it also read every word's code form.
* The standing rules and preferences are read from the moment the retrieval has the query's
  vectors, under the ranking, not after it; and every payload field their searches filter on
  is indexed in the store, so a search does not read each point's payload to test it.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any

import pytest

from memory_service.modules.context import builder as builder_module
from memory_service.modules.context import coverage
from memory_service.modules.retrieval import slots
from memory_service.modules.retrieval.engine import Candidate, RetrievalResult
from tests.unit.test_context_datapath import CTX, QUERY, _builder, _Engine

pytestmark = pytest.mark.unit


def _mem(rid: str, text: str, day: str) -> Candidate:
    return Candidate(
        record_id=rid,
        kind="memory",
        text=text,
        score=1.0,
        retrievers=[],
        payload={"observed_at": f"{day}T09:00:00+00:00", "text": text},
    )


TEXTS = [
    "Reorder point for MK-220 is 40.",
    "We set the MK-220 reorder point to 75 today.",
    "MK-220 ships in cases of 12.",
    "MK-110 reorder point is 30.",
    "The buyer for Region K42 is Omar Haddad.",
    "Update: Lena Vogel has replaced Omar Haddad as the buyer for Region K42.",
    "My regional merchandise manager is Lee Chen.",
    "Mein regionaler Merchandise-Manager ist Ana Ruiz.",
    "Cycle count for aisle B-42: 340 units.",
    "I went hiking in the Alps with Maria last summer and we saw 3 ibexes.",
    "Can you recommend a good lens for portraits? I use a Sony A7 III. " * 40,
]
QUESTIONS = [
    "What's the reorder point for MK-220?",
    "Who is the buyer for Region K42?",
    "Who is my regional merchandise manager?",
    "Wer ist mein regionaler Merchandise-Manager?",
    "What's the current count for aisle B-42?",
    "Can you suggest some accessories that would complement my current photography setup?",
]


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("text", TEXTS)
def test_an_item_the_cheap_parts_turn_down_is_one_read_turns_down(question, text):
    slot = slots.question_slot(question)
    if not slots._may_hold(slot, text):
        reading = slots.read(slot, text)
        assert not (reading.names_all and reading.asks)


def test_identifiers_are_read_only_for_items_that_can_hold_the_slot(monkeypatch):
    read_for: list[str] = []
    anchors = slots._anchors

    def counted(text: str, lang: str) -> Any:
        read_for.append(text)
        return anchors(text, lang)

    monkeypatch.setattr(slots, "_anchors", counted)
    held = [
        _mem("a", "Reorder point for MK-220 is 40.", "2026-09-01"),
        _mem("c", "We set the MK-220 reorder point to 75 today.", "2026-10-09"),
        _mem("b", "MK-220 reorder point is now 60.", "2026-09-20"),
        _mem("x", "I went hiking in the Alps with Maria last summer.", "2026-10-10"),
        _mem("y", "The new espresso machine makes 2 cups at once.", "2026-10-11"),
    ]
    got = slots.latest_values("What's the reorder point for MK-220?", held, newest_of=_said)
    assert [c.record_id for c in got if c.payload.get("earlier_value")] == ["a", "b"]
    # the slot names MK-220 (an identifier): every item still needs its identifiers read, but
    # an item that does not share an asked word never does
    assert held[3].text not in read_for and held[4].text not in read_for


def _said(c: Candidate) -> str:
    return c.payload["observed_at"]


def _unknown_code_before(question: str, evidence: Sequence[str]) -> str | None:
    """``coverage._unknown_code`` as it was, reading every word's code form too."""
    codes = coverage._codes(question)
    if not codes:
        return None
    said = " ".join(coverage._code_form(t) for t in evidence for t in re.split(r"\s+", t))
    flat = coverage._code_form(" ".join(evidence))
    kinds = {coverage._shape(c) for t in evidence for c in coverage._codes(t)}
    missing = sorted(
        c for c in codes if c not in flat and c not in said and coverage._shape(c) in kinds
    )
    return f"no evidence names {', '.join(missing)}" if missing else None


@pytest.mark.parametrize(
    "question",
    [
        "What is the alarm code for ZZ421722?",
        "Which stores stock RT335327?",
        "Write the service report for WO-4120.",
        "Where is PX-90 stored?",
        "How many pre-1920 American coins do I have in my collection?",
        "What's the weather like?",
    ],
)
@pytest.mark.parametrize(
    "evidence",
    [
        ["Stores stock XN494339 at 1641, 4307 and 4610."],
        ["PX 90 is in aisle 4.", "px-90 ships Monday."],
        ["The alarm code for ZZ421723 is 4471.", "Site SITE-9154 opens at 7."],
        ["I bought a 1915 coin and a pre-1920 nickel.", "WO-4120 is done."],
        ["Nothing here names a code."],
        [],
    ],
)
def test_an_unknown_code_is_what_it_was(question, evidence):
    assert coverage._unknown_code(question, evidence) == _unknown_code_before(question, evidence)


class _VectorsEngine(_Engine):
    """Tells the builder the route and the vectors part way through the retrieval, then
    lets the loop run: what the builder started on them runs under the rest."""

    vectors = object()
    standing_started_during_retrieve = False

    async def retrieve(self, ctx: Any, query: str, **kwargs: Any) -> RetrievalResult:
        result = await super().retrieve(ctx, query, **kwargs)
        kwargs["on_encoded"](result.routed, self.vectors)
        await asyncio.sleep(0)
        self.standing_started_during_retrieve = bool(self.standing_calls)
        result.query_vectors = self.vectors  # type: ignore[assignment]
        return result


async def test_the_standing_rules_are_read_under_the_ranking(monkeypatch):
    builder = _builder()
    engine = _VectorsEngine(builder.engine.authz, 2)
    engine.standing_calls = []  # type: ignore[attr-defined]
    builder.engine = engine  # type: ignore[assignment]

    async def standing_blocks(eng: Any, query: str, visibility: Any, vectors: Any) -> list:
        engine.standing_calls.append(vectors)  # type: ignore[attr-defined]
        return []

    monkeypatch.setattr(builder_module, "standing_blocks", standing_blocks)
    try:
        await builder.build(CTX, QUERY)
    finally:
        await builder.close()
    assert engine.standing_started_during_retrieve
    # once, on the vectors the result carries
    assert engine.standing_calls == [engine.vectors]  # type: ignore[attr-defined]


async def test_every_field_the_standing_searches_filter_on_is_indexed():
    from types import SimpleNamespace

    from memory_service.adapters.search.qdrant_store import _PAYLOAD_INDEXES
    from memory_service.domain.script import Script
    from memory_service.modules.context import standing
    from memory_service.modules.retrieval.engine import QueryVectors
    from memory_service.ports.search import SearchFilter, SparseVector, VectorName

    filters: list[SearchFilter] = []

    async def search(collection: str, *args: Any, **kwargs: Any) -> list:
        filters.append(next(a for a in args if isinstance(a, SearchFilter)))
        return []

    engine = SimpleNamespace(
        memory_scope=lambda visibility: ("memories", SearchFilter(tenant_id="acme")),
        indexer=SimpleNamespace(
            spaces=SimpleNamespace(primary_space=SimpleNamespace(name=VectorName.DENSE_ML))
        ),
        store=SimpleNamespace(search_dense=search, search_sparse=search),
    )
    vectors = QueryVectors(
        dense={VectorName.DENSE_ML: [0.1] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        script=Script.LATIN,
    )
    await standing.standing_blocks(engine, "Always send reports as a table", None, vectors)
    fields = {key for flt in filters for key in (*flt.must_any, *flt.should_any)}
    assert len(filters) == 4 and fields
    assert fields <= set(_PAYLOAD_INDEXES)
