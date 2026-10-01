"""A memory is scored with the turns beside it (RetrievalSettings.adjacent_turn_weight)."""

from __future__ import annotations

from memory_service.modules.retrieval.engine import Candidate, with_adjacent_turns


def _turn(record_id: str, score: float, source: str, before: str | None = None) -> Candidate:
    payload: dict[str, object] = {"source_refs": [{"source_type": "message", "source_id": source}]}
    if before:
        payload["preceding_source_id"] = before
    return Candidate(
        record_id=record_id, kind="memory", text=record_id, score=score, payload=payload
    )


def test_a_reply_is_lifted_by_the_question_before_it_and_the_turn_after_it() -> None:
    question = _turn("q", 1.0, "m1")
    reply = _turn("r", 0.2, "m2", before="m1")
    follow = _turn("f", 0.4, "m3", before="m2")
    with_adjacent_turns([question, reply, follow], 0.25)
    assert reply.score == 0.2 + 0.25 * (1.0 + 0.4)
    # each lift reads the scores as fused, not as already lifted
    assert question.score == 1.0 + 0.25 * 0.2
    assert follow.score == 0.4 + 0.25 * 0.2


def test_neighbours_outside_the_pool_and_a_zero_weight_change_nothing() -> None:
    lone = _turn("r", 0.5, "m2", before="m1")
    assert with_adjacent_turns([lone], 0.25)[0].score == 0.5
    pair = [_turn("q", 1.0, "m1"), _turn("r", 0.2, "m2", before="m1")]
    assert [c.score for c in with_adjacent_turns(pair, 0.0)] == [1.0, 0.2]


def test_every_memory_of_a_turn_shares_its_neighbour_once() -> None:
    """A turn's verbatim copy and its extracted fact both carry it; the neighbour gains the
    best of them, not their sum."""
    verbatim = _turn("v", 0.6, "m1")
    fact = _turn("x", 0.3, "m1")
    reply = _turn("r", 0.0, "m2", before="m1")
    with_adjacent_turns([verbatim, fact, reply], 0.5)
    assert reply.score == 0.5 * 0.6
