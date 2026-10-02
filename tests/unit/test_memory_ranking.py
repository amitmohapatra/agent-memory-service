"""The memories' ranking (ADR 0026): reciprocal-rank fusion of every arm, then the session,
speaker and time rules, over hand-made arm lists so every number can be checked by hand."""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval import memory_ranking as mr
from memory_service.ports.search import Retriever, SearchHit, VectorName

pytestmark = pytest.mark.unit


def _hit(
    rid: str, *, text: str = "", who: str = "user:caroline", day: str = "2023-05-08"
) -> SearchHit:
    payload = {"text": text or rid, "subject": who, "observed_at": f"{day}T13:56:00+00:00"}
    return SearchHit(record_id=rid, score=1.0, retriever=Retriever.FUSION, payload=payload)


def _pool(order: dict[VectorName, list[str]], hits: dict[str, SearchHit]) -> mr.ArmPool:
    return mr.ArmPool.of({name: [hits[rid] for rid in rids] for name, rids in order.items()})


HITS = {
    "a": _hit("a", day="2023-05-08"),
    "b": _hit("b", day="2023-06-01", who="user:melanie"),
    "c": _hit("c", day="2023-07-01", text="we met last week"),
}


def test_every_arm_fuses_at_weight_one_and_the_late_arm_at_late() -> None:
    pool = _pool({VectorName.BM25: ["a", "b"], VectorName.COLBERT: ["b"]}, HITS)
    got = mr.fused(pool)
    assert got["a"] == pytest.approx(1 / (mr.K + 1))
    assert got["b"] == pytest.approx(1 / (mr.K + 2) + mr.LATE / (mr.K + 1))


def test_a_session_lifts_every_memory_said_on_its_day() -> None:
    hits = {**HITS, "d": _hit("d", day="2023-05-08")}
    pool = _pool({VectorName.BM25: ["a", "b", "c", "d"]}, hits)
    scores = dict(mr.ranked(pool, "what happened?"))
    base = mr.fused(pool)
    # a is its day's best; d shares that day and gains a's share, c is alone on its day
    assert scores["d"] == pytest.approx(base["d"] + mr.SESSION * base["a"])
    assert scores["c"] == pytest.approx(base["c"] + mr.SESSION * base["c"])


def test_a_named_person_and_a_when_question_lift_their_memories() -> None:
    pool = _pool({VectorName.BM25: ["a", "b", "c"]}, HITS)
    plain = dict(mr.ranked(pool, "what was said?"))
    named = dict(mr.ranked(pool, "what did Melanie say?"))
    unit = 1 / (mr.K + 1)
    assert named["b"] - plain["b"] == pytest.approx(mr.SPEAKER * unit)
    assert named["a"] == pytest.approx(plain["a"])
    when = dict(mr.ranked(pool, "When did we meet?"))
    assert when["c"] - plain["c"] == pytest.approx(mr.TIME * unit)  # "last week" names a time
    assert when["a"] == pytest.approx(plain["a"])


def test_a_memory_leading_the_late_arm_outranks_one_leading_a_single_arm() -> None:
    pool = _pool({VectorName.BM25: ["a", "b"], VectorName.COLBERT: ["b", "a"]}, HITS)
    assert next(rid for rid, _ in mr.ranked(pool, "anything")) == "b"


def test_the_context_keys_late_arm_fuses_at_late_context() -> None:
    pool = _pool({VectorName.COLBERT_CTX: ["a"]}, HITS)
    assert mr.fused(pool)["a"] == pytest.approx(mr.LATE_CONTEXT / (mr.K + 1))


def test_a_memory_in_the_period_the_question_names_is_lifted() -> None:
    june = _hit("j", day="2023-06-10")
    told = _hit("t", day="2023-07-02")
    told.payload["dated_mentions"] = [{"text": "last month", "date": "2023-06-01..2023-06-30"}]
    hits = {"j": june, "t": told, "x": _hit("x", day="2023-09-01")}
    pool = _pool({VectorName.BM25: ["x", "j", "t"]}, hits)
    plain = dict(mr.ranked(pool, "What did Melanie do?"))
    june_q = dict(mr.ranked(pool, "What did Melanie do in June?"))
    unit = 1 / (mr.K + 1)
    # said in June, and said in July about June, are both in the period; September is not
    assert june_q["j"] - plain["j"] == pytest.approx(mr.PERIOD * unit)
    assert june_q["t"] - plain["t"] == pytest.approx(mr.PERIOD * unit)
    assert june_q["x"] == pytest.approx(plain["x"])
