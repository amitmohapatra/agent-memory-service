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


def _form(
    rid: str, *, source: str, text: str = "", owner: str = "user:caroline", **extra
) -> SearchHit:
    payload = {
        "text": text or rid,
        "subject": extra.pop("subject", owner),
        "owner_principal": owner,
        "observed_at": "2023-05-08T13:56:00+00:00",
        "source_refs": [{"source_type": "message", "source_id": source}],
        **extra,
    }
    return SearchHit(record_id=rid, score=1.0, retriever=Retriever.FUSION, payload=payload)


def test_a_turn_and_what_was_read_from_it_are_one_document() -> None:
    hits = {
        "turn": _form("turn", source="obs_1", text="I painted a sunrise. It took all week."),
        "fact": _form("fact", source="obs_1", text="I painted a sunrise.", subject="sunrise"),
        "other": _form("other", source="obs_2"),
    }
    pool = mr.statements(
        _pool(
            {VectorName.BM25: ["other", "turn", "fact"], VectorName.DENSE_EN: ["fact", "other"]},
            hits,
        )
    )
    # the fact fused best on its own and stands for the statement, at its best rank in each arm
    assert set(pool.hits) == {"fact", "other"}
    assert pool.ranks[VectorName.BM25] == {"other": 1, "fact": 2}
    assert pool.ranks[VectorName.DENSE_EN] == {"fact": 1, "other": 2}
    assert [h.record_id for h in pool.forms["fact"]] == ["fact", "turn"]
    # the rules read every form: the turn names its speaker, the fact what it is about
    assert pool.speakers("fact") == {"caroline", "sunrise"}
    named = dict(mr.ranked(pool, "What did Caroline paint?"))
    plain = dict(mr.ranked(pool, "What was painted?"))
    assert named["fact"] - plain["fact"] == pytest.approx(mr.SPEAKER / (mr.K + 1))


def test_forms_of_different_turns_speakers_or_values_stay_apart() -> None:
    hits = {
        "a": _form("a", source="obs_1"),
        "b": _form("b", source="obs_2"),
        "c": _form("c", source="obs_1", owner="user:melanie"),
        "d": _form("d", source="obs_1", earlier_value=True),
        "e": _form("e", source="obs_1", derived=True),
        "f": SearchHit(record_id="f", score=1.0, retriever=Retriever.FUSION, payload={"text": "f"}),
    }
    pool = _pool({VectorName.BM25: list(hits)}, hits)
    assert mr.statements(pool) is pool


def test_an_agent_bound_to_a_user_relays_that_users_statement() -> None:
    hits = {
        "turn": _form("turn", source="obs_1", owner="user:u1"),
        "fact": _form("fact", source="obs_1", owner="agent:u1/harness"),
    }
    pool = mr.statements(_pool({VectorName.BM25: ["turn", "fact"]}, hits))
    assert set(pool.hits) == {"turn"}


def test_a_when_question_in_another_language_lifts_a_memory_that_names_a_time() -> None:
    hits = {
        "a": _hit("a", text="Wir haben Dune gelesen."),
        "c": _hit("c", text="Wir haben uns letzte Woche getroffen."),
    }
    pool = _pool({VectorName.BM25: ["a", "c"]}, hits)
    plain = dict(mr.ranked(pool, "Was haben wir gemacht?"))
    when = dict(mr.ranked(pool, "Wann haben wir uns getroffen?"))
    assert when["c"] - plain["c"] == pytest.approx(mr.TIME / (mr.K + 1))
    assert when["a"] == pytest.approx(plain["a"])
