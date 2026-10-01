"""The memories' learned fusion (ADR 0025): the first stages, the pool, the features and the
final order, over hand-made arm lists so every number can be checked by hand."""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval import learned_fusion as lf
from memory_service.ports.search import Retriever, SearchHit, VectorName

pytestmark = pytest.mark.unit


def _hit(
    rid: str, *, source: str | None = None, before: str | None = None, text: str = ""
) -> SearchHit:
    payload: dict = {"text": text or rid}
    if source:
        payload["source_refs"] = [{"source_type": "message", "source_id": source}]
    if before:
        payload["preceding_source_id"] = before
    return SearchHit(record_id=rid, score=1.0, retriever=Retriever.FUSION, payload=payload)


def _arms(order: dict[VectorName, list[str]], hits: dict[str, SearchHit]):
    return {name: [hits[rid] for rid in rids] for name, rids in order.items()}


HITS = {
    "a": _hit("a", source="m1"),
    "b": _hit("b", source="m2", before="m1"),
    "c": _hit("c", source="m3", before="m2", text="we met last week"),
}


def test_the_coefficients_name_every_feature_and_nothing_else() -> None:
    assert set(lf.COEFFICIENTS) == set(lf.FEATURES)


def test_neighbours_are_read_from_the_preceding_turn_each_memory_names() -> None:
    pool = lf.ArmPool.of(_arms({VectorName.BM25: ["c", "a", "b"]}, HITS))
    assert pool.previous == {"b": "a", "c": "b"}
    assert pool.following == {"a": "b", "b": "c"}
    assert pool.ranks[VectorName.BM25] == {"c": 1, "a": 2, "b": 3}
    assert pool.reciprocal(VectorName.BM25, "c") == 0.5
    assert pool.reciprocal(VectorName.DENSE_ML, "c") == 0.0


def test_a_first_stage_fuses_both_keys_lifts_neighbours_and_adds_the_late_arm() -> None:
    arms = {
        VectorName.BM25: ["a"],
        VectorName.BM25_CTX: ["b"],
        VectorName.COLBERT: ["c"],
    }
    pool = lf.ArmPool.of(_arms(arms, HITS))
    stage = lf.FirstStage(weights={VectorName.BM25: 2.0}, late=6.0, lift=0.5, late_lifted=False)
    scores = lf.first_stage(pool, stage)
    # own key: a = 2/2; context key: b = 2/2; lift 0.5 of each neighbour; late after the lift
    assert scores["a"] == pytest.approx(1.0 + 0.5 * 1.0)
    assert scores["b"] == pytest.approx(1.0 + 0.5 * 1.0 + 0.5 * 0.0)
    assert scores["c"] == pytest.approx(0.0 + 0.5 * 1.0 + 6.0 / 2)
    lifted = lf.first_stage(pool, lf.FirstStage({VectorName.BM25: 2.0}, 6.0, 0.5, True))
    # lifted together: b gains half of c's late score too
    assert lifted["b"] == pytest.approx(1.0 + 0.5 * 1.0 + 0.5 * 3.0)


def test_a_key_scores_nothing_below_its_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lf, "KEY_DEPTH", 1)
    pool = lf.ArmPool.of(_arms({VectorName.BM25: ["a", "c"]}, HITS))
    scores = lf.first_stage(pool, lf.FirstStage({VectorName.BM25: 1.0}, 0.0, 0.0, False))
    assert scores["a"] > 0 and scores["c"] == 0.0


def test_the_pool_is_each_first_stages_top_without_repeats() -> None:
    a = {"x": 3.0, "y": 2.0, "z": 1.0}
    b = {"z": 3.0, "x": 2.0, "y": 1.0}
    assert lf.scoring_pool(a, b, 1) == ["x", "z"]
    assert lf.scoring_pool(a, b, 3) == ["x", "y", "z"]


def test_features_read_ranks_neighbours_and_a_time_for_a_when_question() -> None:
    pool = lf.ArmPool.of(_arms({VectorName.BM25: ["a", "b", "c"]}, HITS))
    a = {"a": 3.0, "b": 2.0, "c": 1.0}
    b = {"a": 1.0, "b": 2.0, "c": 3.0}
    rows = lf.features(
        pool,
        ["a", "b", "c"],
        a=a,
        b=b,
        query="When did we meet?",
    )
    col = {name: [row[i] for row in rows] for i, name in enumerate(lf.FEATURES)}
    assert col["bm25"] == pytest.approx([1 / 2, 1 / 3, 1 / 4])
    assert col["first_a"] == pytest.approx([1 / 2, 1 / 3, 1 / 4])
    assert col["first_b"] == pytest.approx([1 / 4, 1 / 3, 1 / 2])
    assert col["when"] == [1.0, 1.0, 1.0]
    assert col["when_time"] == [0.0, 0.0, 1.0]  # only "c" names a time ("last week")
    # the best first-stage-A score among the turns either side
    assert col["neighbour"] == [2.0, 3.0, 2.0]
    plain = lf.features(pool, ["c"], a=a, b=b, query="Who is Caroline?")
    assert plain[0][lf.FEATURES.index("when_time")] == 0.0


def test_the_order_is_the_scored_pool_then_the_rest_below_it() -> None:
    pool = lf.ArmPool.of(_arms({VectorName.BM25: ["a", "b", "c"]}, HITS))
    rows = [[0.0] * len(lf.FEATURES), [0.0] * len(lf.FEATURES)]
    rows[1][lf.FEATURES.index("colbert")] = 1.0  # "b" is the better candidate
    ranked = lf.order(pool, ["a", "b"], rows, {"a": 1.0, "b": 2.0, "c": 3.0})
    assert [rid for rid, _ in ranked] == ["b", "a", "c"]
    assert ranked[2][1] < min(score for _, score in ranked[:2])


def test_a_candidate_leading_every_arm_ranks_first() -> None:
    best = [0.5] * len(lf.FEATURES)
    worse = [1 / 3] * len(lf.FEATURES)
    for row in (best, worse):
        for name in ("when", "when_time", "neighbour"):
            row[lf.FEATURES.index(name)] = 0.0
    assert lf.probability(best) > lf.probability(worse)
