"""The query-side arms of the accuracy programme, as switches a benchmark can turn on.

D6 steps 2 and 3 each promote a shipped default - ``hybrid_weights`` away from equal weights,
``entity_prefetch`` on - and each may only be promoted by an arm that measured it. So the arm
lives here, in the benchmark's own environment, and the constant in ``config/constants.py``
stays conservative until its gate says otherwise. Nothing here changes what the service does
by default: an arm with no switch set hands ``Overrides`` a ``None`` retrieval tuning, which
is exactly what every result already on disk was produced with.
"""

from __future__ import annotations

import pytest
from benchmark.env import HALVED_RETRIEVAL, BenchEnv

from memory_service.config.constants import RETRIEVAL, derived_k
from memory_service.ports.search import VectorName

pytestmark = pytest.mark.unit


def test_no_switch_leaves_the_frozen_tuning_alone() -> None:
    """The comparability rule: an unswitched arm is the shipped arm, byte for byte."""
    assert BenchEnv().overrides().retrieval is None


def test_fitted_weights_reach_the_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_HYBRID_WEIGHTS", '{"dense_en": 1.0, "dense_ml": 0.5, "bm25": 2}')
    env = BenchEnv.from_environ()
    tuning = env.overrides().retrieval
    assert tuning is not None
    assert tuning.hybrid_weights == {
        VectorName.DENSE_EN: 1.0,
        VectorName.DENSE_ML: 0.5,
        VectorName.BM25: 2.0,
    }
    # the weighting is the only change: depth and the rest of the tuning stay shipped
    assert (tuning.prefetch_k, tuning.fused_k, tuning.final_k) == (
        RETRIEVAL.prefetch_k,
        RETRIEVAL.fused_k,
        RETRIEVAL.final_k,
    )
    assert tuning.entity_prefetch is False


@pytest.mark.parametrize(
    "raw",
    [
        "{not json}",
        "[]",
        "{}",
        '{"dense": 1.0}',
        '{"dense_en": -1}',
        '{"dense_en": true}',
        '{"dense_en": "1.0"}',
    ],
)
def test_a_weighting_that_cannot_be_applied_is_refused(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo would weigh nothing and still be reported as a fitted run."""
    monkeypatch.setenv("BENCH_HYBRID_WEIGHTS", raw)
    with pytest.raises(SystemExit):
        BenchEnv.from_environ()


def test_the_entity_prefetch_is_its_own_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_ENTITY_PREFETCH", "on")
    env = BenchEnv.from_environ()
    tuning = env.overrides().retrieval
    assert tuning is not None
    assert tuning.entity_prefetch is True
    assert tuning.hybrid_weights is None


def test_an_unreadable_switch_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_ENTITY_PREFETCH", "maybe")
    with pytest.raises(SystemExit):
        BenchEnv.from_environ()


def test_a_switch_can_say_off_and_not_only_stay_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control arm of a promoted constant: ``off`` must pin off, not mean "unset".

    Once ``entity_prefetch`` or ``hybrid_weights`` is promoted, an arm that cannot spell the
    old value measures the new default and files it as the control.
    """
    monkeypatch.setenv("BENCH_ENTITY_PREFETCH", "off")
    monkeypatch.setenv("BENCH_HYBRID_WEIGHTS", "equal")
    tuning = BenchEnv.from_environ().overrides().retrieval
    assert tuning is not None
    assert tuning.entity_prefetch is False
    assert tuning.hybrid_weights is None
    # and unset is not "off": it hands back None, the frozen constant untouched
    monkeypatch.delenv("BENCH_ENTITY_PREFETCH")
    monkeypatch.delenv("BENCH_HYBRID_WEIGHTS")
    unset = BenchEnv.from_environ()
    assert (unset.entity_prefetch, unset.hybrid_weights) == (None, None)
    assert unset.overrides().retrieval is None


def test_the_halved_depth_halves_the_work_and_not_the_render() -> None:
    """D6 step 2 spends the fit on candidates, so @10/@20/@50 stay comparable."""
    assert HALVED_RETRIEVAL.prefetch_k == RETRIEVAL.prefetch_k // 2
    assert HALVED_RETRIEVAL.fused_k == RETRIEVAL.fused_k // 2
    assert HALVED_RETRIEVAL.memory_recall_k == RETRIEVAL.memory_recall_k // 2
    assert HALVED_RETRIEVAL.final_k == RETRIEVAL.final_k
    # the store's memory query is max(fused_k, derived_k(memory_recall_k)): exactly halved
    shipped = max(RETRIEVAL.fused_k, derived_k(RETRIEVAL.memory_recall_k))
    halved = max(HALVED_RETRIEVAL.fused_k, derived_k(HALVED_RETRIEVAL.memory_recall_k))
    assert halved * 2 == shipped
    assert BenchEnv(depth="halved").overrides().context is None


def test_a_weighted_arm_at_the_halved_depth_carries_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_DEPTH", "halved")
    monkeypatch.setenv("BENCH_HYBRID_WEIGHTS", '{"bm25": 1.5}')
    tuning = BenchEnv.from_environ().overrides().retrieval
    assert tuning is not None
    assert tuning.prefetch_k == HALVED_RETRIEVAL.prefetch_k
    assert tuning.hybrid_weights == {VectorName.BM25: 1.5}
