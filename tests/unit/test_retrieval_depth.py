"""Retrieval depth is one number, and the two above it are derived from it.

Prefetch and fusion depth exist to give reciprocal rank fusion something to reorder; they are
not a second opinion about how much evidence a caller wants. Written down separately they
drifted - the service shipped 100/100/50 while every judged result on disk was produced at
200/200/100 - so no artifact could say which ratio had been measured. It turns out to be the
same ratio in both, which is what makes this a refactor: the derivation is written down at
2.0 and no depth moves.

Moving the depth *is* a measured change. The roadmap's 63/63/50 (ratio 1.25) is gated on a
judged run reporting evidence_recall >= 0.987 and on tests/eval/test_retrieval_gate.py;
neither has run, so the number here is still the one the artifacts were produced at.
"""

from __future__ import annotations

import pytest

from memory_service.config.constants import (
    DEPTH_RATIO,
    RETRIEVAL,
    RetrievalSettings,
    derived_k,
)


def test_the_shipped_depth_is_derived_from_final_k() -> None:
    assert (RETRIEVAL.prefetch_k, RETRIEVAL.fused_k, RETRIEVAL.final_k) == (100, 100, 50)
    assert derived_k(RETRIEVAL.final_k) == RETRIEVAL.prefetch_k == RETRIEVAL.fused_k


@pytest.mark.parametrize(("final_k", "depth"), [(1, 2), (20, 40), (50, 100), (100, 200)])
def test_moving_final_k_moves_the_depths_above_it(final_k: int, depth: int) -> None:
    assert derived_k(final_k) == depth
    cfg = RetrievalSettings(final_k=final_k)
    assert (cfg.prefetch_k, cfg.fused_k) == (depth, depth)
    assert cfg.prefetch_k >= cfg.final_k


def test_the_ratio_is_the_one_every_artifact_was_measured_at() -> None:
    """The shipped 100/100/50 and the judged 200/200/100 are the same ratio. Changing this
    constant changes retrieval quality and is gated on a judged evidence-recall run."""
    assert DEPTH_RATIO == 2.0


def test_a_pinned_depth_still_wins() -> None:
    """The judged benchmark runs 200/200/100 and must keep running 200/200/100: the
    derivation is the default, not a ceiling imposed on every caller."""
    cfg = RetrievalSettings(final_k=100, prefetch_k=200, fused_k=200)
    assert (cfg.prefetch_k, cfg.fused_k, cfg.final_k) == (200, 200, 100)


def test_copying_a_tuning_derives_the_depth_too() -> None:
    """The trap this closes: pydantic's ``model_copy`` assigns onto the copy and runs no
    validator, so ``RETRIEVAL.model_copy(update={"final_k": 20})`` would have produced a
    final depth of 20 underneath a prefetch of 100 - one knob at construction and two
    everywhere else. Every benchmark ablation and several tests build their tuning this way.
    """
    copied = RETRIEVAL.model_copy(update={"final_k": 20})
    assert (copied.prefetch_k, copied.fused_k, copied.final_k) == (40, 40, 20)
    assert copied.prefetch_k == derived_k(copied.final_k)


def test_copying_something_else_leaves_the_depth_alone() -> None:
    copied = RETRIEVAL.model_copy(update={"dense": False})
    assert (copied.prefetch_k, copied.fused_k, copied.final_k) == (100, 100, 50)
    assert copied.dense is False


def test_a_copy_may_still_pin_its_own_depth() -> None:
    copied = RETRIEVAL.model_copy(update={"final_k": 100, "prefetch_k": 200, "fused_k": 200})
    assert (copied.prefetch_k, copied.fused_k, copied.final_k) == (200, 200, 100)


def test_the_judged_benchmark_depth_is_untouched() -> None:
    from benchmark.env import FINAL_K, FUSED_K, MEMORIES_MAX, PREFETCH_K, TOKEN_BUDGET

    assert (PREFETCH_K, FUSED_K, FINAL_K, MEMORIES_MAX, TOKEN_BUDGET) == (
        200,
        200,
        100,
        100,
        12000,
    )
