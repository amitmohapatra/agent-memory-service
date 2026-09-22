"""Retrieval depth is one number, and the two above it are derived from it.

Prefetch and fusion depth exist to give reciprocal rank fusion something to reorder; they are
not a second opinion about how much evidence a caller wants. Written down separately they
drifted - the service shipped 100/100/50 while every judged result on disk was produced at
200/200/100 - so no artifact could say which ratio had been measured.
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
    assert (RETRIEVAL.prefetch_k, RETRIEVAL.fused_k, RETRIEVAL.final_k) == (63, 63, 50)
    assert derived_k(RETRIEVAL.final_k) == RETRIEVAL.prefetch_k == RETRIEVAL.fused_k


@pytest.mark.parametrize(("final_k", "depth"), [(1, 2), (20, 25), (50, 63), (80, 100), (100, 125)])
def test_moving_final_k_moves_the_depths_above_it(final_k: int, depth: int) -> None:
    """``ceil(1.25 x final_k)``: RRF only reorders inside the prefetch union, so the depth
    above the final one is headroom for the reorder and nothing else."""
    assert derived_k(final_k) == depth
    cfg = RetrievalSettings(final_k=final_k)
    assert (cfg.prefetch_k, cfg.fused_k) == (depth, depth)
    assert cfg.prefetch_k >= cfg.final_k


def test_the_ratio_is_the_one_the_derivation_documents() -> None:
    assert DEPTH_RATIO == 1.25


def test_a_pinned_depth_still_wins() -> None:
    """The judged benchmark runs 200/200/100 and must keep running 200/200/100: the
    derivation is the default, not a ceiling imposed on every caller."""
    cfg = RetrievalSettings(final_k=100, prefetch_k=200, fused_k=200)
    assert (cfg.prefetch_k, cfg.fused_k, cfg.final_k) == (200, 200, 100)


def test_the_judged_benchmark_depth_is_untouched() -> None:
    from benchmark.env import FINAL_K, FUSED_K, MEMORIES_MAX, PREFETCH_K, TOKEN_BUDGET

    assert (PREFETCH_K, FUSED_K, FINAL_K, MEMORIES_MAX, TOKEN_BUDGET) == (
        200,
        200,
        100,
        100,
        12000,
    )
