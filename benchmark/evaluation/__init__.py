"""Evaluation gates and their thresholds.

These lived in ``memory_service.config.settings`` as ``EvaluationSettings`` and
``PerformanceBudgets`` - two sections of the *service's* configuration that the service never
read. Only the benchmarks and the release gate did, so they live with them: a gate threshold
is a fact about how the product is judged, not an operator knob.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Recall@K cutoff for the critical-question gates (``critical_recall_at_k`` must be 1.00).
CRITICAL_RECALL_K = 20

#: Release-gate ceiling on the consolidator's false-merge rate over the labelled pair set.
FALSE_MERGE_RATE_MAX = 0.01


@dataclass(frozen=True)
class PerformanceBudgets:
    """Benchmark targets in milliseconds (p95). Targets, not promises."""

    chat_accept_p95_ms: float = 100
    cached_context_p95_ms: float = 75
    recall_p95_ms: float = 300
    context_bundle_p95_ms: float = 400
    file_accept_p95_ms: float = 200


BUDGETS = PerformanceBudgets()
