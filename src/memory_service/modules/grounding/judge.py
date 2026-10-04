"""The judge's verdict on a run: a grounding report recorded as RUN feedback.

``/v1/verify`` is the one judge. What it found is feedback like any other (``source=judge``),
so the run's outcome follows it unless a person already decided it, and the memories the
grounded claims rest on gain confidence. The record's id is derived from the run, the bundle
and the answer, so verifying the same answer twice records one verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.evidence import EvidenceSource
from memory_service.domain.feedback import (
    Feedback,
    FeedbackEvidenceRef,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
)
from memory_service.domain.grounding import GroundingReport
from memory_service.domain.ids import content_hash, stable_key
from memory_service.modules.feedback.service import FeedbackService
from memory_service.ports.uow import UnitOfWorkFactory

#: Who the verdicts are from, as ``Feedback.reviewer`` names them.
REVIEWER: Final = "judge:grounding"
#: Memory evidence one verdict cites (the feedback record's own bound).
CITED_MAX: Final = 50


def verdict_of(report: GroundingReport) -> FeedbackVerdict:
    """Grounded - no claim unsupported or contradicted - confirms the run; anything else
    rejects it."""
    grounded = report.unsupported == 0 and report.contradicted == 0
    return FeedbackVerdict.CONFIRM if grounded else FeedbackVerdict.REJECT


async def judge_run(
    feedback: FeedbackService,
    uow_factory: UnitOfWorkFactory,
    ctx: MemoryExecutionContext,
    run_id: str,
    bundle_id: str,
    answer: str,
    report: GroundingReport,
    supported: Sequence[str],
) -> str | None:
    """Record the report on ``run_id``; returns the feedback id. An answer with no checkable
    claim (a greeting, a question back) is no verdict on the run: nothing is recorded."""
    if not report.claims:
        return None
    record = Feedback(
        feedback_id=f"fb_{stable_key(ctx.tenant_id, run_id, bundle_id, content_hash(answer))}",
        tenant_id=ctx.tenant_id,
        target_kind=FeedbackTargetKind.RUN,
        target_id=run_id,
        verdict=verdict_of(report),
        score=round(1.0 - report.per_claim_hallucination_rate, 4),
        reviewer=REVIEWER,
        source=FeedbackSource.JUDGE,
        comment=f"{report.supported}/{len(report.claims)} claims supported, "
        f"{report.contradicted} contradicted",
        evidence_refs=[
            FeedbackEvidenceRef(source_type=EvidenceSource.MEMORY, source_id=item_id)
            for item_id in dict.fromkeys(supported)
            if item_id.startswith("mem_")
        ][:CITED_MAX],
        metadata={"bundle_id": bundle_id},
    )
    async with uow_factory() as uow:
        # the service's own verdict: applied as it arrives, never left for review
        stored, _ = await feedback.submit(uow, ctx, record, trusted=True)
        await uow.commit()
    return stored.feedback_id
