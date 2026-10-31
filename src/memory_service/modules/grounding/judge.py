"""The judge's verdict on a run: a grounding report recorded as RUN feedback.

``/v1/verify`` is the one judge. What it found is feedback like any other (``source=judge``):
it waits in the review queue like a vote, and once a tenant admin approves it the run's
outcome follows it unless a person already decided it. It never raises a memory's confidence:
the answer was checked against the memories it was given, so a confirm would only echo them
(a wrong memory reached 1.0 in two runs). A reject cites only the memories that contradict a
claim (``cited``), never one a supported claim rests on, and none at all when the run called
tools: its answer rests on their results, which the context the judge checked against never
held. For the same reason a claim of such a run that the context merely does not hold is no
verdict on it: only a contradicted claim rejects it. The record's id is derived from the run,
the bundle and the answer, so verifying the same answer twice records one verdict.
"""

from __future__ import annotations

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


def cited(report: GroundingReport) -> list[str]:
    """The memories a verdict on ``report`` moves: on a reject, the ones a contradicted claim
    rests on or is contradicted by, less any a supported claim rests on; on a confirm, none."""
    if verdict_of(report) is FeedbackVerdict.CONFIRM:
        return []
    supported = {i for c in report.claims if c.verdict == "supported" for i in c.evidence_ids}
    against = [
        i
        for c in report.claims
        if c.verdict == "contradicted"
        for i in (*c.evidence_ids, *c.contradicted_by)
    ]
    return [i for i in dict.fromkeys(against) if i.startswith("mem_") and i not in supported]


async def judge_run(
    feedback: FeedbackService,
    uow_factory: UnitOfWorkFactory,
    ctx: MemoryExecutionContext,
    run_id: str,
    bundle_id: str,
    answer: str,
    report: GroundingReport,
) -> str | None:
    """Record the report on ``run_id``; returns the feedback id. An answer with no checkable
    claim (a greeting, a question back) is no verdict on the run, and neither is a context
    that held no evidence: every claim is "unsupported" against nothing, including one the
    run's own tools returned, and rejecting such a run made its outcome a failure, so the
    agent never learned from it. Nothing is recorded for either, nor for a run that called
    tools whose claims are only unsupported: they are what its tools returned (B4b)."""
    if not report.claims or report.evidence_count == 0:
        return None
    async with uow_factory() as uow:
        # an answer built from the run's own tool results moves no memory
        tools = await uow.tools.invocations_for_run(ctx.tenant_id, run_id)
        if tools and not report.contradicted and report.unsupported:
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
                for item_id in ([] if tools else cited(report))
            ][:CITED_MAX],
            metadata={"bundle_id": bundle_id},
        )
        # the service's own word on the run, so no person's vote (``run_vouched``); a model's
        # word all the same, so it waits for a tenant admin (``_needs_review``)
        stored, _ = await feedback.submit(uow, ctx, record, trusted=False, run_vouched=True)
        await uow.commit()
    return stored.feedback_id
