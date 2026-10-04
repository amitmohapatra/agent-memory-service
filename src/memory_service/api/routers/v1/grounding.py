"""Public /v1 route: verify an answer against the context it was given (the grounding
cascade), and record the verdict on the run.

The bundle is the evidence: ``bundle_id`` resolves only for the scope the bundle was built for
(``modules.context.handles``), and the answer's handle citations ([m1], [d2]) resolve within
it. This is the one judge: with a run (``run_id``, or the scope's agent run) it writes RUN
feedback with ``source=judge``, which decides the run's outcome unless a person already has.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.api.schemas.context import GroundingReportBody
from memory_service.domain.errors import NotFound, ProviderNotConfigured
from memory_service.modules.grounding.cascade import record_evidence
from memory_service.modules.grounding.judge import judge_run

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 422, 503)

_SCOPE: dict[str, Any] = {
    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
}
_ANSWER = (
    "Adjusted EBITDA increased to EUR 98 million [d1]. "
    "Revenue fell to EUR 400 million because of lower volumes."
)
_VERIFY_EXAMPLE: dict[str, Any] = {
    "scope": {"agent_run_id": "run_01J8ZK"},
    "bundle_id": "6f1c0b9e2a7d4c3b8e5f1a2b3c4d5e6f",
    "answer": _ANSWER,
}
_CLAIM_EXAMPLE: dict[str, Any] = {
    "claim": "Adjusted EBITDA increased to EUR 98 million",
    "verdict": "supported",
    "support": 0.97,
    "contradiction": 0.01,
    "evidence_ids": ["chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"],
    "contradicted_by": [],
    "citations": ["d1"],
    "method": "nli",
    "notes": [],
}
_REPORT_EXAMPLE: dict[str, Any] = {
    "claims": [
        _CLAIM_EXAMPLE,
        {
            "claim": "Revenue fell to EUR 400 million because of lower volumes",
            "verdict": "unsupported",
            "support": 0.08,
            "contradiction": 0.03,
            "evidence_ids": ["chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"],
            "contradicted_by": [],
            "citations": [],
            "method": "nli",
            "notes": [],
        },
    ],
    "supported": 1,
    "unsupported": 1,
    "contradicted": 0,
    "borderline": 0,
    "per_claim_hallucination_rate": 0.5,
    "nli_provider": "nli-onnx-9c4fa74cd6e43a6647ed8eeb",
    "representative": True,
    "judge_consulted": 0,
    "llm_tokens": 0,
    "evidence_count": 1,
    "unused_count": 0,
    "notes": [],
}


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_VERIFY_EXAMPLE]})

    scope: ScopeBody = Field(
        default_factory=ScopeBody,
        description="The lineage the call acts in (thread, session, turn, work, agent, "
        "run). Tenant, workspace and user come from the trusted headers; a "
        "value here must agree with them.",
    )
    bundle_id: str = Field(
        ..., min_length=1, max_length=64, description="the context the answer was given"
    )
    answer: str = Field(
        ...,
        min_length=1,
        max_length=8_000,
        examples=[_ANSWER],
        description="The answer to check, as the model wrote it (1-8000 characters); handle"
        " citations ([m1], [d2]) resolve within the bundle.",
    )
    run_id: str | None = Field(
        default=None,
        max_length=200,
        description="the run the answer ends (default: the scope's agent run): the verdict is "
        "recorded on it as judge feedback",
    )


class VerifyResponse(GroundingReportBody):
    """GroundingReport: one verdict per claim and the per-claim hallucination rate."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_REPORT_EXAMPLE]})

    feedback_id: str | None = Field(
        default=None, description="the RUN feedback the verdict was recorded as (with a run)"
    )


@router.post(
    "/verify",
    response_model=VerifyResponse,
    tags=["retrieval"],
    summary="Verify an answer claim by claim against its context, and judge the run",
    responses=_ERRORS,
)
async def verify(
    request: Request, body: VerifyRequest, container: ContainerDep, _: ServicePrincipalDep
) -> VerifyResponse:
    ctx = build_context(request, container, body.scope)
    cascade = container.services.get("grounding")
    if cascade is None:
        raise ProviderNotConfigured("the NLI classifier is disabled in this process")
    record = await container.services["bundle_records"].load(ctx, body.bundle_id)
    if record is None:
        raise NotFound("bundle not found: it was built for another scope, or over 30 minutes ago")
    async with container.services["llm_assist"].reading(ctx):
        evidence, unused = record_evidence(record)
        report = await cascade.verify(body.answer, evidence, unused=unused)
    supported = [i for c in report.claims if c.verdict == "supported" for i in c.evidence_ids]
    # what a run's verified answer rests on is what its memory pulls were good for
    await container.services["agent_tools"].used(ctx, supported)
    run_id = body.run_id or ctx.agent_run_id
    feedback_id = (
        await judge_run(
            container.services["feedback"],
            container.services["uow_factory"],
            ctx,
            run_id,
            body.bundle_id,
            body.answer,
            report,
            supported,
        )
        if run_id
        else None
    )
    return VerifyResponse(feedback_id=feedback_id, **report.model_dump(mode="json"))
