"""Public /v1 route: verify an answer against evidence (the grounding cascade).

The caller may only verify against evidence it could retrieve: a ``bundle_id`` resolves
inside the caller's tenant, a ``query`` re-runs retrieval under the caller's scope, and
``items`` are text the caller already holds.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.domain.errors import NotFound, ProviderNotConfigured
from memory_service.modules.grounding.cascade import Evidence, bundle_evidence

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 422, 503)

_SCOPE: dict[str, Any] = {
    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
}
_ANSWER = (
    "Adjusted EBITDA increased to EUR 98 million [1]. "
    "Revenue fell to EUR 400 million because of lower volumes."
)
_ITEM: dict[str, Any] = {
    "item_id": "chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "kind": "chunk",
    "text": "Adjusted EBITDA increased to EUR 98 million (FY25: EUR 90 million).",
    "citation": "chunk_id:chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
}
_VERIFY_EXAMPLE: dict[str, Any] = {"scope": _SCOPE, "answer": _ANSWER, "items": [_ITEM]}
_CLAIM_EXAMPLE: dict[str, Any] = {
    "claim": "Adjusted EBITDA increased to EUR 98 million",
    "verdict": "supported",
    "support": 0.97,
    "contradiction": 0.01,
    "evidence_ids": ["chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"],
    "contradicted_by": [],
    "citations": ["1"],
    "method": "nli",
    "notes": [],
}
_REPORT_EXAMPLE: dict[str, Any] = {
    "source": "items",
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
    "nli_provider": "nli-deberta-v3-base-mnli-fever-anli",
    "representative": True,
    "judge_consulted": 0,
    "llm_tokens": 0,
    "evidence_count": 1,
    "unused_count": 0,
    "notes": [],
}


class VerifyItem(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_ITEM]})

    item_id: str = Field(..., min_length=1, examples=[_ITEM["item_id"]])
    text: str = Field(..., min_length=1, max_length=20_000, examples=[_ITEM["text"]])
    kind: str = Field(default="chunk", examples=["chunk"])
    citation: str | None = Field(default=None, examples=[_ITEM["citation"]])

    def to_evidence(self) -> Evidence:
        return Evidence(
            item_id=self.item_id, text=self.text, kind=self.kind, citation=self.citation or ""
        )


class VerifyRequest(BaseModel):
    """Exactly one evidence source: ``bundle_id`` (a bundle from /v1/context, while cached),
    ``items`` (evidence the caller holds; ``unused`` may accompany it) or ``query``
    (retrieval is re-run under the caller's scope)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_VERIFY_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody, examples=[_SCOPE])
    answer: str = Field(..., min_length=1, max_length=40_000, examples=[_ANSWER])
    bundle_id: str | None = Field(default=None, max_length=64, examples=[None])
    items: list[VerifyItem] | None = Field(default=None, max_length=200, examples=[[_ITEM]])
    unused: list[VerifyItem] | None = Field(
        default=None,
        max_length=50,
        description="retrieved-but-unused evidence scanned for contradictions",
        examples=[None],
    )
    query: str | None = Field(default=None, min_length=1, max_length=4000, examples=[None])
    document_ids: list[str] | None = Field(default=None, examples=[None])

    @model_validator(mode="after")
    def _one_source(self) -> VerifyRequest:
        sources = [
            name
            for name, value in (
                ("bundle_id", self.bundle_id),
                ("items", self.items),
                ("query", self.query),
            )
            if value
        ]
        if len(sources) != 1:
            raise ValueError("exactly one of bundle_id, items or query is required")
        if self.unused and sources != ["items"]:
            raise ValueError("unused may only accompany items")
        return self


class ClaimVerdictBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [_CLAIM_EXAMPLE]})

    claim: str
    verdict: str = Field(..., description="supported | unsupported | contradicted | borderline")
    support: float
    contradiction: float
    evidence_ids: list[str] = Field(default_factory=list)
    contradicted_by: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    method: str = Field(..., description="citation | nli | judge")
    notes: list[str] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    """GroundingReport: one verdict per claim and the per-claim hallucination rate."""

    model_config = ConfigDict(json_schema_extra={"examples": [_REPORT_EXAMPLE]})

    source: str = Field(..., description="bundle | items | query")
    claims: list[ClaimVerdictBody]
    supported: int
    unsupported: int
    contradicted: int
    borderline: int
    per_claim_hallucination_rate: float
    nli_provider: str
    representative: bool
    judge_consulted: int
    llm_tokens: int
    evidence_count: int
    unused_count: int
    notes: list[str] = Field(default_factory=list)


@router.post(
    "/verify",
    response_model=VerifyResponse,
    tags=["retrieval"],
    summary="Verify an answer claim by claim against evidence",
    responses=_ERRORS,
)
async def verify(
    request: Request, body: VerifyRequest, container: ContainerDep, _: ServicePrincipalDep
) -> VerifyResponse:
    ctx = build_context(request, container, body.scope)
    cascade = container.services.get("grounding")
    if cascade is None:
        raise ProviderNotConfigured("models.nli.provider=disabled")
    builder = container.services["context_builder"]
    if body.items:
        source = "items"
        evidence = [i.to_evidence() for i in body.items]
        unused = [i.to_evidence() for i in body.unused or []]
    elif body.bundle_id:
        source = "bundle"
        bundle = await builder.cached(ctx, body.bundle_id)
        if bundle is None:
            raise NotFound("bundle not found or no longer cached; pass items or query")
        evidence, unused = bundle_evidence(bundle)
    else:
        source = "query"
        bundle = await builder.build(ctx, str(body.query), document_ids=body.document_ids)
        evidence, unused = bundle_evidence(bundle)
    report = await cascade.verify(body.answer, evidence, unused=unused)
    return VerifyResponse(source=source, **report.model_dump(mode="json"))
