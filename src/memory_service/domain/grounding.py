"""Per-claim grounding: what the cascade says about each assertion of an answer.

An answer is decomposed into claims; each claim gets one verdict from the cheapest stage
that could decide it (citation validation, NLI, LLM judge) plus the evidence it rests on or
collides with. The headline metric is the per-claim hallucination rate, never an
answer-level average.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClaimVerdict = Literal["supported", "unsupported", "contradicted", "borderline"]
GroundingMethod = Literal["citation", "nli", "judge"]


class ClaimReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim: str
    verdict: ClaimVerdict
    support: float = Field(default=0.0, ge=0.0, le=1.0, description="best entailment score")
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0, description="best contradiction")
    evidence_ids: list[str] = Field(default_factory=list, description="items the verdict rests on")
    contradicted_by: list[str] = Field(
        default_factory=list, description="retrieved-but-unused items that contradict the claim"
    )
    citations: list[str] = Field(default_factory=list, description="citation markers in the claim")
    method: GroundingMethod = "nli"
    notes: list[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims: list[ClaimReport] = Field(default_factory=list)
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    borderline: int = 0
    per_claim_hallucination_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="(unsupported + contradicted) / claims"
    )
    nli_provider: str = ""
    representative: bool = Field(
        default=False, description="False when the NLI is a deterministic stand-in"
    )
    judge_consulted: int = Field(default=0, description="borderline claims sent to the LLM judge")
    llm_tokens: int = Field(default=0, description="LLM tokens spent on this report")
    evidence_count: int = 0
    unused_count: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return self.per_claim_hallucination_rate == 0.0
