"""Public response models for ranked context and grounding: the wire form of the domain's
ContextItem, EvidenceReport and GroundingReport.

They mirror the domain models field for field but are separate classes on purpose: the
domain models are free to grow, while these are the contract the SDK and the benchmark
harness rely on, so every closed set is an enum with a description and nothing extra is
allowed through.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import EvidenceStatus, Representation
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.grounding import ClaimVerdict, GroundingMethod
from memory_service.modules.grounding.cascade import EvidenceKind

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

REPRESENTATION_DESCRIPTION = (
    "Which representation of knowledge the item is: CHUNK, TABLE, PARAGRAPH, SECTION, "
    "SUBSECTION or CODE_BLOCK for document passages, MEMORY for a canonical memory, "
    "RELATION for a graph fact, SUMMARY for a rolled-up summary."
)


class ContextItemBody(BaseModel):
    """One ranked piece of context."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    representation: Representation = Field(..., description=REPRESENTATION_DESCRIPTION)
    text: str
    score: float = Field(
        default=0.0,
        description="The raw number the ranking stage produced; its meaning depends on "
        "score_kind, so it is not comparable between items. Use relevance.",
    )
    relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Comparable across the bundle, 0..1, higher is better.",
    )
    score_kind: Literal["cross_encoder", "fusion", "exact"] = Field(
        default="fusion",
        description="Where score came from: a cross-encoder probability (cross_encoder), a "
        "fusion rank score (fusion) or an exact identifier hit (exact).",
    )
    retrievers: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    citation: str = Field(..., description="stable citation key")
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = Field(default=None, description="e.g. 'Financial Results > EBITDA'")
    expanded_from: str | None = Field(default=None, description="item_id this was expanded from")
    expansion_edge: str | None = Field(default=None, description="PARENT | NEXT | DEFINED_BY | ...")
    token_estimate: int = 0
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="structured extras: predicate/subject/object for facts and memories, "
        "fact attributes (period, currency, amount...), contradicts/contributors",
    )


class ConversationWindowBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    rendered: str = ""
    token_estimate: int = 0
    summary: str | None = Field(default=None, description="rolling summary of older messages")


class UnusedEvidenceBody(BaseModel):
    """Retrieved but not packed (reranked out or over budget); the grounding cascade scans
    these for contradictions with the answer."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: EvidenceKind = Field(
        ..., description="Record kind of the unused item: chunk, memory, summary or fact."
    )
    text: str


class ClaimVerdictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_CLAIM_EXAMPLE]})

    claim: str
    verdict: ClaimVerdict = Field(
        ...,
        description="supported: every fact in the claim follows from the evidence; "
        "unsupported: the evidence does not say it; contradicted: retrieved evidence says "
        "otherwise; borderline: the NLI could not decide and no judge settled it.",
    )
    support: float = Field(default=0.0, ge=0.0, le=1.0, description="best entailment score")
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0, description="best contradiction")
    evidence_ids: list[str] = Field(default_factory=list, description="items the verdict rests on")
    contradicted_by: list[str] = Field(
        default_factory=list, description="retrieved-but-unused items that contradict the claim"
    )
    citations: list[str] = Field(default_factory=list, description="citation markers in the claim")
    method: GroundingMethod = Field(
        default="nli",
        description="The cheapest stage that decided the claim: citation (the cited item "
        "settles it), nli (cross-encoder entailment) or judge (LLM, borderline claims only).",
    )
    notes: list[str] = Field(default_factory=list)


class GroundingReportBody(BaseModel):
    """GroundingReport: one verdict per claim and the per-claim hallucination rate."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimVerdictBody] = Field(default_factory=list)
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


class EvidenceReportBody(BaseModel):
    """EvidenceReport: whether the packed evidence can answer the query."""

    model_config = ConfigDict(extra="forbid")

    status: EvidenceStatus = Field(
        ...,
        description="COMPLETE: every required companion passage is present; INCOMPLETE: "
        "evidence exists but a required companion is missing or the subject asked about is "
        "not established; INSUFFICIENT: nothing usable was retrieved, do not answer from it.",
    )
    required_groups: list[str] = Field(default_factory=list)
    satisfied_groups: list[str] = Field(default_factory=list)
    missing_groups: list[str] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list, description="strategies attempted")
    notes: list[str] = Field(default_factory=list)
    unused: list[UnusedEvidenceBody] = Field(
        default_factory=list, description="retrieved-but-unused evidence (bounded)"
    )
    grounding: GroundingReportBody | None = Field(
        default=None, description="per-claim verdicts when an answer was verified"
    )
    llm_tokens: int = Field(default=0, description="LLM tokens spent building this report")
