"""Feedback wire shapes: the contracts ``Feedback`` record in, the stored record out."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from memory_service.api.validation import CustomMetadata, bounded_json
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.feedback import (
    COMMENT_MAX_CHARS,
    CORRECTING_VERDICTS,
    EVIDENCE_REFS_MAX,
    FEEDBACK_JSON_MAX_BYTES,
    REVIEWER_MAX_CHARS,
    Feedback,
    FeedbackEvidenceRef,
    FeedbackProjection,
    FeedbackReview,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
)

BoundedCorrection = Annotated[Any, AfterValidator(bounded_json(FEEDBACK_JSON_MAX_BYTES))]


class FeedbackRequest(BaseModel):
    """``trellis.contracts.Feedback`` as is. The identity fields are optional here because
    the trusted headers decide them; a value that disagrees with its header is refused."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "feedback_id": "fb_01J8ZK3N7R2Q4X5V6W7Y8Z9A0B",
                    "target_kind": "memory",
                    "target_id": "mem_01J8ZK3N7R2Q4X5V6W7Y8Z9A0B",
                    "verdict": "correct",
                    "correction": "The renewal is in March, not May.",
                    "reviewer": "u-123",
                    "source": "human",
                }
            ]
        },
    )

    feedback_id: str | None = Field(default=None, max_length=200)
    tenant_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    agent_run_id: str | None = None
    trace_id: str | None = Field(
        default=None,
        max_length=64,
        description="accepted for the contracts shape; the stored trace is the request's",
    )
    target_kind: FeedbackTargetKind
    target_id: str = Field(min_length=1, max_length=200)
    verdict: FeedbackVerdict
    correction: BoundedCorrection = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    comment: str | None = Field(default=None, max_length=COMMENT_MAX_CHARS)
    reviewer: str | None = Field(default=None, max_length=REVIEWER_MAX_CHARS)
    source: FeedbackSource = FeedbackSource.HUMAN
    evidence_refs: list[FeedbackEvidenceRef] = Field(
        default_factory=list, max_length=EVIDENCE_REFS_MAX
    )
    metadata: CustomMetadata = Field(default_factory=dict)
    created_at: AwareDatetime | None = Field(
        default=None,
        description="accepted for the contracts shape; the stored instant is the service's",
    )

    @model_validator(mode="after")
    def _correction_when_correcting(self) -> FeedbackRequest:
        if self.verdict in CORRECTING_VERDICTS and self.correction in (None, ""):
            raise ValueError(f"a {self.verdict.value} verdict needs a correction")
        return self

    def to_domain(self, ctx: MemoryExecutionContext) -> Feedback:
        fields = self.model_dump(exclude_none=True, exclude={"tenant_id"})
        return Feedback(tenant_id=ctx.tenant_id, **fields)


class FeedbackResponse(BaseModel):
    feedback_id: str
    tenant_id: str
    workspace_id: str | None
    user_id: str | None
    agent_id: str | None
    agent_run_id: str | None
    trace_id: str | None
    target_kind: FeedbackTargetKind
    target_id: str
    verdict: FeedbackVerdict
    correction: Any
    score: float | None
    comment: str | None
    reviewer: str | None
    source: FeedbackSource
    evidence_refs: list[FeedbackEvidenceRef]
    metadata: dict[str, Any]
    created_at: datetime
    projection: FeedbackProjection | None = Field(
        default=None, description="what the projector did; null until it has run"
    )
    review: FeedbackReview | None = Field(
        default=None,
        description="null: applied as it arrived; else pending (changes nothing until a tenant "
        "admin approves it), approved or dismissed",
    )
    author_record: dict[str, int] | None = Field(
        default=None,
        description="in the review queue only: how this author's verdicts fared in review "
        "(pending, approved, dismissed)",
    )

    @classmethod
    def of(
        cls, feedback: Feedback, author_record: dict[str, int] | None = None
    ) -> FeedbackResponse:
        return cls.model_validate({**feedback.model_dump(), "author_record": author_record})


class FeedbackReviewRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"note": "Checked against the signed contract."}]},
    )

    note: str | None = Field(
        default=None, max_length=COMMENT_MAX_CHARS, description="why, for the record"
    )


class FeedbackListResponse(BaseModel):
    feedback: list[FeedbackResponse]
    next_cursor: str | None = Field(
        default=None, description="pass as `cursor` for the next page; null on the last"
    )
