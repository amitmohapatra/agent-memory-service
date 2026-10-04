"""Feedback wire shapes: the contracts ``Feedback`` record in, the stored record out."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

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
from memory_service.domain.instants import UTC_RULE, UtcDateTime

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

    feedback_id: str | None = Field(
        default=None,
        max_length=200,
        description="A client id (at most 200 characters) that makes a retry return the "
        "stored record (200) instead of a duplicate; omitted: the service "
        "generates one (fb_...).",
    )
    tenant_id: str | None = Field(
        default=None,
        description="The tenant; optional (the trusted header decides it) and refused when "
        "it disagrees.",
    )
    workspace_id: str | None = Field(
        default=None,
        description="The workspace the verdict was given in; filled from X-Trellis-"
        "Workspace and refused when it disagrees.",
    )
    user_id: str | None = Field(
        default=None,
        description="The person giving the verdict; filled from X-Trellis-User and refused "
        "when it disagrees.",
    )
    agent_id: str | None = Field(
        default=None, description="The agent whose work is judged, when an agent's."
    )
    agent_run_id: str | None = Field(
        default=None, description="The run whose work is judged, when a run's."
    )
    trace_id: str | None = Field(
        default=None,
        max_length=64,
        description="accepted for the contracts shape; the stored trace is the request's",
    )
    target_kind: FeedbackTargetKind = Field(
        description="What is judged: run (an agent run and the answer it gave), memory, "
        "tool_call (a recorded invocation) or procedure (a learned tool plan)."
    )
    target_id: str = Field(
        min_length=1,
        max_length=200,
        description="The judged object's id: a run id, mem_..., an invocation id (tiv_...) "
        "or a procedure id (prc_...).",
    )
    verdict: FeedbackVerdict = Field(
        description="confirm or approve (it is right), reject (it is wrong), correct or "
        "edit (it should say the correction instead)."
    )
    correction: BoundedCorrection = Field(
        default=None,
        description="What it should say instead: required for correct and edit (text, or "
        "{content: ...} for a memory); bounded JSON.",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0..1, an optional graded judgement (1: fully right).",
    )
    comment: str | None = Field(
        default=None, max_length=COMMENT_MAX_CHARS, description="Why, in words, for the record."
    )
    reviewer: str | None = Field(
        default=None,
        max_length=REVIEWER_MAX_CHARS,
        description="Who judged, as the client names them (a user id, judge:<name>).",
    )
    source: FeedbackSource = Field(
        default=FeedbackSource.HUMAN,
        description="Where the verdict comes from: human, interrupt (a person answering an "
        "approval prompt), judge (an automated evaluator) or system (derived "
        "from a run's status).",
    )
    evidence_refs: list[FeedbackEvidenceRef] = Field(
        default_factory=list,
        max_length=EVIDENCE_REFS_MAX,
        description="What the verdict points at, e.g. the memories an answer cited (at most"
        " 50); each must be readable by the caller.",
    )
    metadata: CustomMetadata = Field(default_factory=dict)
    created_at: UtcDateTime | None = Field(
        default=None,
        description="Accepted for the contracts shape and ignored: the stored instant is the "
        f"service's clock. {UTC_RULE}",
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
    feedback_id: str = Field(description="The record's id (the client's, or fb_...).")
    tenant_id: str = Field(description="The tenant the record belongs to.")
    workspace_id: str | None = Field(description="The workspace the verdict was given in, if any.")
    user_id: str | None = Field(description="The person who gave it, if a person.")
    agent_id: str | None = Field(description="The agent whose work is judged, when an agent's.")
    agent_run_id: str | None = Field(description="The run whose work is judged, when a run's.")
    trace_id: str | None = Field(description="The trace of the request that stored it (32 hex).")
    target_kind: FeedbackTargetKind = Field(
        description="What is judged: run (an agent run and the answer it gave), memory, "
        "tool_call (a recorded invocation) or procedure (a learned tool plan)."
    )
    target_id: str = Field(
        description="The judged object's id: a run id, mem_..., an invocation id (tiv_...) "
        "or a procedure id (prc_...)."
    )
    verdict: FeedbackVerdict = Field(
        description="confirm or approve (it is right), reject (it is wrong), correct or "
        "edit (it should say the correction instead)."
    )
    correction: Any = Field(
        description="What it should say instead: required for correct and edit (text, or "
        "{content: ...} for a memory); bounded JSON."
    )
    score: float | None = Field(description="0..1, an optional graded judgement (1: fully right).")
    comment: str | None = Field(description="Why, in words, for the record.")
    reviewer: str | None = Field(
        description="Who judged, as the client names them (a user id, judge:<name>)."
    )
    source: FeedbackSource = Field(
        description="Where the verdict comes from: human, interrupt (a person answering an "
        "approval prompt), judge (an automated evaluator) or system (derived "
        "from a run's status)."
    )
    evidence_refs: list[FeedbackEvidenceRef] = Field(
        description="What the verdict points at, e.g. the memories an answer cited (at most"
        " 50); each must be readable by the caller."
    )
    metadata: dict[str, Any] = Field(description="The caller-defined JSON sent with it.")
    created_at: datetime = Field(description="When the service stored it (ISO 8601, UTC).")
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
    feedback: list[FeedbackResponse] = Field(description="The page, newest first.")
    next_cursor: str | None = Field(
        default=None, description="pass as `cursor` for the next page; null on the last"
    )
