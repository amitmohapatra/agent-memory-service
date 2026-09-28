"""Feedback: human, judge and interrupt judgements on what the platform did (ADR 0023).

The wire shape is ``trellis.contracts.Feedback`` (0.3.0) as is; this module does not import
the contracts package because the service is usable on its own. Feedback is stored apart
from memory content: a memory is what was learned, feedback is what a person thought of it.
The projector (``modules.feedback``) turns a verdict on a memory into the existing revision
machinery and records what it did in ``projection``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_service.domain.ids import is_valid_id, new_id

#: Bound on the serialised correction and metadata of one record.
FEEDBACK_JSON_MAX_BYTES: Final = 16_000
REVIEWER_MAX_CHARS: Final = 200
COMMENT_MAX_CHARS: Final = 4000
EVIDENCE_REFS_MAX: Final = 50


class FeedbackTargetKind(StrEnum):
    RUN = "run"
    ANSWER = "answer"
    MEMORY = "memory"
    TOOL_CALL = "tool_call"
    BRIEF = "brief"
    PROCEDURE = "procedure"


class FeedbackVerdict(StrEnum):
    CONFIRM = "confirm"
    REJECT = "reject"
    CORRECT = "correct"
    APPROVE = "approve"
    EDIT = "edit"


class FeedbackSource(StrEnum):
    HUMAN = "human"
    JUDGE = "judge"
    INTERRUPT = "interrupt"


class ProjectionAction(StrEnum):
    """What the projector did with a record; ``NONE`` says why it did nothing."""

    NONE = "none"
    MEMORY_REINFORCED = "memory_reinforced"
    MEMORY_RETRACTED = "memory_retracted"
    MEMORY_SUPERSEDED = "memory_superseded"


#: Verdicts that carry a replacement: the projector writes a corrected memory for them.
CORRECTING_VERDICTS: Final = frozenset({FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT})
#: Verdicts that agree with the target as it stands.
AFFIRMING_VERDICTS: Final = frozenset({FeedbackVerdict.CONFIRM, FeedbackVerdict.APPROVE})


class FeedbackEvidenceRef(BaseModel):
    """What a reviewer pointed at: the contracts ``EvidenceRef`` shape, read as data (unknown
    keys ignored, nothing required beyond the source). It is a pointer for people, not the
    memory evidence model, which is why ``observed_at`` may be absent."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    source_type: str = Field(default="memory", max_length=50)
    source_id: str = Field(min_length=1, max_length=200)
    message_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    citation: str | None = Field(default=None, max_length=2000)
    observed_at: AwareDatetime | None = None


class FeedbackProjection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: ProjectionAction
    memory_id: str | None = Field(default=None, description="the memory the verdict landed on")
    superseded_by: str | None = Field(
        default=None, description="the corrected memory, for MEMORY_SUPERSEDED"
    )
    reason: str | None = Field(default=None, max_length=500)
    projected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Feedback(BaseModel):
    """One judgement. ``feedback_id`` is the client's, so a retried submission is one row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feedback_id: str = Field(default_factory=lambda: new_id("feedback"))
    tenant_id: str
    workspace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    agent_run_id: str | None = None
    trace_id: str | None = Field(default=None, max_length=64)

    target_kind: FeedbackTargetKind
    target_id: str = Field(min_length=1, max_length=200)
    verdict: FeedbackVerdict
    correction: Any = Field(
        default=None, description="the replacement for correct/edit; free-form otherwise"
    )
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    comment: str | None = Field(default=None, max_length=COMMENT_MAX_CHARS)
    reviewer: str | None = Field(default=None, max_length=REVIEWER_MAX_CHARS)
    source: FeedbackSource = FeedbackSource.HUMAN
    evidence_refs: list[FeedbackEvidenceRef] = Field(
        default_factory=list, max_length=EVIDENCE_REFS_MAX
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    projection: FeedbackProjection | None = None

    @field_validator("feedback_id", "target_id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not is_valid_id(value):
            raise ValueError(f"invalid identifier {value!r}")
        return value

    @field_validator("score", mode="before")
    @classmethod
    def _score_is_a_number(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("score must be a number between 0 and 1")
        return value

    @model_validator(mode="after")
    def _correction_when_correcting(self) -> Feedback:
        if self.verdict in CORRECTING_VERDICTS and self.correction in (None, ""):
            raise ValueError(f"a {self.verdict.value} verdict needs a correction")
        return self
