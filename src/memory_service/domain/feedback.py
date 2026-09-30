"""Feedback: the learning signal - human, interrupt, judge and system judgements on what the
platform did.

The wire shape is ``trellis.contracts.Feedback`` as is; this module does not import the
contracts package because the service is usable on its own. Feedback is stored apart from
memory content: a memory is what was learned, feedback is what someone thought of it. The
projector (``modules.feedback``) turns a verdict into what it judges - a memory's standing, a
run's outcome, a tool's statistics and approval patterns, a procedure - and records what it
did in ``projection``.

A run's outcome is a projection of its feedback: the harness sends RUN feedback with
``source=system`` from the run's final status, ``/v1/verify`` writes it with ``source=judge``,
and a person overrides both. ``OUTCOME_PRECEDENCE`` orders them; a verdict never replaces an
outcome a higher-ranked source already gave.
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
    #: a run: its outcome, and - through ``evidence_refs`` - the memories its answer cited
    RUN = "run"
    MEMORY = "memory"
    TOOL_CALL = "tool_call"
    PROCEDURE = "procedure"


class FeedbackVerdict(StrEnum):
    CONFIRM = "confirm"
    REJECT = "reject"
    CORRECT = "correct"
    APPROVE = "approve"
    EDIT = "edit"


class FeedbackSource(StrEnum):
    HUMAN = "human"
    #: a person's answer to an interrupt (an approval, a review): a human verdict
    INTERRUPT = "interrupt"
    #: the grounding judge (``POST /v1/verify``)
    JUDGE = "judge"
    #: derived from a run's final status by the harness
    SYSTEM = "system"


#: Who decides a run's outcome: a verdict replaces the stored outcome only when its source
#: ranks at least as high (the last word wins within one rank).
OUTCOME_PRECEDENCE: Final = {
    FeedbackSource.SYSTEM: 1,
    FeedbackSource.JUDGE: 2,
    FeedbackSource.INTERRUPT: 3,
    FeedbackSource.HUMAN: 3,
}


class ProjectionAction(StrEnum):
    """What the projector did with a record; ``NONE`` says why it did nothing."""

    NONE = "none"
    MEMORY_REINFORCED = "memory_reinforced"
    MEMORY_RETRACTED = "memory_retracted"
    MEMORY_SUPERSEDED = "memory_superseded"
    #: a run verdict labelled the run (and moved the confidence of the memories it cited)
    RUN_LABELLED = "run_labelled"
    #: a tool-call verdict counted toward the tool's statistics and approval patterns
    TOOL_CALL_COUNTED = "tool_call_counted"
    #: a procedure verdict rejected the procedure: it is no longer offered
    PROCEDURE_REJECTED = "procedure_rejected"


#: Verdicts that carry a replacement: the projector writes a corrected memory for them.
CORRECTING_VERDICTS: Final = frozenset({FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT})
#: Verdicts that agree with the target as it stands.
AFFIRMING_VERDICTS: Final = frozenset({FeedbackVerdict.CONFIRM, FeedbackVerdict.APPROVE})
#: The approval counter a verdict on a tool call adds to.
APPROVAL_COUNTER: Final = {
    FeedbackVerdict.APPROVE: "approvals",
    FeedbackVerdict.CONFIRM: "approvals",
    FeedbackVerdict.REJECT: "rejections",
    FeedbackVerdict.EDIT: "edits",
    FeedbackVerdict.CORRECT: "edits",
}


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
    memory_ids: list[str] = Field(
        default_factory=list, description="the cited memories a run verdict adjusted"
    )
    run_id: str | None = Field(default=None, description="the run whose outcome was labelled")
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

    def cited_memory_ids(self, limit: int) -> list[str]:
        """The memories the judged target cited, in the order the reviewer named them: the
        evidence references that point at a memory (its source type, or its id prefix)."""
        ids = [
            ref.source_id
            for ref in self.evidence_refs
            if ref.source_type == "memory" or ref.source_id.startswith("mem_")
        ]
        return list(dict.fromkeys(ids))[:limit]

    @model_validator(mode="after")
    def _correction_when_correcting(self) -> Feedback:
        if self.verdict in CORRECTING_VERDICTS and self.correction in (None, ""):
            raise ValueError(f"a {self.verdict.value} verdict needs a correction")
        return self
