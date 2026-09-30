"""The feedback record: the contracts shape, bounded and closed."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_service.api.schemas.feedback import FeedbackRequest
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.feedback import (
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
)
from memory_service.modules.feedback.service import _correction_text

CTX = MemoryExecutionContext(tenant_id="acme", workspace_id="fin", user_id="u1", agent_id="ref")


def _feedback(**fields):
    base = {
        "tenant_id": "acme",
        "target_kind": FeedbackTargetKind.RUN,
        "target_id": "run_1",
        "verdict": FeedbackVerdict.CONFIRM,
    }
    return Feedback(**{**base, **fields})


def test_ids_are_generated_with_the_contracts_prefix_and_validated() -> None:
    record = _feedback()
    assert record.feedback_id.startswith("fb_") and record.source is FeedbackSource.HUMAN
    assert record.created_at.tzinfo is not None and record.projection is None
    with pytest.raises(ValidationError, match="identifier"):
        _feedback(feedback_id="bad id")
    with pytest.raises(ValidationError, match="identifier"):
        _feedback(target_id="../x")
    with pytest.raises(ValidationError):
        _feedback(bogus=1)


def test_a_correction_verdict_needs_a_correction() -> None:
    for verdict in (FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT):
        with pytest.raises(ValidationError, match="needs a correction"):
            _feedback(verdict=verdict)
        with pytest.raises(ValidationError, match="needs a correction"):
            _feedback(verdict=verdict, correction="")
        assert (
            _feedback(verdict=verdict, correction="March, not May").correction == "March, not May"
        )
    assert _feedback(verdict=FeedbackVerdict.REJECT).correction is None


def test_scores_are_numbers_in_the_unit_interval() -> None:
    for score in (0, 1, 0.5, None):
        _feedback(score=score)
    for score in (-0.1, 1.5, True):
        with pytest.raises(ValidationError):
            _feedback(score=score)


def test_the_vocabularies_are_the_contracts_vocabularies() -> None:
    """trellis-contracts 0.3.0 spells these; a member added on one side must be added here."""
    assert {k.value for k in FeedbackTargetKind} == {
        "run",
        "answer",
        "memory",
        "tool_call",
        "brief",
        "procedure",
    }
    assert {v.value for v in FeedbackVerdict} == {"confirm", "reject", "correct", "approve", "edit"}
    assert {s.value for s in FeedbackSource} == {"human", "judge", "interrupt"}
    assert {a.value for a in ProjectionAction} == {
        "none",
        "memory_reinforced",
        "memory_retracted",
        "memory_superseded",
        "memories_adjusted",
        "run_labelled",
        "tool_call_counted",
        "procedure_rejected",
    }


def test_the_request_takes_its_identity_from_the_context() -> None:
    request = FeedbackRequest.model_validate(
        {"target_kind": "memory", "target_id": "mem_1", "verdict": "correct", "correction": "x"}
    )
    record = request.to_domain(CTX)
    assert record.tenant_id == "acme" and record.workspace_id is None  # the service fills these
    assert (
        record.feedback_id.startswith("fb_") and record.created_at.tzinfo is UTC
    ) or record.created_at.tzinfo is not None
    fixed = FeedbackRequest.model_validate(
        {
            "feedback_id": "fb_fixed",
            "tenant_id": "acme",
            "target_kind": "run",
            "target_id": "run_1",
            "verdict": "confirm",
            "created_at": "2026-09-28T07:00:00Z",
        }
    ).to_domain(CTX)
    assert fixed.feedback_id == "fb_fixed" and fixed.created_at == datetime(
        2026, 9, 28, 7, tzinfo=UTC
    )
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate(
            {"target_kind": "run", "target_id": "r", "verdict": "confirm", "extra": 1}
        )
    with pytest.raises(ValidationError):  # bounded correction
        FeedbackRequest.model_validate(
            {
                "target_kind": "run",
                "target_id": "r",
                "verdict": "confirm",
                "correction": "x" * 20_000,
            }
        )


def test_a_correction_yields_text_only_when_it_carries_some() -> None:
    assert _correction_text("  March  ") == "March"
    assert _correction_text({"content": "March", "why": "invoice"}) == "March"
    assert _correction_text({"why": "invoice"}) == ""
    assert _correction_text(["March"]) == ""
    assert _correction_text(None) == ""


def test_the_contracts_record_is_accepted_as_is() -> None:
    """``trellis.contracts.Feedback.model_dump(mode="json")`` (0.3.0): every member, the
    evidence reference shape included, is accepted without stripping anything."""
    contracts_record = {
        "feedback_id": "fb_01J8ZK3N7R2Q4X5V6W7Y8Z9A0B",
        "tenant_id": "acme",
        "workspace_id": "fin",
        "user_id": "u1",
        "agent_id": "refund-agent",
        "agent_run_id": "run_1",
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "target_kind": "answer",
        "target_id": "art_9",
        "verdict": "reject",
        "correction": None,
        "score": 0.2,
        "comment": "the refund amount is wrong",
        "reviewer": "judge:grounded",
        "source": "judge",
        "evidence_refs": [
            {
                "source_type": "message",
                "source_id": "msg_1",
                "message_id": "msg_1",
                "document_id": None,
                "chunk_id": None,
                "page": None,
                "citation": "the invoice says March",
                "observed_at": None,
            }
        ],
        "metadata": {"method": "grounded"},
        "created_at": "2026-09-28T07:00:00Z",
    }
    request = FeedbackRequest.model_validate(contracts_record)
    record = request.to_domain(CTX)
    assert record.comment == "the refund amount is wrong" and record.score == 0.2
    assert record.evidence_refs[0].citation == "the invoice says March"
    assert record.evidence_refs[0].observed_at is None
    assert record.created_at == datetime(2026, 9, 28, 7, tzinfo=UTC)
    # unknown members of a future evidence reference are ignored, not refused
    widened = {**contracts_record, "evidence_refs": [{"source_id": "x", "future": 1}]}
    assert FeedbackRequest.model_validate(widened).evidence_refs[0].source_type == "memory"
