"""FeedbackRepository port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.feedback import (
    Feedback,
    FeedbackProjection,
    FeedbackReview,
    FeedbackTargetGroup,
    FeedbackTargetKind,
    ReviewState,
)


@runtime_checkable
class FeedbackRepository(Protocol):
    async def add(self, feedback: Feedback) -> bool:
        """Store the record; False when this ``feedback_id`` was already stored (a retry)."""
        ...

    async def get(self, tenant_id: str, feedback_id: str) -> Feedback | None: ...

    async def list_for(
        self,
        tenant_id: str,
        *,
        target_kind: FeedbackTargetKind,
        target_id: str,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        """Newest first; ``before`` is the (created_at, feedback_id) keyset of the next page."""
        ...

    async def set_projection(
        self, tenant_id: str, feedback_id: str, projection: FeedbackProjection
    ) -> None: ...

    async def set_review(
        self, tenant_id: str, feedback_id: str, review: FeedbackReview
    ) -> None: ...

    async def list_pending(
        self,
        tenant_id: str,
        *,
        state: ReviewState = ReviewState.PENDING,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        """The review queue (or the verdicts in another review ``state``), newest first,
        keyset paged like ``list_for``."""
        ...

    async def group_by_target(
        self,
        tenant_id: str,
        *,
        state: ReviewState = ReviewState.PENDING,
        before: tuple[datetime, str, str] | None = None,
        limit: int = 100,
    ) -> list[FeedbackTargetGroup]:
        """The verdicts in ``state`` counted per target, the most recently voted first;
        ``before`` is the (latest_at, target_kind, target_id) keyset of the next page."""
        ...

    async def pending_on(
        self,
        tenant_id: str,
        *,
        target_kind: FeedbackTargetKind,
        target_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        voter_only: bool = False,
        votes_only: bool = False,
        limit: int = 1000,
    ) -> list[Feedback]:
        """The pending verdicts on one target, oldest first. With ``voter_only``, only one
        voter's: the person (``user_id``) when there is one, else the agent (``agent_id``);
        with ``votes_only``, only votes (``Feedback.vote``; by source on older records)."""
        ...

    async def applied_votes(
        self,
        tenant_id: str,
        *,
        target_kind: FeedbackTargetKind,
        target_id: str,
        user_id: str | None,
        agent_id: str | None,
        limit: int = 20,
    ) -> list[Feedback]:
        """One voter's votes on one target that a reviewer approved, newest first."""
        ...

    async def review_counts(
        self, tenant_id: str, *, user_id: str | None, agent_id: str | None
    ) -> dict[str, int]:
        """How one author's verdicts fared in review: pending, approved and dismissed."""
        ...
