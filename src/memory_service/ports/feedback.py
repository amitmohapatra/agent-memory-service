"""FeedbackRepository port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.feedback import (
    Feedback,
    FeedbackProjection,
    FeedbackReview,
    FeedbackTargetKind,
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
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        """The review queue: pending verdicts, newest first, keyset paged like ``list_for``."""
        ...

    async def review_counts(
        self, tenant_id: str, *, user_id: str | None, agent_id: str | None
    ) -> dict[str, int]:
        """How one author's verdicts fared in review: pending, approved and dismissed."""
        ...
