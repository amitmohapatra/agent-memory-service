"""FeedbackRepository port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.feedback import (
    Feedback,
    FeedbackProjection,
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
