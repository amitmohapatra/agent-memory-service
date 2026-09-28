"""Feedback rows: one per (tenant, feedback_id); listing is keyset-paged newest first."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from sqlalchemy import literal, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import FeedbackRow
from memory_service.domain.feedback import Feedback, FeedbackProjection, FeedbackTargetKind


def _to_domain(r: FeedbackRow) -> Feedback:
    return Feedback.model_validate(
        {
            "feedback_id": r.feedback_id,
            "tenant_id": r.tenant_id,
            "workspace_id": r.workspace_id,
            "user_id": r.user_id,
            "agent_id": r.agent_id,
            "agent_run_id": r.agent_run_id,
            "trace_id": r.trace_id,
            "target_kind": r.target_kind,
            "target_id": r.target_id,
            "verdict": r.verdict,
            "source": r.source,
            "correction": r.correction,
            "score": r.score,
            "comment": r.comment,
            "reviewer": r.reviewer,
            "evidence_refs": list(r.evidence_refs or []),
            "metadata": dict(r.metadata_ or {}),
            "created_at": r.created_at,
            "projection": r.projection,
        }
    )


class SqlFeedbackRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, feedback: Feedback) -> bool:
        values: dict[str, Any] = {
            **feedback.model_dump(mode="json", exclude={"metadata", "projection", "created_at"}),
            "metadata_": feedback.metadata,
            "created_at": feedback.created_at,
            "projection": feedback.projection.model_dump(mode="json")
            if feedback.projection
            else None,
            "projected_at": feedback.projection.projected_at if feedback.projection else None,
        }
        stmt = (
            insert(FeedbackRow)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["tenant_id", "feedback_id"])
        )
        result = cast(CursorResult[Any], await self.s.execute(stmt))
        return (result.rowcount or 0) > 0

    async def get(self, tenant_id: str, feedback_id: str) -> Feedback | None:
        row = await self.s.get(FeedbackRow, (tenant_id, feedback_id))
        return _to_domain(row) if row is not None else None

    async def list_for(
        self,
        tenant_id: str,
        *,
        target_kind: FeedbackTargetKind,
        target_id: str,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        stmt = select(FeedbackRow).where(
            FeedbackRow.tenant_id == tenant_id,
            FeedbackRow.target_kind == target_kind.value,
            FeedbackRow.target_id == target_id,
        )
        if before is not None:
            stmt = stmt.where(
                tuple_(FeedbackRow.created_at, FeedbackRow.feedback_id)
                < tuple_(literal(before[0]), literal(before[1]))
            )
        stmt = stmt.order_by(FeedbackRow.created_at.desc(), FeedbackRow.feedback_id.desc())
        rows = (await self.s.scalars(stmt.limit(limit))).all()
        return [_to_domain(r) for r in rows]

    async def set_projection(
        self, tenant_id: str, feedback_id: str, projection: FeedbackProjection
    ) -> None:
        await self.s.execute(
            update(FeedbackRow)
            .where(FeedbackRow.tenant_id == tenant_id, FeedbackRow.feedback_id == feedback_id)
            .values(
                projection=projection.model_dump(mode="json"),
                projected_at=projection.projected_at,
            )
        )
