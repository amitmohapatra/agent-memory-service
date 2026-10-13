"""Feedback rows: one per (tenant, feedback_id); listing is keyset-paged newest first."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from sqlalchemy import distinct, func, literal, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import FeedbackRow
from memory_service.domain.feedback import (
    HUMAN_SOURCES,
    THUMBS_UP_VERDICTS,
    Feedback,
    FeedbackProjection,
    FeedbackReview,
    FeedbackTargetGroup,
    FeedbackTargetKind,
    ReviewState,
)


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
            "vote": r.is_vote,
            "review": {
                "state": r.review_state,
                "reviewed_by": r.reviewed_by,
                "reviewed_at": r.reviewed_at,
                "note": r.review_note,
                "replaced_by": r.review_replaced_by,
            }
            if r.review_state
            else None,
        }
    )


def _voter(stmt: Any, user_id: str | None, agent_id: str | None) -> Any:
    """One voter's rows: the person's when there is one (whichever agent carried them), else
    the agent's own with no person behind them."""
    if user_id is not None:
        return stmt.where(FeedbackRow.user_id == user_id)
    return stmt.where(
        FeedbackRow.user_id.is_(None), FeedbackRow.agent_id.is_not_distinct_from(agent_id)
    )


def _is_vote() -> Any:
    """A vote as classified when stored; a record older than the flag by its source."""
    return func.coalesce(
        FeedbackRow.is_vote, FeedbackRow.source.in_([s.value for s in HUMAN_SOURCES])
    )


def _review_columns(review: FeedbackReview | None) -> dict[str, Any]:
    return {
        "review_state": review.state.value if review else None,
        "reviewed_by": review.reviewed_by if review else None,
        "reviewed_at": review.reviewed_at if review else None,
        "review_note": review.note if review else None,
        "review_replaced_by": review.replaced_by if review else None,
    }


class SqlFeedbackRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, feedback: Feedback) -> bool:
        values: dict[str, Any] = {
            **feedback.model_dump(
                mode="json",
                exclude={"metadata", "projection", "created_at", "review", "vote"},
            ),
            "is_vote": feedback.vote,
            **_review_columns(feedback.review),
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
        return await self._page(stmt, before=before, limit=limit)

    async def list_pending(
        self,
        tenant_id: str,
        *,
        state: ReviewState = ReviewState.PENDING,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        stmt = select(FeedbackRow).where(
            FeedbackRow.tenant_id == tenant_id,
            FeedbackRow.review_state == state.value,
        )
        return await self._page(stmt, before=before, limit=limit)

    async def group_by_target(
        self,
        tenant_id: str,
        *,
        state: ReviewState = ReviewState.PENDING,
        before: tuple[datetime, str, str] | None = None,
        limit: int = 100,
    ) -> list[FeedbackTargetGroup]:
        up = FeedbackRow.verdict.in_([v.value for v in THUMBS_UP_VERDICTS])
        # one person is one voter whichever agent carried their vote; an agent with no
        # person behind it is a voter of its own
        voter = func.coalesce(
            literal("user:").concat(FeedbackRow.user_id),
            literal("agent:").concat(func.coalesce(FeedbackRow.agent_id, "")),
        )
        grouped = (
            select(
                FeedbackRow.target_kind.label("target_kind"),
                FeedbackRow.target_id.label("target_id"),
                func.count().label("votes"),
                func.count().filter(up).label("thumbs_up"),
                func.count(distinct(voter)).label("voters"),
                func.max(FeedbackRow.created_at).label("latest_at"),
            )
            .where(
                FeedbackRow.tenant_id == tenant_id,
                FeedbackRow.review_state == state.value,
                # a click its voter replaced is not a vote: counts are of votes
                FeedbackRow.review_replaced_by.is_(None),
            )
            .group_by(FeedbackRow.target_kind, FeedbackRow.target_id)
            .subquery()
        )
        stmt = select(grouped)
        if before is not None:
            stmt = stmt.where(
                tuple_(grouped.c.latest_at, grouped.c.target_kind, grouped.c.target_id)
                < tuple_(literal(before[0]), literal(before[1]), literal(before[2]))
            )
        stmt = stmt.order_by(
            grouped.c.latest_at.desc(), grouped.c.target_kind.desc(), grouped.c.target_id.desc()
        ).limit(limit)
        return [
            FeedbackTargetGroup(
                target_kind=FeedbackTargetKind(r.target_kind),
                target_id=r.target_id,
                votes=int(r.votes),
                thumbs_up=int(r.thumbs_up),
                thumbs_down=int(r.votes) - int(r.thumbs_up),
                voters=int(r.voters),
                latest_at=r.latest_at,
            )
            for r in (await self.s.execute(stmt)).all()
        ]

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
        stmt = select(FeedbackRow).where(
            FeedbackRow.tenant_id == tenant_id,
            FeedbackRow.target_kind == target_kind.value,
            FeedbackRow.target_id == target_id,
            FeedbackRow.review_state == ReviewState.PENDING.value,
        )
        if voter_only:
            stmt = _voter(stmt, user_id, agent_id)
        if votes_only:
            stmt = stmt.where(_is_vote())
        stmt = stmt.order_by(FeedbackRow.created_at, FeedbackRow.feedback_id).limit(limit)
        return [_to_domain(r) for r in (await self.s.scalars(stmt)).all()]

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
        stmt = select(FeedbackRow).where(
            FeedbackRow.tenant_id == tenant_id,
            FeedbackRow.target_kind == target_kind.value,
            FeedbackRow.target_id == target_id,
            FeedbackRow.review_state == ReviewState.APPROVED.value,
            _is_vote(),
        )
        stmt = _voter(stmt, user_id, agent_id).order_by(
            FeedbackRow.created_at.desc(), FeedbackRow.feedback_id.desc()
        )
        return [_to_domain(r) for r in (await self.s.scalars(stmt.limit(limit))).all()]

    async def _page(
        self, stmt: Any, *, before: tuple[datetime, str] | None, limit: int
    ) -> list[Feedback]:
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

    async def set_review(self, tenant_id: str, feedback_id: str, review: FeedbackReview) -> None:
        await self.s.execute(
            update(FeedbackRow)
            .where(FeedbackRow.tenant_id == tenant_id, FeedbackRow.feedback_id == feedback_id)
            .values(**_review_columns(review))
        )

    async def review_counts(
        self, tenant_id: str, *, user_id: str | None, agent_id: str | None
    ) -> dict[str, int]:
        stmt = (
            select(FeedbackRow.review_state, func.count())
            .where(
                FeedbackRow.tenant_id == tenant_id,
                FeedbackRow.review_state.is_not(None),
                # a vote its author replaced is them changing their mind, not a verdict on them
                FeedbackRow.review_replaced_by.is_(None),
                FeedbackRow.user_id.is_not_distinct_from(user_id),
                FeedbackRow.agent_id.is_not_distinct_from(agent_id),
            )
            .group_by(FeedbackRow.review_state)
        )
        counts = {state.value: 0 for state in ReviewState}
        for state, n in (await self.s.execute(stmt)).all():
            counts[str(state)] = int(n)
        return counts
