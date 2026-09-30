"""PostgreSQL persistence of profile blocks and thread summaries."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import ProfileBlockRow, ThreadSummaryRow
from memory_service.domain.profile import ProfileBlock, ThreadSummary


def _block(row: ProfileBlockRow) -> ProfileBlock:
    return ProfileBlock(
        tenant_id=row.tenant_id,
        scope_key=row.scope_key,
        block=row.block,
        text=row.text,
        version=row.version,
        source=row.source,  # type: ignore[arg-type]
        updated_at=row.updated_at,
    )


def _summary(row: ThreadSummaryRow) -> ThreadSummary:
    return ThreadSummary(
        tenant_id=row.tenant_id,
        thread_id=row.thread_id,
        version=row.version,
        text=row.text,
        covers_to_sequence=row.covers_to_sequence,
        model=row.model,
        created_at=row.created_at,
    )


class SqlProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def blocks(self, tenant_id: str, scope_keys: Sequence[str]) -> list[ProfileBlock]:
        if not scope_keys:
            return []
        rows = (
            await self.s.execute(
                select(ProfileBlockRow)
                .where(
                    ProfileBlockRow.tenant_id == tenant_id,
                    ProfileBlockRow.scope_key.in_(list(scope_keys)),
                )
                .order_by(ProfileBlockRow.block)
            )
        ).scalars()
        return [_block(r) for r in rows]

    async def get(self, tenant_id: str, scope_key: str, block: str) -> ProfileBlock | None:
        row = await self.s.get(ProfileBlockRow, (tenant_id, scope_key, block))
        return _block(row) if row is not None else None

    async def put(self, block: ProfileBlock) -> ProfileBlock:
        stmt = insert(ProfileBlockRow).values(**{**block.model_dump(), "version": 1})
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                ProfileBlockRow.tenant_id,
                ProfileBlockRow.scope_key,
                ProfileBlockRow.block,
            ],
            set_={
                "text": stmt.excluded.text,
                "source": stmt.excluded.source,
                "updated_at": stmt.excluded.updated_at,
                "version": ProfileBlockRow.version + 1,
            },
        ).returning(ProfileBlockRow)
        row = (await self.s.scalars(stmt, execution_options={"populate_existing": True})).one()
        return _block(row)


class SqlThreadSummaryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def latest(self, tenant_id: str, thread_id: str) -> ThreadSummary | None:
        row = (
            await self.s.execute(
                select(ThreadSummaryRow)
                .where(
                    ThreadSummaryRow.tenant_id == tenant_id,
                    ThreadSummaryRow.thread_id == thread_id,
                )
                .order_by(ThreadSummaryRow.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return _summary(row) if row is not None else None

    async def add(self, summary: ThreadSummary) -> bool:
        stmt = (
            insert(ThreadSummaryRow)
            .values(**summary.model_dump())
            .on_conflict_do_nothing(
                index_elements=[
                    ThreadSummaryRow.tenant_id,
                    ThreadSummaryRow.thread_id,
                    ThreadSummaryRow.version,
                ]
            )
            .returning(ThreadSummaryRow.version)
        )
        return (await self.s.execute(stmt)).scalar_one_or_none() is not None
