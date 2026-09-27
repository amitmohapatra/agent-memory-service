"""Bounded indexed polling; source content stays in the canonical native database."""

from datetime import datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import BriefRow
from memory_service.domain.briefs import BriefInfo, BriefOutput, BriefSpec, StoredBrief, brief_scope
from memory_service.domain.context import MemoryExecutionContext


def _brief(row: BriefRow) -> StoredBrief:
    return StoredBrief(
        brief_id=row.brief_id,
        context=MemoryExecutionContext.model_validate(row.context),
        spec=BriefSpec.model_validate(row.spec),
        generation=row.generation,
        output=BriefOutput.model_validate(row.output) if row.output is not None else None,
        next_refresh_at=row.next_refresh_at,
        updated_at=row.updated_at,
    )


class SqlBriefRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, tenant_id: str, brief_id: str) -> StoredBrief | None:
        row = await self.session.get(BriefRow, (tenant_id, brief_id))
        return _brief(row) if row is not None else None

    async def add(self, brief: StoredBrief) -> None:
        self.session.add(
            BriefRow(
                tenant_id=brief.context.tenant_id,
                brief_id=brief.brief_id,
                scope_key=brief_scope(brief.context),
                context=brief.context.model_dump(mode="json"),
                spec=brief.spec.model_dump(mode="json"),
                generation=brief.generation,
                output=None,
                next_refresh_at=brief.next_refresh_at,
                updated_at=brief.updated_at,
            )
        )
        await self.session.flush()

    async def save_output(
        self,
        tenant_id: str,
        brief_id: str,
        generation: int,
        output: BriefOutput,
        next_refresh_at: datetime,
    ) -> bool:
        result = await self.session.execute(
            update(BriefRow)
            .where(
                BriefRow.tenant_id == tenant_id,
                BriefRow.brief_id == brief_id,
                BriefRow.generation == generation,
            )
            .values(
                output=output.model_dump(mode="json"),
                next_refresh_at=next_refresh_at,
                updated_at=output.built_at,
            )
            .returning(BriefRow.brief_id)
        )
        return result.scalar_one_or_none() is not None

    async def replace(self, brief: StoredBrief, *, expected_generation: int) -> bool:
        result = await self.session.execute(
            update(BriefRow)
            .where(
                BriefRow.tenant_id == brief.context.tenant_id,
                BriefRow.brief_id == brief.brief_id,
                BriefRow.generation == expected_generation,
            )
            .values(
                spec=brief.spec.model_dump(mode="json"),
                generation=brief.generation,
                output=None,
                next_refresh_at=brief.next_refresh_at,
                updated_at=brief.updated_at,
            )
            .returning(BriefRow.brief_id)
        )
        return result.scalar_one_or_none() is not None

    async def delete(self, tenant_id: str, brief_id: str) -> None:
        await self.session.execute(
            delete(BriefRow).where(BriefRow.tenant_id == tenant_id, BriefRow.brief_id == brief_id)
        )

    async def claim_due(self, now: datetime, *, limit: int) -> list[StoredBrief]:
        rows = (
            await self.session.scalars(
                select(BriefRow)
                .where(BriefRow.next_refresh_at <= now)
                .order_by(BriefRow.next_refresh_at, BriefRow.brief_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        ).all()
        found = [_brief(row) for row in rows]
        for row in rows:
            row.next_refresh_at = now + timedelta(minutes=5)
        return found

    async def list_owned(
        self,
        tenant_id: str,
        scope_key: str,
        *,
        after: str,
        limit: int,
    ) -> list[BriefInfo]:
        # Listing definitions never fetches the potentially large stored source/output JSON.
        rows = await self.session.execute(
            select(BriefRow.brief_id, BriefRow.spec)
            .where(
                BriefRow.tenant_id == tenant_id,
                BriefRow.scope_key == scope_key,
                BriefRow.brief_id > after,
            )
            .order_by(BriefRow.brief_id)
            .limit(limit)
        )
        return [BriefInfo(brief_id=bid, spec=spec) for bid, spec in rows]
