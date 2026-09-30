"""The tenant model policy, and daily usage per tenant and use."""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import LLMPolicyRow, LLMUsageDailyRow
from memory_service.ports.llm import StoredPolicy, UsageDay


def _policy(row: LLMPolicyRow) -> StoredPolicy:
    return StoredPolicy(
        tenant_id=row.tenant_id,
        uses=frozenset(row.uses),
        read_assist=row.read_assist,
        models=dict(row.models or {}),
        revision=row.revision,
        updated_at=row.updated_at,
    )


class SqlLLMPolicyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, tenant_id: str) -> StoredPolicy | None:
        row = await self.session.get(LLMPolicyRow, tenant_id)
        return _policy(row) if row is not None else None

    async def put(
        self,
        tenant_id: str,
        *,
        uses: Sequence[str],
        read_assist: bool,
        models: Mapping[str, str],
    ) -> StoredPolicy:
        statement = insert(LLMPolicyRow).values(
            tenant_id=tenant_id,
            uses=sorted(set(uses)),
            read_assist=read_assist,
            models=dict(models),
            revision=1,
            updated_at=datetime.now(UTC),
        )
        statement = statement.on_conflict_do_update(
            index_elements=["tenant_id"],
            set_={
                "uses": statement.excluded.uses,
                "read_assist": statement.excluded.read_assist,
                "models": statement.excluded.models,
                "revision": LLMPolicyRow.revision + 1,
                "updated_at": statement.excluded.updated_at,
            },
        ).returning(LLMPolicyRow)
        row = (
            await self.session.scalars(statement, execution_options={"populate_existing": True})
        ).one()
        return _policy(row)


class SqlLLMUsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, tenant_id: str, use: str, day: date, tokens: int) -> None:
        statement = insert(LLMUsageDailyRow).values(
            tenant_id=tenant_id, use=use, day=day, tokens=tokens, calls=1
        )
        await self.session.execute(
            statement.on_conflict_do_update(
                index_elements=["tenant_id", "day", "use"],
                set_={
                    "tokens": LLMUsageDailyRow.tokens + statement.excluded.tokens,
                    "calls": LLMUsageDailyRow.calls + 1,
                },
            )
        )

    async def between(self, tenant_id: str, since: date, until: date) -> list[UsageDay]:
        rows = (
            await self.session.scalars(
                select(LLMUsageDailyRow)
                .where(
                    LLMUsageDailyRow.tenant_id == tenant_id,
                    LLMUsageDailyRow.day >= since,
                    LLMUsageDailyRow.day <= until,
                )
                .order_by(LLMUsageDailyRow.day, LLMUsageDailyRow.use)
            )
        ).all()
        return [UsageDay(r.day, r.use, r.tokens, r.calls) for r in rows]
