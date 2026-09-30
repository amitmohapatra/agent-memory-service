"""Model-use policies per key-hierarchy level, and daily usage per tenant and use."""

from collections.abc import Sequence
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import LLMPolicyRow, LLMUsageDailyRow
from memory_service.ports.credentials import ModelIdentity
from memory_service.ports.llm import StoredPolicy, UsageDay


def _policy(row: LLMPolicyRow) -> StoredPolicy:
    return StoredPolicy(
        ModelIdentity(row.tenant_id, row.principal_id),
        frozenset(row.uses),
        row.read_assist,
        row.revision,
        row.updated_at,
    )


class SqlLLMPolicyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def first(self, levels: Sequence[ModelIdentity]) -> StoredPolicy | None:
        if not levels:
            return None
        order = [level.principal_id for level in levels]
        rows = (
            await self.session.scalars(
                select(LLMPolicyRow).where(
                    LLMPolicyRow.tenant_id == levels[0].tenant_id,
                    LLMPolicyRow.principal_id.in_(order),
                )
            )
        ).all()
        by_level = {row.principal_id: row for row in rows}
        found = next((by_level[p] for p in order if p in by_level), None)
        return _policy(found) if found is not None else None

    async def put(
        self, identity: ModelIdentity, *, uses: Sequence[str], read_assist: bool
    ) -> StoredPolicy:
        statement = insert(LLMPolicyRow).values(
            tenant_id=identity.tenant_id,
            principal_id=identity.principal_id,
            uses=sorted(set(uses)),
            read_assist=read_assist,
            revision=1,
            updated_at=datetime.now(UTC),
        )
        statement = statement.on_conflict_do_update(
            index_elements=["tenant_id", "principal_id"],
            set_={
                "uses": statement.excluded.uses,
                "read_assist": statement.excluded.read_assist,
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
