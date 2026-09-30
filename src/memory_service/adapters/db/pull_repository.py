"""PostgreSQL persistence of agent-tool pulls and prefetch counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import Float, Text, cast, select, update
from sqlalchemy.dialects.postgresql import JSONB, array, insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import AgentPullRow, PrefetchStatRow
from memory_service.domain.pulls import AgentPull


def _pull(row: AgentPullRow) -> AgentPull:
    return AgentPull(
        pull_id=row.pull_id,
        tenant_id=row.tenant_id,
        scope_key=row.scope_key,
        run_id=row.run_id,
        pattern=row.pattern,
        tool=row.tool,
        args=dict(row.args or {}),
        result_ids=list(row.result_ids or []),
        used_ids=list(row.used_ids or []),
        created_at=row.created_at,
    )


class SqlPullRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, pull: AgentPull) -> None:
        self.s.add(AgentPullRow(**pull.model_dump()))
        await self.s.flush()

    async def mark_used(self, tenant_id: str, run_id: str, item_ids: Sequence[str]) -> int:
        if not item_ids:
            return 0
        ids = sorted(set(item_ids))
        result = await self.s.execute(
            update(AgentPullRow)
            .where(
                AgentPullRow.tenant_id == tenant_id,
                AgentPullRow.run_id == run_id,
                AgentPullRow.learned_at.is_(None),
                AgentPullRow.result_ids.op("?|")(array(ids, type_=Text)),
            )
            .values(used_ids=AgentPullRow.used_ids.op("||")(cast(ids, JSONB)))
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def settled(self, *, before: datetime, limit: int) -> list[AgentPull]:
        rows = (
            await self.s.execute(
                select(AgentPullRow)
                .where(AgentPullRow.learned_at.is_(None), AgentPullRow.created_at < before)
                .order_by(AgentPullRow.created_at)
                .limit(limit)
            )
        ).scalars()
        return [_pull(r) for r in rows]

    async def fold(self, pulls: Sequence[AgentPull]) -> None:
        counts: Counter[tuple[str, str, str, str]] = Counter()
        uses: Counter[tuple[str, str, str, str]] = Counter()
        for pull in pulls:
            used = set(pull.used_ids)
            for item in dict.fromkeys(pull.result_ids):
                key = (pull.tenant_id, pull.scope_key, pull.pattern, item)
                counts[key] += 1
                uses[key] += int(item in used)
        now = datetime.now(UTC)
        for (tenant, scope, pattern, item), pulled in sorted(counts.items()):
            used_count = uses[(tenant, scope, pattern, item)]
            stmt = insert(PrefetchStatRow).values(
                tenant_id=tenant,
                scope_key=scope,
                pattern=pattern,
                item_id=item,
                pulls=pulled,
                uses=used_count,
                updated_at=now,
            )
            await self.s.execute(
                stmt.on_conflict_do_update(
                    index_elements=[
                        PrefetchStatRow.tenant_id,
                        PrefetchStatRow.scope_key,
                        PrefetchStatRow.pattern,
                        PrefetchStatRow.item_id,
                    ],
                    set_={
                        "pulls": PrefetchStatRow.pulls + pulled,
                        "uses": PrefetchStatRow.uses + used_count,
                        "updated_at": now,
                    },
                )
            )
        for tenant in {p.tenant_id for p in pulls}:
            await self.s.execute(
                update(AgentPullRow)
                .where(
                    AgentPullRow.tenant_id == tenant,
                    AgentPullRow.pull_id.in_([p.pull_id for p in pulls if p.tenant_id == tenant]),
                )
                .values(learned_at=now)
            )

    async def prefetch(
        self,
        tenant_id: str,
        scope_key: str,
        pattern: str,
        *,
        min_pulls: int,
        min_rate: float,
        limit: int,
    ) -> list[str]:
        row = PrefetchStatRow
        rows = await self.s.execute(
            select(row.item_id)
            .where(
                row.tenant_id == tenant_id,
                row.scope_key == scope_key,
                row.pattern == pattern,
                row.pulls >= min_pulls,
                cast(row.uses, Float) >= cast(row.pulls, Float) * min_rate,
            )
            .order_by(row.uses.desc(), row.item_id)
            .limit(limit)
        )
        return [r[0] for r in rows]
