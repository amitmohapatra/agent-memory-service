"""PostgreSQL persistence for tool memory (TOOL_MEMORY.md §30.0-§30.1).

Registration is an idempotent upsert on (tenant, name, schema_hash): the same tool used
from two adapters is one row, a changed schema is a new version, and a policy is only ever
widened by an explicit call — never by a per-call declaration.

Recording is idempotent on (run, step, tool, args_hash): a retried step re-reads its own
row instead of writing a second one, so replayed graphs never inflate the statistics the
suggestions are built from.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Text, func, select, update
from sqlalchemy import false as sa_false
from sqlalchemy.dialects.postgresql import array, insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import RunOutcomeRow, ToolInvocationRow, ToolRow
from memory_service.domain.tools import (
    RunOutcome,
    SubCall,
    ToolDescriptor,
    ToolInvocation,
    ToolOutcomeStats,
    ToolPolicy,
)


def _to_descriptor(row: ToolRow) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id=row.tool_id,
        tenant_id=row.tenant_id,
        workspace_id=row.workspace_id,
        name=row.name,
        version=row.version,
        description=row.description,
        input_schema=row.input_schema,
        output_schema=row.output_schema,
        tags=list(row.tags or []),
        source=row.source,  # type: ignore[arg-type]
        server=row.server,
        policy=ToolPolicy(**(row.policy or {})),
        schema_hash=row.schema_hash,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_invocation(row: ToolInvocationRow) -> ToolInvocation:
    return ToolInvocation(
        invocation_id=row.invocation_id,
        tenant_id=row.tenant_id,
        tool_id=row.tool_id,
        tool_name=row.tool_name,
        tool_version=row.tool_version,
        run_id=row.run_id,
        thread_id=row.thread_id,
        turn_id=row.turn_id,
        workspace_id=row.workspace_id,
        user_id=row.user_id,
        agent_id=row.agent_id,
        principal_id=row.principal_id,
        step=row.step,
        args_redacted=dict(row.args_redacted or {}),
        args_hash=row.args_hash,
        output_summary=row.output_summary,
        output_digest=row.output_digest,
        output_blob_ref=row.output_blob_ref,
        output_fields=dict(row.output_fields or {}),
        status=row.status,  # type: ignore[arg-type]
        error_class=row.error_class,
        latency_ms=row.latency_ms,
        cost=row.cost,
        task=row.task,
        task_pattern=row.task_pattern,
        sub_calls=[SubCall.model_validate(c) for c in (row.sub_calls or [])],
        visibility_keys=list(row.visibility_keys or []),
        occurred_at=row.occurred_at,
    )


class SqlToolRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    # ------------------------------------------------------------------ registry
    async def register(self, descriptor: ToolDescriptor, *, widen_policy: bool) -> ToolDescriptor:
        """Upsert by (tenant, name, schema_hash). ``widen_policy`` is the admin-scope flag: without
        it an existing row keeps its stored policy, so an agent's per-call declaration can never
        make a tool cacheable or mark it side-effect free."""
        desc = descriptor.with_schema_hash() if not descriptor.schema_hash else descriptor
        existing = await self.by_name(desc.tenant_id, desc.name, schema_hash=desc.schema_hash)
        if existing is not None:
            policy = desc.policy if widen_policy else existing.policy
            await self.s.execute(
                update(ToolRow)
                .where(ToolRow.tool_id == existing.tool_id)
                .values(
                    description=desc.description or existing.description,
                    tags=list(desc.tags or existing.tags),
                    policy=policy.model_dump(),
                    server=desc.server or existing.server,
                    updated_at=datetime.now(UTC),
                )
            )
            return existing.model_copy(update={"policy": policy})
        version = await self._next_version(desc.tenant_id, desc.name)
        row = ToolRow(
            tool_id=desc.tool_id,
            tenant_id=desc.tenant_id,
            workspace_id=desc.workspace_id,
            name=desc.name,
            version=version,
            description=desc.description,
            input_schema=desc.input_schema,
            output_schema=desc.output_schema,
            tags=list(desc.tags),
            source=desc.source,
            server=desc.server,
            policy=desc.policy.model_dump(),
            schema_hash=desc.schema_hash,
        )
        self.s.add(row)
        await self.s.flush()
        return desc.model_copy(update={"version": version})

    async def _next_version(self, tenant_id: str, name: str) -> int:
        current = await self.s.scalar(
            select(func.max(ToolRow.version)).where(
                ToolRow.tenant_id == tenant_id, ToolRow.name == name
            )
        )
        return int(current or 0) + 1

    async def by_name(
        self, tenant_id: str, name: str, *, schema_hash: str | None = None
    ) -> ToolDescriptor | None:
        stmt = select(ToolRow).where(ToolRow.tenant_id == tenant_id, ToolRow.name == name)
        if schema_hash is not None:
            stmt = stmt.where(ToolRow.schema_hash == schema_hash)
        stmt = stmt.order_by(ToolRow.version.desc()).limit(1)
        row = (await self.s.execute(stmt)).scalar_one_or_none()
        return _to_descriptor(row) if row is not None else None

    async def get(self, tenant_id: str, tool_id: str) -> ToolDescriptor | None:
        row = (
            await self.s.execute(
                select(ToolRow).where(ToolRow.tenant_id == tenant_id, ToolRow.tool_id == tool_id)
            )
        ).scalar_one_or_none()
        return _to_descriptor(row) if row is not None else None

    async def stats(
        self, tenant_id: str, *, names: Sequence[str] | None = None
    ) -> list[ToolOutcomeStats]:
        stmt = (
            select(
                ToolInvocationRow.tool_name,
                func.count().label("n"),
                func.count().filter(ToolInvocationRow.status == "ok").label("ok"),
                func.percentile_cont(0.5)
                .within_group(ToolInvocationRow.latency_ms)
                .label("median_latency"),
                func.coalesce(func.sum(ToolInvocationRow.cost), 0.0).label("cost"),
                func.max(ToolInvocationRow.occurred_at).label("last_used"),
            )
            .where(ToolInvocationRow.tenant_id == tenant_id)
            .group_by(ToolInvocationRow.tool_name)
        )
        if names:
            stmt = stmt.where(ToolInvocationRow.tool_name.in_(list(names)))
        return [
            ToolOutcomeStats(
                tool_name=r.tool_name,
                invocations=int(r.n),
                successes=int(r.ok),
                failures=int(r.n) - int(r.ok),
                median_latency_ms=float(r.median_latency) if r.median_latency is not None else None,
                total_cost=float(r.cost or 0.0),
                last_used_at=r.last_used,
            )
            for r in (await self.s.execute(stmt)).all()
        ]

    # ------------------------------------------------------------------ invocations
    async def record(self, invocation: ToolInvocation) -> ToolInvocation:
        """Insert once. A retry of the same (run, step, tool, args) returns the stored row."""
        key = invocation.idempotency_key()
        values: dict[str, Any] = {
            "invocation_id": invocation.invocation_id,
            "tenant_id": invocation.tenant_id,
            "tool_id": invocation.tool_id,
            "tool_name": invocation.tool_name,
            "tool_version": invocation.tool_version,
            "run_id": invocation.run_id,
            "thread_id": invocation.thread_id,
            "turn_id": invocation.turn_id,
            "workspace_id": invocation.workspace_id,
            "user_id": invocation.user_id,
            "agent_id": invocation.agent_id,
            "principal_id": invocation.principal_id,
            "step": invocation.step,
            "args_redacted": invocation.args_redacted,
            "args_hash": invocation.args_hash,
            "idempotency_key": key,
            "output_summary": invocation.output_summary,
            "output_digest": invocation.output_digest,
            "output_blob_ref": invocation.output_blob_ref,
            "output_fields": invocation.output_fields,
            "status": invocation.status,
            "error_class": invocation.error_class,
            "latency_ms": invocation.latency_ms,
            "cost": invocation.cost,
            "task": invocation.task,
            "task_pattern": invocation.task_pattern,
            "sub_calls": [c.model_dump() for c in invocation.sub_calls],
            "visibility_keys": invocation.visibility_keys,
            "occurred_at": invocation.occurred_at,
        }
        stmt = (
            insert(ToolInvocationRow)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_tool_invocations_idempotent")
            .returning(ToolInvocationRow.invocation_id)
        )
        inserted = (await self.s.execute(stmt)).scalar_one_or_none()
        if inserted is not None:
            return invocation
        row = (
            await self.s.execute(
                select(ToolInvocationRow).where(
                    ToolInvocationRow.tenant_id == invocation.tenant_id,
                    ToolInvocationRow.idempotency_key == key,
                )
            )
        ).scalar_one()
        return _to_invocation(row)

    async def invocations_for_run(
        self, tenant_id: str, run_id: str, *, scope_keys: Sequence[str] | None = None
    ) -> list[ToolInvocation]:
        stmt = (
            select(ToolInvocationRow)
            .where(ToolInvocationRow.tenant_id == tenant_id, ToolInvocationRow.run_id == run_id)
            .order_by(ToolInvocationRow.step, ToolInvocationRow.occurred_at)
        )
        stmt = _visible(stmt, scope_keys)
        return [_to_invocation(r) for r in (await self.s.execute(stmt)).scalars()]

    async def recent(
        self,
        tenant_id: str,
        *,
        task_pattern: str | None = None,
        tool_name: str | None = None,
        scope_keys: Sequence[str] | None = None,
        limit: int = 200,
    ) -> list[ToolInvocation]:
        stmt = select(ToolInvocationRow).where(ToolInvocationRow.tenant_id == tenant_id)
        if task_pattern is not None:
            stmt = stmt.where(ToolInvocationRow.task_pattern == task_pattern)
        if tool_name is not None:
            stmt = stmt.where(ToolInvocationRow.tool_name == tool_name)
        stmt = _visible(stmt, scope_keys)
        stmt = stmt.order_by(ToolInvocationRow.occurred_at.desc()).limit(limit)
        return [_to_invocation(r) for r in (await self.s.execute(stmt)).scalars()]

    async def mark_indexed(self, tenant_id: str, invocation_ids: Sequence[str]) -> None:
        if not invocation_ids:
            return
        await self.s.execute(
            update(ToolInvocationRow)
            .where(
                ToolInvocationRow.tenant_id == tenant_id,
                ToolInvocationRow.invocation_id.in_(list(invocation_ids)),
            )
            .values(indexed_at=datetime.now(UTC))
        )

    # ------------------------------------------------------------------ outcomes
    async def set_outcome(self, outcome: RunOutcome) -> None:
        stmt = (
            insert(RunOutcomeRow)
            .values(
                tenant_id=outcome.tenant_id,
                run_id=outcome.run_id,
                success=outcome.success,
                note=outcome.note,
                source=outcome.source,
                recorded_at=outcome.recorded_at,
            )
            .on_conflict_do_update(
                index_elements=[RunOutcomeRow.tenant_id, RunOutcomeRow.run_id],
                set_={
                    "success": outcome.success,
                    "note": outcome.note,
                    "source": outcome.source,
                    "recorded_at": outcome.recorded_at,
                },
            )
        )
        await self.s.execute(stmt)

    async def outcome(self, tenant_id: str, run_id: str) -> RunOutcome | None:
        row = (
            await self.s.execute(
                select(RunOutcomeRow).where(
                    RunOutcomeRow.tenant_id == tenant_id, RunOutcomeRow.run_id == run_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return RunOutcome(
            tenant_id=row.tenant_id,
            run_id=row.run_id,
            success=row.success,
            note=row.note,
            source=row.source,  # type: ignore[arg-type]
            recorded_at=row.recorded_at,
        )


def _visible(stmt: Any, scope_keys: Sequence[str] | None) -> Any:
    """Store-side visibility: ``visibility_keys ?| ARRAY[...]``, the same any-of match the graph
    store and memories use. Applied before any ranking, never after."""
    if scope_keys is None:
        return stmt
    if not scope_keys:
        return stmt.where(sa_false())
    return stmt.where(
        ToolInvocationRow.visibility_keys.op("?|")(array([str(k) for k in scope_keys], type_=Text))
    )
