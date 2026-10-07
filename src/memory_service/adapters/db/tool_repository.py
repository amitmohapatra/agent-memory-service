"""PostgreSQL persistence for tool memory: the catalog, invocation records, run outcomes,
running statistics, approval patterns and stored procedures.

The catalog is one row per (tenant, workspace, name), ``workspace_id`` "" for the tenant.
Recording is idempotent on (run, step, tool, args_hash): a retried step re-reads its own row
instead of writing a second one, so replayed graphs never inflate the statistics. Statistics
and approval patterns are counters, one upsert per event.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Text, and_, func, literal, or_, select, tuple_, update
from sqlalchemy import false as sa_false
from sqlalchemy.dialects.postgresql import array, insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import (
    ApprovalPatternRow,
    ProcedureRow,
    RunOutcomeRow,
    ToolInvocationRow,
    ToolRow,
    ToolStatsRow,
)
from memory_service.domain.ids import new_id
from memory_service.domain.learning import ApprovalCounts
from memory_service.domain.tools import (
    RunOutcome,
    StoredProcedure,
    SubCall,
    ToolAnnotations,
    ToolDescriptor,
    ToolInvocation,
    ToolStats,
    agent_audience,
)

_TENANT_WIDE = ""
_VERDICT_COLUMNS = ("approvals", "rejections", "edits")


def _to_descriptor(row: ToolRow) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id=row.tool_id,
        tenant_id=row.tenant_id,
        workspace_id=row.workspace_id or None,
        name=row.name,
        version=row.version,
        description=row.description,
        input_schema=row.input_schema,
        required=list(row.required or []),
        argument_entity_types=dict(row.argument_entity_types or {}),
        side_effects=row.side_effects,  # type: ignore[arg-type]
        source=row.source,
        server=row.server,
        examples=list(row.examples or []),
        redact=list(row.redact or []),
        annotations=ToolAnnotations.model_validate(row.annotations or {}),
        approve_when=row.approve_when,
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


def _to_outcome(row: RunOutcomeRow) -> RunOutcome:
    return RunOutcome(
        tenant_id=row.tenant_id,
        run_id=row.run_id,
        success=row.success,
        note=row.note,
        source=row.source,  # type: ignore[arg-type]
        recorded_at=row.recorded_at,
    )


def _to_stats(row: ToolStatsRow) -> ToolStats:
    return ToolStats(
        tool_name=row.tool_name,
        calls=row.calls,
        successes=row.successes,
        latency_ms_total=row.latency_ms_total,
        latency_calls=row.latency_calls,
        approvals=row.approvals,
        rejections=row.rejections,
        edits=row.edits,
        last_used_at=row.last_used_at,
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


class SqlToolRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    # ------------------------------------------------------------------ catalog
    async def upsert(
        self, descriptor: ToolDescriptor, *, fields: frozenset[str] | None = None
    ) -> tuple[ToolDescriptor, bool]:
        """Insert the entry, or update the stored one with the ``fields`` given (every catalog
        field when None): a publisher that does not send ``side_effects`` or ``approve_when``
        leaves what an administrator set alone."""
        desc = descriptor.with_schema_hash()
        workspace = desc.workspace_id or _TENANT_WIDE
        row = (
            await self.s.execute(
                select(ToolRow).where(
                    ToolRow.tenant_id == desc.tenant_id,
                    ToolRow.workspace_id == workspace,
                    ToolRow.name == desc.name,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            desc = desc.model_copy(update={"tool_id": new_id("tool")})
            self.s.add(_new_row(desc, workspace))
            await self.s.flush()
            return desc, True
        stored = _to_descriptor(row)
        if fields is not None:
            kept = {name: getattr(stored, name) for name in set(desc.catalog_fields()) - fields}
            desc = desc.model_copy(update=kept).with_schema_hash()
        if stored.catalog_fields() == desc.catalog_fields():
            return stored, False
        version = row.version + (1 if row.schema_hash != desc.schema_hash else 0)
        updated = desc.model_copy(
            update={"tool_id": row.tool_id, "version": version, "created_at": stored.created_at}
        )
        await self.s.execute(
            update(ToolRow)
            .where(ToolRow.tool_id == row.tool_id)
            .values(
                **_catalog_values(updated),
                schema_hash=updated.schema_hash,
                version=version,
                updated_at=datetime.now(UTC),
            )
        )
        return updated, True

    async def ensure(self, tenant_id: str, name: str) -> ToolDescriptor:
        found = await self.by_name(tenant_id, name)
        if found is not None:
            return found
        desc = ToolDescriptor(tenant_id=tenant_id, name=name).with_schema_hash()
        await self.s.execute(
            insert(ToolRow)
            .values(**_row_values(desc, _TENANT_WIDE))
            .on_conflict_do_nothing(constraint="uq_tools_catalog_name")
        )
        stored = await self.by_name(tenant_id, name)
        assert stored is not None
        return stored

    async def by_name(
        self, tenant_id: str, name: str, *, workspace_id: str | None = None
    ) -> ToolDescriptor | None:
        row = (
            await self.s.execute(
                select(ToolRow)
                .where(
                    ToolRow.tenant_id == tenant_id,
                    ToolRow.name == name,
                    ToolRow.workspace_id.in_(_workspaces(workspace_id)),
                )
                .order_by(ToolRow.workspace_id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return _to_descriptor(row) if row is not None else None

    async def catalog(
        self,
        tenant_id: str,
        *,
        workspace_id: str | None,
        names: Sequence[str] | None = None,
        limit: int,
        after: str = "",
    ) -> list[ToolDescriptor]:
        stmt = select(ToolRow).where(
            ToolRow.tenant_id == tenant_id,
            ToolRow.workspace_id.in_(_workspaces(workspace_id)),
            ToolRow.name > after,
        )
        if names is not None:
            stmt = stmt.where(ToolRow.name.in_(list(names)))
        rows = (
            await self.s.execute(
                stmt.order_by(ToolRow.name, ToolRow.workspace_id.desc()).limit(limit * 2)
            )
        ).scalars()
        out: dict[str, ToolDescriptor] = {}
        for row in rows:
            out.setdefault(row.name, _to_descriptor(row))
        return list(out.values())[:limit]

    async def catalog_by_ids(self, tenant_id: str, tool_ids: Sequence[str]) -> list[ToolDescriptor]:
        if not tool_ids:
            return []
        rows = (
            await self.s.execute(
                select(ToolRow).where(
                    ToolRow.tenant_id == tenant_id, ToolRow.tool_id.in_(list(tool_ids))
                )
            )
        ).scalars()
        return [_to_descriptor(r) for r in rows]

    # ------------------------------------------------------------------ calls
    async def record(self, invocation: ToolInvocation) -> tuple[ToolInvocation, bool]:
        key = invocation.idempotency_key()
        values: dict[str, Any] = {
            **invocation.model_dump(exclude={"sub_calls"}),
            "sub_calls": [c.model_dump() for c in invocation.sub_calls],
            "idempotency_key": key,
        }
        stmt = (
            insert(ToolInvocationRow)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_tool_invocations_idempotent")
            .returning(ToolInvocationRow.invocation_id)
        )
        if (await self.s.execute(stmt)).scalar_one_or_none() is not None:
            return invocation, True
        row = (
            await self.s.execute(
                select(ToolInvocationRow).where(
                    ToolInvocationRow.tenant_id == invocation.tenant_id,
                    ToolInvocationRow.idempotency_key == key,
                )
            )
        ).scalar_one()
        return _to_invocation(row), False

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

    async def for_pattern(
        self, tenant_id: str, audience: str, pattern: str, *, limit: int
    ) -> list[ToolInvocation]:
        stmt = (
            select(ToolInvocationRow)
            .where(
                ToolInvocationRow.tenant_id == tenant_id,
                ToolInvocationRow.task_pattern == pattern,
                ToolInvocationRow.visibility_keys[0].astext == audience,
            )
            .order_by(ToolInvocationRow.occurred_at.desc())
            .limit(limit)
        )
        return [_to_invocation(r) for r in (await self.s.execute(stmt)).scalars()]

    async def for_agent_pattern(
        self, tenant_id: str, agent_id: str, pattern: str, *, limit: int
    ) -> list[ToolInvocation]:
        """One agent's newest calls of a task pattern, for every user it ran for (the
        caller keeps the agent's own records: ``learning.audience_of``)."""
        stmt = (
            select(ToolInvocationRow)
            .where(
                ToolInvocationRow.tenant_id == tenant_id,
                ToolInvocationRow.task_pattern == pattern,
                ToolInvocationRow.agent_id == agent_id,
            )
            .order_by(ToolInvocationRow.occurred_at.desc())
            .limit(limit)
        )
        return [_to_invocation(r) for r in (await self.s.execute(stmt)).scalars()]

    async def unlearned(self, *, tenant_id: str | None, limit: int) -> list[ToolInvocation]:
        stmt = select(ToolInvocationRow).where(ToolInvocationRow.learned_at.is_(None))
        if tenant_id is not None:
            stmt = stmt.where(ToolInvocationRow.tenant_id == tenant_id)
        stmt = stmt.order_by(ToolInvocationRow.occurred_at).limit(limit)
        return [_to_invocation(r) for r in (await self.s.execute(stmt)).scalars()]

    async def mark_learned(self, tenant_id: str, invocation_ids: Sequence[str]) -> None:
        if not invocation_ids:
            return
        await self.s.execute(
            update(ToolInvocationRow)
            .where(
                ToolInvocationRow.tenant_id == tenant_id,
                ToolInvocationRow.invocation_id.in_(list(invocation_ids)),
            )
            .values(learned_at=datetime.now(UTC))
        )

    # ------------------------------------------------------------------ outcomes
    async def set_outcome(self, outcome: RunOutcome) -> None:
        values = outcome.model_dump()
        stmt = insert(RunOutcomeRow).values(**values)
        await self.s.execute(
            stmt.on_conflict_do_update(
                index_elements=[RunOutcomeRow.tenant_id, RunOutcomeRow.run_id],
                set_={k: stmt.excluded[k] for k in ("success", "note", "source", "recorded_at")},
            )
        )
        await self.s.execute(
            update(ToolInvocationRow)
            .where(
                ToolInvocationRow.tenant_id == outcome.tenant_id,
                ToolInvocationRow.run_id == outcome.run_id,
            )
            .values(learned_at=None)
        )

    async def outcome(self, tenant_id: str, run_id: str) -> RunOutcome | None:
        return (await self.outcomes(tenant_id, [run_id])).get(run_id)

    async def outcomes(self, tenant_id: str, run_ids: Sequence[str]) -> dict[str, RunOutcome]:
        if not run_ids:
            return {}
        rows = (
            await self.s.execute(
                select(RunOutcomeRow).where(
                    RunOutcomeRow.tenant_id == tenant_id, RunOutcomeRow.run_id.in_(list(run_ids))
                )
            )
        ).scalars()
        return {row.run_id: _to_outcome(row) for row in rows}

    # ------------------------------------------------------------------ statistics
    async def count_call(
        self, tenant_id: str, tool_name: str, *, ok: bool, latency_ms: float | None, at: datetime
    ) -> None:
        timed = latency_ms is not None
        stmt = insert(ToolStatsRow).values(
            tenant_id=tenant_id,
            tool_name=tool_name,
            calls=1,
            successes=int(ok),
            latency_ms_total=latency_ms or 0.0,
            latency_calls=int(timed),
            last_used_at=at,
        )
        await self.s.execute(
            stmt.on_conflict_do_update(
                index_elements=[ToolStatsRow.tenant_id, ToolStatsRow.tool_name],
                set_={
                    "calls": ToolStatsRow.calls + 1,
                    "successes": ToolStatsRow.successes + int(ok),
                    "latency_ms_total": ToolStatsRow.latency_ms_total + (latency_ms or 0.0),
                    "latency_calls": ToolStatsRow.latency_calls + int(timed),
                    "last_used_at": func.greatest(ToolStatsRow.last_used_at, at),
                },
            )
        )

    async def count_verdict(self, tenant_id: str, tool_name: str, verdict: str) -> None:
        column = _verdict_column(verdict)
        stmt = insert(ToolStatsRow).values(tenant_id=tenant_id, tool_name=tool_name, **{column: 1})
        await self.s.execute(
            stmt.on_conflict_do_update(
                index_elements=[ToolStatsRow.tenant_id, ToolStatsRow.tool_name],
                set_={column: getattr(ToolStatsRow, column) + 1},
            )
        )

    async def stats(self, tenant_id: str, names: Sequence[str]) -> dict[str, ToolStats]:
        if not names:
            return {}
        rows = (
            await self.s.execute(
                select(ToolStatsRow).where(
                    ToolStatsRow.tenant_id == tenant_id, ToolStatsRow.tool_name.in_(list(names))
                )
            )
        ).scalars()
        return {row.tool_name: _to_stats(row) for row in rows}

    # ------------------------------------------------------------------ approvals
    async def count_approval(
        self, tenant_id: str, agent_id: str, tool_name: str, arg_shape: str, verdict: str
    ) -> None:
        column = _verdict_column(verdict)
        stmt = insert(ApprovalPatternRow).values(
            tenant_id=tenant_id,
            agent_id=agent_id,
            tool_name=tool_name,
            arg_shape=arg_shape,
            updated_at=datetime.now(UTC),
            **{column: 1},
        )
        await self.s.execute(
            stmt.on_conflict_do_update(
                index_elements=[
                    ApprovalPatternRow.tenant_id,
                    ApprovalPatternRow.agent_id,
                    ApprovalPatternRow.tool_name,
                    ApprovalPatternRow.arg_shape,
                ],
                set_={
                    column: getattr(ApprovalPatternRow, column) + 1,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
        )

    async def approval_pattern(
        self, tenant_id: str, agent_id: str, tool_name: str, arg_shape: str
    ) -> ApprovalCounts | None:
        row = await self.s.get(ApprovalPatternRow, (tenant_id, agent_id, tool_name, arg_shape))
        return _to_counts(row) if row is not None else None

    async def approval_patterns(
        self,
        tenant_id: str,
        agent_id: str,
        *,
        tool_name: str | None,
        min_support: int,
        limit: int,
        after: tuple[int, str, str] | None = None,
    ) -> list[ApprovalCounts]:
        row = ApprovalPatternRow
        support = row.approvals + row.rejections + row.edits
        stmt = select(row).where(
            row.tenant_id == tenant_id, row.agent_id == agent_id, support >= min_support
        )
        if tool_name is not None:
            stmt = stmt.where(row.tool_name == tool_name)
        if after is not None:
            last_support, last_tool, last_shape = after
            stmt = stmt.where(
                or_(
                    support < last_support,
                    and_(
                        support == last_support,
                        tuple_(row.tool_name, row.arg_shape)
                        > tuple_(literal(last_tool), literal(last_shape)),
                    ),
                )
            )
        stmt = stmt.order_by(support.desc(), row.tool_name, row.arg_shape).limit(limit)
        return [_to_counts(r) for r in (await self.s.execute(stmt)).scalars()]


def _to_counts(row: ApprovalPatternRow) -> ApprovalCounts:
    return ApprovalCounts(
        agent_id=row.agent_id,
        tool=row.tool_name,
        arg_shape=row.arg_shape,
        approvals=row.approvals,
        rejections=row.rejections,
        edits=row.edits,
    )


def _workspaces(workspace_id: str | None) -> list[str]:
    """The catalog rows a workspace reads: its own and the tenant's."""
    return list(dict.fromkeys((_TENANT_WIDE, workspace_id or _TENANT_WIDE)))


def _verdict_column(verdict: str) -> str:
    if verdict not in _VERDICT_COLUMNS:
        raise ValueError(f"not a verdict count: {verdict}")
    return verdict


def _row_values(desc: ToolDescriptor, workspace: str) -> dict[str, Any]:
    return {
        "tool_id": desc.tool_id,
        "tenant_id": desc.tenant_id,
        "workspace_id": workspace,
        "name": desc.name,
        "version": desc.version,
        "schema_hash": desc.schema_hash,
        **_catalog_values(desc),
    }


def _catalog_values(desc: ToolDescriptor) -> dict[str, Any]:
    """The catalog fields as the row stores them (annotations under their MCP names)."""
    return {
        **desc.catalog_fields(),
        "annotations": desc.annotations.model_dump(by_alias=True, exclude_none=True),
    }


def _new_row(desc: ToolDescriptor, workspace: str) -> ToolRow:
    return ToolRow(**_row_values(desc, workspace))


def _to_procedure(row: ProcedureRow) -> StoredProcedure:
    return StoredProcedure(
        procedure_id=row.procedure_id,
        tenant_id=row.tenant_id,
        scope_key=row.scope_key,
        pattern=row.pattern,
        title=row.title,
        strategy=row.strategy,
        steps=list(row.steps or []),
        bindings=list(row.bindings or []),
        success_rate=row.success_rate,
        support=row.support,
        status=row.status,  # type: ignore[arg-type]
        steps_hash=row.steps_hash,
        distilled=row.distilled,
        owner_principal=row.owner_principal,
        workspace_id=row.workspace_id,
        updated_at=row.updated_at,
        agent_id=row.agent_id,
        users=row.users,
        sole_user=row.sole_user,
    )


class SqlProcedureRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def get(self, tenant_id: str, procedure_id: str) -> StoredProcedure | None:
        row = await self.s.get(ProcedureRow, (tenant_id, procedure_id))
        return _to_procedure(row) if row is not None else None

    async def by_pattern(
        self, tenant_id: str, scope_key: str, pattern: str
    ) -> StoredProcedure | None:
        row = (
            await self.s.execute(
                select(ProcedureRow).where(
                    ProcedureRow.tenant_id == tenant_id,
                    ProcedureRow.scope_key == scope_key,
                    ProcedureRow.pattern == pattern,
                )
            )
        ).scalar_one_or_none()
        return _to_procedure(row) if row is not None else None

    async def save(self, procedure: StoredProcedure) -> None:
        values = procedure.model_dump()
        stmt = insert(ProcedureRow).values(**values)
        fixed = ("tenant_id", "procedure_id")
        mutable = {k: stmt.excluded[k] for k in values if k not in fixed}
        await self.s.execute(
            stmt.on_conflict_do_update(constraint="uq_procedures_pattern", set_=mutable)
        )

    async def visible(
        self,
        tenant_id: str,
        scope_keys: Sequence[str],
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        limit: int,
    ) -> list[StoredProcedure]:
        """Active procedures the reader may read: those of its audience keys, and the agent's
        own learned procedures once two users produced them, or the reader produced them."""
        readable = ProcedureRow.scope_key.in_(list(scope_keys)) if scope_keys else sa_false()
        if agent_id is not None:
            readable = or_(
                readable,
                and_(
                    ProcedureRow.scope_key == agent_audience(tenant_id, agent_id),
                    or_(
                        ProcedureRow.users >= 2,
                        ProcedureRow.sole_user.is_(None),
                        ProcedureRow.sole_user == user_id,
                    ),
                ),
            )
        rows = (
            await self.s.execute(
                select(ProcedureRow)
                .where(
                    ProcedureRow.tenant_id == tenant_id,
                    ProcedureRow.status == "active",
                    readable,
                )
                .order_by(ProcedureRow.updated_at.desc(), ProcedureRow.procedure_id)
                .limit(limit)
            )
        ).scalars()
        return [_to_procedure(r) for r in rows]

    async def learned(
        self, tenant_id: str, *, agent_id: str | None = None, limit: int
    ) -> list[StoredProcedure]:
        """The tenant's learned procedures that are or were offered (active, retired,
        rejected), best supported first; one agent's when ``agent_id`` is given."""
        stmt = select(ProcedureRow).where(
            ProcedureRow.tenant_id == tenant_id, ProcedureRow.status != "candidate"
        )
        if agent_id is not None:
            stmt = stmt.where(ProcedureRow.agent_id == agent_id)
        rows = (
            await self.s.execute(
                stmt.order_by(ProcedureRow.support.desc(), ProcedureRow.procedure_id).limit(limit)
            )
        ).scalars()
        return [_to_procedure(r) for r in rows]

    async def reject(self, tenant_id: str, procedure_id: str) -> bool:
        result = await self.s.execute(
            update(ProcedureRow)
            .where(ProcedureRow.tenant_id == tenant_id, ProcedureRow.procedure_id == procedure_id)
            .values(status="rejected", updated_at=datetime.now(UTC))
        )
        return bool(getattr(result, "rowcount", 0))
