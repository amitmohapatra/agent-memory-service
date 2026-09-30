"""Tool memory service (TOOL_MEMORY.md): the catalog, call records and run outcomes.

The service never runs a tool. The catalog says what each tool is and does (side effects,
the entity types its arguments name); ``record`` stores what an agent called and counts it;
``set_outcome`` labels a run and queues its calls to be learned again. Everything learned from
the records - procedures, graph edges - is the learning job's (``modules.tools.learning``);
the advice read back is ``modules.tools.hints``.

Nothing is read that the caller's visibility keys do not cover, and a call record carries the
same audience keys a memory written by the same caller would.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import (
    RunOutcome,
    SubCall,
    ToolDescriptor,
    ToolInvocation,
    ToolStats,
    ToolStatus,
    stable_hash,
)
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.memory.pipeline import keys_for, scope_for
from memory_service.modules.tenancy.gate import guard_workspace_visibility
from memory_service.modules.tools.patterns import task_pattern
from memory_service.modules.tools.trajectories import flatten
from memory_service.observability.logging import get_logger
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)

TASK_TOOLS_INDEX: Final = "tools.index"
TASK_TOOLS_LEARN: Final = "tools.learn"
INLINE_OUTPUT_LIMIT: Final = 4096
SUMMARY_CHARS: Final = 600
#: Entries one catalog listing or upsert handles.
CATALOG_MAX: Final = 500


class ToolMemoryService:
    def __init__(
        self,
        authz: AuthorizationService,
        *,
        blob: Any = None,
        blob_bucket: str = "memory-tool-outputs",
    ) -> None:
        self.authz = authz
        self.blob = blob
        self.blob_bucket = blob_bucket

    # ------------------------------------------------------------------ catalog
    async def put_catalog(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, entries: Sequence[ToolDescriptor]
    ) -> list[ToolDescriptor]:
        """Upsert entries in the caller's workspace (or tenant-wide without one). Changed
        entries are re-indexed for tool search, and bundles that carried tool hints go stale."""
        stored: list[ToolDescriptor] = []
        changed: list[str] = []
        for entry in entries:
            scoped = entry.model_copy(
                update={"tenant_id": ctx.tenant_id, "workspace_id": ctx.workspace_id}
            )
            saved, was_changed = await uow.tools.upsert(scoped)
            stored.append(saved)
            if was_changed:
                changed.append(saved.tool_id)
        if changed:
            await uow.enqueue(
                JobSpec(
                    task_name=TASK_TOOLS_INDEX,
                    queue=Queue.EMBEDDING,
                    payload={"tenant_id": ctx.tenant_id, "tool_ids": changed},
                    tenant_id=ctx.tenant_id,
                )
            )
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.TENANT, "")
        log.info("tools.catalog", upserted=len(stored), changed=len(changed), **ctx.log_fields())
        return stored

    async def catalog(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, names: Sequence[str] | None
    ) -> list[tuple[ToolDescriptor, ToolStats]]:
        entries = await uow.tools.catalog(
            ctx.tenant_id, workspace_id=ctx.workspace_id, names=names, limit=CATALOG_MAX
        )
        stats = await uow.tools.stats(ctx.tenant_id, [e.name for e in entries])
        return [(e, stats.get(e.name) or ToolStats(tool_name=e.name)) for e in entries]

    # ------------------------------------------------------------------ recording
    async def record(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        tool: str,
        args: dict[str, Any],
        output: Any = None,
        output_summary: str | None = None,
        status: ToolStatus = "ok",
        error_class: str | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        task: str = "",
        step: int | None = None,
        sub_calls: Sequence[dict[str, Any]] | None = None,
        visibility: Visibility = Visibility.PRIVATE,
    ) -> tuple[ToolInvocation, bool]:
        """Persist one call and count it. Idempotent on (run, step, tool, args_hash): a
        replayed graph step re-reads its row instead of inflating the statistics."""
        await guard_workspace_visibility(uow, self.authz, ctx, visibility)
        descriptor = await uow.tools.by_name(
            ctx.tenant_id, tool, workspace_id=ctx.workspace_id
        ) or await uow.tools.ensure(ctx.tenant_id, tool)
        summary, blob_ref, output_digest = await self._store_output(ctx, output, output_summary)
        invocation = ToolInvocation(
            tenant_id=ctx.tenant_id,
            tool_id=descriptor.tool_id,
            tool_name=descriptor.name,
            tool_version=descriptor.version,
            run_id=ctx.agent_run_id,
            thread_id=ctx.thread_id,
            turn_id=ctx.turn_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            principal_id=ctx.principal_id,
            step=step if step is not None else await self._next_step(uow, ctx),
            args_redacted=descriptor.redacted_args(args),
            # The *full* arguments: only the redacted copy is persisted, but the hash has to
            # tell apart calls that differ solely in a redacted field. It is one-way.
            args_hash=stable_hash(args),
            output_summary=summary,
            output_digest=output_digest,
            output_blob_ref=blob_ref,
            output_fields=flatten(output) if output is not None else {},
            status=status,
            error_class=error_class,
            latency_ms=latency_ms,
            cost=cost,
            task=task,
            task_pattern=task_pattern(task) if task else None,
            sub_calls=[SubCall.model_validate(c) for c in (sub_calls or [])],
            visibility_keys=_visibility_keys(ctx, visibility),
        )
        stored, created = await uow.tools.record(invocation)
        if created:
            await uow.tools.count_call(
                ctx.tenant_id,
                stored.tool_name,
                ok=stored.succeeded,
                latency_ms=stored.latency_ms,
                at=stored.occurred_at,
            )
        return stored, created

    async def _next_step(self, uow: UnitOfWork, ctx: MemoryExecutionContext) -> int:
        if not ctx.agent_run_id:
            return 0
        existing = await uow.tools.invocations_for_run(ctx.tenant_id, ctx.agent_run_id)
        return max((i.step for i in existing), default=-1) + 1

    async def _store_output(
        self, ctx: MemoryExecutionContext, output: Any, provided_summary: str | None
    ) -> tuple[str, str | None, str | None]:
        if output is None:
            return (provided_summary or "", None, None)
        text = output if isinstance(output, str) else _render(output)
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        summary = provided_summary or text[:SUMMARY_CHARS]
        blob_ref: str | None = None
        if len(text) > INLINE_OUTPUT_LIMIT and self.blob is not None:
            path = f"{ctx.tenant_id}/tool-output/{digest}.txt"
            try:
                await self.blob.put(
                    self.blob_bucket, path, text.encode("utf-8", "replace"), "text/plain"
                )
                blob_ref = f"{self.blob_bucket}/{path}"
            except Exception as exc:
                log.warning("tools.output_archive_failed", error=type(exc).__name__)
        return (summary, blob_ref, digest)

    # ------------------------------------------------------------------ outcomes
    async def set_outcome(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        run_id: str,
        success: bool,
        note: str | None = None,
    ) -> RunOutcome:
        """Label a run (the last word wins) and queue its calls to be learned again: a label
        is what turns a trajectory into evidence for a procedure."""
        outcome = RunOutcome(tenant_id=ctx.tenant_id, run_id=run_id, success=success, note=note)
        await uow.tools.set_outcome(outcome)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_TOOLS_LEARN,
                queue=Queue.RECONCILE,
                payload={"tenant_id": ctx.tenant_id},
                tenant_id=ctx.tenant_id,
            )
        )
        return outcome


def _visibility_keys(ctx: MemoryExecutionContext, visibility: Visibility) -> list[str]:
    """Tool records are audienced exactly like memories: the same anchor rules and the same
    audience keys, so an agent's tool chatter reaches a user or a group only when it was
    explicitly shared, and a hand-off is visible to the child run and no further. The default
    is PRIVATE, the agent's own durable store: a procedure is learned across its runs."""
    anchor = MemoryCandidate(
        content="", memory_type=MemoryType.TOOL, lifetime=Lifetime.SHORT_TERM, visibility=visibility
    )
    return keys_for(scope_for(anchor, ctx), visibility, ctx)


def _render(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)[:20000]
    except (TypeError, ValueError):
        return str(value)[:20000]
