"""Tool memory service (TOOL_MEMORY.md §30.0-§30.6).

The service registers tools, records what agents did with them, serves a cached output when
(and only when) the tool's policy allows it, mines chains and procedures from the records,
and answers three questions an agent asks mid-task:

    suggest(task)                        which tool, for this kind of task?
    next(task, trajectory_so_far)        given what I have already called, what now?
    plan(task)                           the whole validated chain, up front.

Three rules hold everywhere and are enforced here rather than at the edges: nothing is ever
suggested that the caller did not declare as available; nothing is read that the caller's
visibility keys do not cover; and a policy is only ever widened by an explicit admin-scope
registration, never by a per-call declaration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.tools import (
    RunOutcome,
    SubCall,
    ToolDescriptor,
    ToolInvocation,
    ToolPolicy,
)
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.tools.cache import ToolOutputCache, args_hash_for
from memory_service.modules.tools.patterns import best_match, task_pattern
from memory_service.modules.tools.procedures import (
    Procedure,
    decayed,
    mine_procedure,
    validate_against_registry,
)
from memory_service.modules.tools.trajectories import build_trajectory, flatten
from memory_service.observability.logging import get_logger
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

INLINE_OUTPUT_LIMIT = 4096
SUMMARY_CHARS = 600
MAX_TRAJECTORY_RUNS = 60


@dataclass
class ToolSuggestion:
    tool: str
    confidence: float
    argument_template: dict[str, Any] = field(default_factory=dict)
    supporting_procedures: list[str] = field(default_factory=list)
    supporting_invocations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence_status: str = "NONE"

    def to_payload(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "confidence": round(self.confidence, 4),
            "argument_template": self.argument_template,
            "supporting_procedures": self.supporting_procedures,
            "supporting_invocations": self.supporting_invocations[:10],
            "warnings": self.warnings,
            "evidence_status": self.evidence_status,
        }


class ToolMemoryService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        authz: AuthorizationService,
        *,
        cache: ToolOutputCache,
        blob: Any = None,
        blob_bucket: str = "memory-tool-outputs",
        indexer: Any = None,
        weak_positive_after_hours: float = 24.0,
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.cache = cache
        self.blob = blob
        self.blob_bucket = blob_bucket
        self.indexer = indexer
        self.weak_positive_after_hours = weak_positive_after_hours

    # ------------------------------------------------------------------ registry
    async def register(
        self,
        uow: Any,
        ctx: MemoryExecutionContext,
        descriptor: ToolDescriptor,
        *,
        widen_policy: bool | None = None,
    ) -> ToolDescriptor:
        """Register or upsert a descriptor. Widening a policy needs tenant admin; without it the
        stored policy is preserved and the caller's policy is ignored (never rejected, so an
        adapter can keep declaring what it knows without needing admin rights)."""
        if widen_policy is None:
            widen_policy = await self.authz.is_tenant_admin(ctx)
        desc = descriptor.model_copy(
            update={"tenant_id": ctx.tenant_id, "workspace_id": descriptor.workspace_id}
        ).with_schema_hash()
        stored = await uow.tools.register(desc, widen_policy=bool(widen_policy))
        log.info(
            "tools.registered",
            tool=stored.name,
            version=stored.version,
            source=stored.source,
            widened=bool(widen_policy),
            **ctx.log_fields(),
        )
        return stored

    async def declare(
        self, uow: Any, ctx: MemoryExecutionContext, declared: Sequence[dict[str, Any]]
    ) -> dict[str, ToolDescriptor]:
        """Upsert the tools a caller declares for this request. Conservative by construction:
        a declared tool that is not already registered gets the default policy, which is
        non-deterministic, non-cacheable and of unknown side effects."""
        out: dict[str, ToolDescriptor] = {}
        for item in declared:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            existing = await uow.tools.by_name(ctx.tenant_id, name)
            if existing is not None:
                out[name] = existing
                continue
            descriptor = ToolDescriptor(
                tenant_id=ctx.tenant_id,
                name=name,
                description=str(item.get("description", "")),
                input_schema=item.get("schema") or item.get("input_schema"),
                output_schema=item.get("output_schema"),
                tags=list(item.get("tags") or []),
                source=item.get("source", "manual"),
                server=item.get("server"),
                policy=ToolPolicy(),
            )
            out[name] = await self.register(uow, ctx, descriptor, widen_policy=False)
        return out

    # ------------------------------------------------------------------ recording
    async def record(
        self,
        uow: Any,
        ctx: MemoryExecutionContext,
        *,
        tool: str,
        args: dict[str, Any],
        output: Any = None,
        output_summary: str | None = None,
        status: str = "ok",
        error_class: str | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        task: str = "",
        step: int | None = None,
        sub_calls: Sequence[dict[str, Any]] | None = None,
        visibility: Visibility = Visibility.RUN,
    ) -> ToolInvocation:
        """Persist one call. Idempotent on (run, step, tool, args_hash): a replayed graph step
        re-reads its row instead of inflating the statistics."""
        descriptor = await uow.tools.by_name(ctx.tenant_id, tool)
        if descriptor is None:
            descriptor = await self.register(
                uow,
                ctx,
                ToolDescriptor(tenant_id=ctx.tenant_id, name=tool, policy=ToolPolicy()),
                widen_policy=False,
            )
        digest = args_hash_for(descriptor, args)
        fields = flatten(output) if output is not None else {}
        summary, blob_ref, output_digest = await self._store_output(
            ctx, descriptor, output, output_summary
        )
        resolved_step = step if step is not None else await self._next_step(uow, ctx)
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
            step=resolved_step,
            args_redacted=descriptor.redacted_args(args),
            args_hash=digest,
            output_summary=summary,
            output_digest=output_digest,
            output_blob_ref=blob_ref,
            output_fields=fields,
            status=status,  # type: ignore[arg-type]
            error_class=error_class,
            latency_ms=latency_ms,
            cost=cost,
            task=task,
            task_pattern=task_pattern(task) if task else None,
            sub_calls=[SubCall.model_validate(c) for c in (sub_calls or [])],
            visibility_keys=self._visibility_keys(ctx, visibility),
        )
        stored = await uow.tools.record(invocation)
        if stored.invocation_id == invocation.invocation_id and status == "ok":
            await self.cache.put(
                descriptor,
                digest,
                ctx,
                output_summary=summary,
                output_digest=output_digest,
                output_blob_ref=blob_ref,
                output_fields=fields,
                invocation_id=stored.invocation_id,
            )
        return stored

    async def _next_step(self, uow: Any, ctx: MemoryExecutionContext) -> int:
        if not ctx.agent_run_id:
            return 0
        existing = await uow.tools.invocations_for_run(ctx.tenant_id, ctx.agent_run_id)
        return max((i.step for i in existing), default=-1) + 1

    async def _store_output(
        self,
        ctx: MemoryExecutionContext,
        descriptor: ToolDescriptor,
        output: Any,
        provided_summary: str | None,
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

    def _visibility_keys(self, ctx: MemoryExecutionContext, visibility: Visibility) -> list[str]:
        """Tool records are audienced exactly like memories: the same anchor rules and the same
        audience keys, so an agent's tool chatter reaches a user or a group only when it was
        explicitly shared, and a hand-off is visible to the child run and no further."""
        from memory_service.domain.enums import Lifetime, MemoryType
        from memory_service.modules.memory.pipeline import keys_for, scope_for
        from memory_service.ports.intelligence import MemoryCandidate

        anchor = MemoryCandidate(
            content="",
            memory_type=MemoryType.TOOL,
            lifetime=Lifetime.SHORT_TERM,
            visibility=visibility,
        )
        return keys_for(scope_for(anchor, ctx), visibility, ctx)

    # ------------------------------------------------------------------ outcomes
    async def set_outcome(
        self,
        uow: Any,
        ctx: MemoryExecutionContext,
        *,
        run_id: str,
        success: bool,
        note: str | None = None,
        source: str = "explicit",
    ) -> RunOutcome:
        outcome = RunOutcome(
            tenant_id=ctx.tenant_id,
            run_id=run_id,
            success=success,
            note=note,
            source=source,  # type: ignore[arg-type]
        )
        await uow.tools.set_outcome(outcome)
        return outcome

    async def _outcome_for(self, uow: Any, tenant_id: str, run_id: str) -> bool:
        """A run counts as successful when it was labelled so, or — as a weak positive — when it
        is older than the window, had no failing call, and nobody corrected it."""
        recorded = await uow.tools.outcome(tenant_id, run_id)
        if recorded is not None:
            return recorded.success
        invocations = await uow.tools.invocations_for_run(tenant_id, run_id)
        if not invocations:
            return False
        if any(not i.succeeded for i in invocations):
            return False
        newest = max(i.occurred_at for i in invocations)
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=UTC)
        age_hours = (datetime.now(UTC) - newest).total_seconds() / 3600.0
        return age_hours >= self.weak_positive_after_hours

    # ------------------------------------------------------------------ learning
    async def procedures(
        self, uow: Any, ctx: MemoryExecutionContext, *, task: str, scope_keys: Sequence[str]
    ) -> list[Procedure]:
        """Mine the procedure(s) for the pattern of ``task`` from the records this caller may
        see. Cheap enough to run per request; the periodic job persists the result."""
        pattern = task_pattern(task)
        if not pattern:
            return []
        records = await uow.tools.recent(
            ctx.tenant_id, scope_keys=list(scope_keys), limit=MAX_TRAJECTORY_RUNS * 8
        )
        patterns = sorted({r.task_pattern for r in records if r.task_pattern})
        matched = best_match(pattern, patterns) or pattern
        relevant = [r for r in records if r.task_pattern == matched and r.run_id]
        by_run: dict[str, list[ToolInvocation]] = {}
        for record in relevant:
            by_run.setdefault(str(record.run_id), []).append(record)
        trajectories = []
        for run_id, invocations in list(by_run.items())[:MAX_TRAJECTORY_RUNS]:
            succeeded = await self._outcome_for(uow, ctx.tenant_id, run_id)
            trajectories.append(build_trajectory(run_id, invocations, succeeded=succeeded))
        procedure = mine_procedure(matched, trajectories)
        if procedure is None or decayed(procedure):
            return []
        return [procedure]

    async def plan(
        self,
        uow: Any,
        ctx: MemoryExecutionContext,
        *,
        task: str,
        available_tools: Sequence[dict[str, Any]],
        scope_keys: Sequence[str],
    ) -> dict[str, Any]:
        declared = await self.declare(uow, ctx, available_tools)
        procedures = await self.procedures(uow, ctx, task=task, scope_keys=scope_keys)
        if not procedures:
            return {
                "task_pattern": task_pattern(task),
                "steps": [],
                "valid": False,
                "reason": "no validated procedure yet",
            }
        procedure = procedures[0]
        undeclared = [s.tool for s in procedure.steps if s.tool not in declared]
        if undeclared:
            return {
                "task_pattern": procedure.task_pattern,
                "steps": [],
                "valid": False,
                "reason": (
                    f"procedure uses tools the caller did not declare: {sorted(set(undeclared))}"
                ),
            }
        problems = validate_against_registry(procedure, dict(declared))
        payload = procedure.to_payload()
        payload["valid"] = not problems
        payload["problems"] = problems
        payload["script"] = procedure.render_script()
        payload["rendered"] = procedure.render()
        return payload


def _render(value: Any) -> str:
    import json

    try:
        return json.dumps(value, default=str, sort_keys=True)[:20000]
    except (TypeError, ValueError):
        return str(value)[:20000]
