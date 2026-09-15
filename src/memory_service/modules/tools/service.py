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
from memory_service.modules.tools.cache import CachedOutput, ToolOutputCache, args_hash_for
from memory_service.modules.tools.patterns import best_match, task_pattern
from memory_service.modules.tools.procedures import (
    Procedure,
    ProcedureStep,
    decayed,
    mine_procedure,
    score_tool,
    validate_against_registry,
)
from memory_service.modules.tools.trajectories import build_trajectory, flatten
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
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


@dataclass
class NextStep:
    suggestions: list[ToolSuggestion]
    stop: bool
    matched_procedure: str | None = None
    matched_prefix_length: int = 0


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

    # ------------------------------------------------------------------ cache
    async def lookup(
        self, uow: Any, ctx: MemoryExecutionContext, *, tool: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        descriptor = await uow.tools.by_name(ctx.tenant_id, tool)
        if descriptor is None:
            return {"cached": False, "reason": "tool not registered"}
        if not descriptor.policy.replayable:
            return {
                "cached": False,
                "reason": (
                    "policy forbids replay "
                    f"(deterministic={descriptor.policy.deterministic}, "
                    f"cacheable={descriptor.policy.cacheable}, "
                    f"side_effects={descriptor.policy.side_effects})"
                ),
            }
        digest = args_hash_for(descriptor, args)
        hit: CachedOutput | None = await self.cache.get(descriptor, digest, ctx)
        if hit is None:
            return {"cached": False, "reason": "miss"}
        return {
            "cached": True,
            "age_seconds": round(hit.age_seconds, 3),
            "output_summary": hit.output_summary,
            "output_digest": hit.output_digest,
            "output_blob_ref": hit.output_blob_ref,
            "output_fields": hit.output_fields,
            "invocation_id": hit.invocation_id,
            "cache_scope": descriptor.policy.cache_scope,
        }

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

    # ------------------------------------------------------------------ advice
    async def suggest(
        self,
        uow: Any,
        ctx: MemoryExecutionContext,
        *,
        task: str,
        available_tools: Sequence[dict[str, Any]],
        scope_keys: Sequence[str],
        limit: int = 5,
    ) -> list[ToolSuggestion]:
        with span("tools.suggest"), stage_seconds.labels("tools.suggest").time():
            declared = await self.declare(uow, ctx, available_tools)
            if not declared:
                return []
            procedures = await self.procedures(uow, ctx, task=task, scope_keys=scope_keys)
            procedure = procedures[0] if procedures else None
            stats = {s.tool_name: s for s in await uow.tools.stats(ctx.tenant_id)}
            records = await uow.tools.recent(ctx.tenant_id, scope_keys=list(scope_keys), limit=200)
            recent_failures = {
                r.tool_name for r in records[:20] if not r.succeeded and r.tool_name in declared
            }
            suggestions: list[ToolSuggestion] = []
            for name in declared:
                stat = stats.get(name)
                position = (
                    procedure.tools.index(name) if procedure and name in procedure.tools else None
                )
                confidence = score_tool(
                    in_procedure=position is not None,
                    procedure_position=position,
                    success_rate=stat.success_rate if stat else 0.0,
                    invocations=stat.invocations if stat else 0,
                    last_used_at=stat.last_used_at if stat else None,
                    failed_recently=name in recent_failures,
                )
                if confidence <= 0.0:
                    continue
                warnings: list[str] = []
                if name in recent_failures:
                    warnings.append("failed on a recent call in this scope")
                if stat and stat.failures and stat.success_rate < 0.8:
                    warnings.append(
                        f"success rate {stat.success_rate:.0%} over {stat.invocations} calls"
                    )
                if stat is None:
                    warnings.append("no recorded history for this tool")
                step = procedure.steps[position] if procedure and position is not None else None
                suggestions.append(
                    ToolSuggestion(
                        tool=name,
                        confidence=confidence,
                        argument_template=_template(step),
                        supporting_procedures=[procedure.task_pattern]
                        if position is not None and procedure
                        else [],
                        supporting_invocations=[
                            r.invocation_id for r in records if r.tool_name == name
                        ][:10],
                        warnings=warnings,
                        evidence_status="COMPLETE" if position is not None else "PARTIAL",
                    )
                )
            suggestions.sort(key=lambda s: (-s.confidence, s.tool))
            return suggestions[:limit]

    async def next_step(
        self,
        uow: Any,
        ctx: MemoryExecutionContext,
        *,
        task: str,
        trajectory_so_far: Sequence[dict[str, Any]],
        available_tools: Sequence[dict[str, Any]],
        scope_keys: Sequence[str],
        limit: int = 3,
    ) -> NextStep:
        """The ranked next steps given what has already been called, with bindings resolved from
        the prefix's own outputs. ``stop`` is set when the matched path has no successor."""
        declared = await self.declare(uow, ctx, available_tools)
        procedures = await self.procedures(uow, ctx, task=task, scope_keys=scope_keys)
        prefix = [str(s.get("tool", "")) for s in trajectory_so_far]
        if not procedures:
            suggestions = await self.suggest(
                uow,
                ctx,
                task=task,
                available_tools=available_tools,
                scope_keys=scope_keys,
                limit=limit,
            )
            remaining = [s for s in suggestions if s.tool not in prefix]
            return NextStep(suggestions=remaining, stop=False)
        procedure = procedures[0]
        matched = _longest_suffix_match(prefix, procedure.tools)
        if matched >= len(procedure.steps):
            return NextStep(
                suggestions=[],
                stop=True,
                matched_procedure=procedure.task_pattern,
                matched_prefix_length=matched,
            )
        candidates: list[ToolSuggestion] = []
        for offset, step in enumerate(procedure.steps[matched : matched + limit]):
            if step.tool not in declared:
                continue
            bindings, unmet = _bind_from_prefix(step, trajectory_so_far)
            if unmet:
                continue
            penalty = 0.1 * offset
            candidates.append(
                ToolSuggestion(
                    tool=step.tool,
                    confidence=max(0.0, round(step.success_rate * (1.0 - penalty), 6)),
                    argument_template=bindings,
                    supporting_procedures=[procedure.task_pattern],
                    supporting_invocations=procedure.invocation_ids[:10],
                    warnings=[f"on {e}: {f}" for e, f in step.failure_modes.items() if f],
                    evidence_status="COMPLETE",
                )
            )
        return NextStep(
            suggestions=candidates,
            stop=not candidates and matched >= len(procedure.steps),
            matched_procedure=procedure.task_pattern,
            matched_prefix_length=matched,
        )

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


def _template(step: ProcedureStep | None) -> dict[str, Any]:
    if step is None:
        return {}
    out: dict[str, Any] = {}
    for binding in step.bindings:
        if binding.resolvable_from_trajectory:
            out[binding.argument] = f"${{step{binding.source_step}.{binding.source_field}}}"
        elif binding.literal is not None:
            out[binding.argument] = binding.literal
    return out


def _longest_suffix_match(prefix: Sequence[str], tools: Sequence[str]) -> int:
    """How far along the procedure the caller already is. The longest prefix of ``tools`` that
    is a suffix of what has been called wins, so a retried or reordered start still matches."""
    best = 0
    for length in range(min(len(prefix), len(tools)), 0, -1):
        if list(prefix[-length:]) == list(tools[:length]):
            best = length
            break
    return best


def _bind_from_prefix(
    step: ProcedureStep, trajectory: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Fill the step's arguments from the outputs already produced. Returns the bound arguments
    and the preconditions that could not be met."""
    bound: dict[str, Any] = {}
    unmet: list[str] = []
    for binding in step.bindings:
        if not binding.resolvable_from_trajectory:
            if binding.literal is not None:
                bound[binding.argument] = binding.literal
            continue
        index = binding.source_step
        if index is None or index >= len(trajectory):
            unmet.append(f"step{index}.{binding.source_field}")
            continue
        outputs = trajectory[index].get("output_fields") or {}
        field_name = binding.source_field or ""
        if field_name in outputs:
            bound[binding.argument] = outputs[field_name]
        else:
            unmet.append(f"step{index}.{field_name}")
    return bound, unmet


def _render(value: Any) -> str:
    import json

    try:
        return json.dumps(value, default=str, sort_keys=True)[:20000]
    except (TypeError, ValueError):
        return str(value)[:20000]
