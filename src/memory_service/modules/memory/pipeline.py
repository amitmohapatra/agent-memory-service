"""ObservationPipeline: observation -> extract -> classify -> consolidate -> persist -> index.

Runs inside the ``memory.process_observation`` job. Everything the pipeline writes for one
observation is committed in a single unit of work together with the ``memory.index`` outbox
job, so a crash between steps leaves either nothing or a complete, indexable state (and the
job is retried from the durable observation either way).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from memory_service.config.settings import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    DedupDecision,
    Lifetime,
    MemoryType,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.domain.observation import Observation
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.visibility import visibility_keys
from memory_service.modules.memory.ephemeral import EphemeralMemory
from memory_service.modules.memory.native import normalized_hash
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import memory_decisions_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.intelligence import (
    ConsolidationOutcome,
    MemoryCandidate,
    MemoryIntelligenceProvider,
)
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_MEMORY_INDEX = "memory.index"
TASK_MEMORY_EXPIRE = "memory.expire"

SHORT_TERM_TTL = timedelta(days=7)


def context_from_observation(o: Observation) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id=o.tenant_id,
        workspace_id=o.workspace_id,
        user_id=o.user_id,
        thread_id=o.thread_id,
        session_id=o.session_id,
        turn_id=o.turn_id,
        work_id=o.work_id,
        task_id=o.task_id,
        agent_id=o.agent_id,
        agent_group_id=o.agent_group_id,
        agent_run_id=o.agent_run_id,
        parent_agent_run_id=o.parent_agent_run_id,
        trace_id=o.trace_id or "",
    )


def scope_for(candidate: MemoryCandidate, ctx: MemoryExecutionContext) -> Scope:
    """Anchor: where the memory belongs (distinct from who may read it)."""
    vis = candidate.visibility or Visibility.PRIVATE
    mt = candidate.memory_type
    if mt in (MemoryType.AGENT, MemoryType.TOOL, MemoryType.WORKING) and ctx.agent_id:
        level = ScopeLevel.AGENT
    elif mt in (MemoryType.USER, MemoryType.PREFERENCE) and ctx.user_id:
        level = ScopeLevel.USER
    elif vis is Visibility.AGENT_GROUP and ctx.agent_group_id:
        level = ScopeLevel.AGENT_GROUP
    elif vis is Visibility.WORK and ctx.work_id:
        level = ScopeLevel.WORK
    elif vis is Visibility.THREAD and ctx.thread_id:
        level = ScopeLevel.THREAD
    elif vis is Visibility.GROUP and ctx.group_ids:
        level = ScopeLevel.GROUP
    elif vis is Visibility.WORKSPACE and ctx.workspace_id:
        level = ScopeLevel.WORKSPACE
    elif vis in (Visibility.TENANT, Visibility.GLOBAL):
        level = ScopeLevel.TENANT
    elif ctx.agent_id and vis is Visibility.PRIVATE:
        level = ScopeLevel.AGENT
    elif ctx.thread_id:
        level = ScopeLevel.THREAD
    elif ctx.user_id:
        level = ScopeLevel.USER
    elif ctx.workspace_id:
        level = ScopeLevel.WORKSPACE
    else:
        level = ScopeLevel.TENANT
    return Scope(
        level=level,
        tenant_id=ctx.tenant_id,
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id if level in (ScopeLevel.USER, ScopeLevel.AGENT) else None,
        group_id=ctx.group_ids[0] if level is ScopeLevel.GROUP and ctx.group_ids else None,
        thread_id=ctx.thread_id if level in (ScopeLevel.THREAD, ScopeLevel.AGENT) else None,
        work_id=ctx.work_id if level is ScopeLevel.WORK else None,
        agent_id=ctx.agent_id if level is ScopeLevel.AGENT else None,
        agent_group_id=ctx.agent_group_id if level is ScopeLevel.AGENT_GROUP else None,
    )


def keys_for(scope: Scope, visibility: Visibility, ctx: MemoryExecutionContext) -> list[str]:
    """Audience keys for a memory. The owner principal can always read what it wrote
    (a memory shared with a thread or workspace stays visible to its author even if the
    author later loses that membership); everyone else needs the visibility's audience."""
    keys = visibility_keys(
        ctx.tenant_id,
        visibility,
        owner_principal=ctx.principal_id,
        scope=scope,
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        group_id=ctx.group_ids[0] if ctx.group_ids else None,
        thread_id=ctx.thread_id,
        work_id=ctx.work_id,
        agent_group_id=ctx.agent_group_id,
        agent_run_id=ctx.agent_run_id,
    )
    owner = f"principal:{ctx.tenant_id}/{ctx.principal_id}"
    if owner in keys or visibility is Visibility.PRIVATE:
        return keys
    return [*keys, owner]


def build_memory(
    candidate: MemoryCandidate, ctx: MemoryExecutionContext, *, now: datetime
) -> CanonicalMemory:
    scope = scope_for(candidate, ctx)
    vis = candidate.visibility or Visibility.PRIVATE
    expires = now + SHORT_TERM_TTL if candidate.lifetime is Lifetime.SHORT_TERM else None
    if candidate.valid_to and (expires is None or candidate.valid_to < expires):
        expires = candidate.valid_to
    return CanonicalMemory(
        tenant_id=ctx.tenant_id,
        scope=scope,
        visibility=vis,
        owner_principal=ctx.principal_id,
        lifetime=candidate.lifetime,
        memory_type=candidate.memory_type,
        content=candidate.content,
        normalized_hash=normalized_hash(candidate.content),
        subject=candidate.subject,
        predicate=candidate.predicate,
        object=candidate.object,
        temporal=TemporalState(
            valid_from=candidate.valid_from,
            valid_to=candidate.valid_to,
            observed_at=candidate.evidence[0].observed_at
            if candidate.evidence and candidate.evidence[0].observed_at
            else now,
        ),
        evidence=list(candidate.evidence),
        confidence=candidate.confidence,
        importance=candidate.importance,
        system_metadata={
            "provider": candidate.provider,
            "category": candidate.category,
            "entities": list(candidate.entities),
            "provider_ref": candidate.provider_ref,
            "expires_at": expires.isoformat() if expires else None,
        },
        created_at=now,
        updated_at=now,
    )


class ObservationPipeline:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        provider: MemoryIntelligenceProvider,
        *,
        settings: MemoryIntelligenceSettings,
        working: EphemeralMemory | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.provider = provider
        self.cfg = settings
        self.working = working

    async def run(self, payload: dict[str, Any]) -> list[ConsolidationOutcome]:
        tenant_id, observation_id = payload["tenant_id"], payload["observation_id"]
        async with self.uow_factory() as uow:
            observation = await uow.observations.get(tenant_id, observation_id)
        if observation is None:
            log.warning("memory.observation_missing", tenant_id=tenant_id, id=observation_id)
            return []
        if observation.processed_at is not None:
            return []  # idempotent replay
        ctx = context_from_observation(observation)
        with (
            span("memory.process", tenant_id=tenant_id, kind=observation.kind.value),
            stage_seconds.labels("memory.process").time(),
        ):
            candidates = [
                await self.provider.classify(self._apply_hints(c, observation), ctx)
                for c in await self.provider.extract(observation, ctx)
            ]
            outcomes: list[ConsolidationOutcome] = []
            async with self.uow_factory() as uow:
                affected = await self._apply_all(uow, ctx, candidates, outcomes)
                if affected:
                    await uow.enqueue(
                        JobSpec(
                            task_name=TASK_MEMORY_INDEX,
                            queue=Queue.EMBEDDING,
                            payload={"tenant_id": tenant_id, "memory_ids": sorted(affected)},
                            idempotency_key=f"memidx:{observation_id}",
                            tenant_id=tenant_id,
                        )
                    )
                    await self._bump(uow, ctx)
                status = "PROCESSED" if candidates else "NO_MEMORY"
                await uow.observations.mark_processed(tenant_id, observation_id, status=status)
                await uow.commit()
        for o in outcomes:
            memory_decisions_total.labels(o.decision.value).inc()
        log.info(
            "memory.processed",
            tenant_id=tenant_id,
            observation_id=observation_id,
            candidates=len(candidates),
            decisions={o.decision.value: 1 for o in outcomes},
        )
        return outcomes

    @staticmethod
    def _apply_hints(c: MemoryCandidate, o: Observation) -> MemoryCandidate:
        h = o.hints
        update: dict[str, Any] = {}
        if h.memory_type:
            update["memory_type"] = h.memory_type
        elif o.agent_id and c.memory_type in (MemoryType.USER, MemoryType.PREFERENCE):
            # "my timezone is UTC" said by an agent is about the agent: it becomes the agent's
            # working memory, never a USER memory of the human it acts for (no chat pollution)
            update["memory_type"] = MemoryType.AGENT
        if h.lifetime:
            update["lifetime"] = h.lifetime
        if h.visibility:
            update["visibility"] = h.visibility
        if h.importance is not None:
            update["importance"] = h.importance
        return c.model_copy(update=update) if update else c

    async def _apply_all(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        candidates: list[MemoryCandidate],
        outcomes: list[ConsolidationOutcome],
    ) -> set[str]:
        affected: set[str] = set()
        now = datetime.now(UTC)
        for cand in candidates:
            if cand.lifetime is Lifetime.EPHEMERAL:
                if self.working is not None:
                    await self.working.remember(ctx, cand)
                outcomes.append(
                    ConsolidationOutcome(
                        decision=DedupDecision.IGNORE, candidate=cand, reason="ephemeral (cache)"
                    )
                )
                continue
            scope = scope_for(cand, ctx)
            existing = await uow.memories.candidates(
                ctx.tenant_id,
                scope_key=scope.key(),
                normalized_hash=normalized_hash(cand.content),
                subject=cand.subject,
                limit=self.cfg.dedup_candidate_k,
            )
            outcome = await self.provider.consolidate(cand, existing, ctx)
            outcomes.append(outcome)
            affected |= await self._apply(uow, ctx, outcome, existing, now=now)
        return affected

    async def _apply(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        outcome: ConsolidationOutcome,
        existing: list[CanonicalMemory],
        *,
        now: datetime,
    ) -> set[str]:
        cand = outcome.candidate
        target = next((m for m in existing if m.memory_id == outcome.target_memory_id), None)
        match outcome.decision:
            case DedupDecision.CREATE:
                memory = build_memory(cand, ctx, now=now)
                await uow.memories.add(
                    memory, visibility_keys=keys_for(memory.scope, memory.visibility, ctx)
                )
                return {memory.memory_id}
            case DedupDecision.REINFORCE | DedupDecision.MERGE if target is not None:
                target.reinforcement_count += 1
                contributors = list(target.system_metadata.get("contributors", []))
                if (
                    ctx.principal_id not in contributors
                    and ctx.principal_id != target.owner_principal
                ):
                    # independent corroboration by another principal is worth more than a repeat
                    contributors.append(ctx.principal_id)
                    target.system_metadata["contributors"] = contributors
                    target.confidence = min(1.0, target.confidence + 0.15)
                else:
                    target.confidence = min(1.0, target.confidence + 0.05)
                target.importance = max(target.importance, cand.importance)
                for ev in cand.evidence:
                    if ev not in target.evidence:
                        target.evidence.append(ev)
                if outcome.decision is DedupDecision.MERGE and len(cand.content) > len(
                    target.content
                ):
                    target.system_metadata.setdefault("merged_from", []).append(target.content)
                    target.content = cand.content
                    target.normalized_hash = normalized_hash(cand.content)
                target.updated_at = now
                await uow.memories.update(target)
                return {target.memory_id}
            case DedupDecision.SUPERSEDE | DedupDecision.UPDATE if target is not None:
                memory = build_memory(cand, ctx, now=now)
                memory.temporal = memory.temporal.model_copy(
                    update={"supersedes": target.memory_id}
                )
                memory.reinforcement_count = 1
                await uow.memories.add(
                    memory, visibility_keys=keys_for(memory.scope, memory.visibility, ctx)
                )
                target.temporal = target.temporal.model_copy(
                    update={
                        "status": TemporalStatus.SUPERSEDED,
                        "superseded_by": memory.memory_id,
                        "valid_to": target.temporal.valid_to or memory.temporal.valid_from or now,
                    }
                )
                target.updated_at = now
                await uow.memories.update(target)
                return {memory.memory_id, target.memory_id}
            case DedupDecision.CONTRADICT if target is not None:
                memory = build_memory(cand, ctx, now=now)
                memory.temporal = memory.temporal.model_copy(
                    update={"contradicts": [target.memory_id]}
                )
                await uow.memories.add(
                    memory, visibility_keys=keys_for(memory.scope, memory.visibility, ctx)
                )
                target.temporal = target.temporal.model_copy(
                    update={"contradicts": [*target.temporal.contradicts, memory.memory_id]}
                )
                target.updated_at = now
                await uow.memories.update(target)
                return {memory.memory_id, target.memory_id}
            case _:
                return set()

    @staticmethod
    async def _bump(uow: UnitOfWork, ctx: MemoryExecutionContext) -> None:
        if ctx.user_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.USER, ctx.user_id)
        if ctx.thread_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.THREAD, ctx.thread_id)
        if ctx.agent_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.AGENT, ctx.agent_id)
