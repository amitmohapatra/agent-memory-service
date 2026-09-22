"""Reflection pass on landing: when a new memory is admitted, its top-k related memories
(same scope and subject, k <= 8) are re-evaluated deterministically inside the same unit of
work: same slot and value -> strengthen; same slot with disagreeing numbers or negation ->
weaken and link as contradicting; an older value of a single-valued slot -> supersede.
Then the derived memories are maintained: a belief over a multi-valued slot with enough
support, and the entity summary of the subject. The pass records what it did on the landed
memory and refuses to run twice on it, so a replay changes nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memory_service.config.settings import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.memory import DERIVED_MEMORY_TYPES, CanonicalMemory
from memory_service.modules.memory.derived import BeliefService, EntitySummaryService
from memory_service.modules.memory.native import (
    _SINGLE_VALUED,
    _clean_object,
    has_negation,
    numbers,
)
from memory_service.observability.logging import get_logger
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)

_STRENGTHEN = 0.05
_WEAKEN = 0.1
_CONFIDENCE_FLOOR = 0.05


def is_single_valued(predicate: str | None) -> bool:
    return bool(predicate) and (predicate in _SINGLE_VALUED or predicate.startswith("favourite_"))


class LandingReflection:
    def __init__(
        self,
        settings: MemoryIntelligenceSettings,
        *,
        beliefs: BeliefService | None = None,
        summaries: EntitySummaryService | None = None,
    ) -> None:
        self.cfg = settings
        self.beliefs = beliefs or BeliefService()
        self.summaries = summaries or EntitySummaryService()

    async def on_landed(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        memory: CanonicalMemory,
        *,
        now: datetime,
    ) -> set[str]:
        """Returns the ids of every memory changed or created (for indexing)."""
        if (
            self.cfg.landing_reflection_k <= 0
            or memory.memory_type in DERIVED_MEMORY_TYPES
            or not memory.subject
            or memory.temporal.status is not TemporalStatus.CURRENT
            or memory.deleted_at is not None
            or "landing" in memory.system_metadata
        ):
            return set()
        related = await uow.memories.related(
            ctx.tenant_id,
            scope_key=memory.scope.key(),
            subject=memory.subject,
            exclude=[memory.memory_id],
            limit=min(8, self.cfg.landing_reflection_k),
        )
        facts = [r for r in related if r.memory_type not in DERIVED_MEMORY_TYPES]
        report: dict[str, Any] = {
            "reflected_at": now.isoformat(),
            "evaluated": [r.memory_id for r in facts],
            "strengthened": [],
            "weakened": [],
            "superseded": [],
        }
        affected: set[str] = set()
        m_obj = _clean_object(memory.object or "")
        for r in facts:
            if not memory.predicate or r.predicate != memory.predicate:
                continue
            r_obj = _clean_object(r.object or "")
            if m_obj and r_obj and m_obj == r_obj:
                r.confidence = min(1.0, r.confidence + _STRENGTHEN)
                r.reinforcement_count += 1
                report["strengthened"].append(r.memory_id)
            elif (
                is_single_valued(memory.predicate)
                and m_obj != r_obj
                and (
                    numbers(r.content) != numbers(memory.content)
                    or has_negation(r.content) != has_negation(memory.content)
                )
            ):
                r.confidence = max(_CONFIDENCE_FLOOR, r.confidence - _WEAKEN)
                if memory.memory_id not in r.temporal.contradicts:
                    r.temporal = r.temporal.model_copy(
                        update={"contradicts": [*r.temporal.contradicts, memory.memory_id]}
                    )
                if r.memory_id not in memory.temporal.contradicts:
                    memory.temporal = memory.temporal.model_copy(
                        update={"contradicts": [*memory.temporal.contradicts, r.memory_id]}
                    )
                report["weakened"].append(r.memory_id)
            elif (
                m_obj
                and r_obj
                and is_single_valued(memory.predicate)
                and r.created_at <= memory.created_at
            ):
                r.temporal = r.temporal.model_copy(
                    update={
                        "status": TemporalStatus.SUPERSEDED,
                        "superseded_by": memory.memory_id,
                        "valid_to": r.temporal.valid_to or memory.temporal.valid_from or now,
                    }
                )
                if memory.temporal.supersedes is None:
                    memory.temporal = memory.temporal.model_copy(update={"supersedes": r.memory_id})
                report["superseded"].append(r.memory_id)
            else:
                continue
            r.updated_at = now
            await uow.memories.update(r)
            affected.add(r.memory_id)
        live = [memory, *(r for r in facts if r.temporal.status is TemporalStatus.CURRENT)]
        if memory.predicate and not is_single_valued(memory.predicate):
            support = [m for m in live if m.predicate == memory.predicate]
            if len(support) >= self.cfg.belief_min_support:
                belief, changed = await self.beliefs.upsert(
                    uow,
                    ctx,
                    scope=memory.scope,
                    subject=memory.subject,
                    predicate=memory.predicate,
                    sources=support,
                    now=now,
                )
                report["belief"] = belief.memory_id
                if changed:
                    affected.add(belief.memory_id)
                    if belief.temporal.supersedes:
                        affected.add(belief.temporal.supersedes)
        if len(live) >= self.cfg.entity_summary_min_facts:
            summary, changed = await self.summaries.rebuild(
                uow, ctx, scope=memory.scope, subject=memory.subject, facts=live, now=now
            )
            if summary is not None:
                report["entity_summary"] = summary.memory_id
                if changed:
                    affected.add(summary.memory_id)
                    if summary.temporal.supersedes:
                        affected.add(summary.temporal.supersedes)
        memory.system_metadata["landing"] = report
        memory.updated_at = now
        await uow.memories.update(memory)
        affected.add(memory.memory_id)
        log.debug(
            "memory.landing",
            tenant_id=ctx.tenant_id,
            memory_id=memory.memory_id,
            strengthened=len(report["strengthened"]),
            weakened=len(report["weakened"]),
            superseded=len(report["superseded"]),
        )
        return affected
