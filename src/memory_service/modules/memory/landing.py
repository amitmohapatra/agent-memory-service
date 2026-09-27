"""Maintain source-backed summaries after admission, without reinforcing facts twice.

The observation pipeline already owns reinforcement and supersession. This pass only
maintains derived representations, within one author/audience and a bounded source set.
"""

from __future__ import annotations

from datetime import datetime

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, TemporalStatus
from memory_service.domain.memory import CanonicalMemory, unverified_representation
from memory_service.modules.memory.derived import (
    BeliefService,
    EntitySummaryService,
    source_statement,
)
from memory_service.modules.memory.native import _SINGLE_VALUED
from memory_service.ports.uow import UnitOfWork


class LandingReflection:
    def __init__(self, settings: MemoryIntelligenceSettings) -> None:
        self.cfg = settings
        self.beliefs = BeliefService()
        # No network calls under admission locks. Model reflection is background work.
        self.summaries = EntitySummaryService()

    async def on_landed(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        memory: CanonicalMemory,
        *,
        now: datetime,
    ) -> set[str]:
        if not memory.subject or not self._eligible(memory):
            return set()
        await uow.serialize(f"derived-source:{ctx.tenant_id}:{memory.scope.key()}:{memory.subject}")
        sources = await uow.memories.related(
            ctx.tenant_id,
            scope_key=memory.scope.key(),
            subject=memory.subject,
            owner_principal=memory.owner_principal,
            visibility_keys=memory.system_metadata.get("visibility_keys", []),
            include_derived=False,
            limit=self.cfg.consolidation_max_sources,
        )
        sources = self._bounded_sources(memory.subject, sources)
        affected: set[str] = set()
        predicate = memory.predicate
        if predicate and predicate not in _SINGLE_VALUED and not predicate.startswith("favourite_"):
            support = [m for m in sources if m.predicate == predicate]
            if len(support) >= self.cfg.belief_min_support:
                belief, changed = await self.beliefs.upsert(
                    uow,
                    ctx,
                    scope=memory.scope,
                    subject=memory.subject,
                    predicate=predicate,
                    sources=support,
                    now=now,
                )
                if changed:
                    affected.add(belief.memory_id)
                    if belief.temporal.supersedes:
                        affected.add(belief.temporal.supersedes)
        if len(sources) >= self.cfg.entity_summary_min_facts:
            summary, changed = await self.summaries.rebuild(
                uow,
                ctx,
                scope=memory.scope,
                subject=memory.subject,
                facts=sources,
                now=now,
            )
            if summary is not None and changed:
                affected.add(summary.memory_id)
                if summary.temporal.supersedes:
                    affected.add(summary.temporal.supersedes)
        return affected

    def _eligible(self, memory: CanonicalMemory) -> bool:
        return not (
            not self.cfg.consolidation_enabled
            or memory.memory_type in {MemoryType.BELIEF, MemoryType.ENTITY_SUMMARY}
            or memory.system_metadata.get("category") == "verbatim_turn"
            or unverified_representation(memory.system_metadata)
            or memory.temporal.status is not TemporalStatus.CURRENT
            or memory.deleted_at is not None
        )

    def _bounded_sources(
        self, subject: str, sources: list[CanonicalMemory]
    ) -> list[CanonicalMemory]:
        # Keep whole statements within the embedding/context budget, without
        # truncating away negations or a correction at the end of a source.
        bounded = []
        size = len(subject) + 64
        for source in sources:
            statement_size = len(source_statement(source)) + 1
            if size + statement_size <= self.cfg.consolidation_max_chars:
                bounded.append(source)
                size += statement_size
        return bounded
