"""Derived memories: beliefs and entity summaries.

A *belief* is a generalisation over several supporting memories (same subject and
multi-valued predicate, or a model insight): it carries a confidence, the supporting
memory ids and a ``revised_from`` chain. A belief is revised, never duplicated: a new
version supersedes the previous one and points back at it.

An *entity summary* is the one maintained memory per entity (``subject``), rebuilt
deterministically from the entity's current facts and replaced (SUPERSEDE) whenever those
facts change. These extractive representations preserve complete source statements;
model-generated insights are produced separately by the background ReflectionService.

Derived access uses the intersection of source audiences, including incomparable
RUN/THREAD/GROUP scopes, and is checked again when sources are persisted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from statistics import fmean
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import (
    CanonicalMemory,
    Scope,
    TemporalState,
    aggregate_statement,
    dated_statement,
)
from memory_service.modules.memory.native import normalized_hash
from memory_service.modules.memory.revisions import supersede
from memory_service.observability.logging import get_logger
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)

BELIEF_CATEGORY = "belief"
ENTITY_SUMMARY_CATEGORY = "entity_summary"


def source_audience(sources: Sequence[CanonicalMemory]) -> list[str]:
    """Audience intersection, since RUN, THREAD, USER and GROUP are not a total order."""
    return (
        sorted(
            set.intersection(*[set(m.system_metadata.get("visibility_keys", [])) for m in sources])
        )
        if sources
        else []
    )


def source_slot(
    ctx: MemoryExecutionContext,
    scope: Scope,
    subject: str,
    predicate: str,
    memory_type: MemoryType,
    sources: Sequence[CanonicalMemory],
) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                scope.key(),
                ctx.principal_id,
                subject,
                predicate,
                memory_type.value,
                source_audience(sources),
            ]
        ).encode()
    ).hexdigest()


def _memory_evidence(sources: Sequence[CanonicalMemory]) -> list[EvidenceRef]:
    return [
        EvidenceRef(source_type="memory", source_id=m.memory_id, observed_at=m.temporal.observed_at)
        for m in sources
    ]


def source_statement(memory: CanonicalMemory) -> str:
    return dated_statement(memory.temporal.observed_at.date().isoformat(), memory.content)


def _dated(sources: Sequence[CanonicalMemory]) -> list[tuple[str, str]]:
    """``(day, content)`` pairs: what the shared aggregate formatter reads."""
    return [(m.temporal.observed_at.date().isoformat(), m.content) for m in sources]


def _source_statements(sources: Sequence[CanonicalMemory]) -> str:
    """Keep each relative-date statement beside its own observation date."""
    return "\n".join(dict.fromkeys(source_statement(m) for m in sources))


def _derived(
    ctx: MemoryExecutionContext,
    *,
    memory_type: MemoryType,
    scope: Scope,
    sources: Sequence[CanonicalMemory],
    content: str,
    subject: str,
    predicate: str,
    confidence: float,
    importance: float,
    now: datetime,
    category: str,
    extra: dict[str, Any],
) -> tuple[CanonicalMemory, list[str]]:
    if not sources:
        raise ValueError("Derived memory requires source memories")
    visibility = sources[0].visibility
    keys = source_audience(sources)
    if not keys or any(m.tenant_id != ctx.tenant_id for m in sources):
        raise ValueError("Derived sources need a common audience within one tenant")
    slot = source_slot(ctx, scope, subject, predicate, memory_type, sources)
    expiries = [m.system_metadata.get("expires_at") for m in sources]
    expiry = min((datetime.fromisoformat(x) for x in expiries if isinstance(x, str)), default=None)
    observed = [m.temporal.observed_at for m in sources]
    first_observed, last_observed = min(observed), max(observed)
    memory = CanonicalMemory(
        tenant_id=ctx.tenant_id,
        scope=scope,
        visibility=visibility,
        owner_principal=ctx.principal_id,
        lifetime=Lifetime.LONG_TERM,
        memory_type=memory_type,
        content=content,
        normalized_hash=normalized_hash(content),
        subject=subject,
        predicate=predicate,
        # A synthesis can span distinct events: it has no single inferred valid interval.
        # Its evidence clock is the newest source, while creation time remains `now`.
        temporal=TemporalState(observed_at=last_observed),
        evidence=_memory_evidence(sources),
        confidence=round(min(1.0, confidence), 4),
        importance=round(min(1.0, importance), 4),
        system_metadata={
            "provider": "native",
            "category": category,
            "entities": [subject],
            "supporting_memory_ids": sorted(m.memory_id for m in sources),
            "contributors": sorted({m.owner_principal for m in sources}),
            "derived_slot": slot,
            "source_revisions": {m.memory_id: m.revision for m in sources},
            "source_observed_from": first_observed.isoformat(),
            "source_observed_to": last_observed.isoformat(),
            "expires_at": expiry.isoformat() if expiry else None,
            **extra,
        },
        created_at=now,
        updated_at=now,
    )
    return memory, keys


class BeliefService:
    """Revise-not-duplicate upsert of one belief per (scope, subject, predicate)."""

    @staticmethod
    def derive_content(subject: str, predicate: str, sources: Sequence[CanonicalMemory]) -> str:
        ordered = sorted(sources, key=lambda m: (m.temporal.observed_at, m.memory_id))
        return aggregate_statement(subject, predicate, _dated(ordered))

    async def upsert(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        scope: Scope,
        subject: str,
        predicate: str,
        sources: Sequence[CanonicalMemory],
        content: str | None = None,
        confidence: float | None = None,
        now: datetime,
        source: str = "native",
    ) -> tuple[CanonicalMemory, bool]:
        """Returns the current belief and whether anything changed."""
        content = content or self.derive_content(subject, predicate, sources)
        if confidence is None:
            n = len(sources)
            confidence = min(0.95, 0.4 + 0.1 * n) * fmean(m.confidence for m in sources)
        slot = source_slot(ctx, scope, subject, predicate, MemoryType.BELIEF, sources)
        await uow.serialize(f"derived:{ctx.tenant_id}:{slot}")
        existing = await uow.memories.current_derived(ctx.tenant_id, slot)
        if (
            existing is not None
            and existing.normalized_hash == normalized_hash(content)
            and existing.system_metadata.get("source_revisions")
            == {m.memory_id: m.revision for m in sources}
        ):
            return existing, False
        chain = (
            [*existing.system_metadata.get("revision_chain", []), existing.memory_id]
            if existing is not None
            else []
        )
        belief, keys = _derived(
            ctx,
            memory_type=MemoryType.BELIEF,
            scope=scope,
            sources=sources,
            content=content,
            subject=subject,
            predicate=predicate,
            confidence=confidence,
            importance=max((m.importance for m in sources), default=0.5),
            now=now,
            category=BELIEF_CATEGORY,
            extra={
                "belief_source": source,
                "revised_from": existing.memory_id if existing is not None else None,
                "revision_chain": chain,
            },
        )
        if existing is not None:
            await supersede(uow, existing, belief, now=now)
        await uow.memories.add(belief, visibility_keys=keys)
        log.info(
            "memory.belief_%s" % ("revised" if existing else "created"),
            tenant_id=ctx.tenant_id,
            subject=subject,
            predicate=predicate,
            support=len(sources),
        )
        return belief, True


class EntitySummaryService:
    """One maintained ENTITY_SUMMARY per (scope, subject), rebuilt from current facts."""

    @staticmethod
    def derive_content(subject: str, facts: Sequence[CanonicalMemory]) -> str:
        ordered = sorted(
            facts, key=lambda m: (m.predicate or "~", m.temporal.observed_at, m.memory_id)
        )
        return f"{subject} — recent source statements:\n" + _source_statements(ordered)

    async def rebuild(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        scope: Scope,
        subject: str,
        facts: Sequence[CanonicalMemory],
        now: datetime,
    ) -> tuple[CanonicalMemory | None, bool]:
        """Returns the current summary (None when there are no facts) and whether it changed."""
        slot = source_slot(ctx, scope, subject, "summary", MemoryType.ENTITY_SUMMARY, facts)
        await uow.serialize(f"derived:{ctx.tenant_id}:{slot}")
        existing = await uow.memories.current_derived(ctx.tenant_id, slot)
        if not facts:
            return existing, False
        extractive = self.derive_content(subject, facts)
        if (
            existing is not None
            and existing.system_metadata.get("extractive") == extractive
            and existing.system_metadata.get("source_revisions")
            == {m.memory_id: m.revision for m in facts}
        ):
            return existing, False
        content = extractive
        summary, keys = _derived(
            ctx,
            memory_type=MemoryType.ENTITY_SUMMARY,
            scope=scope,
            sources=facts,
            content=content,
            subject=subject,
            predicate="summary",
            confidence=fmean(m.confidence for m in facts),
            importance=max(m.importance for m in facts),
            now=now,
            category=ENTITY_SUMMARY_CATEGORY,
            extra={"extractive": extractive, "fact_count": len(facts)},
        )
        if existing is not None:
            await supersede(uow, existing, summary, now=now)
        await uow.memories.add(summary, visibility_keys=keys)
        return summary, True
