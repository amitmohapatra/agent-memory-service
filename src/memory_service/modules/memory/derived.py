"""Derived memories: beliefs and entity summaries.

A *belief* is a generalisation over several supporting memories (same subject and
multi-valued predicate, or a model insight): it carries a confidence, the supporting
memory ids and a ``revised_from`` chain. A belief is revised, never duplicated: a new
version supersedes the previous one and points back at it.

An *entity summary* is the one maintained memory per entity (``subject``), rebuilt
deterministically from the entity's current facts and replaced (SUPERSEDE) whenever those
facts change. With the ``summaries`` LLM use enabled the deterministic text is refined into
one paragraph; the deterministic text stays stored and is what decides whether a rebuild
is needed, so re-running on unchanged facts changes nothing.

Both follow the *narrowest source*: a derived memory is never more visible than the least
visible memory it was built from.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from statistics import fmean
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, TemporalStatus, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.modules.context.summaries import SUMMARY_SCHEMA, accept_abstractive
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.native import _clean_object, normalized_hash
from memory_service.modules.memory.pipeline import keys_for
from memory_service.observability.logging import get_logger
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)

BELIEF_CATEGORY = "belief"
ENTITY_SUMMARY_CATEGORY = "entity_summary"

_NARROWNESS: dict[Visibility, int] = {
    Visibility.PRIVATE: 0,
    Visibility.RUN: 1,
    Visibility.USER: 2,
    Visibility.GROUP: 3,
    Visibility.AGENT_GROUP: 4,
    Visibility.THREAD: 5,
    Visibility.WORK: 6,
    Visibility.WORKSPACE: 7,
    Visibility.TENANT: 8,
    Visibility.GLOBAL: 9,
}
_ENTITY_SUMMARY_SYSTEM = (
    "You write the one-paragraph profile of an entity for a memory index, from a list of its "
    "current facts. Keep every fact and value, add nothing, at most {max_chars} characters, "
    'no preamble. Return JSON only: {{"summary": "..."}}.'
)
_SUMMARY_MAX_CHARS = 700
_MAX_VALUES = 12


def narrowest_visibility(sources: Sequence[CanonicalMemory]) -> Visibility:
    return min(
        (m.visibility for m in sources), key=lambda v: _NARROWNESS[v], default=Visibility.PRIVATE
    )


def _value_of(m: CanonicalMemory) -> str:
    obj = _clean_object(m.object or "")
    return obj or " ".join(m.content.split())[:120]


def _label(text: str) -> str:
    return text.replace("_", " ")


def _memory_evidence(sources: Sequence[CanonicalMemory]) -> list[EvidenceRef]:
    return [
        EvidenceRef(source_type="memory", source_id=m.memory_id, observed_at=m.temporal.observed_at)
        for m in sources
    ]


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
    visibility = narrowest_visibility(sources)
    try:
        keys = keys_for(scope, visibility, ctx)
    except ValueError:
        visibility = Visibility.PRIVATE
        keys = keys_for(scope, visibility, ctx)
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
        temporal=TemporalState(observed_at=now, valid_from=now),
        evidence=_memory_evidence(sources),
        confidence=round(min(1.0, confidence), 4),
        importance=round(min(1.0, importance), 4),
        system_metadata={
            "provider": "native",
            "category": category,
            "entities": [subject],
            "supporting_memory_ids": sorted(m.memory_id for m in sources),
            "contributors": sorted({m.owner_principal for m in sources}),
            **extra,
        },
        created_at=now,
        updated_at=now,
    )
    return memory, keys


async def _supersede(
    uow: UnitOfWork, old: CanonicalMemory, new: CanonicalMemory, *, now: datetime
) -> None:
    new.temporal = new.temporal.model_copy(update={"supersedes": old.memory_id})
    old.temporal = old.temporal.model_copy(
        update={
            "status": TemporalStatus.SUPERSEDED,
            "superseded_by": new.memory_id,
            "valid_to": old.temporal.valid_to or now,
        }
    )
    old.updated_at = now
    await uow.memories.update(old)


class BeliefService:
    """Revise-not-duplicate upsert of one belief per (scope, subject, predicate)."""

    @staticmethod
    def derive_content(subject: str, predicate: str, sources: Sequence[CanonicalMemory]) -> str:
        ordered = sorted(sources, key=lambda m: (m.created_at, m.memory_id))
        values: list[str] = []
        for m in ordered:
            v = _value_of(m)
            if v not in values:
                values.append(v)
        listed = "; ".join(values[:_MAX_VALUES])
        return f"{subject} {_label(predicate)}: {listed} ({len(ordered)} supporting memories)"

    async def current(
        self, uow: UnitOfWork, tenant_id: str, *, scope: Scope, subject: str, predicate: str
    ) -> CanonicalMemory | None:
        rows = await uow.memories.list_scope(
            tenant_id,
            scope_keys=[scope.key()],
            memory_types=[MemoryType.BELIEF.value],
            limit=200,
        )
        return next((m for m in rows if m.subject == subject and m.predicate == predicate), None)

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
        existing = await self.current(
            uow, ctx.tenant_id, scope=scope, subject=subject, predicate=predicate
        )
        if existing is not None and existing.normalized_hash == normalized_hash(content):
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
            await _supersede(uow, existing, belief, now=now)
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

    def __init__(self, assist: LLMAssist | None = None) -> None:
        self.assist = assist or LLMAssist.disabled()

    @staticmethod
    def derive_content(subject: str, facts: Sequence[CanonicalMemory]) -> str:
        ordered = sorted(facts, key=lambda m: (m.predicate or "~", m.created_at, m.memory_id))
        parts: list[str] = []
        for m in ordered:
            piece = f"{_label(m.predicate)} {_value_of(m)}" if m.predicate else _value_of(m)
            if piece not in parts:
                parts.append(piece)
        return f"{subject}: " + "; ".join(parts)

    async def current(
        self, uow: UnitOfWork, tenant_id: str, *, scope: Scope, subject: str
    ) -> CanonicalMemory | None:
        rows = await uow.memories.list_scope(
            tenant_id,
            scope_keys=[scope.key()],
            memory_types=[MemoryType.ENTITY_SUMMARY.value],
            limit=200,
        )
        return next((m for m in rows if m.subject == subject), None)

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
        existing = await self.current(uow, ctx.tenant_id, scope=scope, subject=subject)
        if not facts:
            return existing, False
        extractive = self.derive_content(subject, facts)
        if existing is not None and existing.system_metadata.get("extractive") == extractive:
            return existing, False
        content = extractive
        if self.assist.wants("summaries"):
            refined = await self.assist.structured(
                "summaries",
                system=_ENTITY_SUMMARY_SYSTEM.format(max_chars=_SUMMARY_MAX_CHARS),
                user=f"Entity: {subject}\nFacts:\n" + "\n".join(f"- {m.content}" for m in facts),
                schema=SUMMARY_SCHEMA,
                max_tokens=max(128, _SUMMARY_MAX_CHARS // 2),
            )
            content = accept_abstractive(refined, max_chars=_SUMMARY_MAX_CHARS) or extractive
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
            await _supersede(uow, existing, summary, now=now)
        await uow.memories.add(summary, visibility_keys=keys)
        return summary, True
