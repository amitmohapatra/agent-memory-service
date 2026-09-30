"""Derived memories: what the background ReflectionService writes from several sources.

A derived memory carries its supporting memory ids and a ``revised_from`` chain; it is
revised, never duplicated. Its access is the intersection of its sources' audiences,
including incomparable RUN/THREAD/GROUP scopes, and is checked again when sources are
persisted. (The on-landing consolidation that minted extractive beliefs and entity
summaries here was measured and removed: docs/MEASUREMENTS.md, section 8.6.)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import (
    CanonicalMemory,
    Scope,
    TemporalState,
)
from memory_service.modules.memory.native import normalized_hash
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


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
