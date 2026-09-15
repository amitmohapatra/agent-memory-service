"""ReflectionService: higher-level insights over a principal's recent memories.

There is no native counterpart: reflection exists only when the ``reflection`` LLM use is
enabled. For every (tenant, principal) with memories created inside the window the model is
asked for a few insights; each one becomes a memory of its own with evidence pointing at
the source memories, and is indexed through the same ``memory.index`` job as any other
memory. A scope is reflected again only once it has memories newer than its last insight,
and an insight whose normalized content already exists in the scope is not stored twice.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.pipeline import TASK_MEMORY_INDEX, build_memory, keys_for
from memory_service.observability.logging import get_logger
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_MEMORY_REFLECT = "memory.reflect"
REFLECTION_CATEGORY = "reflection"

_MEMORY_TYPES = [t.value for t in MemoryType if t is not MemoryType.CUSTOM]
_SYSTEM = (
    "You reflect on a principal's recent memories and derive up to {n} higher-level insights: "
    "patterns, stable preferences, goals or facts that several memories support together. "
    "Each insight is one compact statement in the third person, must be grounded only in the "
    "listed memories, and cites the ids it is derived from. Do not restate a single memory; "
    "return an empty list when nothing meaningful emerges. memory_type is one of USER, "
    "PREFERENCE, SEMANTIC, PROCEDURAL, EPISODIC or TASK."
)
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["insights"],
    "properties": {
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["content", "memory_type", "source_memory_ids"],
                "properties": {
                    "content": {"type": "string"},
                    "memory_type": {"type": "string", "enum": _MEMORY_TYPES},
                    "source_memory_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}
_MAX_PROMPT_CHARS = 6000
_MAX_MEMORY_CHARS = 200
_MAX_INSIGHT_CHARS = 1000


class ReflectionService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        *,
        assist: LLMAssist | None = None,
        window: timedelta = timedelta(hours=24),
        max_memories: int = 40,
        max_insights: int = 3,
        scan_limit: int = 1000,
    ) -> None:
        self.uow_factory = uow_factory
        self.assist = assist or LLMAssist.disabled()
        self.window = window
        self.max_memories = max_memories
        self.max_insights = max_insights
        self.scan_limit = scan_limit

    async def reflect_all(self, *, now: datetime | None = None) -> list[str]:
        """Reflect every (tenant, principal) with fresh memories; returns new memory ids."""
        if not self.assist.wants("reflection"):
            return []
        now = now or datetime.now(UTC)
        async with self.uow_factory() as uow:
            recent = await uow.memories.list_recent(since=now - self.window, limit=self.scan_limit)
        sources: dict[tuple[str, str], list[CanonicalMemory]] = {}
        reflected_at: dict[tuple[str, str], datetime] = {}
        for m in recent:
            key = (m.tenant_id, m.owner_principal)
            if m.system_metadata.get("category") == REFLECTION_CATEGORY:
                reflected_at[key] = max(reflected_at.get(key, m.created_at), m.created_at)
            else:
                sources.setdefault(key, []).append(m)
        created: list[str] = []
        for key, mems in sources.items():
            last = reflected_at.get(key)
            if last is not None and all(m.created_at <= last for m in mems):
                continue
            created += await self.reflect(key[0], key[1], mems[: self.max_memories], now=now)
        return created

    async def reflect(
        self,
        tenant_id: str,
        principal: str,
        memories: list[CanonicalMemory],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        if not memories or not self.assist.wants("reflection"):
            return []
        now = now or datetime.now(UTC)
        out = await self.assist.structured(
            "reflection",
            system=_SYSTEM.format(n=self.max_insights),
            user=self._prompt(principal, memories),
            schema=_SCHEMA,
            max_tokens=600,
        )
        if out is None:
            return []
        by_id = {m.memory_id: m for m in memories}
        ctx = _context_for(tenant_id, principal, memories)
        stored: list[str] = []
        async with self.uow_factory() as uow:
            for raw in list(out.get("insights") or [])[: self.max_insights]:
                memory = self._insight(raw, by_id, ctx, now=now)
                if memory is None or await self._exists(uow, memory):
                    continue
                await uow.memories.add(
                    memory, visibility_keys=keys_for(memory.scope, memory.visibility, ctx)
                )
                stored.append(memory.memory_id)
            if stored:
                await uow.enqueue(
                    JobSpec(
                        task_name=TASK_MEMORY_INDEX,
                        queue=Queue.EMBEDDING,
                        payload={"tenant_id": tenant_id, "memory_ids": sorted(stored)},
                        idempotency_key=f"memidx:reflect:{stored[0]}",
                        tenant_id=tenant_id,
                    )
                )
                if ctx.user_id:
                    await uow.revisions.bump(tenant_id, RevisionKind.USER, ctx.user_id)
                if ctx.agent_id:
                    await uow.revisions.bump(tenant_id, RevisionKind.AGENT, ctx.agent_id)
                await uow.commit()
        if stored:
            log.info(
                "memory.reflected", tenant_id=tenant_id, principal=principal, count=len(stored)
            )
        return stored

    @staticmethod
    def _prompt(principal: str, memories: list[CanonicalMemory]) -> str:
        lines = [f"Principal: {principal}", "Recent memories (id | type | content):"]
        size = sum(len(line) for line in lines)
        for m in memories:
            content = re.sub(r"\s+", " ", m.content)[:_MAX_MEMORY_CHARS]
            line = f"- {m.memory_id} | {m.memory_type.value} | {content}"
            if size + len(line) > _MAX_PROMPT_CHARS:
                break
            lines.append(line)
            size += len(line)
        return "\n".join(lines)

    def _insight(
        self,
        raw: Any,
        by_id: dict[str, CanonicalMemory],
        ctx: MemoryExecutionContext,
        *,
        now: datetime,
    ) -> CanonicalMemory | None:
        if not isinstance(raw, dict):
            return None
        content = raw.get("content")
        mt = raw.get("memory_type")
        if not isinstance(content, str) or not isinstance(mt, str) or mt not in _MEMORY_TYPES:
            return None
        content = re.sub(r"\s+", " ", content).strip()[:_MAX_INSIGHT_CHARS]
        ids = raw.get("source_memory_ids")
        source_ids = sorted({i for i in ids if isinstance(i, str) and i in by_id}) if ids else []
        if not content or not source_ids:
            return None
        sources = [by_id[i] for i in source_ids]
        candidate = MemoryCandidate(
            content=content,
            memory_type=MemoryType(mt),
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.PRIVATE,
            subject=f"user:{ctx.user_id}" if ctx.user_id else ctx.principal_id,
            predicate="insight",
            evidence=[
                EvidenceRef(
                    source_type="memory",
                    source_id=m.memory_id,
                    observed_at=m.temporal.observed_at,
                )
                for m in sources
            ],
            confidence=0.6,
            importance=0.6,
            category=REFLECTION_CATEGORY,
        )
        memory = build_memory(candidate, ctx, now=now)
        memory.system_metadata["source_memory_ids"] = source_ids
        memory.system_metadata["contributors"] = sorted({m.owner_principal for m in sources})
        return memory

    @staticmethod
    async def _exists(uow: UnitOfWork, memory: CanonicalMemory) -> bool:
        existing = await uow.memories.candidates(
            memory.tenant_id,
            scope_key=memory.scope.key(),
            normalized_hash=memory.normalized_hash,
            limit=5,
        )
        return any(m.normalized_hash == memory.normalized_hash for m in existing)


def _context_for(
    tenant_id: str, principal: str, memories: list[CanonicalMemory]
) -> MemoryExecutionContext:
    """The principal's own context, rebuilt from its id and the anchors of its memories."""
    kind, _, ident = principal.partition(":")
    workspace_id = next((m.scope.workspace_id for m in memories if m.scope.workspace_id), None)
    if kind == "user":
        return MemoryExecutionContext(tenant_id=tenant_id, workspace_id=workspace_id, user_id=ident)
    if kind == "agent":
        user_id = next((m.scope.user_id for m in memories if m.scope.user_id), None)
        return MemoryExecutionContext(
            tenant_id=tenant_id, workspace_id=workspace_id, user_id=user_id, agent_id=ident
        )
    return MemoryExecutionContext(tenant_id=tenant_id, workspace_id=workspace_id)
