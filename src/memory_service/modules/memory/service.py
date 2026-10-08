"""MemoryService: observations in (learned from later), stated memories in (stored now), and
read/list/supersede/forget under authorization.

Two write verbs, deliberately different. An *observation* is evidence: the pipeline decides
asynchronously what, if anything, it teaches. A *stated* memory (``remember``) is the caller
saying "this is true": stored verbatim, synchronously, as one memory, with no extraction or
admission gate - then indexed and linked into the graph off the request path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import Conflict, NotFound, ScopeDenied
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.fiscal import FiscalCalendar
from memory_service.domain.ids import content_hash, new_id
from memory_service.domain.memory import CanonicalMemory, TemporalState
from memory_service.domain.observation import ProcessingHints
from memory_service.domain.text import sanitise
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import validate_requested_visibility
from memory_service.modules.memory.native import default_visibility, normalized_hash
from memory_service.modules.memory.pipeline import (
    TASK_MEMORY_INDEX,
    build_memory,
    keys_for,
    supersede,
)
from memory_service.modules.memory.revisions import bump_memory_revisions
from memory_service.modules.tenancy.gate import guard_workspace_visibility
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork

if TYPE_CHECKING:
    from memory_service.modules.memory.forgetting import ForgettingService

#: A statement is taken at its word more than an extraction is; below 1.0 so that
#: corroboration can still raise it.
STATED_CONFIDENCE = 0.9
STATED_IMPORTANCE = 0.7
STATED_CATEGORY = "stated"


@dataclass(frozen=True)
class RememberAck:
    memory_id: str
    deduplicated: bool
    job_ids: list[str] = field(default_factory=list)


def _stated(
    ctx: MemoryExecutionContext, memory_id: str, content: str, now: datetime
) -> EvidenceRef:
    """Evidence for a stated memory: the principal's own statement, at this instant."""
    return EvidenceRef(
        source_type=EvidenceSource.STATEMENT,
        source_id=memory_id,
        agent_id=ctx.agent_id,
        agent_run_id=ctx.agent_run_id,
        observed_at=now,
        source_hash=content_hash(content),
    )


class MemoryService:
    def __init__(
        self, authz: AuthorizationService, *, fiscal: FiscalCalendar | None = None
    ) -> None:
        self.authz = authz
        #: the deployment's retail calendar, which a stated memory's fiscal phrases resolve in
        self.fiscal = fiscal

    async def remember(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        content: str,
        memory_type: MemoryType,
        lifetime: Lifetime,
        visibility: Visibility | None = None,
        subject: str | None = None,
        entities: Sequence[str] = (),
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> RememberAck:
        """Store ``content`` verbatim as one memory of the caller's, now. The same content in
        the same scope is the memory already stored (``deduplicated``), not a second one."""
        hints = ProcessingHints(visibility=visibility)
        validate_requested_visibility(ctx, hints)
        await guard_workspace_visibility(uow, self.authz, ctx, visibility)
        content = sanitise(content)
        now = datetime.now(UTC)
        memory_id = new_id("memory")
        about_user = memory_type in (MemoryType.USER, MemoryType.PREFERENCE) and ctx.user_id
        candidate = MemoryCandidate(
            content=content,
            memory_type=memory_type,
            lifetime=lifetime,
            # a user stating something is that user's own memory; an agent's keeps its rules
            visibility=visibility
            or default_visibility(memory_type, ctx, user_statement=not ctx.is_agent),
            subject=subject or (f"user:{ctx.user_id}" if about_user else None),
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=STATED_CONFIDENCE,
            importance=STATED_IMPORTANCE,
            evidence=[_stated(ctx, memory_id, content, now)],
            entities=list(dict.fromkeys(entities)),
            provider="statement",
            category=STATED_CATEGORY,
        )
        memory = build_memory(candidate, ctx, now=now, fiscal=self.fiscal).model_copy(
            update={"memory_id": memory_id, "custom_metadata": custom_metadata or {}}
        )
        scope_key = memory.scope.key()
        await uow.serialize(f"remember:{ctx.tenant_id}:{scope_key}:{memory.normalized_hash}")
        existing = await uow.memories.current_with_hash(
            ctx.tenant_id,
            scope_key=scope_key,
            owner_principal=memory.owner_principal,
            normalized_hash=memory.normalized_hash,
        )
        if existing is not None:
            return RememberAck(existing.memory_id, deduplicated=True)
        await uow.memories.add(
            memory, visibility_keys=keys_for(memory.scope, memory.visibility, ctx)
        )
        job_ids = await self._index(uow, ctx, [memory], key=f"memidx:{memory_id}")
        return RememberAck(memory_id, deduplicated=False, job_ids=job_ids)

    async def supersede(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        memory_id: str,
        *,
        content: str,
        reason: str,
    ) -> CanonicalMemory:
        """Replace a memory's content with a new version, bi-temporally: the new memory holds
        from now, the old one is closed at now and points at it, and both stay readable in a
        temporal view. The owner (or the user an agent acts for, or a tenant admin) only."""
        old = await self.get_memory(uow, ctx, memory_id)
        await self._require_owner(ctx, old, "supersede")
        if old.temporal.status is not TemporalStatus.CURRENT:
            raise Conflict(
                f"memory {memory_id} is {old.temporal.status.value}, not CURRENT",
                details={"superseded_by": old.temporal.superseded_by},
            )
        content = sanitise(content)
        now = datetime.now(UTC)
        new_memory_id = new_id("memory")
        new = CanonicalMemory(
            memory_id=new_memory_id,
            tenant_id=old.tenant_id,
            scope=old.scope,
            visibility=old.visibility,
            owner_principal=old.owner_principal,
            lifetime=old.lifetime,
            memory_type=old.memory_type,
            custom_type=old.custom_type,
            content=content,
            normalized_hash=normalized_hash(content),
            subject=old.subject,
            predicate=old.predicate,
            temporal=TemporalState(observed_at=now),
            evidence=[_stated(ctx, new_memory_id, content, now)],
            confidence=STATED_CONFIDENCE,
            importance=old.importance,
            system_metadata={
                "provider": "statement",
                "category": STATED_CATEGORY,
                "entities": list(old.system_metadata.get("entities") or []),
                "expires_at": old.system_metadata.get("expires_at"),
                "supersede_reason": reason,
            },
            custom_metadata=dict(old.custom_metadata),
            created_at=now,
            updated_at=now,
        )
        # the same statement kept twice (a turn and a rule's reading in its words) is
        # replaced once: both close at the new version
        closed = [old, *await uow.memories.twins(ctx.tenant_id, old)]
        supersede(old, new, now=now)
        for twin in closed[1:]:
            supersede(twin, new.model_copy(), now=now)
        for memory in closed:
            memory.system_metadata["supersede_reason"] = reason
        # the new version keeps exactly the audience the old one had
        audience = await uow.memories.visibility_keys(ctx.tenant_id, memory_id)
        await uow.memories.add(new, visibility_keys=audience)
        for memory in closed:
            await uow.memories.update(memory)
        await self._index(uow, ctx, [new, *closed], key=f"memidx:supersede:{new.memory_id}")
        return new

    async def _index(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        memories: list[CanonicalMemory],
        *,
        key: str,
    ) -> list[str]:
        """Queue the search index and graph enrichment for written memories and move the
        revisions their readers' caches are keyed on."""
        outbox_id = await uow.enqueue(
            JobSpec(
                task_name=TASK_MEMORY_INDEX,
                queue=Queue.EMBEDDING,
                payload={
                    "tenant_id": ctx.tenant_id,
                    "memory_ids": sorted(m.memory_id for m in memories),
                },
                idempotency_key=key,
                tenant_id=ctx.tenant_id,
            )
        )
        await bump_memory_revisions(uow, memories)
        return [f"obx_{outbox_id}"] if outbox_id is not None else []

    async def _require_owner(
        self, ctx: MemoryExecutionContext, memory: CanonicalMemory, action: str
    ) -> None:
        if memory.owner_principal in (ctx.principal_id, f"user:{ctx.user_id}"):
            return
        if not await self.authz.is_tenant_admin(ctx):
            raise ScopeDenied(
                f"only the owner (or a tenant admin) can {action} a memory",
                details={"principal": ctx.principal_id},
            )

    async def _visible(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, memory: CanonicalMemory
    ) -> bool:
        keys = await uow.memories.visibility_keys(ctx.tenant_id, memory.memory_id)
        spec = await self.authz.visibility(ctx, revisions=uow.revisions)
        return spec.allows(memory.tenant_id, keys)

    async def get_memory(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, memory_id: str
    ) -> CanonicalMemory:
        memory = await uow.memories.get(ctx.tenant_id, memory_id)
        if memory is None:
            raise NotFound(f"memory {memory_id} not found")
        if not await self._visible(uow, ctx, memory):
            # The message must not reveal whether the memory exists. ``principal`` is the
            # caller's own identity, so it leaks nothing — and it is the answer to the most
            # common cause of this 403: reading an agent's memory without naming the agent
            # (``?agent_id=``), which makes the caller ``user:alice`` rather than
            # ``agent:alice/research``.
            raise ScopeDenied(
                f"memory {memory_id} is not visible in this scope",
                details={"principal": ctx.principal_id},
            )
        return memory

    async def list_memories(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[CanonicalMemory]:
        """Memories anchored to the caller's agent, user, thread, agent group or tenant,
        newest created first. ``before`` is the (created_at, memory_id) keyset of the next
        page; the page holds ``limit`` *visible* rows, however many hidden rows lie between
        them, and a row written or reinforced during the walk is never skipped because the
        keyset never moves."""
        from memory_service.domain.enums import ScopeLevel
        from memory_service.domain.memory import Scope

        anchors: list[Scope] = []
        if ctx.agent_id:
            anchors.append(
                Scope(
                    level=ScopeLevel.AGENT,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    thread_id=ctx.thread_id,
                    agent_id=ctx.agent_id,
                )
            )
        if ctx.user_id:
            anchors.append(
                Scope(
                    level=ScopeLevel.USER,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                )
            )
        if ctx.thread_id:
            anchors.append(
                Scope(
                    level=ScopeLevel.THREAD,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    thread_id=ctx.thread_id,
                )
            )
        if ctx.agent_group_id:
            anchors.append(
                Scope(
                    level=ScopeLevel.AGENT_GROUP,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    agent_group_id=ctx.agent_group_id,
                )
            )
        if ctx.workspace_id:
            # the team's shared memory; what the caller may read of it, ``spec`` decides
            anchors.append(
                Scope(
                    level=ScopeLevel.WORKSPACE,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                )
            )
        # Everyone in the tenant shares this one, so it is always an anchor. The WORKSPACE
        # anchor used to stand in for it whenever a caller had a workspace, which is why its
        # absence was not noticed until workspace stopped being an audience.
        anchors.append(
            Scope(
                level=ScopeLevel.TENANT,
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
            )
        )
        spec = await self.authz.visibility(ctx, revisions=uow.revisions)
        scope_keys = [s.key() for s in anchors]
        out: list[CanonicalMemory] = []
        batch = max(limit * 2, 50)
        while len(out) < limit:
            rows = await uow.memories.list_scope(
                ctx.tenant_id,
                scope_keys=scope_keys,
                memory_types=memory_types,
                current_only=not include_superseded,
                before=before,
                limit=batch,
            )
            for m in rows:
                if spec.allows(m.tenant_id, m.system_metadata.get("visibility_keys", [])):
                    out.append(m)
                if len(out) >= limit:
                    break
            if len(rows) < batch:
                break  # the store is exhausted: fewer than a batch came back
            before = (rows[-1].created_at, rows[-1].memory_id)
        return out

    async def restore(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        memory_id: str,
        forgetting: ForgettingService,
    ) -> CanonicalMemory:
        """Bring back a memory automatic forgetting archived: CURRENT and searchable again.
        The same people who may forget a memory may restore it. A memory that is not
        archived is returned as it is; one that was deleted stays deleted (404)."""
        memory = await self.get_memory(uow, ctx, memory_id)
        await self._require_owner(ctx, memory, "restore")
        restored = await forgetting.restore(uow, ctx.tenant_id, memory_id)
        return restored or memory

    async def forget(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, memory_id: str
    ) -> CanonicalMemory | None:
        """Soft-delete. Allowed for the owner principal, the user an agent acts for, or a
        tenant admin; the search index entry is removed by the index job. Idempotent: a
        memory that is already forgotten returns None instead of NotFound."""
        if await uow.memories.is_forgotten(ctx.tenant_id, memory_id):
            return None
        memory = await self.get_memory(uow, ctx, memory_id)
        await self._require_owner(ctx, memory, "forget")
        # the same statement kept twice (a turn and a rule's reading in its words) is
        # forgotten once: the twin left behind would still answer with the forgotten words
        twins = await uow.memories.twins(ctx.tenant_id, memory, current_only=False)
        for gone in (memory, *twins):
            await uow.memories.forget(ctx.tenant_id, gone.memory_id)
        await self._index(uow, ctx, [memory, *twins], key=f"memidx:forget:{memory_id}")
        return memory
