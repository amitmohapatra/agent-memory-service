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
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    ObservationKind,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import Conflict, NotFound, ScopeDenied
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import content_hash, new_id
from memory_service.domain.memory import CanonicalMemory, TemporalState
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.domain.text import sanitise
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import validate_requested_visibility
from memory_service.modules.conversation.service import TASK_PROCESS_OBSERVATION
from memory_service.modules.memory.native import default_visibility, normalized_hash
from memory_service.modules.memory.pipeline import (
    TASK_MEMORY_INDEX,
    build_memory,
    created_event,
    keys_for,
    supersede,
)
from memory_service.modules.memory.revisions import bump_memory_revisions, retract
from memory_service.modules.tenancy.gate import guard_workspace_visibility
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork
from memory_service.ports.webhooks import EventPublisher

#: A statement is taken at its word more than an extraction is; below 1.0 so that
#: corroboration can still raise it.
STATED_CONFIDENCE = 0.9
STATED_IMPORTANCE = 0.7
STATED_CATEGORY = "stated"


@dataclass(frozen=True)
class ObservationAck:
    observation_id: str
    job_ids: list[str] = field(default_factory=list)


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
        source_type="statement",
        source_id=memory_id,
        agent_id=ctx.agent_id,
        agent_run_id=ctx.agent_run_id,
        observed_at=now,
        source_hash=content_hash(content),
    )


class MemoryService:
    def __init__(self, authz: AuthorizationService, events: EventPublisher | None = None) -> None:
        self.authz = authz
        self.events = events

    async def submit_observation(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        kind: ObservationKind,
        content: str,
        hints: ProcessingHints | None = None,
        custom_metadata: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
        source_system: str | None = None,
        source_id: str | None = None,
        tool_run_id: str | None = None,
    ) -> ObservationAck:
        validate_requested_visibility(ctx, hints)
        # A team's shared memory is written by its members. The anchor alone is a
        # caller-supplied string; without this, anyone in the tenant could publish into any
        # team and every member would read it.
        await guard_workspace_visibility(uow, self.authz, ctx, getattr(hints, "visibility", None))
        # Agents write back whatever their tools produced. A NUL byte anywhere in that text
        # makes PostgreSQL reject the INSERT outright, so a single stray 0x00 in a tool result
        # turned a write into a 500 instead of a stored observation. The document path has
        # been sanitising since an uploaded file did the same thing; this one never was.
        content = sanitise(content)
        observation = Observation(
            **ctx.provenance(),
            kind=kind,
            content=content,
            # hashed *after* sanitising, so the same text submitted twice — once with a stray
            # control character, once without — is recognised as the duplicate it is
            content_hash=content_hash(content),
            tool_run_id=tool_run_id,
            source_system=source_system,
            source_id=source_id,
            hints=hints or ProcessingHints(),
            custom_metadata=custom_metadata or {},
            occurred_at=occurred_at or datetime.now(UTC),
        )
        await uow.observations.add(observation)
        outbox_id = await uow.enqueue(
            JobSpec(
                task_name=TASK_PROCESS_OBSERVATION,
                queue=Queue.CHAT_FAST,
                payload={
                    "tenant_id": ctx.tenant_id,
                    "observation_id": observation.observation_id,
                    "trace_id": ctx.trace_id,
                },
                idempotency_key=f"obs:{observation.observation_id}",
                tenant_id=ctx.tenant_id,
            )
        )
        return ObservationAck(
            observation.observation_id, [f"obx_{outbox_id}"] if outbox_id is not None else []
        )

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
            visibility=visibility or default_visibility(memory_type, ctx),
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
        memory = build_memory(candidate, ctx, now=now).model_copy(
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
        if self.events is not None:
            await self.events.publish(uow, created_event(memory))
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
        supersede(old, new, now=now)
        old.system_metadata["supersede_reason"] = reason
        # the new version keeps exactly the audience the old one had
        audience = await uow.memories.visibility_keys(ctx.tenant_id, memory_id)
        await uow.memories.add(new, visibility_keys=audience)
        await uow.memories.update(old)
        await self._index(uow, ctx, [new, old], key=f"memidx:supersede:{new.memory_id}")
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

    async def retract(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, memory_id: str, *, reason: str
    ) -> CanonicalMemory:
        """Withdraw a memory that is no longer true without replacing it: it leaves retrieval
        and stays readable in a temporal view. Same rule as supersede: the owner (or the user
        an agent acts for, or a tenant admin), and only while it is CURRENT."""
        memory = await self.get_memory(uow, ctx, memory_id)
        await self._require_owner(ctx, memory, "retract")
        if memory.temporal.status is not TemporalStatus.CURRENT:
            raise Conflict(f"memory {memory_id} is {memory.temporal.status.value}, not CURRENT")
        memory.system_metadata["retract_reason"] = reason
        await retract(
            uow, memory, now=datetime.now(UTC), events=self.events, data={"reason": reason}
        )
        await self._index(uow, ctx, [memory], key=f"memidx:retract:{memory_id}")
        return memory

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
        await uow.memories.forget(ctx.tenant_id, memory_id)
        await self._index(uow, ctx, [memory], key=f"memidx:forget:{memory_id}")
        return memory
