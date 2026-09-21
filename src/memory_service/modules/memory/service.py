"""MemoryService: submit observations, read/list/forget memories under authorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.domain.errors import NotFound, ScopeDenied, ValidationFailed
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.text import sanitise
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import visibility_keys
from memory_service.modules.conversation.service import TASK_PROCESS_OBSERVATION
from memory_service.modules.memory.pipeline import TASK_MEMORY_INDEX
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork


@dataclass(frozen=True)
class ObservationAck:
    observation_id: str
    job_ids: list[str] = field(default_factory=list)


def _validate_visibility(ctx: MemoryExecutionContext, hints: ProcessingHints | None) -> None:
    """Reject a requested visibility this context cannot satisfy, at submission time.

    Audience keys are built from the context's anchors, so a visibility whose anchor is
    missing (AGENT_GROUP without an agent group, WORKSPACE without a workspace...) cannot be
    expressed. Without this check the observation is acknowledged with a 202 and the failure
    surfaces only when ``memory.process_observation`` runs — by which time the caller is long
    gone and no memory was ever created. Validating here turns silent data loss into a 422
    that names the missing anchor.
    """
    requested = getattr(hints, "visibility", None) if hints is not None else None
    if requested is None:
        return
    try:
        visibility_keys(
            ctx.tenant_id,
            requested,
            owner_principal=ctx.principal_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            group_id=ctx.group_ids[0] if ctx.group_ids else None,
            thread_id=ctx.thread_id,
            work_id=ctx.work_id,
            agent_group_id=ctx.agent_group_id,
            agent_run_id=ctx.agent_run_id,
        )
    except ValueError as exc:
        raise ValidationFailed(str(exc), details={"visibility": str(requested)}) from exc


class MemoryService:
    def __init__(self, authz: AuthorizationService) -> None:
        self.authz = authz

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
        _validate_visibility(ctx, hints)
        # Agents write back whatever their tools produced. A NUL byte anywhere in that text
        # makes PostgreSQL reject the INSERT outright, so a single stray 0x00 in a tool result
        # turned a write into a 500 instead of a stored observation. The document path has
        # been sanitising since an uploaded file did the same thing; this one never was.
        content = sanitise(content)
        observation = Observation(
            tenant_id=ctx.tenant_id,
            kind=kind,
            content=content,
            # hashed *after* sanitising, so the same text submitted twice — once with a stray
            # control character, once without — is recognised as the duplicate it is
            content_hash=content_hash(content),
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            thread_id=ctx.thread_id,
            session_id=ctx.session_id,
            turn_id=ctx.turn_id,
            work_id=ctx.work_id,
            task_id=ctx.task_id,
            agent_id=ctx.agent_id,
            agent_group_id=ctx.agent_group_id,
            agent_run_id=ctx.agent_run_id,
            parent_agent_run_id=ctx.parent_agent_run_id,
            principal_id=ctx.principal_id,
            trace_id=ctx.trace_id,
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
        limit: int = 100,
    ) -> list[CanonicalMemory]:
        """Memories anchored to the caller's own scopes (user, thread, agent, work, workspace)."""
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
        if ctx.work_id:
            anchors.append(
                Scope(
                    level=ScopeLevel.WORK,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    work_id=ctx.work_id,
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
            anchors.append(
                Scope(
                    level=ScopeLevel.WORKSPACE,
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                )
            )
        rows = await uow.memories.list_scope(
            ctx.tenant_id,
            scope_keys=[s.key() for s in anchors],
            memory_types=memory_types,
            current_only=not include_superseded,
            limit=limit * 2,
        )
        spec = await self.authz.visibility(ctx, revisions=uow.revisions)
        out = []
        for m in rows:
            keys = m.system_metadata.get("visibility_keys", [])
            if spec.allows(m.tenant_id, keys):
                out.append(m)
            if len(out) >= limit:
                break
        return out

    async def forget(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, memory_id: str
    ) -> CanonicalMemory | None:
        """Soft-delete. Allowed for the owner principal, the user an agent acts for, or a
        tenant admin; the search index entry is removed by the index job. Idempotent: a
        memory that is already forgotten returns None instead of NotFound."""
        if await uow.memories.is_forgotten(ctx.tenant_id, memory_id):
            return None
        memory = await self.get_memory(uow, ctx, memory_id)
        owner_ok = memory.owner_principal in (ctx.principal_id, f"user:{ctx.user_id}")
        if not owner_ok and not await self.authz.is_tenant_admin(ctx):
            raise ScopeDenied(
                "only the owner (or a tenant admin) can forget a memory",
                details={"principal": ctx.principal_id},
            )
        await uow.memories.forget(ctx.tenant_id, memory_id)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_MEMORY_INDEX,
                queue=Queue.EMBEDDING,
                payload={"tenant_id": ctx.tenant_id, "memory_ids": [memory_id]},
                idempotency_key=f"memidx:forget:{memory_id}",
                tenant_id=ctx.tenant_id,
            )
        )
        if memory.scope.user_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.USER, memory.scope.user_id)
        if memory.scope.thread_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.THREAD, memory.scope.thread_id)
        if memory.scope.agent_id:
            await uow.revisions.bump(ctx.tenant_id, RevisionKind.AGENT, memory.scope.agent_id)
        return memory
