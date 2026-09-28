"""Feedback: store a judgement, decide who may see it, and learn from it off the request path.

The projector runs as the ``feedback.project`` job enqueued in the transaction that stored
the record, so a verdict is never lost and never slows the request that carried it. On a
memory it reuses the revision machinery: an affirming verdict reinforces, a rejection
retracts, a correction writes a new memory that supersedes the old one. Every projection
bumps the memory revisions, so a cached bundle that showed the old memory stops being served.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.errors import NotFound, ScopeDenied, ValidationFailed
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.feedback import (
    AFFIRMING_VERDICTS,
    CORRECTING_VERDICTS,
    Feedback,
    FeedbackProjection,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
)
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.webhooks import Event, WebhookEvent
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.jobs.names import TASK_MEMORY_INDEX
from memory_service.modules.memory.native import normalized_hash
from memory_service.modules.memory.revisions import bump_memory_revisions, supersede
from memory_service.modules.memory.service import MemoryService
from memory_service.observability.logging import get_logger
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory
from memory_service.ports.webhooks import EventPublisher

log = get_logger(__name__)

TASK_FEEDBACK_PROJECT = "feedback.project"
#: How much one affirming verdict moves a memory's confidence.
REINFORCEMENT_STEP: Final = 0.1
#: The record fields the trusted headers decide; a body that disagrees is refused.
IDENTITY_FIELDS: Final = ("tenant_id", "workspace_id", "user_id")
EVIDENCE_SOURCE: Final = "feedback"


def _correction_text(correction: Any) -> str:
    if isinstance(correction, str):
        return correction.strip()
    if isinstance(correction, dict) and isinstance(correction.get("content"), str):
        return str(correction["content"]).strip()
    return ""


class FeedbackService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        authz: AuthorizationService,
        memory: MemoryService,
        events: EventPublisher,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.memory = memory
        self.events = events
        self.clock = clock

    # ------------------------------------------------------------------ writes
    async def submit(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, feedback: Feedback
    ) -> tuple[Feedback, bool]:
        """Store the record and schedule its projection. Returns the stored record and
        whether it was new (a retry with the same ``feedback_id`` returns the original).

        The record only claims who it is from: its identity fields must agree with the
        trusted headers, and a memory target must be readable by the caller."""
        for name in IDENTITY_FIELDS:
            claimed, trusted = getattr(feedback, name), getattr(ctx, name)
            if claimed is not None and trusted is not None and claimed != trusted:
                raise ValidationFailed(
                    f"feedback {name} does not match the trusted header", details={"field": name}
                )
        if feedback.target_kind is FeedbackTargetKind.MEMORY:
            await self._authorize_memory_verdict(uow, ctx, feedback)
        existing = await uow.feedback.get(ctx.tenant_id, feedback.feedback_id)
        if existing is not None:
            return existing, False
        stored = feedback.model_copy(
            update={
                "workspace_id": feedback.workspace_id or ctx.workspace_id,
                "user_id": feedback.user_id or ctx.user_id,
                "agent_id": feedback.agent_id or ctx.agent_id,
                "agent_run_id": feedback.agent_run_id or ctx.agent_run_id,
                # provenance is the request's, never the body's (ADR 0022)
                "trace_id": ctx.trace_id,
                "created_at": self.clock(),
                "projection": None,
            }
        )
        await uow.feedback.add(stored)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_FEEDBACK_PROJECT,
                queue=Queue.RECONCILE,
                payload={"tenant_id": stored.tenant_id, "feedback_id": stored.feedback_id},
                idempotency_key=f"feedback:{stored.tenant_id}:{stored.feedback_id}",
                tenant_id=stored.tenant_id,
            )
        )
        await self.events.publish(
            uow,
            Event(
                type=WebhookEvent.FEEDBACK_RECEIVED,
                tenant_id=stored.tenant_id,
                workspace_id=stored.workspace_id,
                data=self.event_data(stored),
            ),
        )
        return stored, True

    async def _authorize_memory_verdict(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, feedback: Feedback
    ) -> None:
        """An affirming verdict needs what reading needs. A verdict that retracts or rewrites
        the memory is the forget rule: the owner, the user an agent acts for, or a tenant
        admin; a reader may not rewrite what a colleague remembered."""
        memory = await self.memory.get_memory(uow, ctx, feedback.target_id)  # 404 / 403
        if feedback.verdict in AFFIRMING_VERDICTS:
            return
        owner_ok = memory.owner_principal in (ctx.principal_id, f"user:{ctx.user_id}")
        if not owner_ok and not await self.authz.is_tenant_admin(ctx):
            raise ScopeDenied(
                f"only the owner (or a tenant admin) can {feedback.verdict.value} a memory",
                details={"principal": ctx.principal_id},
            )

    @staticmethod
    def event_data(feedback: Feedback) -> dict[str, Any]:
        """What a webhook learns about a record: its identity, target and verdict, never the
        correction text (the receiver reads the record through the API if it needs it)."""
        return {
            "feedback_id": feedback.feedback_id,
            "target_kind": feedback.target_kind.value,
            "target_id": feedback.target_id,
            "verdict": feedback.verdict.value,
            "source": feedback.source.value,
            "agent_id": feedback.agent_id,
            "agent_run_id": feedback.agent_run_id,
            "projection": feedback.projection.model_dump(mode="json")
            if feedback.projection
            else None,
        }

    # ------------------------------------------------------------------ reads
    async def get(self, uow: UnitOfWork, ctx: MemoryExecutionContext, feedback_id: str) -> Feedback:
        record = await uow.feedback.get(ctx.tenant_id, feedback_id)
        if record is None or not await self._visible(uow, ctx, record):
            raise NotFound(f"feedback {feedback_id} not found")
        return record

    async def list_for(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        target_kind: FeedbackTargetKind,
        target_id: str,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        """Feedback on one target. A memory target is checked once (as ``GET`` does) and then
        every record on it is visible; on other kinds each record is checked on its own."""
        if target_kind is FeedbackTargetKind.MEMORY:
            await self.memory.get_memory(uow, ctx, target_id)
            return await uow.feedback.list_for(
                ctx.tenant_id,
                target_kind=target_kind,
                target_id=target_id,
                before=before,
                limit=limit,
            )
        admin = await self.authz.is_tenant_admin(ctx)
        out: list[Feedback] = []
        while len(out) < limit:
            rows = await uow.feedback.list_for(
                ctx.tenant_id,
                target_kind=target_kind,
                target_id=target_id,
                before=before,
                limit=limit,
            )
            out.extend(r for r in rows if admin or self._own(ctx, r))
            if len(rows) < limit:
                break
            before = (rows[-1].created_at, rows[-1].feedback_id)
        return out[:limit]

    async def _visible(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, record: Feedback
    ) -> bool:
        if record.target_kind is FeedbackTargetKind.MEMORY:
            try:
                await self.memory.get_memory(uow, ctx, record.target_id)
            except (NotFound, ScopeDenied):
                return False
            return True
        return self._own(ctx, record) or await self.authz.is_tenant_admin(ctx)

    @staticmethod
    def _own(ctx: MemoryExecutionContext, record: Feedback) -> bool:
        """The author, or a teammate in the workspace the feedback was given in."""
        if ctx.user_id is not None and record.user_id == ctx.user_id:
            return True
        return ctx.workspace_id is not None and record.workspace_id == ctx.workspace_id

    # ------------------------------------------------------------------ projector
    async def project(self, tenant_id: str, feedback_id: str) -> FeedbackProjection | None:
        """The ``feedback.project`` job. Idempotent and serialised: the record and, for a
        memory target, the memory are locked for the transaction, so a re-dispatched job or
        two verdicts on one memory apply one after another and never fork the chain."""
        async with self.uow_factory() as uow:
            await uow.serialize(f"feedback:{tenant_id}:{feedback_id}")
            record = await uow.feedback.get(tenant_id, feedback_id)
            if record is None:
                log.warning("feedback.missing", tenant_id=tenant_id, feedback_id=feedback_id)
                return None
            if record.projection is not None:
                return record.projection
            if record.target_kind is FeedbackTargetKind.MEMORY:
                await uow.serialize(f"memory:{tenant_id}:{record.target_id}")
            now = self.clock()
            projection = await self._project(uow, record, now=now)
            await uow.feedback.set_projection(tenant_id, feedback_id, projection)
            projected = record.model_copy(update={"projection": projection})
            await self.events.publish(
                uow,
                Event(
                    type=WebhookEvent.FEEDBACK_PROJECTED,
                    tenant_id=tenant_id,
                    workspace_id=record.workspace_id,
                    occurred_at=now,
                    data=self.event_data(projected),
                ),
            )
            await uow.commit()
        log.info(
            "feedback.projected",
            tenant_id=tenant_id,
            feedback_id=feedback_id,
            action=projection.action.value,
        )
        return projection

    async def _project(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        if record.target_kind is not FeedbackTargetKind.MEMORY:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason=f"{record.target_kind.value} feedback is recorded, not projected",
                projected_at=now,
            )
        memory = await uow.memories.get(record.tenant_id, record.target_id)
        if memory is None or memory.deleted_at is not None:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=record.target_id,
                reason="the memory is gone",
                projected_at=now,
            )
        if memory.temporal.status is not TemporalStatus.CURRENT:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=memory.memory_id,
                reason=f"the memory is {memory.temporal.status.value.lower()}",
                projected_at=now,
            )
        if record.verdict in AFFIRMING_VERDICTS:
            return await self._reinforce(uow, memory, now=now)
        if record.verdict is FeedbackVerdict.REJECT:
            return await self._retract(uow, memory, record, now=now)
        if record.verdict in CORRECTING_VERDICTS:
            return await self._correct(uow, memory, record, now=now)
        raise AssertionError(f"unhandled verdict {record.verdict}")  # pragma: no cover

    async def _reinforce(
        self, uow: UnitOfWork, memory: CanonicalMemory, *, now: datetime
    ) -> FeedbackProjection:
        memory.reinforcement_count += 1
        memory.confidence = min(1.0, memory.confidence + REINFORCEMENT_STEP)
        memory.updated_at = now
        await uow.memories.update(memory)
        await bump_memory_revisions(uow, [memory])
        return FeedbackProjection(
            action=ProjectionAction.MEMORY_REINFORCED, memory_id=memory.memory_id, projected_at=now
        )

    async def _retract(
        self, uow: UnitOfWork, memory: CanonicalMemory, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        memory.temporal = memory.temporal.model_copy(
            update={"status": TemporalStatus.RETRACTED, "valid_to": memory.temporal.valid_to or now}
        )
        memory.updated_at = now
        await uow.memories.update(memory)
        await self._reindex(uow, memory)
        await self.events.publish(
            uow,
            Event(
                type=WebhookEvent.MEMORY_RETRACTED,
                tenant_id=memory.tenant_id,
                workspace_id=memory.scope.workspace_id,
                occurred_at=now,
                data={"memory_id": memory.memory_id, "feedback_id": record.feedback_id},
            ),
        )
        return FeedbackProjection(
            action=ProjectionAction.MEMORY_RETRACTED, memory_id=memory.memory_id, projected_at=now
        )

    async def _correct(
        self, uow: UnitOfWork, memory: CanonicalMemory, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        content = _correction_text(record.correction)
        if not content:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=memory.memory_id,
                reason="the correction carries no text (a string or {content: ...})",
                projected_at=now,
            )
        evidence = [
            *memory.evidence,
            EvidenceRef(
                source_type=EVIDENCE_SOURCE,
                source_id=record.feedback_id,
                agent_id=record.agent_id,
                agent_run_id=record.agent_run_id,
                observed_at=now,
            ),
        ]
        corrected = memory.model_copy(
            update={
                "memory_id": CanonicalMemory.model_fields["memory_id"].get_default(
                    call_default_factory=True
                ),
                "content": content,
                "normalized_hash": normalized_hash(content),
                "confidence": max(memory.confidence, 0.9),
                "reinforcement_count": 1,
                "access_count": 0,
                "last_accessed_at": None,
                "evidence": evidence,
                "temporal": memory.temporal.model_copy(
                    update={"observed_at": now, "valid_from": now, "valid_to": None}
                ),
                # a fresh row: no inherited TTL, derived slot, index state or admission trail
                "system_metadata": {
                    "provider": memory.system_metadata.get("provider", "native"),
                    "corrected_by": record.feedback_id,
                    "reviewer": record.reviewer,
                },
                "created_at": now,
                "updated_at": now,
                "revision": 1,
            }
        )
        keys = list(memory.system_metadata.get("visibility_keys", []))
        # link first: ``add`` copies the temporal state into the row as it is at that moment
        await supersede(uow, memory, corrected, now=now)
        await uow.memories.add(corrected, visibility_keys=keys)
        await self._reindex(uow, memory, corrected)
        await self.events.publish(
            uow,
            Event(
                type=WebhookEvent.MEMORY_SUPERSEDED,
                tenant_id=memory.tenant_id,
                workspace_id=memory.scope.workspace_id,
                occurred_at=now,
                data={
                    "memory_id": memory.memory_id,
                    "superseded_by": corrected.memory_id,
                    "feedback_id": record.feedback_id,
                },
            ),
        )
        return FeedbackProjection(
            action=ProjectionAction.MEMORY_SUPERSEDED,
            memory_id=memory.memory_id,
            superseded_by=corrected.memory_id,
            projected_at=now,
        )

    @staticmethod
    async def _reindex(uow: UnitOfWork, *memories: CanonicalMemory) -> None:
        ids = sorted(m.memory_id for m in memories)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_MEMORY_INDEX,
                queue=Queue.EMBEDDING,
                payload={"tenant_id": memories[0].tenant_id, "memory_ids": ids},
                idempotency_key="memidx:feedback:" + ",".join(ids),
                tenant_id=memories[0].tenant_id,
            )
        )
        await bump_memory_revisions(uow, memories)
