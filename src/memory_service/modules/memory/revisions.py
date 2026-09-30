"""Content revisions follow audiences, including readers other than a memory's owner."""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from memory_service.domain.enums import TemporalStatus
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.webhooks import Event, WebhookEvent
from memory_service.ports.uow import UnitOfWork
from memory_service.ports.webhooks import EventPublisher


def memory_revision_keys(memory: CanonicalMemory) -> set[tuple[RevisionKind, str]]:
    keys = {
        (kind, ident)
        for kind, ident in (
            (RevisionKind.USER, memory.scope.user_id),
            (RevisionKind.THREAD, memory.scope.thread_id),
            (RevisionKind.AGENT, memory.scope.agent_id),
        )
        if ident
    }
    for audience in memory.system_metadata.get("visibility_keys", []):
        kind, _, value = audience.partition(":")
        if kind in {"tenant", "agroup", "run", "runup", "thread", "workspace"}:
            # These audiences can cross owner/agent identities. Existing readers already
            # subscribe to TENANT; no scope-resolution round trip is needed on cache hits.
            keys.add((RevisionKind.TENANT, ""))
            # Threadless searches include every granted thread, while their cache key
            # has no single thread ID. The thread counter alone cannot invalidate them.
        elif kind == "user":
            tenant, separator, identifier = value.partition("/")
            if tenant == memory.tenant_id and separator and identifier:
                keys.add((RevisionKind(kind), identifier))
        elif kind == "principal":
            tenant, _, principal = value.partition("/")
            if tenant != memory.tenant_id:
                continue
            principal_kind, _, identifier = principal.partition(":")
            if principal_kind == "user" and identifier:
                keys.add((RevisionKind.USER, identifier))
            elif principal_kind == "agent" and identifier:
                # Both bound agent:user/id and unattended agent:id subscribe to AGENT:id.
                keys.add((RevisionKind.AGENT, identifier.rsplit("/", 1)[-1]))
            else:
                keys.add((RevisionKind.TENANT, ""))
    return keys or {(RevisionKind.TENANT, "")}


async def bump_memory_revisions(uow: UnitOfWork, memories: Iterable[CanonicalMemory]) -> None:
    touched = {
        (memory.tenant_id, kind, identifier)
        for memory in memories
        for kind, identifier in memory_revision_keys(memory)
    }
    for tenant_id, kind, identifier in sorted(touched):
        await uow.revisions.bump(tenant_id, kind, identifier)


async def supersede(
    uow: UnitOfWork, old: CanonicalMemory, new: CanonicalMemory, *, now: datetime
) -> None:
    """Link ``new`` as the current revision of ``old``: the old row closes its validity and
    points forward, the new one points back, so the chain reads in both directions."""
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


async def retract(
    uow: UnitOfWork,
    memory: CanonicalMemory,
    *,
    now: datetime,
    events: EventPublisher | None,
    data: dict[str, Any],
) -> None:
    """Withdraw a memory without deleting it: RETRACTED, its validity closed at ``now``, the
    row kept for the temporal view, and ``memory.retracted`` published with ``data``."""
    memory.temporal = memory.temporal.model_copy(
        update={"status": TemporalStatus.RETRACTED, "valid_to": memory.temporal.valid_to or now}
    )
    memory.updated_at = now
    await uow.memories.update(memory)
    if events is not None:
        await events.publish(
            uow,
            Event(
                type=WebhookEvent.MEMORY_RETRACTED,
                tenant_id=memory.tenant_id,
                workspace_id=memory.scope.workspace_id,
                occurred_at=now,
                data={"memory_id": memory.memory_id, **data},
            ),
        )
