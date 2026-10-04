"""Content revisions follow audiences, including readers other than a memory's owner."""

from collections.abc import Iterable
from datetime import datetime

from memory_service.domain.enums import TemporalStatus
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind, audience_revision_keys
from memory_service.ports.uow import UnitOfWork


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
    keys |= audience_revision_keys(
        memory.tenant_id, memory.system_metadata.get("visibility_keys", [])
    )
    return keys or {(RevisionKind.TENANT, "")}


def touched_revisions(
    memories: Iterable[CanonicalMemory],
) -> list[tuple[str, RevisionKind, str]]:
    """``(tenant, kind, id)`` of every revision these memories' readers cache on, once each
    and in a fixed order (concurrent bumpers lock the rows in the same order)."""
    return sorted(
        {
            (memory.tenant_id, kind, identifier)
            for memory in memories
            for kind, identifier in memory_revision_keys(memory)
        }
    )


async def bump_memory_revisions(uow: UnitOfWork, memories: Iterable[CanonicalMemory]) -> None:
    for tenant_id, kind, identifier in touched_revisions(memories):
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
) -> None:
    """Withdraw a memory without deleting it: RETRACTED, its validity closed at ``now``, the
    row kept for the temporal view, and ``memory.retracted`` published with ``data``."""
    memory.temporal = memory.temporal.model_copy(
        update={"status": TemporalStatus.RETRACTED, "valid_to": memory.temporal.valid_to or now}
    )
    memory.updated_at = now
    await uow.memories.update(memory)
