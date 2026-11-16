"""Content revisions follow audiences, including readers other than a memory's owner."""

from collections.abc import Iterable

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
