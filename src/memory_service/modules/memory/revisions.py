"""Content revisions follow audiences, including readers other than a memory's owner."""

from collections.abc import Iterable

from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
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
    for audience in memory.system_metadata.get("visibility_keys", []):
        kind, _, value = audience.partition(":")
        if kind in {"tenant", "agroup", "run", "runup", "thread"}:
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
