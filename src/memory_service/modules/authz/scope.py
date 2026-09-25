"""ScopeResolver: turns a MemoryExecutionContext into an AuthorizedScope via list_objects.

Shared by the OpenFGA and in-memory providers. Lists are bounded by ``max_listed_objects``;
when a list overflows, ``truncated`` is set and retrieval falls back to per-object checks.
"""

from __future__ import annotations

from typing import Protocol

from memory_service.domain.context import MemoryExecutionContext
from memory_service.ports.authorization import AuthorizedScope


class _Lister(Protocol):
    async def list_objects(self, user: str, relation: str, object_type: str) -> list[str]: ...


class ScopeResolver:
    def __init__(self, provider: _Lister, *, max_listed_objects: int = 2000):
        self.provider = provider
        self.max = max_listed_objects

    async def _list(self, user: str, relation: str, object_type: str) -> tuple[list[str], bool]:
        objects = await self.provider.list_objects(user, relation, object_type)
        truncated = len(objects) > self.max
        return [o.split(":", 1)[1] for o in objects[: self.max]], truncated

    async def resolve(self, ctx: MemoryExecutionContext) -> AuthorizedScope:
        principal = ctx.principal_id
        truncated = False
        threads: list[str] = []
        documents: list[str] = []
        agents: list[str] = []

        subjects = [principal]
        # an agent acts on behalf of the user it runs for: union of both principals' access
        if ctx.is_agent and ctx.user_id:
            subjects.append(f"user:{ctx.user_id}")

        for subject in subjects:
            for relation, object_type, sink in (
                ("can_read", "thread", threads),
                ("can_read", "document", documents),
            ):
                found, over = await self._list(subject, relation, object_type)
                truncated = truncated or over
                sink.extend(x for x in found if x not in sink)
        if ctx.is_agent and ctx.agent_id:
            agents.append(ctx.agent_id)
        # only objects belonging to this tenant are ever exposed: object ids are tenant-prefixed
        prefix = f"{ctx.tenant_id}/"

        def own(items: list[str]) -> list[str]:
            return sorted(i[len(prefix) :] for i in items if i.startswith(prefix))

        return AuthorizedScope(
            tenant_id=ctx.tenant_id,
            principal=principal,
            thread_ids=own(threads),
            document_ids=own(documents),
            agent_ids=agents,
            run_ids=[r for r in (ctx.agent_run_id, ctx.parent_agent_run_id) if r],
            agent_group_ids=[ctx.agent_group_id] if ctx.agent_group_id else [],
            user_id=ctx.user_id,
            truncated=truncated,
        )


def object_id(tenant_id: str, ident: str) -> str:
    """Object identifiers in the authorization store are tenant-prefixed: ``acme/thr_1``."""
    return f"{tenant_id}/{ident}"
