"""AuthorizationService: the one place that turns identities into decisions and filters.

- ``scope(ctx)``            -> AuthorizedScope (cached by tenant/user/agent revision)
- ``visibility(ctx)``       -> VisibilitySpecification for store-side filtering
- ``require(ctx, relation, object)`` -> raises ScopeDenied on a negative decision
- ``grant_*``               -> writes relationship tuples when objects are created

Object ids in the relationship store are tenant-prefixed (``thread:acme/thr_1``) so a
tenant boundary is structural, not a convention.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.scope import ScopeResolver, object_id
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import authz_denials_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.authorization import (
    AccessCheck,
    AuthorizationProvider,
    AuthorizedScope,
    RelationTuple,
)
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.repositories import RevisionRepository

log = get_logger(__name__)


class AuthorizationService:
    def __init__(
        self,
        provider: AuthorizationProvider,
        cache: CacheProvider | None,
        *,
        max_listed_objects: int = 2000,
        cache_ttl_seconds: int = 60,
        trust_header_groups: bool = True,
        decision_cache: bool = True,
    ) -> None:
        self.provider = provider
        self.cache = cache if decision_cache else None
        self.cache_ttl = cache_ttl_seconds
        self.resolver = ScopeResolver(
            provider, max_listed_objects=max_listed_objects, trust_header_groups=trust_header_groups
        )

    # -- scope -----------------------------------------------------------------
    @staticmethod
    def _scope_cache_key(ctx: MemoryExecutionContext, revision_fingerprint: str) -> str:
        return f"authz:scope:{ctx.tenant_id}:{ctx.scope_fingerprint()}:{revision_fingerprint}"

    async def scope(
        self, ctx: MemoryExecutionContext, *, revisions: RevisionRepository | None = None
    ) -> AuthorizedScope:
        fingerprint = "0"
        if revisions is not None:
            keys = [(RevisionKind.TENANT, ""), (RevisionKind.USER, ctx.user_id or "")]
            if ctx.agent_id:
                keys.append((RevisionKind.AGENT, ctx.agent_id))
            values = await revisions.get_many(ctx.tenant_id, keys)
            fingerprint = ",".join(f"{k}={v}" for k, v in sorted(values.items()))
        key = self._scope_cache_key(ctx, fingerprint)
        if self.cache is not None:
            try:
                raw = await self.cache.get(key)
            except CacheUnavailable:
                raw = None
            if raw is not None:
                return AuthorizedScope.model_validate_json(raw)
        with (
            span("authz.scope", tenant_id=ctx.tenant_id),
            stage_seconds.labels("authz.scope").time(),
        ):
            scope = await self.resolver.resolve(ctx)
        if self.cache is not None:
            with contextlib.suppress(CacheUnavailable):
                await self.cache.set(
                    key, scope.model_dump_json().encode(), ttl_seconds=self.cache_ttl
                )
        return scope

    async def visibility(
        self, ctx: MemoryExecutionContext, *, revisions: RevisionRepository | None = None
    ) -> VisibilitySpecification:
        return VisibilitySpecification.from_scope(await self.scope(ctx, revisions=revisions))

    # -- decisions --------------------------------------------------------------
    async def allowed(
        self, ctx: MemoryExecutionContext, relation: str, obj_type: str, ident: str
    ) -> bool:
        obj = f"{obj_type}:{object_id(ctx.tenant_id, ident)}"
        subjects = [ctx.principal_id]
        if ctx.is_agent and ctx.user_id:
            subjects.append(f"user:{ctx.user_id}")
        checks = [AccessCheck(user=s, relation=relation, object=obj) for s in subjects]
        with span("authz.check", relation=relation), stage_seconds.labels("authz.check").time():
            results = await self.provider.batch_check(checks)
        return any(results)

    async def require(
        self, ctx: MemoryExecutionContext, relation: str, obj_type: str, ident: str
    ) -> None:
        if not await self.allowed(ctx, relation, obj_type, ident):
            authz_denials_total.labels(relation).inc()
            log.info("authz.denied", relation=relation, object_type=obj_type, **ctx.log_fields())
            raise ScopeDenied(
                "Access denied", details={"relation": relation, "object_type": obj_type}
            )

    async def filter_allowed(
        self, ctx: MemoryExecutionContext, relation: str, obj_type: str, idents: Sequence[str]
    ) -> list[str]:
        """Per-object checks for the truncated-scope fallback (bounded by the caller)."""
        if not idents:
            return []
        checks = [
            AccessCheck(
                user=ctx.principal_id,
                relation=relation,
                object=f"{obj_type}:{object_id(ctx.tenant_id, i)}",
            )
            for i in idents
        ]
        results = await self.provider.batch_check(checks)
        return [i for i, ok in zip(idents, results, strict=True) if ok]

    # -- grants -----------------------------------------------------------------
    async def grant_thread(
        self, ctx: MemoryExecutionContext, thread_id: str, *, workspace_id: str | None = None
    ) -> None:
        obj = f"thread:{object_id(ctx.tenant_id, thread_id)}"
        tuples = [RelationTuple(user=f"tenant:{ctx.tenant_id}", relation="tenant", object=obj)]
        if workspace_id:
            tuples.append(
                RelationTuple(
                    user=f"workspace:{object_id(ctx.tenant_id, workspace_id)}",
                    relation="workspace",
                    object=obj,
                )
            )
        if ctx.user_id:
            tuples.append(RelationTuple(user=f"user:{ctx.user_id}", relation="owner", object=obj))
        if ctx.is_agent:
            tuples.append(RelationTuple(user=ctx.principal_id, relation="participant", object=obj))
        await self.provider.write(tuples)

    async def grant_document(
        self,
        ctx: MemoryExecutionContext,
        document_id: str,
        *,
        thread_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        obj = f"document:{object_id(ctx.tenant_id, document_id)}"
        tuples = [RelationTuple(user=f"tenant:{ctx.tenant_id}", relation="tenant", object=obj)]
        if ctx.user_id:
            tuples.append(RelationTuple(user=f"user:{ctx.user_id}", relation="owner", object=obj))
        if thread_id:
            tuples.append(
                RelationTuple(
                    user=f"thread:{object_id(ctx.tenant_id, thread_id)}",
                    relation="thread",
                    object=obj,
                )
            )
        if workspace_id:
            tuples.append(
                RelationTuple(
                    user=f"workspace:{object_id(ctx.tenant_id, workspace_id)}",
                    relation="workspace",
                    object=obj,
                )
            )
        await self.provider.write(tuples)

    async def grant_memory(
        self, ctx: MemoryExecutionContext, memory_id: str, *, owner: str | None = None
    ) -> None:
        obj = f"memory:{object_id(ctx.tenant_id, memory_id)}"
        await self.provider.write(
            [
                RelationTuple(user=f"tenant:{ctx.tenant_id}", relation="tenant", object=obj),
                RelationTuple(user=owner or ctx.principal_id, relation="owner", object=obj),
            ]
        )

    async def grant_membership(
        self,
        tenant_id: str,
        user_id: str,
        *,
        groups: Sequence[str] = (),
        workspaces: Sequence[str] = (),
        admin: bool = False,
        revisions: RevisionRepository | None = None,
    ) -> None:
        """Bootstrap helper used by imports/admin: tenant membership, groups and workspaces.

        Pass ``revisions`` (inside the caller's unit of work) so the user's cached
        AuthorizedScope is invalidated immediately instead of after the cache TTL.
        """
        tuples = [
            RelationTuple(
                user=f"user:{user_id}",
                relation="admin" if admin else "member",
                object=f"tenant:{tenant_id}",
            )
        ]
        for g in groups:
            tuples.append(
                RelationTuple(
                    user=f"user:{user_id}",
                    relation="member",
                    object=f"group:{object_id(tenant_id, g)}",
                )
            )
        for w in workspaces:
            tuples.append(
                RelationTuple(
                    user=f"user:{user_id}",
                    relation="member",
                    object=f"workspace:{object_id(tenant_id, w)}",
                )
            )
        await self.provider.write(tuples)
        if revisions is not None:
            await revisions.bump(tenant_id, RevisionKind.USER, user_id)

    def describe(self) -> str:
        return json.dumps(
            {"provider": type(self.provider).__name__, "cache": self.cache is not None}
        )
