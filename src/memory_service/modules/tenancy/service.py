"""Onboarding and team administration.

A platform operator creates a tenant and receives its first admin key. A tenant admin issues
keys for its services, creates workspaces (teams) and groups, and admits or removes members.
Every write runs in the unit of work the router opened; the authorization tuples are written
beside the rows, and the membership revision is bumped so a change is seen on the caller's
next request rather than at a cache's expiry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory_service.domain.errors import Conflict, NotFound, ValidationFailed
from memory_service.domain.ids import is_valid_id, new_id
from memory_service.domain.tenancy import (
    MAX_KEYS_PER_TENANT,
    ApiKey,
    Group,
    GroupMember,
    IssuedKey,
    KeyRole,
    MemberRole,
    Tenant,
    TenantStatus,
    Workspace,
    WorkspaceMember,
    is_valid_tenant_id,
    mint_token,
    parse_principal,
)
from memory_service.domain.text import sanitise
from memory_service.modules.authz.service import AuthorizationService
from memory_service.observability.logging import get_logger
from memory_service.ports.uow import UnitOfWork

log = get_logger(__name__)
#: ``created_by`` / ``added_by`` are String(512)
ACTOR_MAX = 512


def _now() -> datetime:
    return datetime.now(UTC)


def _require_id(value: str | None, kind: str) -> str:
    if value is None:
        return new_id(kind)
    return _valid(value, kind)


def _clean_name(name: str) -> str:
    """Control characters are stripped; a name that is nothing else is no name (a 422, not
    the 500 the domain model's own validation would be)."""
    cleaned = sanitise(name).strip()
    if not cleaned:
        raise ValidationFailed("name is empty once control characters are removed")
    return cleaned


def _actor(service_id: str) -> str:
    """Who did it, bounded to the column: a JWT subject can be longer than 512 characters."""
    return service_id[:ACTOR_MAX]


def _valid(value: str, kind: str) -> str:
    """An identifier the client named: refused before it reaches a query, so a NUL byte or
    a path-shaped string is a 422 and never a database error."""
    if not is_valid_id(value):
        raise ValidationFailed(f"invalid {kind}_id: {value!r}")
    return value


class TenancyService:
    def __init__(self, authz: AuthorizationService) -> None:
        self.authz = authz

    # -- tenants (platform) -------------------------------------------------------
    async def create_tenant(
        self,
        uow: UnitOfWork,
        *,
        name: str,
        created_by: str,
        tenant_id: str | None = None,
        retention_days: int | None = None,
        rate_limit_per_minute: int | None = None,
    ) -> tuple[Tenant, IssuedKey]:
        """The tenant and its first admin key, shown once."""
        tenant_id = _require_id(tenant_id, "tenant")
        if not is_valid_tenant_id(tenant_id):
            raise ValidationFailed(f"invalid tenant_id: {tenant_id!r} (reserved, or contains ':')")
        if await uow.tenants.get(tenant_id) is not None:
            raise Conflict(f"tenant {tenant_id} already exists")
        tenant = Tenant(
            tenant_id=tenant_id,
            name=_clean_name(name),
            retention_days=retention_days,
            rate_limit_per_minute=rate_limit_per_minute,
        )
        await uow.tenants.add(tenant)
        admin = await self.issue_key(
            uow, tenant_id, role=KeyRole.ADMIN, name="initial admin", created_by=created_by
        )
        log.info("tenant.created", tenant_id=tenant_id)
        return tenant, admin

    async def get_tenant(self, uow: UnitOfWork, tenant_id: str) -> Tenant:
        tenant = await uow.tenants.get(_valid(tenant_id, "tenant"))
        if tenant is None:
            raise NotFound("Tenant not found")
        return tenant

    async def list_tenants(
        self, uow: UnitOfWork, *, after: str = "", limit: int = 100
    ) -> list[Tenant]:
        return await uow.tenants.list(after=_valid(after, "tenant") if after else "", limit=limit)

    async def update_tenant(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        name: str | None = None,
        status: TenantStatus | None = None,
        retention_days: int | None = None,
        rate_limit_per_minute: int | None = None,
        clear_retention: bool = False,
        clear_rate_limit: bool = False,
    ) -> tuple[Tenant, bool]:
        """The updated tenant, and whether its status changed (every key of a suspended
        tenant must stop working on its next request, so the caller invalidates them)."""
        current = await self.get_tenant(uow, tenant_id)
        changes: dict[str, object] = {"updated_at": _now()}
        if name is not None:
            changes["name"] = _clean_name(name)
        if status is not None:
            changes["status"] = status
        if retention_days is not None or clear_retention:
            changes["retention_days"] = None if clear_retention else retention_days
        if rate_limit_per_minute is not None or clear_rate_limit:
            changes["rate_limit_per_minute"] = None if clear_rate_limit else rate_limit_per_minute
        tenant = current.model_copy(update=changes)
        await uow.tenants.update(tenant_id, changes)
        return tenant, tenant.status != current.status

    # -- keys ---------------------------------------------------------------------
    async def issue_key(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        role: KeyRole,
        name: str,
        created_by: str,
        workspace_id: str | None = None,
        expires_in_days: int | None = None,
    ) -> IssuedKey:
        if role is KeyRole.PLATFORM:
            raise ValidationFailed("the platform role is configured, never issued")
        if role is KeyRole.ADMIN and workspace_id is not None:
            # Administration is tenant-wide (it creates workspaces and issues keys), so a
            # "bound" admin key would be a binding in name only. Bind service keys.
            raise ValidationFailed("admin keys are tenant-wide; bind service keys only")
        tenant = await self.get_tenant(uow, tenant_id)
        if tenant.status != "active":
            raise Conflict(f"tenant {tenant_id} is {tenant.status}")
        if workspace_id is not None:
            await self.get_workspace(uow, tenant_id, workspace_id)
        # one issuance at a time per tenant, so the cap is a cap and not a check-then-insert
        await uow.serialize(f"tenant-keys:{tenant_id}")
        if await uow.api_keys.count_live(tenant_id) >= MAX_KEYS_PER_TENANT:
            # every instance holds the live set in memory and a suspension tombstones each
            # key; a tenant is rotating keys, not collecting them
            raise Conflict(
                f"tenant {tenant_id} holds {MAX_KEYS_PER_TENANT} live keys; revoke one first"
            )
        key_id, token, secret_hash = mint_token()
        key = ApiKey(
            key_id=key_id,
            tenant_id=tenant_id,
            role=role,
            name=_clean_name(name),
            workspace_id=workspace_id,
            secret_hash=secret_hash,
            created_by=_actor(created_by),
            expires_at=_now() + timedelta(days=expires_in_days) if expires_in_days else None,
        )
        await uow.api_keys.add(key)
        log.info("api_key.issued", tenant_id=tenant_id, key_id=key_id, role=role.value)
        return IssuedKey(key=key, token=token)

    async def list_keys(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        after: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[ApiKey]:
        return await uow.api_keys.list(tenant_id, after=after, limit=limit)

    async def revoke_key(self, uow: UnitOfWork, tenant_id: str, key_id: str) -> bool:
        """Idempotent: revoking a revoked or unknown key of this tenant is not an error, so a
        retry after a lost response converges. Returns whether a live key of this tenant was
        revoked; the caller invalidates the verifier's cache entry only then, so another
        tenant's key id names nothing here."""
        revoked = await uow.api_keys.revoke(tenant_id, _valid(key_id, "key"), at=_now())
        if revoked:
            log.info("api_key.revoked", tenant_id=tenant_id, key_id=key_id)
        return revoked

    # -- workspaces ---------------------------------------------------------------
    async def create_workspace(
        self, uow: UnitOfWork, tenant_id: str, *, name: str, workspace_id: str | None = None
    ) -> Workspace:
        workspace_id = _require_id(workspace_id, "workspace")
        if await uow.workspaces.ever_existed(tenant_id, workspace_id):
            # A deleted workspace keeps its rows for the record, and a new team must not
            # inherit an old team's identity in audit trails and memory anchors.
            raise Conflict(f"workspace id {workspace_id} exists or was used before")
        if await uow.threads.any_in_workspace(tenant_id, workspace_id) or (
            await uow.documents.any_in_workspace(tenant_id, workspace_id)
        ):
            # Threads and documents were labelled with this id before it named a team; a
            # team created over it would inherit them through the model's rewrites.
            raise Conflict(f"workspace id {workspace_id} is in use as an anchor; choose another")
        workspace = Workspace(
            workspace_id=workspace_id, tenant_id=tenant_id, name=_clean_name(name)
        )
        await uow.workspaces.add(workspace)
        await self.authz.grant_workspace(tenant_id, workspace_id)
        log.info("workspace.created", tenant_id=tenant_id, workspace_id=workspace_id)
        return workspace

    async def get_workspace(self, uow: UnitOfWork, tenant_id: str, workspace_id: str) -> Workspace:
        workspace = await uow.workspaces.get(tenant_id, _valid(workspace_id, "workspace"))
        if workspace is None:
            raise NotFound("Workspace not found")
        return workspace

    async def list_workspaces(
        self, uow: UnitOfWork, tenant_id: str, *, after: str = "", limit: int = 100
    ) -> list[Workspace]:
        return await uow.workspaces.list(
            tenant_id, after=_valid(after, "workspace") if after else "", limit=limit
        )

    async def delete_workspace(
        self, uow: UnitOfWork, tenant_id: str, workspace_id: str
    ) -> list[str]:
        """Every member loses the audience at once and every key bound to the workspace is
        revoked; the rows stay for the record. Returns the revoked key ids so the caller can
        drop their cache entries after commit. Deleting a deleted workspace is a no-op, so a
        retry after a lost 204 converges."""
        workspace_id = _valid(workspace_id, "workspace")
        await uow.serialize(f"workspace-members:{tenant_id}/{workspace_id}")
        if await uow.workspaces.get(tenant_id, workspace_id) is None:
            if await uow.workspaces.ever_existed(tenant_id, workspace_id):
                return []
            raise NotFound("Workspace not found")
        for member in await uow.workspaces.members(tenant_id, workspace_id):
            await self.authz.revoke_workspace_member(
                tenant_id, workspace_id, member.principal, revisions=uow.revisions
            )
        now = _now()
        revoked = [
            key.key_id
            for key in await uow.api_keys.list(tenant_id)
            if key.workspace_id == workspace_id and key.usable_at(now)
        ]
        for key_id in revoked:
            await uow.api_keys.revoke(tenant_id, key_id, at=now)
        await uow.workspaces.soft_delete(tenant_id, workspace_id, at=now)
        log.info(
            "workspace.deleted", tenant_id=tenant_id, workspace_id=workspace_id, keys=len(revoked)
        )
        return revoked

    async def set_member(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        workspace_id: str,
        principal: str,
        *,
        role: MemberRole,
        added_by: str,
    ) -> WorkspaceMember:
        try:
            kind, ident = parse_principal(principal)
        except ValueError as exc:
            raise ValidationFailed(str(exc)) from exc
        if role == "admin" and kind != "user":
            # the model grants workspace admin to users only (deploy/openfga/model.fga)
            raise ValidationFailed("only a user: principal can administer a workspace")
        # membership changes of one team are serialised: two administrators changing one
        # principal at once must not leave the union of their roles in the tuples
        await uow.serialize(f"workspace-members:{tenant_id}/{workspace_id}")
        await self.get_workspace(uow, tenant_id, workspace_id)
        if kind == "group":
            await self.get_group(uow, tenant_id, ident)
        member = WorkspaceMember(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal=principal,
            role=role,
            added_by=_actor(added_by),
        )
        previous = await uow.workspaces.put_member(member)
        await self.authz.set_workspace_member(
            tenant_id, workspace_id, principal, role, previous=previous, revisions=uow.revisions
        )
        return member

    async def remove_member(
        self, uow: UnitOfWork, tenant_id: str, workspace_id: str, principal: str
    ) -> None:
        try:
            parse_principal(principal)
        except ValueError as exc:
            raise ValidationFailed(str(exc)) from exc
        await uow.serialize(f"workspace-members:{tenant_id}/{workspace_id}")
        await self.get_workspace(uow, tenant_id, workspace_id)
        await uow.workspaces.remove_member(tenant_id, workspace_id, principal)
        await self.authz.revoke_workspace_member(
            tenant_id, workspace_id, principal, revisions=uow.revisions
        )

    async def members(
        self, uow: UnitOfWork, tenant_id: str, workspace_id: str
    ) -> list[WorkspaceMember]:
        await self.get_workspace(uow, tenant_id, workspace_id)
        return await uow.workspaces.members(tenant_id, workspace_id)

    # -- groups -------------------------------------------------------------------
    async def create_group(
        self, uow: UnitOfWork, tenant_id: str, *, name: str, group_id: str | None = None
    ) -> Group:
        group_id = _require_id(group_id, "group")
        if await uow.groups.ever_existed(tenant_id, group_id):
            raise Conflict(f"group id {group_id} exists or was used before")
        group = Group(group_id=group_id, tenant_id=tenant_id, name=_clean_name(name))
        await uow.groups.add(group)
        await self.authz.grant_group(tenant_id, group_id)
        return group

    async def get_group(self, uow: UnitOfWork, tenant_id: str, group_id: str) -> Group:
        group = await uow.groups.get(tenant_id, _valid(group_id, "group"))
        if group is None:
            raise NotFound("Group not found")
        return group

    async def list_groups(
        self, uow: UnitOfWork, tenant_id: str, *, after: str = "", limit: int = 100
    ) -> list[Group]:
        return await uow.groups.list(
            tenant_id, after=_valid(after, "group") if after else "", limit=limit
        )

    async def delete_group(self, uow: UnitOfWork, tenant_id: str, group_id: str) -> None:
        """Its users leave the group and the group leaves every workspace it was admitted
        to, so no dangling ``group:`` member remains in a workspace's listing or tuples.
        Deleting a deleted group is a no-op."""
        group_id = _valid(group_id, "group")
        if await uow.groups.get(tenant_id, group_id) is None:
            if await uow.groups.ever_existed(tenant_id, group_id):
                return
            raise NotFound("Group not found")
        for member in await uow.groups.members(tenant_id, group_id):
            await self.authz.revoke_group_member(
                tenant_id, group_id, member.user_id, revisions=uow.revisions
            )
        principal = f"group:{group_id}"
        for workspace_id in await uow.workspaces.memberships_of(tenant_id, principal):
            await uow.workspaces.remove_member(tenant_id, workspace_id, principal)
            await self.authz.revoke_workspace_member(
                tenant_id, workspace_id, principal, revisions=uow.revisions
            )
        await uow.groups.soft_delete(tenant_id, group_id, at=_now())

    async def add_group_user(
        self, uow: UnitOfWork, tenant_id: str, group_id: str, user_id: str, *, added_by: str
    ) -> GroupMember:
        if not is_valid_id(user_id):
            raise ValidationFailed(f"invalid user_id: {user_id!r}")
        await self.get_group(uow, tenant_id, group_id)
        member = GroupMember(
            tenant_id=tenant_id, group_id=group_id, user_id=user_id, added_by=_actor(added_by)
        )
        await uow.groups.put_member(member)
        await self.authz.set_group_member(tenant_id, group_id, user_id, revisions=uow.revisions)
        return member

    async def remove_group_user(
        self, uow: UnitOfWork, tenant_id: str, group_id: str, user_id: str
    ) -> None:
        await self.get_group(uow, tenant_id, group_id)
        await uow.groups.remove_member(tenant_id, group_id, _valid(user_id, "user"))
        await self.authz.revoke_group_member(tenant_id, group_id, user_id, revisions=uow.revisions)

    async def group_members(
        self, uow: UnitOfWork, tenant_id: str, group_id: str
    ) -> list[GroupMember]:
        await self.get_group(uow, tenant_id, group_id)
        return await uow.groups.members(tenant_id, group_id)
