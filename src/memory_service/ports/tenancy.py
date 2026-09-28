"""Repositories for the platform layer: tenants, API keys, workspaces, groups, read audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.audit import ReadAuditEntry
from memory_service.domain.tenancy import (
    ApiKey,
    Group,
    GroupMember,
    MemberRole,
    Tenant,
    Workspace,
    WorkspaceMember,
)


@runtime_checkable
class TenantRepository(Protocol):
    async def add(self, tenant: Tenant) -> None: ...

    async def get(self, tenant_id: str) -> Tenant | None: ...

    async def list(self, *, after: str = "", limit: int = 100) -> list[Tenant]: ...

    async def update(self, tenant_id: str, changes: Mapping[str, object]) -> None:
        """Write only ``changes``, so two administrators changing different fields at once
        both land (a whole-row write would let the last one erase the first)."""
        ...

    async def suspended_tenants(self) -> list[str]: ...

    async def rate_limits(self) -> dict[str, int]:
        """``tenant_id -> rate_limit_per_minute`` for every tenant that overrides it."""
        ...

    async def retention_policies(self) -> dict[str, int]:
        """``tenant_id -> retention_days`` for every tenant that set one."""
        ...


@runtime_checkable
class ApiKeyRepository(Protocol):
    async def add(self, key: ApiKey) -> None: ...

    async def get(self, key_id: str) -> ApiKey | None: ...

    async def list(
        self, tenant_id: str, *, after: tuple[datetime, str] | None = None, limit: int = 100
    ) -> list[ApiKey]:
        """Oldest first by (created_at, key_id); ``after`` is that keyset of the next page."""
        ...

    async def revoke(self, tenant_id: str, key_id: str, *, at: datetime) -> bool: ...

    async def touch(self, key_id: str, *, at: datetime) -> None: ...

    async def key_tenants(self, tenant_ids: Sequence[str]) -> dict[str, str]:
        """``key_id -> tenant_id`` for the live keys of these tenants (the rate limiter)."""
        ...

    async def live_key_ids(self) -> list[str]:
        """Every unrevoked, unexpired key id (the verifier's unknown-id budget)."""
        ...

    async def count_live(self, tenant_id: str) -> int:
        """Unrevoked, unexpired keys of the tenant (the issuance cap)."""
        ...


@runtime_checkable
class WorkspaceRepository(Protocol):
    async def add(self, workspace: Workspace) -> None: ...

    async def get(self, tenant_id: str, workspace_id: str) -> Workspace | None: ...

    async def list(self, tenant_id: str, *, after: str = "", limit: int = 100) -> list[Workspace]:
        """By workspace id; ``after`` is the last id of the previous page."""
        ...

    async def soft_delete(self, tenant_id: str, workspace_id: str, *, at: datetime) -> bool: ...

    async def ever_existed(self, tenant_id: str, workspace_id: str) -> bool:
        """True for a live or a deleted row: identifiers are never reused."""
        ...

    async def put_member(self, member: WorkspaceMember) -> MemberRole | None:
        """Upsert; returns the role the principal held before, if any."""
        ...

    async def remove_member(self, tenant_id: str, workspace_id: str, principal: str) -> bool: ...

    async def members(self, tenant_id: str, workspace_id: str) -> list[WorkspaceMember]: ...

    async def memberships_of(self, tenant_id: str, principal: str) -> list[str]:
        """Workspace ids a principal (user:, agent: or group:) is a member of."""
        ...


@runtime_checkable
class GroupRepository(Protocol):
    async def add(self, group: Group) -> None: ...

    async def get(self, tenant_id: str, group_id: str) -> Group | None: ...

    async def list(self, tenant_id: str, *, after: str = "", limit: int = 100) -> list[Group]:
        """By group id; ``after`` is the last id of the previous page."""
        ...

    async def soft_delete(self, tenant_id: str, group_id: str, *, at: datetime) -> bool: ...

    async def ever_existed(self, tenant_id: str, group_id: str) -> bool: ...

    async def put_member(self, member: GroupMember) -> None: ...

    async def remove_member(self, tenant_id: str, group_id: str, user_id: str) -> bool: ...

    async def members(self, tenant_id: str, group_id: str) -> list[GroupMember]: ...


@runtime_checkable
class ReadAuditRepository(Protocol):
    async def add_many(self, entries: Sequence[ReadAuditEntry]) -> None: ...

    async def list(
        self,
        tenant_id: str,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[ReadAuditEntry]:
        """Newest first. ``before`` walks older pages; ``after`` is a since-filter."""
        ...

    async def purge_before(self, before: datetime, *, limit: int = 5000) -> int: ...
