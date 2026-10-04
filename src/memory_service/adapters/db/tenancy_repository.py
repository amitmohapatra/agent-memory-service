"""SQL repositories for tenants, API keys, workspaces and the read audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from sqlalchemy import and_, delete, func, literal, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import (
    ApiKeyRow,
    ReadAuditRow,
    TenantRow,
    WorkspaceMemberRow,
    WorkspaceRow,
)
from memory_service.adapters.db.repositories import _rowcount
from memory_service.domain.audit import ReadAuditEntry
from memory_service.domain.errors import Conflict, ValidationFailed
from memory_service.domain.tenancy import (
    ApiKey,
    KeyRole,
    MemberRole,
    Tenant,
    Workspace,
    WorkspaceMember,
)


def _tenant(row: TenantRow) -> Tenant:
    return Tenant(
        tenant_id=row.tenant_id,
        name=row.name,
        status=row.status,  # type: ignore[arg-type]
        retention_days=row.retention_days,
        rate_limit_per_minute=row.rate_limit_per_minute,
        admission_gate=row.admission_gate,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _key(row: ApiKeyRow) -> ApiKey:
    return ApiKey(
        key_id=row.key_id,
        tenant_id=row.tenant_id,
        role=KeyRole(row.role),
        name=row.name,
        workspace_id=row.workspace_id,
        secret_hash=row.secret_hash,
        created_by=row.created_by,
        created_at=row.created_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        last_used_at=row.last_used_at,
        may_act_as=list(row.may_act_as),
    )


def _workspace(row: WorkspaceRow) -> Workspace:
    return Workspace(
        workspace_id=row.workspace_id,
        tenant_id=row.tenant_id,
        name=row.name,
        created_at=row.created_at,
        deleted_at=row.deleted_at,
    )


def _workspace_member(row: WorkspaceMemberRow) -> WorkspaceMember:
    return WorkspaceMember(
        tenant_id=row.tenant_id,
        workspace_id=row.workspace_id,
        principal=row.principal,
        role=row.role,  # type: ignore[arg-type]
        added_by=row.added_by,
        added_at=row.added_at,
    )


async def _insert(session: AsyncSession, row: object, what: str) -> None:
    """Add and flush; a primary-key collision is a 409, not a 500.

    The services check for an existing id first, but two requests can pass that check
    together; the database decides, and the loser gets the same answer a sequential
    duplicate gets.
    """
    session.add(row)
    try:
        await session.flush()
    except IntegrityError as exc:
        raise Conflict(f"{what} already exists") from exc
    except DataError as exc:
        # a value the column cannot hold (a NUL byte, an over-long string) is the caller's
        raise ValidationFailed(f"{what}: a field is not storable") from exc


async def _execute(session: AsyncSession, stmt: object, what: str) -> None:
    """An upsert whose value the column cannot hold is a 422, not a 500."""
    try:
        await session.execute(stmt)  # type: ignore[arg-type]
    except DataError as exc:
        raise ValidationFailed(f"{what}: a field is not storable") from exc


def _live():  # type: ignore[no-untyped-def]
    """Unrevoked and unexpired: what the registry and the issuance cap count as a key."""
    return and_(
        ApiKeyRow.revoked_at.is_(None),
        or_(ApiKeyRow.expires_at.is_(None), ApiKeyRow.expires_at > func.now()),
    )


def _read(row: ReadAuditRow) -> ReadAuditEntry:
    return ReadAuditEntry(
        tenant_id=row.tenant_id,
        credential=row.credential,
        principal=row.principal,
        kind=row.kind,  # type: ignore[arg-type]
        query_hash=row.query_hash,
        scope_fingerprint=row.scope_fingerprint,
        record_ids=list(row.record_ids or []),
        at=row.at,
    )


class SqlTenantRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, tenant: Tenant) -> None:
        await _insert(self.s, TenantRow(**tenant.model_dump()), f"tenant {tenant.tenant_id}")

    async def get(self, tenant_id: str) -> Tenant | None:
        row = await self.s.get(TenantRow, tenant_id)
        return _tenant(row) if row is not None else None

    async def list(self, *, after: str = "", limit: int = 100) -> list[Tenant]:
        stmt = (
            select(TenantRow)
            .where(TenantRow.tenant_id > after)
            .order_by(TenantRow.tenant_id)
            .limit(limit)
        )
        return [_tenant(r) for r in (await self.s.scalars(stmt)).all()]

    async def update(self, tenant_id: str, changes: Mapping[str, object]) -> None:
        await self.s.execute(
            update(TenantRow).where(TenantRow.tenant_id == tenant_id).values(**changes)
        )

    async def suspended_tenants(self) -> list[str]:
        stmt = select(TenantRow.tenant_id).where(TenantRow.status == "suspended")
        return list((await self.s.scalars(stmt)).all())

    async def rate_limits(self) -> dict[str, int]:
        stmt = select(TenantRow.tenant_id, TenantRow.rate_limit_per_minute).where(
            TenantRow.rate_limit_per_minute.is_not(None)
        )
        return {tenant_id: int(limit) for tenant_id, limit in (await self.s.execute(stmt)).all()}

    async def retention_policies(self) -> dict[str, int]:
        stmt = select(TenantRow.tenant_id, TenantRow.retention_days).where(
            TenantRow.retention_days.is_not(None), TenantRow.status == "active"
        )
        return {tenant_id: int(days) for tenant_id, days in (await self.s.execute(stmt)).all()}


class SqlApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, key: ApiKey) -> None:
        await _insert(
            self.s, ApiKeyRow(**{**key.model_dump(), "role": key.role.value}), f"key {key.key_id}"
        )

    async def get(self, key_id: str) -> ApiKey | None:
        row = await self.s.get(ApiKeyRow, key_id)
        return _key(row) if row is not None else None

    async def list(
        self, tenant_id: str, *, after: tuple[datetime, str] | None = None, limit: int = 100
    ) -> list[ApiKey]:
        stmt = select(ApiKeyRow).where(ApiKeyRow.tenant_id == tenant_id)
        if after is not None:
            stmt = stmt.where(
                tuple_(ApiKeyRow.created_at, ApiKeyRow.key_id)
                > tuple_(literal(after[0]), literal(after[1]))
            )
        stmt = stmt.order_by(ApiKeyRow.created_at, ApiKeyRow.key_id).limit(limit)
        return [_key(r) for r in (await self.s.scalars(stmt)).all()]

    async def revoke(self, tenant_id: str, key_id: str, *, at: datetime) -> bool:
        result = await self.s.execute(
            update(ApiKeyRow)
            .where(
                ApiKeyRow.key_id == key_id,
                ApiKeyRow.tenant_id == tenant_id,
                ApiKeyRow.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        return _rowcount(result) > 0

    async def set_may_act_as(
        self, tenant_id: str, key_id: str, principals: Sequence[str]
    ) -> ApiKey | None:
        row = (
            await self.s.execute(
                update(ApiKeyRow)
                .where(ApiKeyRow.key_id == key_id, ApiKeyRow.tenant_id == tenant_id)
                .values(may_act_as=list(principals))
                .returning(ApiKeyRow)
            )
        ).scalar_one_or_none()
        return _key(row) if row is not None else None

    async def touch(self, key_id: str, *, at: datetime) -> None:
        await self.s.execute(
            update(ApiKeyRow).where(ApiKeyRow.key_id == key_id).values(last_used_at=at)
        )

    async def live_key_ids(self) -> list[str]:
        stmt = select(ApiKeyRow.key_id).where(_live())
        return list((await self.s.scalars(stmt)).all())

    async def count_live(self, tenant_id: str) -> int:
        stmt = select(func.count()).where(ApiKeyRow.tenant_id == tenant_id, _live())
        return int((await self.s.scalar(stmt)) or 0)

    async def key_tenants(self, tenant_ids: Sequence[str]) -> dict[str, str]:
        if not tenant_ids:
            return {}
        stmt = select(ApiKeyRow.key_id, ApiKeyRow.tenant_id).where(
            ApiKeyRow.tenant_id.in_(list(tenant_ids)), _live()
        )
        return dict((await self.s.execute(stmt)).tuples().all())


class SqlWorkspaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, workspace: Workspace) -> None:
        await _insert(
            self.s, WorkspaceRow(**workspace.model_dump()), f"workspace {workspace.workspace_id}"
        )

    async def get(self, tenant_id: str, workspace_id: str) -> Workspace | None:
        row = await self.s.get(WorkspaceRow, (tenant_id, workspace_id))
        return _workspace(row) if row is not None and row.deleted_at is None else None

    async def list(self, tenant_id: str, *, after: str = "", limit: int = 100) -> list[Workspace]:
        stmt = (
            select(WorkspaceRow)
            .where(
                WorkspaceRow.tenant_id == tenant_id,
                WorkspaceRow.deleted_at.is_(None),
                WorkspaceRow.workspace_id > after,
            )
            .order_by(WorkspaceRow.workspace_id)
            .limit(limit)
        )
        return [_workspace(r) for r in (await self.s.scalars(stmt)).all()]

    async def soft_delete(self, tenant_id: str, workspace_id: str, *, at: datetime) -> bool:
        result = await self.s.execute(
            update(WorkspaceRow)
            .where(
                WorkspaceRow.tenant_id == tenant_id,
                WorkspaceRow.workspace_id == workspace_id,
                WorkspaceRow.deleted_at.is_(None),
            )
            .values(deleted_at=at)
        )
        return _rowcount(result) > 0

    async def ever_existed(self, tenant_id: str, workspace_id: str) -> bool:
        return await self.s.get(WorkspaceRow, (tenant_id, workspace_id)) is not None

    async def memberships_of(self, tenant_id: str, principal: str) -> list[str]:
        stmt = (
            select(WorkspaceMemberRow.workspace_id)
            .where(
                WorkspaceMemberRow.tenant_id == tenant_id,
                WorkspaceMemberRow.principal == principal,
            )
            .order_by(WorkspaceMemberRow.workspace_id)
        )
        return list((await self.s.scalars(stmt)).all())

    async def put_member(self, member: WorkspaceMember) -> MemberRole | None:
        previous = await self.s.scalar(
            select(WorkspaceMemberRow.role).where(
                WorkspaceMemberRow.tenant_id == member.tenant_id,
                WorkspaceMemberRow.workspace_id == member.workspace_id,
                WorkspaceMemberRow.principal == member.principal,
            )
        )
        stmt = pg_insert(WorkspaceMemberRow).values(**member.model_dump())
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "workspace_id", "principal"],
            set_={
                "role": stmt.excluded.role,
                "added_by": stmt.excluded.added_by,
                "added_at": stmt.excluded.added_at,
            },
        )
        await _execute(self.s, stmt, "workspace member")
        return previous  # type: ignore[return-value]

    async def remove_member(self, tenant_id: str, workspace_id: str, principal: str) -> bool:
        result = await self.s.execute(
            delete(WorkspaceMemberRow).where(
                WorkspaceMemberRow.tenant_id == tenant_id,
                WorkspaceMemberRow.workspace_id == workspace_id,
                WorkspaceMemberRow.principal == principal,
            )
        )
        return _rowcount(result) > 0

    async def members(self, tenant_id: str, workspace_id: str) -> list[WorkspaceMember]:
        stmt = (
            select(WorkspaceMemberRow)
            .where(
                WorkspaceMemberRow.tenant_id == tenant_id,
                WorkspaceMemberRow.workspace_id == workspace_id,
            )
            .order_by(WorkspaceMemberRow.principal)
        )
        return [_workspace_member(r) for r in (await self.s.scalars(stmt)).all()]


class SqlReadAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add_many(self, entries: Sequence[ReadAuditEntry]) -> None:
        if not entries:
            return
        self.s.add_all(ReadAuditRow(**entry.model_dump()) for entry in entries)
        await self.s.flush()

    async def list(
        self,
        tenant_id: str,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[ReadAuditEntry]:
        stmt = select(ReadAuditRow).where(ReadAuditRow.tenant_id == tenant_id)
        if after is not None:
            stmt = stmt.where(ReadAuditRow.at > after)
        if before is not None:
            stmt = stmt.where(ReadAuditRow.at < before)
        stmt = stmt.order_by(ReadAuditRow.at.desc(), ReadAuditRow.id.desc()).limit(limit)
        return [_read(r) for r in (await self.s.scalars(stmt)).all()]

    async def purge_before(self, before: datetime, *, limit: int = 5000) -> int:
        ids = (
            await self.s.scalars(
                select(ReadAuditRow.id)
                .where(ReadAuditRow.at < before)
                .order_by(ReadAuditRow.at)
                .limit(limit)
            )
        ).all()
        if not ids:
            return 0
        result = await self.s.execute(delete(ReadAuditRow).where(ReadAuditRow.id.in_(ids)))
        return _rowcount(result)
