"""Request and response bodies for onboarding and team administration."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.audit import ReadKind
from memory_service.domain.tenancy import (
    ANY_PRINCIPAL,
    MAY_ACT_AS_MAX,
    ApiKey,
    IssuedKey,
    KeyRole,
    MemberRole,
    Tenant,
    TenantStatus,
    Workspace,
    WorkspaceMember,
)

_STATUS = (
    "active serves requests; suspended stops every credential of the tenant on its next "
    "request (403 tenant is suspended), refuses new keys and is skipped by retention"
)
_MAY_ACT_AS = (
    "the principals a request with this key may act for (user:<id>, agent:<id>), or * for "
    "every principal of the tenant; empty: only the key itself"
)
_ADMISSION = (
    "score each memory candidate extracted from a message or event (worthiness, novelty, "
    "confidence, expected utility) and store only the admitted ones; a deferred one is kept "
    "when it is said again. Off (the default) stores every deduplicated candidate. Statements "
    "made with POST /v1/memories are never gated"
)
_QUOTA = (
    "requests per minute per credential of this tenant, replacing the service default; "
    "0 disables the limiter for the tenant (to stop serving it, suspend it)"
)


class CreateTenantRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"tenant_id": "acme", "name": "Acme Corp"}]},
    )

    tenant_id: str | None = Field(default=None, description="omit for a generated id")
    name: str = Field(min_length=1, max_length=200)
    retention_days: int | None = Field(
        default=None, ge=1, description="forget canonical memories older than this many days"
    )
    rate_limit_per_minute: int | None = Field(default=None, ge=0, description=_QUOTA)
    admission_gate: bool = Field(default=False, description=_ADMISSION)


class UpdateTenantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    status: TenantStatus | None = Field(default=None, description=_STATUS)
    retention_days: int | None = Field(default=None, ge=1)
    clear_retention: bool = False
    rate_limit_per_minute: int | None = Field(default=None, ge=0, description=_QUOTA)
    clear_rate_limit: bool = False
    admission_gate: bool | None = Field(default=None, description=_ADMISSION)


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    status: TenantStatus = Field(description=_STATUS)
    retention_days: int | None
    rate_limit_per_minute: int | None
    admission_gate: bool = Field(description=_ADMISSION)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, tenant: Tenant) -> TenantResponse:
        return cls.model_validate(tenant.model_dump())


class IssueKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"role": "service", "name": "support-harness"}]},
    )

    role: Literal[KeyRole.ADMIN, KeyRole.SERVICE] = Field(
        description="admin manages the tenant; service acts for its users (platform is "
        "configured, never issued)"
    )
    name: str = Field(min_length=1, max_length=200)
    workspace_id: str | None = Field(
        default=None,
        description="service keys only: pin the key to one workspace, so requests may name "
        "no other; the key still acts for the tenant's users within it",
    )
    expires_in_days: int | None = Field(default=None, ge=1, le=3660)
    may_act_as: list[str] = Field(
        default_factory=lambda: [ANY_PRINCIPAL],
        max_length=MAY_ACT_AS_MAX,
        description=_MAY_ACT_AS,
    )


class UpdateKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"may_act_as": ["user:u-123"]}]}
    )

    may_act_as: list[str] = Field(max_length=MAY_ACT_AS_MAX, description=_MAY_ACT_AS)


class KeySelfResponse(BaseModel):
    """Who a key is, for the platform's other services (agent-runs) to authenticate by."""

    key_id: str = Field(description="the key's id (never the secret)")
    tenant_id: str | None = Field(
        description="the tenant the key speaks for; a development key's is the development "
        "tenant it acts in unless a request names another; null for a credential that names "
        "the tenant per request (the platform key, an issuer's token)"
    )
    principal: str = Field(description="who the caller is; a principal the key may always be")
    role: str = Field(description="platform, admin, service, trusted_dev or jwt")
    may_act_as: list[str] = Field(description=_MAY_ACT_AS)


class ApiKeyResponse(BaseModel):
    key_id: str
    tenant_id: str
    role: KeyRole = Field(description="admin or service; platform is never a stored key")
    name: str
    workspace_id: str | None
    created_by: str
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    last_used_at: datetime | None
    may_act_as: list[str] = Field(description=_MAY_ACT_AS)

    @classmethod
    def of(cls, key: ApiKey) -> ApiKeyResponse:
        return cls.model_validate(key.model_dump(exclude={"secret_hash"}))


class IssuedKeyResponse(ApiKeyResponse):
    token: str | None = Field(
        description="the secret, shown exactly once; store it now. Null when the response is "
        "an idempotent replay (Idempotent-Replayed: true): the secret is never shown twice"
    )

    @classmethod
    def issued(cls, issued: IssuedKey) -> IssuedKeyResponse:
        return cls.model_validate(
            {**issued.key.model_dump(exclude={"secret_hash"}), "token": issued.token}
        )


class CreatedTenantResponse(BaseModel):
    tenant: TenantResponse
    admin_key: IssuedKeyResponse


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"workspace_id": "finance", "name": "Finance"}]},
    )

    workspace_id: str | None = Field(default=None, description="omit for a generated id")
    name: str = Field(min_length=1, max_length=200)


class WorkspaceResponse(BaseModel):
    workspace_id: str
    tenant_id: str
    name: str
    created_at: datetime

    @classmethod
    def of(cls, workspace: Workspace) -> WorkspaceResponse:
        return cls.model_validate(workspace.model_dump())


class SetMemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"role": "member"}]})

    role: MemberRole = Field(
        default="member",
        description="admin (users only) manages the workspace; member reads and writes its "
        "shared memory; viewer reads it",
    )


class WorkspaceMemberResponse(BaseModel):
    workspace_id: str
    principal: str
    role: MemberRole = Field(
        description="admin, member or viewer; every role reads the workspace, admin and "
        "member write into it"
    )
    added_by: str
    added_at: datetime

    @classmethod
    def of(cls, member: WorkspaceMember) -> WorkspaceMemberResponse:
        return cls.model_validate(member.model_dump())


class ReadAuditResponse(BaseModel):
    credential: str = Field(
        description="the authenticated caller: key:<id>, platform, dev:<hash> or a JWT subject"
    )
    principal: str = Field(description="who the caller acted for: user:<id> or agent:<id>")
    kind: ReadKind = Field(description="recall lists the records served; context records the ask")
    query_hash: str
    scope_fingerprint: str
    record_ids: list[str]
    at: datetime
