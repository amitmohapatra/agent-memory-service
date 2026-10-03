"""Tenants, workspaces and API keys: the platform layer's own records.

A tenant is a customer boundary. A workspace is a team inside it whose members share what
they store there; a group is a set of users a workspace admits at once. An API key is how a
calling service proves which tenant - and optionally which workspace - it acts for: the
secret is shown once at issue time, only its SHA-256 is stored, and comparison is
constant-time.
"""

from __future__ import annotations

import hmac
import re
import secrets
import string
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_service.domain.ids import content_hash, is_valid_id

KEY_PREFIX = "mk"
KEY_ID_LENGTH = 16
_KEY_ID_ALPHABET = string.ascii_lowercase + string.digits
_KEY_ID = re.compile(rf"^[{_KEY_ID_ALPHABET}]{{{KEY_ID_LENGTH}}}$")
#: Live keys a tenant may hold. Keys are cheap to issue and every instance keeps the live
#: set in memory; a tenant that needs more is rotating instead of revoking.
MAX_KEYS_PER_TENANT = 1000
#: What a workspace may admit. Agents are named bare (``agent:<id>``) because a workspace
#: grant is durable and an agent bound to a user already inherits that user's membership.
PRINCIPAL_KINDS = frozenset({"user", "agent"})
#: ``may_act_as`` entry meaning every principal of the tenant.
ANY_PRINCIPAL = "*"
#: Principals one key may be allowed to act as.
MAY_ACT_AS_MAX = 100

MemberRole = Literal["admin", "member", "viewer"]
TenantStatus = Literal["active", "suspended"]
#: The platform's own scope: the bootstrap principal's id and the onboarding idempotency
#: namespace. Reserved, so no tenant can be created under it.
PLATFORM_SCOPE = "platform"
RESERVED_TENANT_IDS = frozenset({PLATFORM_SCOPE})


class KeyRole(StrEnum):
    """What a key may do.

    ``platform`` is the bootstrap operator: it onboards tenants and is never a row. ``admin``
    manages one tenant's workspaces and keys. ``service`` acts for that tenant's
    users and agents - it is what a harness holds.
    """

    PLATFORM = "platform"
    ADMIN = "admin"
    SERVICE = "service"


def _now() -> datetime:
    return datetime.now(UTC)


class Tenant(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    name: str = Field(min_length=1, max_length=200)
    status: TenantStatus = "active"
    retention_days: int | None = Field(
        default=None,
        ge=1,
        description="canonical records older than this are forgotten by the retention "
        "sweep; None keeps everything",
    )
    rate_limit_per_minute: int | None = Field(
        default=None, ge=0, description="overrides service.rate_limit_per_minute for this tenant"
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("tenant_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not is_valid_tenant_id(value):
            raise ValueError(f"invalid tenant_id: {value!r}")
        return value


class ApiKey(BaseModel):
    """A key's record. ``secret_hash`` is all that is ever stored of the secret."""

    model_config = ConfigDict(frozen=True)

    key_id: str
    tenant_id: str
    role: KeyRole
    name: str = Field(min_length=1, max_length=200)
    workspace_id: str | None = None
    secret_hash: str = Field(repr=False)
    created_by: str
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    #: the principals a request made with this key may act for (``user:<id>``,
    #: ``agent:<id>``, or ``*`` for any of the tenant's); empty: only the key itself
    may_act_as: list[str] = Field(default_factory=lambda: [ANY_PRINCIPAL])

    @field_validator("may_act_as")
    @classmethod
    def _principals(cls, value: list[str]) -> list[str]:
        return acting_principals(value)

    @property
    def principal(self) -> str:
        """Who the key is: what a service records as the author of what it writes."""
        return f"key:{self.key_id}"

    def usable_at(self, now: datetime) -> bool:
        return self.revoked_at is None and (self.expires_at is None or now < self.expires_at)


class IssuedKey(BaseModel):
    """What issuing returns: the record and the one-time plaintext token."""

    model_config = ConfigDict(frozen=True)

    key: ApiKey
    token: str = Field(repr=False)


class Workspace(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    tenant_id: str
    name: str = Field(min_length=1, max_length=200)
    created_at: datetime = Field(default_factory=_now)
    deleted_at: datetime | None = None


class WorkspaceMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    workspace_id: str
    principal: str = Field(description="user:<id> | agent:<id>")
    role: MemberRole = "member"
    added_by: str
    added_at: datetime = Field(default_factory=_now)


def is_valid_tenant_id(value: str) -> bool:
    """An id, without ``:`` - it joins tenant and key in cache keys (``idem:<tenant>:<key>``),
    so a tenant carrying one would alias another tenant's entries - and not a reserved one."""
    return is_valid_id(value) and ":" not in value and value not in RESERVED_TENANT_IDS


def acting_principals(values: Sequence[str]) -> list[str]:
    """A ``may_act_as`` list: ``*`` or principals, de-duplicated in order; refuses anything
    else and more than ``MAY_ACT_AS_MAX``."""
    if len(values) > MAY_ACT_AS_MAX:
        raise ValueError(f"a key may act for at most {MAY_ACT_AS_MAX} principals")
    for principal in values:
        if principal != ANY_PRINCIPAL:
            parse_principal(principal)
    return list(dict.fromkeys(values))


def parse_principal(principal: str) -> tuple[str, str]:
    """``user:u1`` -> ``("user", "u1")``; refuses anything that is not a member kind."""
    kind, sep, ident = principal.partition(":")
    if not sep or kind not in PRINCIPAL_KINDS or not is_valid_id(ident):
        raise ValueError(f"invalid principal {principal!r}: expected user:<id> or agent:<id>")
    return kind, ident


def hash_secret(secret: str) -> str:
    return content_hash(secret)


def secret_matches(secret: str, secret_hash: str) -> bool:
    return hmac.compare_digest(hash_secret(secret), secret_hash)


def mint_token() -> tuple[str, str, str]:
    """A fresh ``(key_id, token, secret_hash)``. The token reads ``mk_<key_id>.<secret>``:
    the id is the lookup handle and is safe to log; the secret never is."""
    key_id = "".join(secrets.choice(_KEY_ID_ALPHABET) for _ in range(KEY_ID_LENGTH))
    secret = secrets.token_urlsafe(32)
    return key_id, f"{KEY_PREFIX}_{key_id}.{secret}", hash_secret(secret)


def bare_credential(credential: str) -> str:
    """The token itself, whichever carrier brought it (``X-API-Key`` or ``Bearer``), so one
    credential is one identity however it is sent."""
    scheme, _, rest = credential.partition(" ")
    return rest.strip() if scheme.lower() == "bearer" else credential


def parse_token(token: str) -> tuple[str, str] | None:
    """``(key_id, secret)`` from a token, or None when it is not one of ours."""
    prefix, sep, rest = token.partition("_")
    if not sep or prefix != KEY_PREFIX:
        return None
    key_id, sep, secret = rest.partition(".")
    if not sep or not _KEY_ID.fullmatch(key_id) or not secret:
        return None
    return key_id, secret
