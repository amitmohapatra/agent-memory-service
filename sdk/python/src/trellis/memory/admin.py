"""Administration: onboarding tenants (the platform key) and one tenant's keys, workspaces,
model keys and policies, and read audit (a tenant admin key).

None of it is bound to a request scope: :class:`MemoryClient` holds ``admin`` and ``tenant``,
and a bound context reaches the same objects through ``ctx.advanced``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any

from trellis.memory.models import (
    AgentKeyStatus,
    ApiKeyInfo,
    CreatedTenant,
    IssuedKey,
    KeyInfo,
    KeyRole,
    MemberRole,
    ModelPolicy,
    ModelUsage,
    Page,
    ReadAuditRecord,
    TenantInfo,
    WorkspaceInfo,
    WorkspaceMemberInfo,
)
from trellis.memory.transport import HEADER_TENANT

if TYPE_CHECKING:
    from trellis.memory.client import MemoryClient


# --- platform administration -----------------------------------------------------


class AdminAPI:
    """Onboarding, for the bootstrap key: ``POST /v1/admin/tenants`` and friends."""

    def __init__(self, client: MemoryClient) -> None:
        self._t = client.transport

    async def create_tenant(
        self,
        name: str,
        *,
        tenant_id: str | None = None,
        retention_days: int | None = None,
        rate_limit_per_minute: int | None = None,
        idempotency_key: str | None = None,
    ) -> CreatedTenant:
        """The tenant and its first admin key. The key's token is returned once: keep it.

        Pass ``idempotency_key`` to make a retry safe: the replay carries the same tenant with
        ``admin_key.token`` set to None (the secret is never shown twice).
        """
        payload = {
            "name": name,
            "tenant_id": tenant_id,
            "retention_days": retention_days,
            "rate_limit_per_minute": rate_limit_per_minute,
        }
        return CreatedTenant.model_validate(
            await self._t.request(
                "POST", "/v1/admin/tenants", json=payload, idempotency_key=idempotency_key
            )
        )

    async def tenants(
        self, *, after: str = "", limit: int = 100, cursor: str | None = None
    ) -> list[TenantInfo]:
        return (await self.tenants_page(after=after, limit=limit, cursor=cursor)).items

    async def tenants_page(
        self, *, after: str = "", limit: int = 100, cursor: str | None = None
    ) -> Page[TenantInfo]:
        params = {"after": after or None, "limit": limit, "cursor": cursor}
        data, next_cursor = await self._t.request_page(
            "/v1/admin/tenants", params={k: v for k, v in params.items() if v is not None}
        )
        return Page[TenantInfo](
            items=[TenantInfo.model_validate(t) for t in data], next_cursor=next_cursor
        )

    async def get_tenant(self, tenant_id: str) -> TenantInfo:
        return TenantInfo.model_validate(
            await self._t.request("GET", f"/v1/admin/tenants/{tenant_id}")
        )

    async def update_tenant(self, tenant_id: str, **changes: Any) -> TenantInfo:
        """``name``, ``status``, ``retention_days`` / ``clear_retention``,
        ``rate_limit_per_minute`` / ``clear_rate_limit``."""
        return TenantInfo.model_validate(
            await self._t.request("PATCH", f"/v1/admin/tenants/{tenant_id}", json=changes)
        )


class TenantAPI:
    """Administration of one tenant: its keys, workspaces (teams) and read audit."""

    def __init__(self, client: MemoryClient, *, tenant_id: str | None = None) -> None:
        self._t = client.transport
        self._headers = {HEADER_TENANT: tenant_id} if tenant_id else {}
        self.keys = KeysAPI(self)
        self.workspaces = WorkspacesAPI(self)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._t.request(method, path, headers=self._headers, **kwargs)

    async def _page(self, path: str, **params: Any) -> tuple[Any, str | None]:
        query = {k: v for k, v in params.items() if v is not None}
        return await self._t.request_page(path, params=query, headers=self._headers)

    async def model_key_status(self) -> AgentKeyStatus:
        """The tenant's model key: what every agent without a key of its own calls with."""
        return AgentKeyStatus.model_validate(await self._request("GET", "/v1/model-key"))

    async def set_model_key(
        self, virtual_key: str, *, idempotency_key: str | None = None
    ) -> AgentKeyStatus:
        data = await self._request(
            "PUT",
            "/v1/model-key",
            json={"virtual_key": virtual_key},
            idempotency_key=idempotency_key,
        )
        return AgentKeyStatus.model_validate(data)

    async def revoke_model_key(self, *, idempotency_key: str | None = None) -> AgentKeyStatus:
        data = await self._request("DELETE", "/v1/model-key", idempotency_key=idempotency_key)
        return AgentKeyStatus.model_validate(data)

    async def model_policy(self) -> ModelPolicy:
        """The tenant's model policy: the uses the model may be called for, whether reads
        are model-assisted, and the model each use calls."""
        return ModelPolicy.model_validate(await self._request("GET", "/v1/model-key/policy"))

    async def set_model_policy(
        self,
        uses: Sequence[str],
        *,
        read_assist: bool,
        models: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> ModelPolicy:
        """``models`` names the gateway model (provider/model) a use calls, e.g.
        ``{"grounding_judge": "gemini/gemini-3.8-flash"}``; the others call the default."""
        data = await self._request(
            "PUT",
            "/v1/model-key/policy",
            json={"uses": list(uses), "read_assist": read_assist, "models": dict(models or {})},
            idempotency_key=idempotency_key,
        )
        return ModelPolicy.model_validate(data)

    async def model_usage(
        self, *, since: date | None = None, until: date | None = None
    ) -> ModelUsage:
        """Tokens and calls per day and use (default: the last 30 days)."""
        params = {
            name: value.isoformat()
            for name, value in (("since", since), ("until", until))
            if value is not None
        }
        return ModelUsage.model_validate(
            await self._request("GET", "/v1/model-key/usage", params=params)
        )

    async def reads(
        self, *, after: Any = None, before: Any = None, limit: int = 100, cursor: str | None = None
    ) -> list[ReadAuditRecord]:
        """Who read which records, newest first. Page older entries with the cursor (or
        ``before=<the last entry's at>``); ``after`` is a since-filter."""
        return (await self.reads_page(after=after, before=before, limit=limit, cursor=cursor)).items

    async def reads_page(
        self, *, after: Any = None, before: Any = None, limit: int = 100, cursor: str | None = None
    ) -> Page[ReadAuditRecord]:
        params: dict[str, Any] = {"limit": limit, "cursor": cursor}
        for name, value in (("after", after), ("before", before)):
            if value is not None:
                params[name] = value.isoformat() if hasattr(value, "isoformat") else value
        data, next_cursor = await self._page("/v1/reads", **params)
        return Page[ReadAuditRecord](
            items=[ReadAuditRecord.model_validate(r) for r in data], next_cursor=next_cursor
        )


class KeysAPI:
    def __init__(self, tenant: TenantAPI) -> None:
        self._tenant = tenant

    async def issue(
        self,
        role: KeyRole,
        name: str,
        *,
        workspace_id: str | None = None,
        expires_in_days: int | None = None,
        may_act_as: Sequence[str] = ("*",),
        idempotency_key: str | None = None,
    ) -> IssuedKey:
        """A new key; its ``token`` is shown once. ``may_act_as`` names the principals its
        requests may act for (``user:<id>``, ``agent:<id>``, ``*`` for any). With
        ``idempotency_key`` a retry returns the same key and ``token=None``; without it every
        call issues another key."""
        payload = {
            "role": role,
            "name": name,
            "workspace_id": workspace_id,
            "expires_in_days": expires_in_days,
            "may_act_as": list(may_act_as),
        }
        return IssuedKey.model_validate(
            await self._tenant._request(
                "POST", "/v1/keys", json=payload, idempotency_key=idempotency_key
            )
        )

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> list[ApiKeyInfo]:
        return (await self.page(limit=limit, cursor=cursor)).items

    async def page(self, *, limit: int = 100, cursor: str | None = None) -> Page[ApiKeyInfo]:
        data, next_cursor = await self._tenant._page("/v1/keys", limit=limit, cursor=cursor)
        return Page[ApiKeyInfo](
            items=[ApiKeyInfo.model_validate(k) for k in data], next_cursor=next_cursor
        )

    async def update(self, key_id: str, *, may_act_as: Sequence[str]) -> ApiKeyInfo:
        """Change whom a key may act for; it applies on the key's next request."""
        data = await self._tenant._request(
            "PATCH", f"/v1/keys/{key_id}", json={"may_act_as": list(may_act_as)}
        )
        return ApiKeyInfo.model_validate(data)

    async def revoke(self, key_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/keys/{key_id}")

    async def whoami(self) -> KeyInfo:
        """Who the client's own key is (``GET /v1/keys/self``)."""
        return KeyInfo.model_validate(await self._tenant._request("GET", "/v1/keys/self"))


class WorkspacesAPI:
    def __init__(self, tenant: TenantAPI) -> None:
        self._tenant = tenant

    async def create(
        self, name: str, *, workspace_id: str | None = None, idempotency_key: str | None = None
    ) -> WorkspaceInfo:
        payload = {"name": name, "workspace_id": workspace_id}
        return WorkspaceInfo.model_validate(
            await self._tenant._request(
                "POST", "/v1/workspaces", json=payload, idempotency_key=idempotency_key
            )
        )

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> list[WorkspaceInfo]:
        return (await self.page(limit=limit, cursor=cursor)).items

    async def page(self, *, limit: int = 100, cursor: str | None = None) -> Page[WorkspaceInfo]:
        data, next_cursor = await self._tenant._page("/v1/workspaces", limit=limit, cursor=cursor)
        return Page[WorkspaceInfo](
            items=[WorkspaceInfo.model_validate(w) for w in data], next_cursor=next_cursor
        )

    async def get(self, workspace_id: str) -> WorkspaceInfo:
        return WorkspaceInfo.model_validate(
            await self._tenant._request("GET", f"/v1/workspaces/{workspace_id}")
        )

    async def delete(self, workspace_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/workspaces/{workspace_id}")

    async def set_member(
        self, workspace_id: str, principal: str, *, role: MemberRole = "member"
    ) -> WorkspaceMemberInfo:
        """``principal`` is ``user:<id>`` or ``agent:<id>``."""
        return WorkspaceMemberInfo.model_validate(
            await self._tenant._request(
                "PUT", f"/v1/workspaces/{workspace_id}/members/{principal}", json={"role": role}
            )
        )

    async def remove_member(self, workspace_id: str, principal: str) -> None:
        await self._tenant._request("DELETE", f"/v1/workspaces/{workspace_id}/members/{principal}")

    async def members(self, workspace_id: str) -> list[WorkspaceMemberInfo]:
        data = await self._tenant._request("GET", f"/v1/workspaces/{workspace_id}/members")
        return [WorkspaceMemberInfo.model_validate(m) for m in data]
