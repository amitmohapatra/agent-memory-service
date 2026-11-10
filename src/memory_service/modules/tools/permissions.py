"""A person's tool permissions: "Always allow" for a tool, kept per tenant and user.

The harness writes one when a person answers an approval with "Always allow", with the
service credential it acts with (only that credential may: an agent's or a user's own key
cannot grant itself a tool), and reads the user's list when a run of theirs first asks. Until
the permission is revoked, the tool's calls asked of the same person or role run without
asking in every conversation of the user. The service keeps them; the harness decides with
them (an administrator's ``approve_when`` and an irreversible tool still ask).
"""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.domain.ids import new_id
from memory_service.ports.tools import ToolPermission
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory


class ToolPermissions:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self.uow_factory = uow_factory

    async def of_user(self, tenant_id: str, user_id: str) -> list[ToolPermission]:
        async with self.uow_factory() as uow:
            return await uow.tool_permissions.of_user(tenant_id, user_id)

    async def allow(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        user_id: str,
        tool: str,
        *,
        assignee: str | None,
        allowed_by: str,
    ) -> ToolPermission:
        """The user's permission for ``tool`` asked of ``assignee``: stored, or the one
        already stored."""
        return await uow.tool_permissions.allow(
            ToolPermission(
                permission_id=new_id("tool_permission"),
                tenant_id=tenant_id,
                user_id=user_id,
                tool=tool,
                assignee=assignee,
                allowed_by=allowed_by,
                created_at=datetime.now(UTC),
            )
        )

    async def find(
        self, uow: UnitOfWork, tenant_id: str, permission_id: str
    ) -> ToolPermission | None:
        return await uow.tool_permissions.get(tenant_id, permission_id)

    async def revoke(self, uow: UnitOfWork, tenant_id: str, permission_id: str) -> None:
        await uow.tool_permissions.revoke(tenant_id, permission_id)
