"""Public /v1/tool-permissions routes: a person's "Always allow" for a tool
(``modules/tools/permissions.py``).

The harness grants one with its service credential when the person answers an approval with
"Always allow", and lists the user's when a run of theirs first asks; anyone acting for the user
lists and revokes them. Only the credential that acts for everyone in the tenant grants
(:func:`~memory_service.api.deps.acts_for_anyone`): an agent's or a user's own key is refused,
so no agent allows itself a tool.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    acts_for_anyone,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import NO_CONTENT, run_idempotent
from memory_service.api.params import PermissionIdPath
from memory_service.domain.errors import AuthorizationFailed, NotFound
from memory_service.modules.tools.permissions import ToolPermissions
from memory_service.ports.tools import ToolPermission

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
_READ_ERRORS = error_responses(401, 403, 422, 503)


class AllowToolRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"user": "alice", "tool": "crm-update_quote", "allowed_by": "alice"}]
        },
    )

    user: str = Field(
        min_length=1,
        max_length=200,
        description="The user the tool is allowed for; when X-Trellis-User is sent, it must "
        "name the same user.",
    )
    tool: str = Field(min_length=1, max_length=200, description="The tool, as its calls name it.")
    assignee: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="The person or role the tool's approvals are asked of; null: an approval "
        "anyone may answer. A call asked of someone else still asks.",
    )
    allowed_by: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Who answered the approval with Always allow; null: the user.",
    )


class ToolPermissionOut(BaseModel):
    permission_id: str = Field(description="The permission's id (tpm_...).")
    user: str = Field(description="The user the tool is allowed for.")
    tool: str = Field(description="The tool allowed.")
    assignee: str | None = Field(
        description="The person or role its approvals are asked of; null: anyone."
    )
    allowed_by: str = Field(description="Who allowed it.")
    created_at: datetime = Field(description="When it was allowed (UTC).")

    @classmethod
    def of(cls, permission: ToolPermission) -> ToolPermissionOut:
        return cls(
            permission_id=permission.permission_id,
            user=permission.user_id,
            tool=permission.tool,
            assignee=permission.assignee,
            allowed_by=permission.allowed_by,
            created_at=permission.created_at,
        )


class ToolPermissionsResponse(BaseModel):
    permissions: list[ToolPermissionOut] = Field(
        description="The user's permissions, oldest first."
    )


def _service(container) -> ToolPermissions:  # type: ignore[no-untyped-def]
    return container.services["tool_permissions"]


@router.get(
    "/tool-permissions",
    response_model=ToolPermissionsResponse,
    tags=["tools"],
    responses=_READ_ERRORS,
    summary="List the tools a user allowed always (Always allow)",
)
async def list_tool_permissions(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    user: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
            description="The user whose permissions to list; when X-Trellis-User is sent, it "
            "must name the same user.",
        ),
    ],
) -> ToolPermissionsResponse:
    ctx = build_context(request, container, ScopeBody(user_id=user))
    found = await _service(container).of_user(ctx.tenant_id, user)
    return ToolPermissionsResponse(permissions=[ToolPermissionOut.of(p) for p in found])


@router.put(
    "/tool-permissions",
    response_model=ToolPermissionOut,
    tags=["tools"],
    responses=_ERRORS,
    summary="Allow a tool for a user always (idempotent; the service credential only)",
    description="The harness's service credential (a service key that may act for anyone, or "
    "a development key) grants it; any other credential - an agent's or a user's own key, an "
    "administrator's - is refused with 403. Allowing a tool already allowed returns the "
    "stored permission.",
)
async def allow_tool(
    request: Request,
    body: AllowToolRequest,
    container: ContainerDep,
    principal: ServicePrincipalDep,
):
    ctx = build_context(request, container, ScopeBody(user_id=body.user))
    if not acts_for_anyone(principal):
        raise AuthorizationFailed(
            "only the service credential that acts for everyone in the tenant allows a tool"
        )

    async def write(uow):  # type: ignore[no-untyped-def]
        stored = await _service(container).allow(
            uow,
            ctx.tenant_id,
            body.user,
            body.tool,
            assignee=body.assignee,
            allowed_by=body.allowed_by or body.user,
        )
        return 200, ToolPermissionOut.of(stored).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=body.model_dump(mode="json"),
        handler=write,
    )


@router.delete(
    "/tool-permissions/{permission_id}",
    status_code=204,
    tags=["tools"],
    responses=_ERRORS,
    summary="Revoke a permission: the tool asks again from the user's next run",
    description="By anyone acting for the permission's user. 404 when the tenant has no such "
    "permission, or X-Trellis-User names another user.",
)
async def revoke_tool_permission(
    permission_id: PermissionIdPath,
    request: Request,
    container: ContainerDep,
    ctx: HeaderContextDep,
):
    service = _service(container)

    async def write(uow):  # type: ignore[no-untyped-def]
        # looked up in the write, so a retry with the same Idempotency-Key replays the 204
        found = await service.find(uow, ctx.tenant_id, permission_id)
        if found is None or ctx.user_id not in (None, found.user_id):
            raise NotFound("Tool permission not found")
        build_context(request, container, ScopeBody(user_id=found.user_id))  # may act for them
        await service.revoke(uow, ctx.tenant_id, permission_id)
        return NO_CONTENT, {}, None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload={"permission_id": permission_id},
        handler=write,
    )
