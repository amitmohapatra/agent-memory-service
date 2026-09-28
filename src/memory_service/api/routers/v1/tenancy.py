"""Tenant administration: keys, workspaces (teams), groups and the read audit.

The tenant is the credential's own; the platform key may administer any tenant by naming it
in ``X-Trellis-Tenant``. Membership changes take effect on the next request: the authorization
tuples are written with the rows and the membership revision is bumped in the same unit of work.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse

from memory_service.api.deps import (
    AdministeredTenantDep,
    ContainerDep,
    ServicePrincipalDep,
    request_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.pagination import CursorQuery, decode_cursor, encode_cursor, link_next, page
from memory_service.api.schemas.tenancy import (
    ApiKeyResponse,
    CreateGroupRequest,
    CreateWorkspaceRequest,
    GroupMemberResponse,
    GroupResponse,
    IssuedKeyResponse,
    IssueKeyRequest,
    ReadAuditResponse,
    SetMemberRequest,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)

router = APIRouter(tags=["tenancy"])
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)


def _service(container):  # type: ignore[no-untyped-def]
    return container.services["tenancy"]


def _without_token(body: dict) -> dict:
    """What a replay may return: the record, never the secret shown on the first response."""
    return {**body, "token": None}


# -- keys -----------------------------------------------------------------------------


@router.post(
    "/keys",
    response_model=IssuedKeyResponse,
    status_code=201,
    responses=_ERRORS,
    summary="Issue an admin or service key for this tenant (the secret is shown once)",
    description="Send `Idempotency-Key` to make a retry safe: it returns the same key record "
    "with `Idempotent-Replayed: true` and `token: null`; the secret is never shown twice. "
    "Without the header, every call issues a new key.",
)
async def issue_key(
    request: Request,
    body: IssueKeyRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    principal: ServicePrincipalDep,
) -> JSONResponse:
    ctx = request_context(request, tenant_id)
    payload = body.model_dump(mode="json")

    async def handler(uow):  # type: ignore[no-untyped-def]
        issued = await _service(container).issue_key(
            uow,
            tenant_id,
            role=body.role,
            name=body.name,
            workspace_id=body.workspace_id,
            expires_in_days=body.expires_in_days,
            created_by=principal.service_id,
        )

        async def after_commit() -> None:
            container.services["tenant_registry"].observe_key(issued.key.key_id, tenant_id)
            # a "no such key" marker may be cached from an earlier guess at this id
            await container.services["api_keys"].forget_missing(issued.key.key_id)

        return 201, IssuedKeyResponse.issued(issued).model_dump(mode="json"), after_commit

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=payload,
        handler=handler,
        stored_body=_without_token,
    )


@router.get(
    "/keys",
    response_model=list[ApiKeyResponse],
    responses=_ERRORS,
    summary="List keys, oldest first (cursor paged)",
)
async def list_keys(
    request: Request,
    response: Response,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ApiKeyResponse]:
    position = decode_cursor(cursor, fields={"created_at": datetime, "key_id": str})
    after = (position["created_at"], position["key_id"]) if position else None
    async with container.services["uow_factory"]() as uow:
        keys = await _service(container).list_keys(uow, tenant_id, after=after, limit=limit + 1)
    items, next_cursor = page(
        keys,
        limit=limit,
        position=lambda k: {"created_at": k.created_at.isoformat(), "key_id": k.key_id},
    )
    link_next(request, response, next_cursor)
    return [ApiKeyResponse.of(k) for k in items]


@router.delete(
    "/keys/{key_id}",
    status_code=204,
    responses=_ERRORS,
    summary="Revoke a key; it fails on its next request from any instance",
)
async def revoke_key(
    key_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> None:
    verifier = container.services["api_keys"]
    async with container.services["uow_factory"]() as uow:
        revoked = await _service(container).revoke_key(uow, tenant_id, key_id)
        if revoked:
            # Tombstone before the commit, strictly: a reader between the two steps re-reads,
            # and a cache that is away turns this into a 503 to retry rather than a 204
            # while the key still serves from cache.
            await verifier.invalidate(key_id, strict=True)
        await uow.commit()
    if revoked:
        # only a key of this tenant: another tenant's key id must not be touched from here
        await verifier.invalidate(key_id)
        container.services["tenant_registry"].forget_key(key_id)


# -- workspaces -----------------------------------------------------------------------


@router.post(
    "/workspaces",
    response_model=WorkspaceResponse,
    status_code=201,
    responses=_ERRORS,
    summary="Create a workspace: a team whose members share what they store in it",
)
async def create_workspace(
    request: Request,
    body: CreateWorkspaceRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> JSONResponse:
    ctx = request_context(request, tenant_id)
    payload = body.model_dump(mode="json")

    async def handler(uow):  # type: ignore[no-untyped-def]
        workspace = await _service(container).create_workspace(
            uow, tenant_id, name=body.name, workspace_id=body.workspace_id
        )
        return 201, WorkspaceResponse.of(workspace).model_dump(mode="json"), None

    return await run_idempotent(
        request, container, ctx, key=request.state.idempotency_key, payload=payload, handler=handler
    )


@router.get(
    "/workspaces",
    response_model=list[WorkspaceResponse],
    responses=_ERRORS,
    summary="List workspaces",
)
async def list_workspaces(
    request: Request,
    response: Response,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[WorkspaceResponse]:
    position = decode_cursor(cursor, fields=("workspace_id",))
    async with container.services["uow_factory"]() as uow:
        workspaces = await _service(container).list_workspaces(
            uow, tenant_id, after=position["workspace_id"] if position else "", limit=limit + 1
        )
    items, next_cursor = page(
        workspaces, limit=limit, position=lambda w: {"workspace_id": w.workspace_id}
    )
    link_next(request, response, next_cursor)
    return [WorkspaceResponse.of(w) for w in items]


@router.get(
    "/workspaces/{workspace_id}",
    response_model=WorkspaceResponse,
    responses=_ERRORS,
    summary="Get a workspace",
)
async def get_workspace(
    workspace_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> WorkspaceResponse:
    async with container.services["uow_factory"]() as uow:
        return WorkspaceResponse.of(
            await _service(container).get_workspace(uow, tenant_id, workspace_id)
        )


@router.delete(
    "/workspaces/{workspace_id}",
    status_code=204,
    responses=_ERRORS,
    summary="Delete a workspace; every member loses its audience and every key bound to it "
    "is revoked at once",
)
async def delete_workspace(
    workspace_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> None:
    verifier = container.services["api_keys"]
    async with container.services["uow_factory"]() as uow:
        revoked = await _service(container).delete_workspace(uow, tenant_id, workspace_id)
        for key_id in revoked:
            # as in revoke_key: strictly and before the commit
            await verifier.invalidate(key_id, strict=True)
        await uow.commit()
    for key_id in revoked:
        await verifier.invalidate(key_id)
        container.services["tenant_registry"].forget_key(key_id)


@router.get(
    "/workspaces/{workspace_id}/members",
    response_model=list[WorkspaceMemberResponse],
    responses=_ERRORS,
    summary="List a workspace's members",
)
async def list_members(
    workspace_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> list[WorkspaceMemberResponse]:
    async with container.services["uow_factory"]() as uow:
        members = await _service(container).members(uow, tenant_id, workspace_id)
    return [WorkspaceMemberResponse.of(m) for m in members]


@router.put(
    "/workspaces/{workspace_id}/members/{principal_ref}",
    response_model=WorkspaceMemberResponse,
    responses=_ERRORS,
    summary="Admit a user, agent or group (user:<id> | agent:<id> | group:<id>) with one role",
)
async def set_member(
    workspace_id: str,
    principal_ref: str,
    body: SetMemberRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    principal: ServicePrincipalDep,
) -> WorkspaceMemberResponse:
    async with container.services["uow_factory"]() as uow:
        member = await _service(container).set_member(
            uow,
            tenant_id,
            workspace_id,
            principal_ref,
            role=body.role,
            added_by=principal.service_id,
        )
        await uow.commit()
    return WorkspaceMemberResponse.of(member)


@router.delete(
    "/workspaces/{workspace_id}/members/{principal_ref}",
    status_code=204,
    responses=_ERRORS,
    summary="Remove a member; its next request no longer reads the workspace",
)
async def remove_member(
    workspace_id: str, principal_ref: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> None:
    async with container.services["uow_factory"]() as uow:
        await _service(container).remove_member(uow, tenant_id, workspace_id, principal_ref)
        await uow.commit()


# -- groups ---------------------------------------------------------------------------


@router.post(
    "/groups",
    response_model=GroupResponse,
    status_code=201,
    responses=_ERRORS,
    summary="Create a group of users a workspace can admit at once",
)
async def create_group(
    request: Request,
    body: CreateGroupRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> JSONResponse:
    ctx = request_context(request, tenant_id)
    payload = body.model_dump(mode="json")

    async def handler(uow):  # type: ignore[no-untyped-def]
        group = await _service(container).create_group(
            uow, tenant_id, name=body.name, group_id=body.group_id
        )
        return 201, GroupResponse.of(group).model_dump(mode="json"), None

    return await run_idempotent(
        request, container, ctx, key=request.state.idempotency_key, payload=payload, handler=handler
    )


@router.get(
    "/groups",
    response_model=list[GroupResponse],
    responses=_ERRORS,
    summary="List groups (cursor paged)",
)
async def list_groups(
    request: Request,
    response: Response,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[GroupResponse]:
    position = decode_cursor(cursor, fields=("group_id",))
    async with container.services["uow_factory"]() as uow:
        groups = await _service(container).list_groups(
            uow, tenant_id, after=position["group_id"] if position else "", limit=limit + 1
        )
    items, next_cursor = page(groups, limit=limit, position=lambda g: {"group_id": g.group_id})
    link_next(request, response, next_cursor)
    return [GroupResponse.of(g) for g in items]


@router.delete("/groups/{group_id}", status_code=204, responses=_ERRORS, summary="Delete a group")
async def delete_group(
    group_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> None:
    async with container.services["uow_factory"]() as uow:
        await _service(container).delete_group(uow, tenant_id, group_id)
        await uow.commit()


@router.get(
    "/groups/{group_id}/members",
    response_model=list[GroupMemberResponse],
    responses=_ERRORS,
    summary="List a group's users",
)
async def list_group_members(
    group_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> list[GroupMemberResponse]:
    async with container.services["uow_factory"]() as uow:
        members = await _service(container).group_members(uow, tenant_id, group_id)
    return [GroupMemberResponse.of(m) for m in members]


@router.put(
    "/groups/{group_id}/members/{user_id}",
    response_model=GroupMemberResponse,
    responses=_ERRORS,
    summary="Add a user to a group",
)
async def add_group_user(
    group_id: str,
    user_id: str,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    principal: ServicePrincipalDep,
) -> GroupMemberResponse:
    async with container.services["uow_factory"]() as uow:
        member = await _service(container).add_group_user(
            uow, tenant_id, group_id, user_id, added_by=principal.service_id
        )
        await uow.commit()
    return GroupMemberResponse.of(member)


@router.delete(
    "/groups/{group_id}/members/{user_id}",
    status_code=204,
    responses=_ERRORS,
    summary="Remove a user from a group",
)
async def remove_group_user(
    group_id: str, user_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> None:
    async with container.services["uow_factory"]() as uow:
        await _service(container).remove_group_user(uow, tenant_id, group_id, user_id)
        await uow.commit()


# -- read audit -----------------------------------------------------------------------


@router.get(
    "/reads",
    response_model=list[ReadAuditResponse],
    responses=_ERRORS,
    summary="Who read which records, newest first (cursor paged; the keyset is the instant, so "
    "entries sharing one instant across a page boundary need a larger page)",
)
async def list_reads(
    request: Request,
    response: Response,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    after: Annotated[
        datetime | None, Query(description="only entries newer than this instant (a since-filter)")
    ] = None,
    before: Annotated[
        datetime | None,
        Query(description="only entries older than this instant: the cursor for the next page"),
    ] = None,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[ReadAuditResponse]:
    position = decode_cursor(cursor, fields={"before": datetime})
    if position is not None:
        before = position["before"]
    await container.services["read_audit"].flush()
    async with container.services["uow_factory"]() as uow:
        entries = await uow.read_audit.list(
            tenant_id, after=_aware(after), before=_aware(before), limit=limit + 1
        )
    items = entries[:limit]
    next_cursor = (
        encode_cursor({"before": items[-1].at.isoformat()}) if len(entries) > limit else None
    )
    link_next(request, response, next_cursor)
    return [ReadAuditResponse.model_validate(e.model_dump()) for e in items]


def _aware(instant: datetime | None) -> datetime | None:
    """A naive instant means UTC; the store compares against timestamptz."""
    if instant is not None and instant.tzinfo is None:
        return instant.replace(tzinfo=UTC)
    return instant
