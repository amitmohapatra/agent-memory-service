"""Who may write into a workspace, and when a workspace id means a team at all.

A workspace id is an anchor any caller can name in a header. Before teams existed that was
harmless: nothing granted anyone access through it. Now a workspace row makes ``workspace``
tuples live - members read the team's documents, its admins read and write its threads - so
publishing into a *team* requires membership, and the tuples are written only for a team.
A bare anchor with no team row keeps its old meaning (a label, granting nothing), so existing
callers are untouched, and creating a team later does not adopt what was labelled before it.
"""

from __future__ import annotations

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.errors import NotFound, ValidationFailed
from memory_service.domain.tenancy import Workspace
from memory_service.modules.authz.service import AuthorizationService
from memory_service.ports.uow import UnitOfWork


async def require_workspace_member(
    uow: UnitOfWork,
    authz: AuthorizationService,
    ctx: MemoryExecutionContext,
    *,
    team_only: bool = True,
) -> Workspace | None:
    """Refuse unless the caller is a ``member`` (admins compute to it) of ``ctx.workspace_id``.

    Returns the team, or None when the id is a bare anchor. ``team_only=True`` (threads,
    documents): a bare anchor passes and the caller grants nothing through it.
    ``team_only=False`` (WORKSPACE-visibility writes): the team must exist, since the
    audience is its members.
    """
    if not ctx.workspace_id:
        if team_only:
            return None
        raise ValidationFailed(
            "WORKSPACE visibility requires workspace_id", details={"visibility": "WORKSPACE"}
        )
    workspace = await uow.workspaces.get(ctx.tenant_id, ctx.workspace_id)
    if workspace is None:
        if team_only:
            return None
        raise NotFound("Workspace not found")
    await authz.require(ctx, "member", "workspace", ctx.workspace_id)
    return workspace


async def guard_workspace_visibility(
    uow: UnitOfWork,
    authz: AuthorizationService,
    ctx: MemoryExecutionContext,
    visibility: Visibility | None,
) -> None:
    """Every path that mints a WORKSPACE audience - observations, messages, tool records -
    passes through here, so the membership rule has one home."""
    if visibility is Visibility.WORKSPACE:
        await require_workspace_member(uow, authz, ctx, team_only=False)
