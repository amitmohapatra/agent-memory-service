"""WORKSPACE: a team's shared audience, granted and revoked through the authorization service.

Keys, the resolver, narrowing to the workspace being worked in, group and agent membership,
and - the property that did not exist before - a removed member is denied on the next scope.
"""

from __future__ import annotations

import pytest

from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ScopeLevel, Visibility
from memory_service.domain.memory import Scope
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import (
    VisibilitySpecification,
    readable_by,
    visibility_keys,
)
from memory_service.ports.authorization import AuthorizedScope

pytestmark = pytest.mark.unit


def test_workspace_keys_and_anchor() -> None:
    scope = Scope(level=ScopeLevel.WORKSPACE, tenant_id="acme", workspace_id="finance")
    assert visibility_keys(
        "acme", Visibility.WORKSPACE, owner_principal="user:u1", scope=scope
    ) == ["workspace:acme/finance"]
    # the author keeps access after leaving, like USER and AGENT_GROUP (the oracle's "or author")
    assert readable_by("acme", Visibility.WORKSPACE, owner_principal="user:u1", scope=scope) == [
        "workspace:acme/finance",
        "principal:acme/user:u1",
    ]
    with pytest.raises(ValueError, match="WORKSPACE visibility requires workspace_id"):
        visibility_keys("acme", Visibility.WORKSPACE, owner_principal="user:u1")


def test_a_request_inside_a_workspace_reads_that_team_only() -> None:
    scope = AuthorizedScope(
        tenant_id="acme", principal="user:u1", user_id="u1", workspace_ids=["finance", "legal"]
    )
    everywhere = VisibilitySpecification.from_scope(scope)
    assert {"workspace:acme/finance", "workspace:acme/legal"} <= everywhere.keys
    inside = VisibilitySpecification.from_scope(scope, current_workspace_id="finance")
    assert "workspace:acme/finance" in inside.keys and "workspace:acme/legal" not in inside.keys
    # naming a workspace one is not a member of grants nothing
    outsider = VisibilitySpecification.from_scope(scope, current_workspace_id="ops")
    assert not any(k.startswith("workspace:") for k in outsider.keys)


def _service() -> AuthorizationService:
    return AuthorizationService(MemoryAuthorizationProvider(), None, decision_cache=False)


def _ctx(**fields) -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id="acme", **fields)


async def test_members_are_resolved_and_a_removed_member_is_denied_next_time() -> None:
    authz = _service()
    await authz.grant_workspace("acme", "finance")
    await authz.set_workspace_member("acme", "finance", "user:u1", "member")
    assert (await authz.scope(_ctx(user_id="u1"))).workspace_ids == ["finance"]
    assert (await authz.scope(_ctx(user_id="u2"))).workspace_ids == []
    await authz.revoke_workspace_member("acme", "finance", "user:u1")
    assert (await authz.scope(_ctx(user_id="u1"))).workspace_ids == []


async def test_one_role_per_principal_and_every_role_reads() -> None:
    authz = _service()
    await authz.grant_workspace("acme", "finance")
    previous = None
    for role in ("viewer", "member", "admin"):
        await authz.set_workspace_member(
            "acme",
            "finance",
            "user:u1",
            role,
            previous=previous,  # type: ignore[arg-type]
        )
        assert (await authz.scope(_ctx(user_id="u1"))).workspace_ids == ["finance"]
        previous = role
    held = {t.relation for t in authz.provider.dump() if t.user == "user:u1"}  # type: ignore[attr-defined]
    assert held == {"admin"}, "changing role replaces it rather than accumulating"


async def test_group_membership_reaches_the_workspace_and_revokes_with_the_group() -> None:
    authz = _service()
    await authz.grant_workspace("acme", "finance")
    await authz.grant_group("acme", "analysts")
    await authz.set_group_member("acme", "analysts", "u3")
    await authz.set_workspace_member("acme", "finance", "group:analysts", "member")
    assert (await authz.scope(_ctx(user_id="u3"))).workspace_ids == ["finance"]
    await authz.revoke_group_member("acme", "analysts", "u3")
    assert (await authz.scope(_ctx(user_id="u3"))).workspace_ids == []


async def test_an_unattended_agent_can_be_a_member_and_a_bound_agent_inherits_its_user() -> None:
    authz = _service()
    await authz.grant_workspace("acme", "finance")
    await authz.set_workspace_member("acme", "finance", "agent:ingest", "member")
    assert (await authz.scope(_ctx(agent_id="ingest"))).workspace_ids == ["finance"]
    await authz.set_workspace_member("acme", "finance", "user:u1", "member")
    assert (await authz.scope(_ctx(user_id="u1", agent_id="research"))).workspace_ids == ["finance"]
    assert (await authz.scope(_ctx(user_id="u9", agent_id="research"))).workspace_ids == []


async def test_a_tenant_admin_reads_every_workspace() -> None:
    authz = _service()
    await authz.grant_membership("acme", "boss", admin=True)
    await authz.grant_workspace("acme", "finance")
    await authz.grant_workspace("acme", "legal")
    assert (await authz.scope(_ctx(user_id="boss"))).workspace_ids == ["finance", "legal"]


async def test_cross_tenant_workspaces_never_resolve() -> None:
    authz = _service()
    await authz.grant_workspace("globex", "finance")
    await authz.set_workspace_member("globex", "finance", "user:u1", "member")
    assert (await authz.scope(_ctx(user_id="u1"))).workspace_ids == []


async def test_changing_a_role_drops_only_the_role_held_before() -> None:
    """A batch that deletes a tuple that does not exist fails and falls back to one write
    per tuple; naming only the previous role keeps every admit and change a single write."""
    authz = _service()
    await authz.grant_workspace("acme", "finance")
    batches: list[tuple[list, list]] = []
    original = authz.provider.write

    async def spy(writes, deletes=()):  # type: ignore[no-untyped-def]
        batches.append((list(writes), list(deletes)))
        return await original(writes, deletes)

    authz.provider.write = spy  # type: ignore[method-assign]
    await authz.set_workspace_member("acme", "finance", "user:u1", "viewer")
    assert batches[-1][1] == [], "a first admission deletes nothing"
    await authz.set_workspace_member("acme", "finance", "user:u1", "member", previous="viewer")
    assert [t.relation for t in batches[-1][1]] == ["viewer"], "only the role held before"
    await authz.set_workspace_member("acme", "finance", "user:u1", "member", previous="member")
    assert batches[-1][1] == [], "the same role again deletes nothing"
    assert (await authz.scope(_ctx(user_id="u1"))).workspace_ids == ["finance"]
