"""Release-blocking isolation gates.

    cross-tenant unauthorized retrieval = 0
    cross-user unauthorized retrieval   = 0
    private-agent leakage               = 0

Exercised at the authorization + visibility layer with an independent oracle and with
property-based generation. Retrieval-level versions live in later milestones' suites.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification, visibility_keys
from memory_service.ports.authorization import AuthorizedScope

pytestmark = pytest.mark.security

TENANTS = ["acme", "globex"]
USERS = ["u1", "u2"]
AGENTS = ["research", "writer"]


def _oracle(reader: dict, obj: dict) -> bool:
    """Independent statement of the visibility rules (does not use the implementation)."""
    if reader["tenant"] != obj["tenant"]:
        return False
    v = obj["visibility"]
    if v == "GLOBAL" or v == "TENANT":
        return True
    if v == "PRIVATE":
        return reader["principal"] == obj["owner"]
    if v == "USER":
        return reader["user"] == obj["user"]
    if v == "GROUP":
        return obj["group"] in reader["groups"]
    if v == "THREAD":
        return obj["thread"] in reader["threads"]
    if v == "WORKSPACE":
        return obj["workspace"] in reader["workspaces"]
    if v == "WORK":
        return obj["work"] in reader["works"]
    if v == "AGENT_GROUP":
        return obj["agent_group"] in reader["agent_groups"]
    raise AssertionError(v)


reader_st = st.fixed_dictionaries(
    {
        "tenant": st.sampled_from(TENANTS),
        "user": st.sampled_from(USERS),
        "is_agent": st.booleans(),
        "agent": st.sampled_from(AGENTS),
        "groups": st.lists(st.sampled_from(["legal", "finance"]), max_size=2),
        "threads": st.lists(st.sampled_from(["thr1", "thr2"]), max_size=2),
        "workspaces": st.lists(st.sampled_from(["ws1", "ws2"]), max_size=2),
        "works": st.lists(st.sampled_from(["w1"]), max_size=1),
        "agent_groups": st.lists(st.sampled_from(["crew"]), max_size=1),
    }
)
object_st = st.fixed_dictionaries(
    {
        "tenant": st.sampled_from(TENANTS),
        "visibility": st.sampled_from([v.value for v in Visibility]),
        "owner": st.sampled_from(["user:u1", "user:u2", "agent:research", "agent:writer"]),
        "user": st.sampled_from(USERS),
        "group": st.sampled_from(["legal", "finance"]),
        "thread": st.sampled_from(["thr1", "thr2"]),
        "workspace": st.sampled_from(["ws1", "ws2"]),
        "work": st.sampled_from(["w1", "w2"]),
        "agent_group": st.sampled_from(["crew", "other"]),
    }
)


def _spec(reader: dict) -> VisibilitySpecification:
    principal = f"agent:{reader['agent']}" if reader["is_agent"] else f"user:{reader['user']}"
    scope = AuthorizedScope(
        tenant_id=reader["tenant"],
        principal=principal,
        user_id=reader["user"],
        workspace_ids=reader["workspaces"],
        group_ids=reader["groups"],
        thread_ids=reader["threads"],
        work_ids=reader["works"],
        agent_group_ids=reader["agent_groups"],
    )
    reader["principal"] = principal
    return VisibilitySpecification.from_scope(scope)


def _keys(obj: dict) -> list[str]:
    return visibility_keys(
        obj["tenant"],
        Visibility(obj["visibility"]),
        owner_principal=obj["owner"],
        user_id=obj["user"],
        group_id=obj["group"],
        thread_id=obj["thread"],
        workspace_id=obj["workspace"],
        work_id=obj["work"],
        agent_group_id=obj["agent_group"],
    )


@given(reader=reader_st, obj=object_st)
@h_settings(max_examples=400, deadline=None)
def test_visibility_matches_independent_oracle(reader: dict, obj: dict) -> None:
    spec = _spec(reader)
    assert spec.allows(obj["tenant"], _keys(obj)) == _oracle(reader, obj)


def test_exhaustive_cross_tenant_never_leaks() -> None:
    leaks = 0
    for tenant_r, tenant_o, vis in itertools.product(TENANTS, TENANTS, Visibility):
        if tenant_r == tenant_o:
            continue
        reader = {
            "tenant": tenant_r,
            "user": "u1",
            "is_agent": False,
            "agent": "research",
            "groups": ["legal"],
            "threads": ["thr1"],
            "workspaces": ["ws1"],
            "works": ["w1"],
            "agent_groups": ["crew"],
        }
        obj = {
            "tenant": tenant_o,
            "visibility": vis.value,
            "owner": "user:u1",
            "user": "u1",
            "group": "legal",
            "thread": "thr1",
            "workspace": "ws1",
            "work": "w1",
            "agent_group": "crew",
        }
        if _spec(reader).allows(obj["tenant"], _keys(obj)):
            leaks += 1
    assert leaks == 0


def test_private_agent_memory_invisible_to_everyone_else() -> None:
    keys = visibility_keys("acme", Visibility.PRIVATE, owner_principal="agent:research")
    readers = [
        {
            "tenant": "acme",
            "user": "u1",
            "is_agent": True,
            "agent": "writer",
            "groups": [],
            "threads": ["thr1"],
            "workspaces": ["ws1"],
            "works": [],
            "agent_groups": ["crew"],
        },
        {
            "tenant": "acme",
            "user": "u1",
            "is_agent": False,
            "agent": "research",
            "groups": [],
            "threads": ["thr1"],
            "workspaces": ["ws1"],
            "works": [],
            "agent_groups": [],
        },
    ]
    assert not any(_spec(r).allows("acme", keys) for r in readers)
    owner = {
        "tenant": "acme",
        "user": "u1",
        "is_agent": True,
        "agent": "research",
        "groups": [],
        "threads": [],
        "workspaces": [],
        "works": [],
        "agent_groups": [],
    }
    assert _spec(owner).allows("acme", keys)


async def test_scope_resolution_is_tenant_bound() -> None:
    """Same user id in two tenants: the scope for tenant A contains nothing from tenant B."""
    provider = MemoryAuthorizationProvider()
    from memory_service.ports.authorization import RelationTuple as R

    await provider.write(
        [
            R(user="tenant:acme", relation="tenant", object="thread:acme/thr1"),
            R(user="user:u1", relation="owner", object="thread:acme/thr1"),
            R(user="tenant:globex", relation="tenant", object="thread:globex/thr9"),
            R(user="user:u1", relation="owner", object="thread:globex/thr9"),
            R(user="user:u1", relation="member", object="group:globex/secret"),
        ]
    )
    svc = AuthorizationService(provider, None)
    scope = await svc.scope(MemoryExecutionContext(tenant_id="acme", user_id="u1"))
    assert scope.thread_ids == ["thr1"] and scope.group_ids == []
    scope_g = await svc.scope(MemoryExecutionContext(tenant_id="globex", user_id="u1"))
    assert scope_g.thread_ids == ["thr9"] and scope_g.group_ids == ["secret"]
    # require() denies objects of the other tenant even with a matching id
    from memory_service.domain.errors import ScopeDenied

    with pytest.raises(ScopeDenied):
        await svc.require(
            MemoryExecutionContext(tenant_id="acme", user_id="u1"), "can_read", "thread", "thr9"
        )
    await svc.require(
        MemoryExecutionContext(tenant_id="globex", user_id="u1"), "can_read", "thread", "thr9"
    )


async def test_agent_inherits_user_access_but_not_user_private_memories() -> None:
    provider = MemoryAuthorizationProvider()
    from memory_service.ports.authorization import RelationTuple as R

    await provider.write(
        [
            R(user="tenant:acme", relation="tenant", object="thread:acme/thr1"),
            R(user="user:u1", relation="owner", object="thread:acme/thr1"),
        ]
    )
    svc = AuthorizationService(provider, None)
    agent_ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1").child_agent(
        agent_id="research"
    )
    scope = await svc.scope(agent_ctx)
    assert scope.thread_ids == ["thr1"]  # via the user it acts for
    spec = VisibilitySpecification.from_scope(scope)
    assert spec.allows(
        "acme", visibility_keys("acme", Visibility.USER, owner_principal="user:u1", user_id="u1")
    )
    assert not spec.allows(
        "acme", visibility_keys("acme", Visibility.PRIVATE, owner_principal="user:u1")
    )
    assert spec.allows(
        "acme", visibility_keys("acme", Visibility.PRIVATE, owner_principal="agent:research")
    )
