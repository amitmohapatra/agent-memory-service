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
from memory_service.modules.authz.visibility import VisibilitySpecification, readable_by
from memory_service.ports.authorization import AuthorizedScope

pytestmark = pytest.mark.security

TENANTS = ["acme", "globex"]
USERS = ["u1", "u2"]
AGENTS = ["research", "writer"]


def _oracle(reader: dict, obj: dict) -> bool:
    """Independent statement of the visibility rules (does not use the implementation).

    ``author`` is the escape hatch a stored row carries: whoever wrote a memory keeps read
    access to it even if they later lose the audience it was shared with. THREAD is excluded
    - there the author IS the audience, so the escape cancelled the scope entirely and a
    memory belonging to one conversation could be read from all of them.
    """
    if reader["tenant"] != obj["tenant"]:
        return False
    v = obj["visibility"]
    author = reader["principal"] == obj["owner"]
    if v == "GLOBAL" or v == "TENANT":
        return True
    if v == "PRIVATE":
        return author
    if v == "USER":
        return reader["user"] == obj["user"] or author
    if v == "GROUP":
        return obj["group"] in reader["groups"] or author
    if v == "THREAD":
        return obj["thread"] in reader["threads"]
    if v == "WORKSPACE":
        return obj["workspace"] in reader["workspaces"] or author
    if v == "WORK":
        return obj["work"] in reader["works"] or author
    if v == "AGENT_GROUP":
        return obj["agent_group"] in reader["agent_groups"] or author
    if v == "RUN":  # the writing run and the run that spawned it - identity does not enter
        return obj["run"] in reader["runs"]
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
        "runs": st.lists(st.sampled_from(["run1", "run2"]), max_size=2),
    }
)
object_st = st.fixed_dictionaries(
    {
        "tenant": st.sampled_from(TENANTS),
        "visibility": st.sampled_from([v.value for v in Visibility]),
        "owner_agent": st.one_of(st.none(), st.sampled_from(AGENTS)),
        "user": st.sampled_from(USERS),
        "group": st.sampled_from(["legal", "finance"]),
        "thread": st.sampled_from(["thr1", "thr2"]),
        "workspace": st.sampled_from(["ws1", "ws2"]),
        "work": st.sampled_from(["w1", "w2"]),
        "agent_group": st.sampled_from(["crew", "other"]),
        "run": st.sampled_from(["run1", "run2", "run3"]),
    }
).map(
    # The owner is DERIVED from the user anchor, never drawn independently: keys_for builds
    # both from one execution context, so a row owned by user:u2 whose USER anchor is u1
    # cannot exist. Drawing them apart invented objects the service cannot mint and the
    # property then failed on them. An agent principal is bound to its user for the same
    # reason - see domain/context.py:141-153, where an unbound agent id let any caller
    # assume any agent's identity.
    lambda o: {
        **o,
        "owner": (
            f"agent:{o['user']}/{o['owner_agent']}" if o["owner_agent"] else f"user:{o['user']}"
        ),
    }
)


def _spec(reader: dict) -> VisibilitySpecification:
    principal = (
        f"agent:{reader['user']}/{reader['agent']}"
        if reader["is_agent"]
        else f"user:{reader['user']}"
    )
    scope = AuthorizedScope(
        tenant_id=reader["tenant"],
        principal=principal,
        user_id=reader["user"],
        thread_ids=reader["threads"],
        agent_group_ids=reader["agent_groups"],
        run_ids=reader.get("runs", []),
    )
    reader["principal"] = principal
    return VisibilitySpecification.from_scope(scope)


def _keys(obj: dict) -> list[str]:
    """The keys a STORED row carries - which is ``readable_by``, not ``visibility_keys``.

    This used to call ``visibility_keys``, which omits the author's own principal key that
    every stored non-PRIVATE row actually has. Both sides of the comparison then agreed on a
    key shape nothing in the system produces, and the gate passed while proving nothing.
    """
    return readable_by(
        obj["tenant"],
        Visibility(obj["visibility"]),
        owner_principal=obj["owner"],
        user_id=obj["user"],
        thread_id=obj["thread"],
        agent_group_id=obj["agent_group"],
        agent_run_id=obj.get("run", "run0"),
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
    keys = readable_by("acme", Visibility.PRIVATE, owner_principal="agent:u1/research")
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
        ]
    )
    svc = AuthorizationService(provider, None)
    scope = await svc.scope(MemoryExecutionContext(tenant_id="acme", user_id="u1"))
    assert scope.thread_ids == ["thr1"]
    scope_g = await svc.scope(MemoryExecutionContext(tenant_id="globex", user_id="u1"))
    assert scope_g.thread_ids == ["thr9"], "the same user id in another tenant is another scope"
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
        "acme", readable_by("acme", Visibility.USER, owner_principal="user:u1", user_id="u1")
    )
    assert not spec.allows(
        "acme", readable_by("acme", Visibility.PRIVATE, owner_principal="user:u1")
    )
    # Its own private memories, under the principal it actually writes with. That principal
    # is bound to the user the agent runs for (``agent:u1/research``), so the same agent_id
    # acting for a different user is a different principal.
    assert spec.allows(
        "acme",
        readable_by("acme", Visibility.PRIVATE, owner_principal=agent_ctx.principal_id),
    )
    assert agent_ctx.principal_id == "agent:u1/research"
    # The unbound form is what made a private memory readable by anyone who named the agent:
    # nothing may write it, and nothing may read it.
    assert not spec.allows(
        "acme", readable_by("acme", Visibility.PRIVATE, owner_principal="agent:research")
    )


def test_a_thread_memory_is_not_readable_from_another_thread_by_its_author() -> None:
    """The leak an integrator reported: a new conversation recalling the previous one's turns.

    Every non-PRIVATE row carries its author's ``principal:`` key so that losing a
    membership does not lose access to what you wrote. Applied to THREAD it was
    self-defeating - the author matched their own key from any thread, the thread key was
    never reached, and ``visibility=THREAD`` meant "this thread, or anywhere if you wrote
    it".

    Dropping that key is only half of it, because the author still has to read the memory in
    the thread it belongs to, and a thread used to be granted in exactly one place
    (``conversation/service.py``, POST /v1/threads). An observation naming a thread now
    ensures it, so the author reaches their own memory through the grant instead of through
    authorship - a key that says which conversation, not which person.

    The property test above could not catch any of this: it built its objects with
    ``visibility_keys``, which never appended the author key, so both sides of the
    comparison agreed on a key shape no stored row has.
    """
    author = "user:u1"
    keys = readable_by("acme", Visibility.THREAD, owner_principal=author, thread_id="thrA")
    assert f"principal:acme/{author}" not in keys, "the author key defeats thread scoping"

    def _reader(threads: list[str]) -> VisibilitySpecification:
        return VisibilitySpecification.from_scope(
            AuthorizedScope(
                tenant_id="acme",
                principal=author,
                user_id="u1",
                thread_ids=threads,
                agent_group_ids=[],
                run_ids=[],
            )
        )

    assert _reader(["thrA"]).allows("acme", keys), "the thread it belongs to reads it"
    assert not _reader(["thrB"]).allows("acme", keys), "another thread must not, author or not"


def _scope_for(principal: str, threads: list[str]) -> AuthorizedScope:
    return AuthorizedScope(
        tenant_id="acme",
        principal=principal,
        user_id=principal.split(":", 1)[1],
        thread_ids=threads,
        agent_group_ids=[],
        run_ids=[],
    )


def test_a_query_inside_a_thread_carries_only_that_thread() -> None:
    """Being authorized for a thread is not the same as working in it.

    ``from_scope`` put EVERY granted thread in the caller's audience, so someone with twenty
    conversations carried twenty thread keys into every query and a memory scoped to one was
    readable from all of them. The caller's current thread narrows it.
    """
    scope = _scope_for("user:u1", ["thrA", "thrB", "thrC"])
    everywhere = VisibilitySpecification.from_scope(scope)
    assert {"thread:acme/thrA", "thread:acme/thrB"} <= everywhere.keys

    inside = VisibilitySpecification.from_scope(scope, current_thread_id="thrA")
    assert "thread:acme/thrA" in inside.keys
    assert "thread:acme/thrB" not in inside.keys
    assert "thread:acme/thrC" not in inside.keys


def test_naming_a_thread_grants_nothing() -> None:
    """The narrowing is an INTERSECTION, and it has to stay one.

    Measured against the running service before this was written: a second user who names
    someone else's thread matches no key and reads nothing. If ``current_thread_id`` were
    added to the audience directly rather than intersected with what was granted, that would
    become a genuine cross-user leak - strictly worse than the same-user contamination this
    change is fixing.
    """
    stranger = VisibilitySpecification.from_scope(
        _scope_for("user:u2", []), current_thread_id="thrA"
    )
    assert "thread:acme/thrA" not in stranger.keys

    keys = readable_by("acme", Visibility.THREAD, owner_principal="user:u1", thread_id="thrA")
    assert not stranger.allows("acme", keys), "naming a thread is not being granted it"
