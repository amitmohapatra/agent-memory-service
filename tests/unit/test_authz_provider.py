"""Zanzibar semantics of the in-memory provider, shared golden checks for OpenFGA."""

from __future__ import annotations

import pytest

from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider
from memory_service.ports.authorization import AccessCheck, RelationTuple

T = lambda u, r, o: RelationTuple(user=u, relation=r, object=o)  # noqa: E731

# (user, relation, object, expected) evaluated against GOLDEN_TUPLES; also run against real
# OpenFGA in tests/contract/test_openfga_contract.py (docker marker).
GOLDEN_TUPLES = [
    T("user:admin1", "admin", "tenant:acme"),
    T("user:u1", "member", "tenant:acme"),
    T("user:u2", "member", "tenant:acme"),
    T("user:u3", "member", "tenant:globex"),
    T("tenant:acme", "tenant", "group:acme/legal"),
    T("user:u2", "member", "group:acme/legal"),
    T("tenant:acme", "tenant", "workspace:acme/ws1"),
    T("group:acme/legal#member", "member", "workspace:acme/ws1"),
    T("tenant:acme", "tenant", "thread:acme/thr1"),
    T("workspace:acme/ws1", "workspace", "thread:acme/thr1"),
    T("user:u1", "owner", "thread:acme/thr1"),
    T("agent:research", "participant", "thread:acme/thr1"),
    T("tenant:acme", "tenant", "thread:acme/thr2"),
    T("user:u2", "owner", "thread:acme/thr2"),
    T("tenant:acme", "tenant", "document:acme/doc1"),
    T("thread:acme/thr1", "thread", "document:acme/doc1"),
    T("tenant:acme", "tenant", "memory:acme/mem1"),
    T("agent:research", "owner", "memory:acme/mem1"),
    T("tenant:globex", "tenant", "thread:globex/thr9"),
    T("user:u3", "owner", "thread:globex/thr9"),
]

GOLDEN_CHECKS = [
    ("user:u1", "can_read", "thread:acme/thr1", True),  # owner
    ("user:u1", "can_write", "thread:acme/thr1", True),
    ("agent:research", "can_read", "thread:acme/thr1", True),  # participant
    ("agent:research", "can_write", "thread:acme/thr1", True),
    ("user:u2", "can_read", "thread:acme/thr1", False),  # legal is workspace member, not admin
    ("user:admin1", "can_read", "thread:acme/thr1", True),  # tenant admin via ttu
    ("user:admin1", "can_write", "thread:acme/thr1", True),  # tenant admin -> workspace admin
    ("user:u1", "can_read", "thread:acme/thr2", False),  # other user's thread
    ("user:u3", "can_read", "thread:acme/thr1", False),  # other tenant
    ("user:admin1", "can_read", "thread:globex/thr9", False),  # admin of a different tenant
    ("user:u2", "viewer", "workspace:acme/ws1", True),  # group#member userset
    ("user:u1", "viewer", "workspace:acme/ws1", False),
    ("user:u1", "can_read", "document:acme/doc1", True),  # via thread can_read
    ("agent:research", "can_read", "document:acme/doc1", True),
    ("user:u2", "can_read", "document:acme/doc1", False),
    ("agent:research", "can_read", "memory:acme/mem1", True),
    ("agent:writer", "can_read", "memory:acme/mem1", False),  # private to another agent
    ("user:u1", "can_read", "memory:acme/mem1", False),  # agent-private not visible to user
    ("user:admin1", "can_delete", "memory:acme/mem1", True),
    ("user:u1", "nonexistent", "thread:acme/thr1", False),
    ("user:u1", "can_read", "unknown:acme/x", False),
]


@pytest.fixture
async def provider() -> MemoryAuthorizationProvider:
    p = MemoryAuthorizationProvider()
    await p.write(GOLDEN_TUPLES)
    return p


@pytest.mark.parametrize(("user", "relation", "obj", "expected"), GOLDEN_CHECKS)
async def test_golden_checks(
    provider: MemoryAuthorizationProvider, user, relation, obj, expected
) -> None:
    assert await provider.check(AccessCheck(user=user, relation=relation, object=obj)) is expected


async def test_batch_check_preserves_order(provider: MemoryAuthorizationProvider) -> None:
    checks = [AccessCheck(user=u, relation=r, object=o) for u, r, o, _ in GOLDEN_CHECKS]
    assert await provider.batch_check(checks) == [e for *_, e in GOLDEN_CHECKS]


async def test_list_objects(provider: MemoryAuthorizationProvider) -> None:
    assert await provider.list_objects("user:u1", "can_read", "thread") == ["thread:acme/thr1"]
    assert await provider.list_objects("user:admin1", "can_read", "thread") == [
        "thread:acme/thr1",
        "thread:acme/thr2",
    ]
    assert await provider.list_objects("user:u3", "can_read", "thread") == ["thread:globex/thr9"]
    assert await provider.list_objects("user:u2", "member", "group") == ["group:acme/legal"]


async def test_delete_tuple_revokes(provider: MemoryAuthorizationProvider) -> None:
    await provider.write([], delete=[T("user:u1", "owner", "thread:acme/thr1")])
    assert (
        await provider.check(
            AccessCheck(user="user:u1", relation="can_read", object="thread:acme/thr1")
        )
        is False
    )


async def test_invalid_tuples_rejected() -> None:
    p = MemoryAuthorizationProvider()
    with pytest.raises(ValueError, match="may not be directly assigned"):
        await p.write([T("agent:a", "owner", "thread:acme/t")])  # owner is [user] only
    with pytest.raises(ValueError, match="unknown relation"):
        await p.write([T("user:u", "bogus", "thread:acme/t")])
    with pytest.raises(ValueError, match="unknown object type"):
        await p.write([T("user:u", "owner", "planet:acme/t")])


async def test_cycles_terminate() -> None:
    p = MemoryAuthorizationProvider()
    await p.write(
        [
            T("group:acme/a#member", "member", "workspace:acme/w"),
            T("user:x", "member", "group:acme/a"),
        ]
    )
    assert (
        await p.check(AccessCheck(user="user:x", relation="viewer", object="workspace:acme/w"))
        is True
    )
    assert (
        await p.check(AccessCheck(user="user:y", relation="viewer", object="workspace:acme/w"))
        is False
    )
