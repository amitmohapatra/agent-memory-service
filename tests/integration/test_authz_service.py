import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.ports.authorization import RelationTuple as R

pytestmark = pytest.mark.integration


async def test_a_grant_invalidates_the_scope_and_a_memory_write_does_not(
    container, uow_factory
) -> None:
    """The two halves of the same defect, against the real provider and cache.

    A grant used to bump nothing, so a caller who had just been given a thread waited out
    the sixty-second TTL to see it. Meanwhile the scope was keyed on the TENANT and USER
    revisions, which every memory write bumps - 730 times over 369 ingested turns on the
    benchmark tenant - so content churn threw the cache away constantly and each miss cost
    five sequential ListObjects calls. Under load that returned 503s.
    """
    svc: AuthorizationService = container.services["authz"]
    provider = container.authorization
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1")
    await provider.write(
        [
            R(user="tenant:acme", relation="tenant", object="thread:acme/thr1"),
            R(user="user:u1", relation="owner", object="thread:acme/thr1"),
        ]
    )
    async with uow_factory() as uow:
        assert (await svc.scope(ctx, revisions=uow.revisions)).thread_ids == ["thr1"]

        # a grant that is told about the unit of work is visible on the next read
        await svc.grant_thread(ctx, "thr2", revisions=uow.revisions)
        assert (await svc.scope(ctx, revisions=uow.revisions)).thread_ids == ["thr1", "thr2"]

        # ...and a grant made without one is still stale within the TTL, which is the
        # documented escape hatch for callers that hold no transaction
        await svc.grant_thread(ctx, "thr3")
        assert (await svc.scope(ctx, revisions=uow.revisions)).thread_ids == ["thr1", "thr2"]

        # a memory write bumps the content revisions and must change nothing here
        await uow.revisions.bump("acme", RevisionKind.USER, "u1")
        await uow.revisions.bump("acme", RevisionKind.TENANT, "")
        assert (await svc.scope(ctx, revisions=uow.revisions)).thread_ids == ["thr1", "thr2"]
        await uow.commit()


async def test_grants_create_expected_tuples(container) -> None:
    svc: AuthorizationService = container.services["authz"]
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
    await svc.grant_thread(ctx, "thr1", workspace_id="ws1")
    await svc.grant_document(ctx, "doc1", thread_id="thr1", workspace_id="ws1")
    agent_ctx = ctx.child_agent(agent_id="research")
    await svc.grant_memory(agent_ctx, "mem1")
    tuples = {(t.user, t.relation, t.object) for t in container.authorization.dump()}
    assert ("user:u1", "owner", "thread:acme/thr1") in tuples
    assert ("workspace:acme/ws1", "workspace", "thread:acme/thr1") in tuples
    assert ("thread:acme/thr1", "thread", "document:acme/doc1") in tuples
    # The agent principal is bound to the user it runs for: agent_id arrives in the request
    # body and is not authenticated, so an unbound "agent:research" would be the same
    # principal for every user who named that agent.
    assert ("agent:u1/research", "owner", "memory:acme/mem1") in tuples
    assert ("agent:research", "owner", "memory:acme/mem1") not in tuples
    assert await svc.allowed(ctx, "can_read", "document", "doc1")
    assert await svc.allowed(agent_ctx, "can_read", "memory", "mem1")
    assert not await svc.allowed(ctx, "can_read", "memory", "mem1")
