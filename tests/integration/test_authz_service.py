import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.ports.authorization import RelationTuple as R

pytestmark = pytest.mark.integration


async def test_scope_cache_invalidated_by_revision(container, uow_factory) -> None:
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
        scope = await svc.scope(ctx, revisions=uow.revisions)
        assert scope.thread_ids == ["thr1"]
        # new grant without a revision bump is served from cache (stale by design within TTL)
        await svc.grant_thread(ctx, "thr2")
        assert (await svc.scope(ctx, revisions=uow.revisions)).thread_ids == ["thr1"]
        await uow.revisions.bump("acme", RevisionKind.USER, "u1")
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
