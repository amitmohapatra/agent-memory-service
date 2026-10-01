"""A bundle handle only resolves for the scope that built it.

`POST /v1/verify` accepts a `bundle_id` instead of a query, and the replay (now the bundle's
record, `BundleRecords.load`) used to look that handle up under the TENANT alone. A same-tenant caller holding another principal's handle
was served that principal's bundle with no re-check: a cross-principal read of evidence ids,
counts, and a supported/contradicted oracle over someone else's memories.

The automatic path was never exposed - the bundle_id it computes already folds in the scope
fingerprint - so only the replay took the id on trust. The e2e suite covered the cross-TENANT
replay and not this one.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.revisions import RevisionKind

pytestmark = pytest.mark.integration


def _ctx(user_id: str, **extra) -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id="acme", user_id=user_id, workspace_id="ws1", **extra)


async def _build(container, ctx, query="what did we decide about the brief?"):
    """Build and flush. The cache write is a background task, so without draining the
    negative cases below would pass for the wrong reason - nothing would be cached at all."""
    builder = container.services["context_builder"]
    bundle = await builder.build(ctx, query)
    await builder.drain()
    return bundle.bundle_id


async def test_the_caller_that_built_it_can_replay_it(container) -> None:
    alice = _ctx("alice")
    bundle_id = await _build(container, alice)
    assert bundle_id
    assert await container.services["bundle_records"].load(alice, bundle_id) is not None


async def test_another_user_in_the_same_tenant_cannot(container) -> None:
    """The defect: same tenant, different principal, someone else's handle."""
    alice = _ctx("alice")
    bundle_id = await _build(container, alice)
    mallory = _ctx("mallory")
    assert await container.services["bundle_records"].load(mallory, bundle_id) is None, (
        "another principal replayed alice's bundle"
    )


async def test_an_agent_run_cannot_replay_the_users_bundle(container) -> None:
    """Agent lineage is part of the scope, so it is part of the handle."""
    alice = _ctx("alice")
    bundle_id = await _build(container, alice)
    agent = _ctx("alice", agent_id="plansmart", agent_run_id="run_1")
    assert await container.services["bundle_records"].load(agent, bundle_id) is None


async def test_an_unknown_handle_is_simply_absent(container) -> None:
    assert await container.services["bundle_records"].load(_ctx("alice"), "nope") is None


async def test_the_record_outlives_a_content_change(container) -> None:
    """Unlike the bundle cache, the record keeps resolving after the memory changes: an agent
    that just updated ``m1`` goes on to forget ``m2`` (``modules/context/handles.py``)."""
    alice = _ctx("alice")
    bundle_id = await _build(container, alice)
    async with container.services["uow_factory"]() as uow:
        await uow.revisions.bump(alice.tenant_id, RevisionKind.USER, alice.user_id)
        await uow.commit()
    assert await container.services["bundle_records"].load(alice, bundle_id) is not None
