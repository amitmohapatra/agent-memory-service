"""A bundle handle only resolves for the scope that built it.

`POST /v1/verify` accepts a `bundle_id` instead of a query, and `ContextBuilder.cached` looked
that handle up under the TENANT alone. A same-tenant caller holding another principal's handle
was served that principal's bundle with no re-check: a cross-principal read of evidence ids,
counts, and a supported/contradicted oracle over someone else's memories.

The automatic path was never exposed - the bundle_id it computes already folds in the scope
fingerprint - so only the replay took the id on trust. The e2e suite covered the cross-TENANT
replay and not this one.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext

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
    assert await container.services["context_builder"].cached(alice, bundle_id) is not None


async def test_another_user_in_the_same_tenant_cannot(container) -> None:
    """The defect: same tenant, different principal, someone else's handle."""
    alice = _ctx("alice")
    bundle_id = await _build(container, alice)
    mallory = _ctx("mallory")
    assert await container.services["context_builder"].cached(mallory, bundle_id) is None, (
        "another principal replayed alice's bundle"
    )


async def test_an_agent_run_cannot_replay_the_users_bundle(container) -> None:
    """Agent lineage is part of the scope, so it is part of the handle."""
    alice = _ctx("alice")
    bundle_id = await _build(container, alice)
    agent = _ctx("alice", agent_id="plansmart", agent_run_id="run_1")
    assert await container.services["context_builder"].cached(agent, bundle_id) is None


async def test_an_unknown_handle_is_simply_absent(container) -> None:
    assert await container.services["context_builder"].cached(_ctx("alice"), "nope") is None
