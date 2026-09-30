"""The pinned profile and the thread summary as an agent uses them, through the SDK only."""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import ConflictError, MemoryError, ValidationError

pytestmark = pytest.mark.e2e


async def _tenant(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return admin, sdk(app, service.token)


@pytest.mark.covers("profile.get_profile", "profile.edit_profile_block")
async def test_an_agent_reads_sets_and_edits_its_pinned_blocks(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("buyer")
    assert await agent.profile() == []

    user = await agent.profile.edit("user", "name: Ann\ndelivery address: Hauptstr. 1")
    persona = await agent.profile.edit("agent.persona", "Terse. Always quotes the PO number.")
    assert (user.version, persona.version) == (1, 1)
    edited = await agent.profile.edit("user", "Ringstr. 9", old="Hauptstr. 1")
    assert edited.text.endswith("Ringstr. 9") and edited.version == 2
    assert [b.block for b in await agent.profile()] == ["agent.persona", "user"]

    # the user block is the user's: another agent of theirs sees it, the persona is this agent's
    other = harness.bind(user_id="u1").agent("planner")
    assert [b.block for b in await other.profile()] == ["user"]
    assert [b.block for b in await harness.bind(user_id="u2").profile()] == []


@pytest.mark.covers_error("profile.edit_profile_block")
async def test_a_stale_edit_conflicts_and_a_block_outside_the_scope_is_refused(
    app, running
) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("buyer")
    await agent.profile.edit("user", "name: Ann")
    with pytest.raises(ConflictError):
        await agent.profile.edit("user", "name: Rob", old="name: Bob")
    with pytest.raises(ValidationError):
        await harness.bind(user_id="u1").profile.edit("agent", "no agent in this scope")
    with pytest.raises(MemoryError) as unknown:
        await agent.profile.edit("tenant", "not a block")
    assert unknown.value.status == 422


@pytest.mark.covers("threads.get_thread")
async def test_a_thread_gains_a_durable_summary_every_twenty_messages(app, running) -> None:
    _, harness = await _tenant(app)
    chat = harness.bind(user_id="u1", thread_id="thr_long")
    await chat.history.add([("USER", "Turn 0: about the paper order.")])
    assert (await chat.history.thread()).summary is None
    for i in range(1, 20):
        await chat.history.add(
            [("USER" if i % 2 == 0 else "ASSISTANT", f"Turn {i}: about the paper order.")]
        )
    summary = (await chat.history.thread()).summary
    assert summary is not None and summary.covers_to_sequence == 20
    assert summary.version == 1 and summary.model == "extractive"
    assert summary.text.startswith("user: Turn 0: about the paper order.")
