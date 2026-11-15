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


def _user_block(blocks) -> str:
    return next((b.text for b in blocks if b.block == "user"), "")


@pytest.mark.covers("profile.get_profile", "memory.forget_memory")
async def test_a_profile_line_goes_with_the_memory_it_was_learned_from(app, running) -> None:
    """Derived data follows its source: the pinned ``user`` block is composed from the
    user's USER and PREFERENCE memories, and forgetting one left its line in the block (no
    refresh was queued for a forgotten memory, and a user with none left kept the block). A
    line another memory still holds stays. (A forgotten memory cannot be restored, and a
    user's stated preference is protected from archiving, so no restore applies here.)"""
    _, harness = await _tenant(app)
    user = harness.bind(user_id="u1")
    seats = await user.remember("I prefer window seats.", memory_type="PREFERENCE")
    tea = await user.remember("I prefer green tea.", memory_type="PREFERENCE")
    container = app.state.container
    running.portal.call(container.tasks.drain)
    assert "window seats" in _user_block(await user.profile())

    await user.forget(seats.memory_id)
    running.portal.call(container.tasks.drain)
    held = _user_block(await user.profile())
    assert "window seats" not in held and "green tea" in held
    assert "window seats" not in (await user.context("seat preference")).rendered

    await user.forget(tea.memory_id)  # the last one: nothing of theirs is left to learn from
    running.portal.call(container.tasks.drain)
    assert _user_block(await user.profile()) == ""


@pytest.mark.covers("profile.edit_profile_block", "memory.forget_memory")
async def test_a_line_written_into_the_profile_by_hand_outlives_the_memory_it_repeats(
    app, running
) -> None:
    """MEM-05. A line a person or an agent writes into the block is their own statement, not
    derived from a memory: docs/api/profile.md, "Once a person or an agent has edited the
    block, ... an edit is never lost or undone", and ``ProfileService.refresh_user``, "An
    edited block keeps its edit (a person's or an agent's words, not derived)". Forgetting the
    memory the line repeats leaves it (a learned line goes:
    ``test_a_profile_line_goes_with_the_memory_it_was_learned_from``); the editor removes it
    by editing."""
    _, harness = await _tenant(app)
    user = harness.bind(user_id="u1")
    seats = await user.remember("I prefer window seats.", memory_type="PREFERENCE")
    container = app.state.container
    running.portal.call(container.tasks.drain)
    assert _user_block(await user.profile()) == "- I prefer window seats."

    written = await user.profile.edit("user", "- I prefer window seats.\n- Call me Sam.")
    await user.forget(seats.memory_id)
    running.portal.call(container.tasks.drain)
    assert _user_block(await user.profile()) == written.text
    assert "window seats" in (await user.context("seat preference")).rendered

    await user.profile.edit("user", "", old="- I prefer window seats.\n")
    assert _user_block(await user.profile()) == "- Call me Sam."


@pytest.mark.covers("profile.edit_profile_block", "memory.forget_memory")
async def test_a_line_the_job_appends_to_an_edited_block_goes_with_its_memory(app, running) -> None:
    """Derived data follows its source (ADR 0032), in an edited block too: a line the profile
    job appended from a memory leaves when that memory is forgotten, and the editor's own
    lines stay. A job's line a person then edits is theirs, and stays (the hand-written
    line: ``test_a_line_written_into_the_profile_by_hand_outlives_the_memory_it_repeats``)."""
    _, harness = await _tenant(app)
    user = harness.bind(user_id="u1")
    written = await user.profile.edit("user", "name: Sam\n- I travel light.")
    seats = await user.remember("I prefer window seats.", memory_type="PREFERENCE")
    tea = await user.remember("I prefer green tea.", memory_type="PREFERENCE")
    container = app.state.container
    running.portal.call(container.tasks.drain)
    learned = _user_block(await user.profile()).splitlines()
    assert learned[:2] == written.text.splitlines()
    assert sorted(learned[2:]) == ["- I prefer green tea.", "- I prefer window seats."]

    await user.forget(seats.memory_id)
    running.portal.call(container.tasks.drain)
    assert _user_block(await user.profile()) == f"{written.text}\n- I prefer green tea."

    await user.profile.edit("user", "- I prefer green tea, no sugar.", old="- I prefer green tea.")
    await user.forget(tea.memory_id)  # the last one: the job has nothing left to learn from
    running.portal.call(container.tasks.drain)
    assert _user_block(await user.profile()) == f"{written.text}\n- I prefer green tea, no sugar."
