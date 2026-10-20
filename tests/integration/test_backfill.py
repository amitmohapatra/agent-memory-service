"""``tools.backfill`` on a store an earlier build wrote (ADR 0041): replacement across a
user's conversations in event-time order, the index again, nothing deleted; idempotent,
resumable, one user never touching another, and a user with thousands of memories."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from memory_service.domain.enums import TemporalStatus
from memory_service.domain.memory import earlier_value
from memory_service.tools.backfill import Checkpoint, backfill
from tests.integration.test_replacement_across_conversations import ScriptedNLI
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration


def _old_build(container) -> None:
    """Writes as a build before ADR 0038 did: no replacement across conversations."""
    container.services["observation_pipeline"].replacement = None


def _this_build(container) -> ScriptedNLI:
    from memory_service.modules.memory.replacement import CrossConversationReplacement

    nli = ScriptedNLI(0.95)
    container.services["observation_pipeline"].replacement = CrossConversationReplacement(nli)
    return nli


async def _states(container, uow_factory, ctx) -> dict[str, str]:
    """Every memory of the user's own, by content: CURRENT, EARLIER (current, replaced) or
    its temporal status."""
    from sqlalchemy import select

    from memory_service.adapters.db.orm import MemoryRow

    async with container.database.session_factory() as session:
        ids = (
            await session.scalars(
                select(MemoryRow.memory_id).where(
                    MemoryRow.tenant_id == ctx.tenant_id,
                    MemoryRow.owner_principal == ctx.principal_id,
                    MemoryRow.deleted_at.is_(None),
                )
            )
        ).all()
    async with uow_factory() as uow:
        memories = await uow.memories.get_many(ctx.tenant_id, list(ids))
    out = {}
    for m in memories:
        state = m.temporal.status.value
        if m.temporal.status is TemporalStatus.CURRENT and earlier_value(m.system_metadata):
            state = "EARLIER"
        out[m.content] = state
    return out


SAID = (
    "Our main supplier of shipping crates is Brightway.",
    "We ended the Brightway agreement last week. Shipping crates now come from Corrimax only.",
)


async def test_an_old_store_gets_its_replacements_and_a_second_run_changes_nothing(
    container, uow_factory, tmp_path
):
    _old_build(container)
    u1 = _chat("u1")
    for text in SAID:
        await _say(container, uow_factory, _chat("u1"), text)
    # another user's statements of the same thing are theirs alone
    for text in SAID[:1]:
        await _say(container, uow_factory, _chat("u2"), text)
    before = await _states(container, uow_factory, u1)
    assert before[SAID[0]] == "CURRENT"  # what the earlier build left

    _this_build(container)
    state = tmp_path / "backfill.json"
    report = await backfill(container, state=state)
    after = await _states(container, uow_factory, u1)
    assert after[SAID[0]] in ("EARLIER", "SUPERSEDED") and after[SAID[1]] == "CURRENT"
    assert report.replaced >= 1 and not report.failures
    # nothing deleted
    assert set(after) >= set(before)
    # the other user's statement is untouched
    assert (await _states(container, uow_factory, _chat("u2")))[SAID[0]] == "CURRENT"
    # finished users are recorded: a rerun with the same state skips them
    assert json.loads(state.read_text())["done"]
    again = await backfill(container, state=state)
    assert again.users == 0 and again.skipped_users == report.users
    # and a fresh run over the same store changes nothing (idempotent)
    fresh = await backfill(container)
    assert fresh.replaced == 0 and await _states(container, uow_factory, u1) == after


async def test_a_run_stopped_midway_resumes_after_the_last_message_done(
    container, uow_factory, tmp_path
):
    _old_build(container)
    for text in SAID:
        await _say(container, uow_factory, _chat("u1"), text)
    _this_build(container)
    state = tmp_path / "backfill.json"
    users = await backfill(container, state=tmp_path / "probe.json", user="nobody")
    assert users.users == 0
    # a checkpoint left by a run stopped after the first message
    async with uow_factory() as uow:
        statements = await uow.memories.statements_of(
            "acme", "user:u1", before=datetime.max.replace(tzinfo=UTC), limit=100
        )
    first = min(m.temporal.observed_at for m in statements)
    Checkpoint(state).progress("acme", "user:u1", first.isoformat())
    report = await backfill(container, state=state)
    assert report.users == 1 and report.replaced >= 1
    after = await _states(container, uow_factory, _chat("u1"))
    assert after[SAID[0]] in ("EARLIER", "SUPERSEDED")


@pytest.mark.timeout(900)
async def test_a_user_with_thousands_of_memories(container, uow_factory):
    from datetime import timedelta

    from memory_service.domain.evidence import EvidenceRef
    from memory_service.domain.ids import new_id

    _old_build(container)
    await _say(container, uow_factory, _chat("big"), "Aisle 0 holds bin code B00000.")
    async with uow_factory() as uow:
        (seed,) = [
            m
            for m in await uow.memories.statements_of(
                "acme", "user:big", before=datetime.max.replace(tzinfo=UTC), limit=10
            )
            if m.system_metadata.get("category") == "verbatim_turn"
        ]
        for n in range(1, 2000):
            copy = seed.model_copy(
                deep=True,
                update={
                    "memory_id": new_id("memory"),
                    "content": f"Aisle {n} holds bin code B{n:05d}.",
                    "evidence": [
                        EvidenceRef(
                            source_type="message",
                            source_id=new_id("observation"),
                            observed_at=seed.temporal.observed_at + timedelta(minutes=n),
                        )
                    ],
                    "temporal": seed.temporal.model_copy(
                        update={"observed_at": seed.temporal.observed_at + timedelta(minutes=n)}
                    ),
                },
            )
            await uow.memories.add(copy, visibility_keys=seed.system_metadata["visibility_keys"])
        await uow.commit()
    _this_build(container)
    report = await backfill(container, user="big", batch=500)
    assert report.users == 1 and report.messages == 2000 and not report.failures
    assert report.indexed == 2000
    # every aisle has its own bin: no statement replaced another
    assert report.replaced == 0


async def test_the_graph_facts_of_what_it_replaced_are_retired(container, uow_factory):
    """The write path's index job enriches what it indexed, and enrichment retires the facts
    of a memory no longer current; the backfill does the same for every memory it touched."""
    _old_build(container)
    u1 = _chat("u1")
    for text in SAID:
        await _say(container, uow_factory, _chat("u1"), text)
    _this_build(container)
    graph = container.services["graph"]
    enriched: list[str] = []
    real = graph.enrich_memories

    async def recording(tenant_id, memory_ids):
        enriched.extend(memory_ids)
        return await real(tenant_id, memory_ids)

    graph.enrich_memories = recording
    try:
        await backfill(container)
    finally:
        graph.enrich_memories = real
    async with uow_factory() as uow:
        memories = await uow.memories.get_many(u1.tenant_id, enriched)
    replaced = [m for m in memories if m.content == SAID[0]]
    assert replaced and all(m.memory_id in enriched for m in replaced)


async def test_a_state_from_an_earlier_revision_runs_every_user_again(
    container, uow_factory, tmp_path
):
    """A release that changes what the backfill applies bumps ``BACKFILL_REVISION``: the
    state file a run of the earlier revision left must not skip the users it finished."""
    from memory_service.tools.backfill import BACKFILL_REVISION

    _old_build(container)
    for text in SAID:
        await _say(container, uow_factory, _chat("u1"), text)
    _this_build(container)
    state = tmp_path / "backfill.json"
    first = await backfill(container, state=state)
    assert first.users >= 1 and json.loads(state.read_text())["revision"] == BACKFILL_REVISION
    # the same revision: every user it finished is skipped
    same = await backfill(container, state=state)
    assert same.users == 0 and same.skipped_users == first.users
    # the state an earlier revision wrote: every user runs again (and, idempotent, changes
    # nothing more)
    stored = json.loads(state.read_text())
    state.write_text(json.dumps({**stored, "revision": BACKFILL_REVISION - 1}))
    again = await backfill(container, state=state)
    assert again.users == first.users and again.skipped_users == 0
    assert again.replaced == 0 and not again.failures


def test_an_empty_store_exits_0_without_loading_a_model(monkeypatch, tmp_path, capsys):
    """A fresh install: compose runs the backfill before any user exists. It reads the store
    with a bare connection, finds nobody and exits 0 - no container, no model."""
    import sys

    import memory_service.application.container as app
    from memory_service.tools import backfill as module
    from tests.conftest import DB_URL

    async def no_container(*args, **kwargs):
        raise AssertionError("nothing to do: no model may be loaded")

    monkeypatch.setattr(app, "build_container", no_container)
    monkeypatch.setenv("MEMORY__DATABASE__URL", DB_URL)
    state = tmp_path / "volume" / "state.json"
    monkeypatch.setattr(
        sys, "argv", ["backfill", "--state", str(state), "--tenant", "a-tenant-nobody-wrote"]
    )
    assert module.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["users"] == 0 and report["skipped_users"] == 0 and not report["failures"]


async def test_the_user_block_is_composed_again(container, uow_factory):
    """The ``user`` block an earlier build composed is composed again from what is current
    after the replacement."""
    _old_build(container)
    for text in SAID:
        await _say(container, uow_factory, _chat("u1"), text)
    _this_build(container)
    profile = container.services["profile"]
    refreshed: list[tuple[str, str]] = []
    real = profile.refresh_user

    async def recording(tenant_id, user_id):
        refreshed.append((tenant_id, user_id))
        return await real(tenant_id, user_id)

    profile.refresh_user = recording
    try:
        await backfill(container, user="u1")
    finally:
        profile.refresh_user = real
    assert refreshed == [("acme", "u1")]
