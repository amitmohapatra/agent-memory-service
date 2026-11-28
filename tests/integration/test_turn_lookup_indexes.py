"""The turn lookups read their indexes (migration 0041): a search's neighbours and what a
model's reading finds look turns up by the message they keep and by the turn they reply to,
inside JSON. Unindexed, each lookup read every live turn of the tenant. Each statement the
repository sends is planned with sequential scans off: a query that drifts from its index's
expression or predicate no longer names the index, and this fails."""

from __future__ import annotations

import pytest
from sqlalchemy import event

pytestmark = pytest.mark.integration


async def _sent(container, uow_factory) -> list[tuple[str, object]]:
    """The statements the turn lookups send, with their parameters."""
    sent: list[tuple[str, object]] = []
    sync = container.database.engine.sync_engine

    def grab(conn, cursor, statement, parameters, context, executemany) -> None:  # noqa: ANN001
        if "verbatim_turn" in statement:
            sent.append((statement, parameters))

    event.listen(sync, "before_cursor_execute", grab)
    try:
        async with uow_factory() as uow:
            await uow.memories.turns_around("acme", said=["obs_a"], replying_to=["obs_b"])
            await uow.memories.turns_of("acme", ["obs_a"])
    finally:
        event.remove(sync, "before_cursor_execute", grab)
    return sent


async def _plan(conn, statement: str, parameters: object) -> str:
    rows = await conn.exec_driver_sql(f"EXPLAIN {statement}", parameters)
    return "\n".join(r[0] for r in rows)


async def test_no_turn_lookup_reads_the_whole_table(container, uow_factory) -> None:
    sent = await _sent(container, uow_factory)
    assert len(sent) == 3
    async with container.database.engine.connect() as conn:
        await conn.exec_driver_sql("SET enable_seqscan = off")
        for statement, parameters in sent:
            plan = await _plan(conn, statement, parameters)
            assert "Seq Scan on memories" not in plan, plan


async def test_the_lookups_by_message_read_the_message_index(container, uow_factory) -> None:
    """On an empty table the tenant's replies index costs least for every lookup; with it set
    aside (in a transaction rolled back), the lookups by message read the index built for
    them - their expression and predicate match it."""
    by_message, replying, read_from = await _sent(container, uow_factory)
    async with container.database.engine.connect() as conn:
        held = await conn.begin()
        try:
            await conn.exec_driver_sql("SET LOCAL enable_seqscan = off")
            assert "ix_memories_turn_replies" in await _plan(conn, *replying)
            await conn.exec_driver_sql("DROP INDEX ix_memories_turn_replies")
            for statement, parameters in (by_message, read_from):
                plan = await _plan(conn, statement, parameters)
                assert "ix_memories_turn_messages" in plan, plan
        finally:
            await held.rollback()
