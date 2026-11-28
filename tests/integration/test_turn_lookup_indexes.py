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


#: The indexes the turn lookups are built to read (migration 0041).
OURS = ("ix_memories_turn_messages", "ix_memories_turn_replies")


async def test_each_turn_lookup_can_read_its_index(container, uow_factory) -> None:
    """Each lookup's expression and predicate match its index: with every other index of
    the table set aside (in a transaction rolled back) and sequential scans off, the lookups
    by message read ix_memories_turn_messages and the one by reply ix_memories_turn_replies.
    Which index the planner picks among several depends on the table's statistics; whether a
    query can use the one built for it does not."""
    by_message, replying, read_from = await _sent(container, uow_factory)
    async with container.database.engine.connect() as conn:
        held = await conn.begin()
        try:
            others = await conn.exec_driver_sql(
                "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE i.indrelid = 'memories'::regclass AND NOT i.indisprimary "
                "AND NOT EXISTS (SELECT 1 FROM pg_constraint k WHERE k.conindid = i.indexrelid)"
            )
            for (name,) in others.all():
                if name not in OURS:
                    await conn.exec_driver_sql(f'DROP INDEX "{name}"')
            await conn.exec_driver_sql("SET LOCAL enable_seqscan = off")
            assert "ix_memories_turn_replies" in await _plan(conn, *replying)
            # the replies index leads with the tenant too: aside, as the others are
            await conn.exec_driver_sql('DROP INDEX "ix_memories_turn_replies"')
            for statement, parameters in (by_message, read_from):
                plan = await _plan(conn, statement, parameters)
                assert "ix_memories_turn_messages" in plan, plan
        finally:
            await held.rollback()
