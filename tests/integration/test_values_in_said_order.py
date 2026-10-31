"""Of two values of one slot, the one said later stays current, whatever order they are read in.

N3, reproduced at the API without a model: "I work at Globex now." (said 5 days ago) sent
first, then "I work at ACME Corp." (said 45 days ago) in a separate call. Globex was closed
and ACME kept current: replacement followed the order of reading, not when each was said. The
earlier-said value is now Globex's history, closed where Globex begins, so "where did I work six
weeks ago?" still answers ACME - in one conversation and across a user's conversations (ADR
0038), call by call and in one batch (whose messages are still read in order).

And two conversations of one user read side by side each compared itself with a store the
other had not committed to, and both values stayed current: one user's statements are written
one message at a time; other users' are not held up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.enums import MessageRole, TemporalStatus
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.retrieval.engine import PointInTime
from tests.integration.test_memory import _memories
from tests.integration.test_replacement_across_conversations import _scripted, _slot
from tests.integration.test_statements_follow_the_user import _chat

pytestmark = pytest.mark.integration

GLOBEX = "I work at Globex now."
ACME = "I work at ACME Corp."
NOW = datetime.now(UTC)
FIVE_DAYS_AGO, SIX_WEEKS_AGO, FORTY_FIVE_DAYS_AGO = (
    NOW - timedelta(days=5),
    NOW - timedelta(days=42),
    NOW - timedelta(days=45),
)


async def _stored(container, uow_factory, ctx, said: Sequence[tuple[str, datetime | None]]):
    """``history.add`` with these messages (``POST /v1/messages``), each said at its time
    (``None``: not given, the time it is stored): one transaction, nothing read yet. The
    observation ids, in order."""
    register_handlers(container)
    ctx = ctx.model_copy(update={"session_id": None, "turn_id": None})
    async with uow_factory() as uow:
        results = [
            await container.services["conversation"].append_message(
                uow, ctx, role=MessageRole.USER, content=content, occurred_at=when
            )
            for content, when in said
        ]
        await uow.commit()
    return [r.ack.observation_id for r in results]


async def _sent(container, uow_factory, ctx, said: Sequence[tuple[str, datetime | None]]):
    """One ``history.add`` call, and its memories formed and indexed."""
    await _stored(container, uow_factory, ctx, said)
    await container.tasks.drain()  # memory.process_observation
    await container.tasks.drain()  # memory.index


async def _employers(container, uow_factory, ctx) -> dict[str, CanonicalMemory]:
    """The user's ``works_at`` readings by employer."""
    return {
        m.object: m
        for m in await _memories(uow_factory, ctx, container, include_superseded=True)
        if m.predicate == "works_at"
    }


async def _assert_globex_current_acme_its_history(container, uow_factory, ctx) -> None:
    by = await _employers(container, uow_factory, ctx)
    acme, globex = by["acme corp"], by["globex"]
    assert globex.temporal.status is TemporalStatus.CURRENT
    assert acme.temporal.status is TemporalStatus.SUPERSEDED
    assert acme.temporal.superseded_by == globex.memory_id
    # closed where Globex begins: when it was said, not when either was read
    assert acme.temporal.valid_to == globex.temporal.valid_from == FIVE_DAYS_AGO


async def test_an_earlier_said_value_sent_later_does_not_replace_the_current_one(
    container, uow_factory
) -> None:
    """N3, one conversation, two calls."""
    chat = _chat()
    await _sent(container, uow_factory, chat, [(GLOBEX, FIVE_DAYS_AGO)])
    await _sent(container, uow_factory, chat, [(ACME, FORTY_FIVE_DAYS_AGO)])
    await _assert_globex_current_acme_its_history(container, uow_factory, chat)


async def test_across_conversations_too(container, uow_factory) -> None:
    """N3 with each statement in its own conversation of the user (ADR 0038). With a trained
    head, a turn kept beside its reading goes with it: it is read against the later statement
    by its reading's slot, though that reading was stored as history at once."""
    _scripted(container, 0.97, values=("ACME", "Globex"))
    now_chat, then_chat = _chat(), _chat()
    old = "I work at ACME Corp. My shift starts at six."
    await _sent(container, uow_factory, now_chat, [(GLOBEX, FIVE_DAYS_AGO)])
    await _sent(container, uow_factory, then_chat, [(old, FORTY_FIVE_DAYS_AGO)])
    await _assert_globex_current_acme_its_history(container, uow_factory, now_chat)
    [turn] = [
        m
        for m in await _memories(uow_factory, then_chat, container, include_superseded=True)
        if m.content == old
    ]
    assert turn.temporal.status is TemporalStatus.SUPERSEDED
    assert turn.temporal.valid_to == FIVE_DAYS_AGO


async def test_a_batch_said_out_of_order_keeps_the_later_said_value(container, uow_factory) -> None:
    """One batch, still read in the order stored (one conversation's lock): the later-said
    value is current although it is read first."""
    chat = _chat()
    await _sent(
        container, uow_factory, chat, [(GLOBEX, FIVE_DAYS_AGO), (ACME, FORTY_FIVE_DAYS_AGO)]
    )
    await _assert_globex_current_acme_its_history(container, uow_factory, chat)


@pytest.mark.parametrize("when", ["same moment", "not given"])
async def test_without_a_later_time_the_value_read_later_replaces(
    container, uow_factory, when
) -> None:
    """Said at the same moment, or with no time given (each stored, and said, as it comes):
    the order of reading decides, as before."""
    chat = _chat()
    at = FIVE_DAYS_AGO if when == "same moment" else None
    await _sent(container, uow_factory, chat, [(GLOBEX, at)])
    await _sent(container, uow_factory, chat, [(ACME, at)])
    assert await _slot(container, uow_factory, chat, "works_at") == {
        "globex": TemporalStatus.SUPERSEDED,
        "acme corp": TemporalStatus.CURRENT,
    }


async def test_where_did_i_work_six_weeks_ago_is_still_acme(container, uow_factory) -> None:
    """A search as of a moment (``PointInTime``: what was true then) reads each value over
    the time it held: ACME until Globex began, Globex since."""
    chat = _chat()
    await _sent(container, uow_factory, chat, [(GLOBEX, FIVE_DAYS_AGO)])
    await _sent(container, uow_factory, chat, [(ACME, FORTY_FIVE_DAYS_AGO)])
    by = await _employers(container, uow_factory, chat)
    engine = container.services["retrieval"]
    for as_of, employer, other in (
        (SIX_WEEKS_AGO, "acme corp", "globex"),
        (NOW - timedelta(days=1), "globex", "acme corp"),
    ):
        res = await engine.retrieve(
            chat, "Where did I work?", kinds=("memory",), at=PointInTime(as_of=as_of)
        )
        ids = {c.record_id for c in res.candidates}
        assert by[employer].memory_id in ids and by[other].memory_id not in ids, as_of


def _meeting(container, monkeypatch, parties: int, timeout: float = 2.0) -> list[bool]:
    """Each message's ``works_at`` reading, about to be compared with what its unit of work
    read, waits up to ``timeout`` for ``parties`` of them to get there: side by side, each
    compares itself with a store the others have not committed to. Whether each one waited in
    vain (another was held back), in order of arrival."""
    provider = container.services["observation_pipeline"].provider
    consolidate = provider.consolidate
    arrived: list[bool] = []
    everyone = asyncio.Event()

    async def meet(candidate, existing, ctx):
        if candidate.predicate == "works_at":
            n = len(arrived)
            arrived.append(False)
            if n + 1 == parties:
                everyone.set()
            try:
                await asyncio.wait_for(everyone.wait(), timeout)
            except TimeoutError:
                arrived[n] = True
        return await consolidate(candidate, existing, ctx)

    monkeypatch.setattr(provider, "consolidate", meet)
    return arrived


async def _read_at_once(container, observation_ids: Sequence[str]) -> None:
    pipeline = container.services["observation_pipeline"]
    await asyncio.gather(
        *(pipeline.run({"tenant_id": "acme", "observation_id": o}) for o in observation_ids)
    )
    await container.tasks.drain()  # the jobs find their observations read: nothing more
    await container.tasks.drain()  # memory.index


async def test_two_conversations_of_one_user_read_at_once_keep_one_current_value(
    container, uow_factory, monkeypatch
) -> None:
    """Two of a user's conversations, read side by side (the queue runs one conversation's
    messages one at a time, not one user's): both values of the one-valued slot stayed
    current. The second waits for the first to commit, and compares itself with it."""
    first, second = _chat(), _chat()
    [acme] = await _stored(container, uow_factory, first, [(ACME, FORTY_FIVE_DAYS_AGO)])
    [globex] = await _stored(container, uow_factory, second, [(GLOBEX, FIVE_DAYS_AGO)])
    waited = _meeting(container, monkeypatch, parties=2)
    await _read_at_once(container, [acme, globex])

    assert waited == [True, False], "one after the other"
    by = await _employers(container, uow_factory, first)
    current = [o for o, m in by.items() if m.temporal.status is TemporalStatus.CURRENT]
    assert current == ["globex"]
    await _assert_globex_current_acme_its_history(container, uow_factory, second)


async def test_two_users_are_read_side_by_side(container, uow_factory, monkeypatch) -> None:
    mine, theirs = _chat("u1"), _chat("u2")
    [acme] = await _stored(container, uow_factory, mine, [(ACME, FORTY_FIVE_DAYS_AGO)])
    [globex] = await _stored(container, uow_factory, theirs, [(GLOBEX, FIVE_DAYS_AGO)])
    waited = _meeting(container, monkeypatch, parties=2)
    await _read_at_once(container, [acme, globex])

    assert waited == [False, False], "neither waits for the other"
    for chat, employer in ((mine, "acme corp"), (theirs, "globex")):
        by = await _employers(container, uow_factory, chat)
        assert list(by) == [employer]
        assert by[employer].temporal.status is TemporalStatus.CURRENT
