"""Of two values of one slot, the newer is the one said later, not the one read later.

"I work at Globex now." (said 5 days ago) was sent first, then "I work at ACME Corp." (said
45 days ago) in a separate call: consolidation closed Globex and kept ACME current, because
replacement followed the order the statements were read in. When each was said
(``TemporalState.since``: the date it names, else when it was said) decides; the order of
reading only breaks a tie. A value said before the current one is its history, closed where the
current one begins - on the consolidation path and on the cross-conversation one (ADR 0038).

And while one message's statements are compared and written, the user's others wait: their
conversations are read side by side, and each compared itself with a store the other had not
committed to.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, TemporalStatus
from memory_service.domain.memory import CanonicalMemory, TemporalState
from memory_service.modules.memory.pipeline import _one_writer_per_user, _replace, supersede
from memory_service.modules.memory.replacement import Decision, Grade, apply

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", thread_id="thr_a")


def _works_at(employer: str, *, days_ago: float, **temporal: Any) -> CanonicalMemory:
    said = NOW - timedelta(days=days_ago)
    return CanonicalMemory.model_validate(
        {
            "tenant_id": "acme",
            "content": f"I work at {employer}.",
            "memory_type": MemoryType.USER,
            "lifetime": "LONG_TERM",
            "visibility": "USER",
            "scope": {"level": "USER", "tenant_id": "acme", "user_id": "u1"},
            "owner_principal": "user:u1",
            "normalized_hash": employer,
            "subject": "user:u1",
            "predicate": "works_at",
            "object": employer.lower(),
            "temporal": {"observed_at": said, **temporal},
            "evidence": [{"source_type": "message", "source_id": employer, "observed_at": said}],
            "system_metadata": {"category": "attribute"},
        }
    )


class _Rows:
    """The memories table ``_replace`` writes through: what it added and updated."""

    def __init__(self) -> None:
        self.added: list[CanonicalMemory] = []
        self.updated: list[CanonicalMemory] = []
        self.twins_asked = 0

    async def twins(self, *_: object) -> list[CanonicalMemory]:
        self.twins_asked += 1
        return []

    async def add(self, memory: CanonicalMemory, **_: object) -> None:
        self.added.append(memory)

    async def update(self, memory: CanonicalMemory) -> None:
        self.updated.append(memory)


class _Writes:
    def __init__(self) -> None:
        self.memories = _Rows()
        self.locks: list[str] = []

    async def serialize(self, *keys: str) -> None:
        self.locks += keys


def test_a_statement_holds_from_the_date_it_names_else_from_when_it_was_said() -> None:
    said = NOW - timedelta(days=5)
    assert TemporalState(observed_at=said).since == said
    dated = NOW - timedelta(days=30)
    assert TemporalState(observed_at=said, valid_from=dated).since == dated


async def test_an_earlier_said_value_read_later_is_kept_as_the_current_ones_history() -> None:
    """N3: Globex (5 days ago) read first, ACME (45 days ago) read after it."""
    globex, acme = _works_at("Globex", days_ago=5), _works_at("ACME Corp", days_ago=45)
    uow = _Writes()
    written = await _replace(uow, CTX, globex, acme, now=NOW)  # type: ignore[arg-type]

    assert written == {acme.memory_id, globex.memory_id}
    assert globex.temporal.status is TemporalStatus.CURRENT
    assert globex.temporal.valid_to is None and globex.temporal.superseded_by is None
    # it holds from when it was said, so "where did I work six weeks ago?" is ACME
    assert globex.temporal.valid_from == NOW - timedelta(days=5)
    assert globex.temporal.supersedes == acme.memory_id
    assert acme.temporal.status is TemporalStatus.SUPERSEDED
    assert acme.temporal.superseded_by == globex.memory_id
    assert acme.temporal.valid_to == NOW - timedelta(days=5), "closed where Globex begins"
    assert uow.memories.added == [acme] and uow.memories.updated == [globex]
    assert uow.memories.twins_asked == 0, "the current value and its twins are not closed"


async def test_a_later_said_value_replaces_the_earlier_from_when_it_was_said() -> None:
    acme, globex = _works_at("ACME Corp", days_ago=45), _works_at("Globex", days_ago=5)
    uow = _Writes()
    await _replace(uow, CTX, acme, globex, now=NOW)  # type: ignore[arg-type]

    assert globex.temporal.status is TemporalStatus.CURRENT
    assert globex.temporal.supersedes == acme.memory_id
    assert acme.temporal.status is TemporalStatus.SUPERSEDED
    assert acme.temporal.superseded_by == globex.memory_id
    # not from when it was read (``now``): Globex holds from 5 days ago, ACME until then
    assert globex.temporal.valid_from == acme.temporal.valid_to == NOW - timedelta(days=5)
    assert uow.memories.added == [globex] and uow.memories.updated == [acme]


async def test_on_a_tie_the_value_read_later_replaces() -> None:
    """The same moment (two statements of one day, dated alike): the order of reading."""
    first, second = _works_at("ACME Corp", days_ago=5), _works_at("Globex", days_ago=5)
    await _replace(_Writes(), CTX, first, second, now=NOW)  # type: ignore[arg-type]
    assert second.temporal.status is TemporalStatus.CURRENT
    assert first.temporal.status is TemporalStatus.SUPERSEDED
    assert first.temporal.superseded_by == second.memory_id


async def test_a_stated_date_decides_before_when_it_was_said() -> None:
    """ "I have worked at ACME Corp since 1 October", said today, is older than "I work at
    Globex now." said on 7 October; said today with no date, it would be the newer."""
    globex = _works_at("Globex", days_ago=20)
    acme = _works_at("ACME Corp", days_ago=0, valid_from=NOW - timedelta(days=26))
    await _replace(_Writes(), CTX, globex, acme, now=NOW)  # type: ignore[arg-type]
    assert globex.temporal.status is TemporalStatus.CURRENT
    assert acme.temporal.valid_to == NOW - timedelta(days=20)


def test_the_current_value_keeps_what_it_already_replaced() -> None:
    """Initech (20 days ago) replaced by Globex (5 days ago); then ACME (45 days ago) is read:
    Globex still points back at Initech."""
    initech, globex = _works_at("Initech", days_ago=20), _works_at("Globex", days_ago=5)
    supersede(initech, globex, now=NOW)
    acme = _works_at("ACME Corp", days_ago=45)
    supersede(acme, globex, now=NOW)
    assert globex.temporal.supersedes == initech.memory_id
    assert acme.temporal.superseded_by == globex.memory_id


def test_a_value_replaced_across_conversations_closes_where_the_newer_begins() -> None:
    """ADR 0038's path: closed at when the newer statement was said, not at when it was read."""
    acme, globex = _works_at("ACME Corp", days_ago=45), _works_at("Globex", days_ago=5)
    apply(acme, globex, Decision(Grade.SUPERSEDED, 0.97, "contradiction"), now=NOW)
    assert acme.temporal.status is TemporalStatus.SUPERSEDED
    assert acme.temporal.valid_to == NOW - timedelta(days=5)


async def test_one_users_statements_are_compared_and_written_one_message_at_a_time() -> None:
    uow, other, nobody = _Writes(), _Writes(), _Writes()
    await _one_writer_per_user(uow, CTX)  # type: ignore[arg-type]
    await _one_writer_per_user(  # type: ignore[arg-type]
        other, CTX.model_copy(update={"user_id": "u2", "thread_id": "thr_c"})
    )
    await _one_writer_per_user(  # type: ignore[arg-type]
        nobody, MemoryExecutionContext(tenant_id="acme", agent_id="bot", thread_id="thr_x")
    )
    another_thread = _Writes()
    await _one_writer_per_user(  # type: ignore[arg-type]
        another_thread, CTX.model_copy(update={"thread_id": "thr_b"})
    )
    assert uow.locks == another_thread.locks == ["statements:acme:u1"], "every conversation"
    assert other.locks == ["statements:acme:u2"], "another user never waits"
    assert nobody.locks == [], "no user: the queue's conversation lock is enough"
