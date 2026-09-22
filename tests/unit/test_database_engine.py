"""How the engine treats a pooled connection, and what it carries on every statement.

Nothing here opens a connection: creating an engine does not, which is what makes the pool's
own configuration testable without a database.
"""

from __future__ import annotations

import pytest

from memory_service.adapters.db.engine import Database, _is_dead
from memory_service.config.constants import DATABASE
from memory_service.config.settings import DatabaseSettings

pytestmark = pytest.mark.unit


def _database() -> Database:
    return Database(DatabaseSettings())


def test_the_pool_recycles_instead_of_pinging() -> None:
    """A pre-ping is a round trip per checkout, and a request takes two or three against a
    database on another host."""
    pool = _database().engine.pool
    assert pool._pre_ping is False
    assert pool._recycle == DATABASE.pool_recycle_seconds == 1800


def test_the_pool_is_sized_per_process() -> None:
    settings = DatabaseSettings(pool_size=8, max_overflow=8)
    engine = Database(settings).engine
    assert engine.pool.size() == 8
    assert engine.pool._max_overflow == 8


def test_a_connection_known_to_be_dead_is_recognised_without_asking_the_server() -> None:
    class _Conn:
        def __init__(self, closed: bool = False, broken: bool = False) -> None:
            self.closed = closed
            self.broken = broken

    assert _is_dead(_Conn(closed=True))
    assert _is_dead(_Conn(broken=True))
    assert not _is_dead(_Conn())
    assert not _is_dead(object()), "a driver that reports neither is treated as usable"


def test_a_dead_connection_is_discarded_on_checkout() -> None:
    """The half of pool_pre_ping that costs nothing: SQLAlchemy answers DisconnectionError
    from a checkout listener by throwing that connection away and taking another one."""
    from sqlalchemy import exc

    engine = _database().engine.sync_engine
    listeners = engine.pool.dispatch.checkout
    assert listeners, "no checkout listener is installed"

    class _Conn:
        closed = True
        broken = False

    with pytest.raises(exc.DisconnectionError):
        for listener in listeners:
            listener(_Conn(), None, None)


def test_the_sqlalchemy_tracer_is_not_installed_when_nothing_collects_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The instrumentation wraps every statement whether or not an exporter exists, and the
    shipped default exporter is 'none'."""
    from memory_service.adapters.db import engine as engine_module

    monkeypatch.setattr(engine_module, "_tracing_is_on", lambda: False)
    engine = Database(DatabaseSettings()).engine.sync_engine
    assert not engine.dispatch.before_cursor_execute

    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    monkeypatch.setattr(engine_module, "_tracing_is_on", lambda: True)
    try:
        traced = Database(DatabaseSettings()).engine.sync_engine
        assert traced.dispatch.before_cursor_execute, "an exporter must still get its spans"
    finally:
        SQLAlchemyInstrumentor().uninstrument()
