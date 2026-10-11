"""The worker's queue connector rides out a database out of reach (``ResilientConnector``).

A pool timeout in one heartbeat stopped the worker and the process exited: under memory
pressure the database answered late for a minute and the job worker was gone until someone
restarted it. The live behaviour against PostgreSQL is in
``tests/integration/test_procrastinate_queue.py``; these pin the retry rules."""

from __future__ import annotations

from typing import Any

import psycopg
import psycopg_pool
import pytest

from memory_service.adapters.tasks import procrastinate_queue as pq
from memory_service.observability.metrics import REGISTRY

pytestmark = pytest.mark.unit


def _out_of_reach() -> Exception:
    exc = pq.pexc.ConnectorException()
    exc.__cause__ = psycopg_pool.PoolTimeout("couldn't get a connection after 5.00 sec")
    return exc


def _disk_full() -> Exception:
    exc = pq.pexc.ConnectorException()
    exc.__cause__ = psycopg.errors.DiskFull("could not extend file: No space left on device")
    return exc


def _connector(outage: float | None) -> pq.ResilientConnector:
    connector = pq.ResilientConnector(conninfo="")  # never opened: the calls are stand-ins
    connector.outage_seconds = outage
    connector.backoff = (0.01, 0.02)
    return connector


def _retries(operation: str) -> float:
    found = REGISTRY.get_sample_value("memory_worker_db_retries_total", {"operation": operation})
    return found or 0.0


def _failing(times: int, error: Any = _out_of_reach) -> tuple[list[int], Any]:
    calls: list[int] = []

    async def call(*args: Any, **kwargs: Any) -> str:
        calls.append(1)
        if len(calls) <= times:
            raise error()
        return "ok"

    return calls, call


async def test_a_query_that_found_no_connection_is_tried_again_and_counted() -> None:
    calls, call = _failing(2)
    before = _retries("query")
    assert await _connector(5)._ride_out(call) == "ok"
    assert len(calls) == 3 and _retries("query") == before + 2


async def test_a_write_to_a_full_disk_is_tried_again_and_the_outage_is_a_gauge() -> None:
    """The worker died on a job-status write when PostgreSQL's disk filled: the statement was
    rolled back, so once there is room again running it again is safe."""
    seen: list[float] = []
    calls, failing = _failing(2, error=_disk_full)

    async def call() -> str:
        seen.append(REGISTRY.get_sample_value("memory_worker_db_unavailable_seconds") or 0.0)
        return await failing()

    assert await _connector(5)._ride_out(call) == "ok"
    assert len(calls) == 3 and seen[2] > 0, "out of reach while it retried"
    assert REGISTRY.get_sample_value("memory_worker_db_unavailable_seconds") == 0


async def test_past_the_outage_budget_the_error_is_raised() -> None:
    calls, call = _failing(1000)
    with pytest.raises(pq.pexc.ConnectorException):
        await _connector(0.1)._ride_out(call)
    assert 1 < len(calls) < 1000


async def test_only_a_statement_that_never_ran_is_tried_again() -> None:
    """Any other database error may have come after the statement ran: it is raised at once."""
    calls, call = _failing(1, error=pq.pexc.ConnectorException)
    with pytest.raises(pq.pexc.ConnectorException):
        await _connector(5)._ride_out(call)
    assert len(calls) == 1


async def test_the_apis_queue_raises_at_once() -> None:
    calls, call = _failing(1)
    with pytest.raises(pq.pexc.ConnectorException):
        await _connector(None)._ride_out(call)
    assert len(calls) == 1


async def test_a_lost_listen_connection_is_opened_again(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, listen = _failing(2, error=pq.pexc.ConnectorException)
    parent = pq.ResilientConnector.__mro__[1]
    monkeypatch.setattr(parent, "listen_notify", listen)
    before = _retries("listen")

    async def notified(**_: Any) -> None:
        return None

    await _connector(5).listen_notify(notified, ["procrastinate_any_queue_v1"])
    assert len(calls) == 3 and _retries("listen") == before + 2
