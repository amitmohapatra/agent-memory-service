"""Every process must agree on which authorization store it is talking to.

``_ensure_store`` runs from ``ping()`` - the first readiness probe of every process. On a
fresh stack that is three uvicorn workers and the job worker arriving together, each calling
``list_stores``, each finding nothing named ``memory-service``, and each creating one, because
OpenFGA does not make store names unique. Tuples written through one store are invisible to a
check served through another, so the same request is allowed or denied depending on which
process answers it - an authorization split with no error anywhere to say it happened.

The race cannot be prevented from the client, so it is made to converge: every process picks
the same store out of whatever exists.
"""

from __future__ import annotations

import pytest

from memory_service.adapters.authz.openfga_provider import (
    STORE_NAME,
    OpenFGAAuthorizationProvider,
)

pytestmark = pytest.mark.unit


def _provider() -> OpenFGAAuthorizationProvider:
    """The instance without its constructor: this exercises store selection, not connection."""
    return OpenFGAAuthorizationProvider.__new__(OpenFGAAuthorizationProvider)


class _Store:
    def __init__(self, store_id: str, name: str) -> None:
        self.id = store_id
        self.name = name


class _Stores:
    def __init__(self, stores):
        self.stores = stores


class _Client:
    """Enough OpenFGA to answer list/create, and a counter for how often it created."""

    def __init__(self, stores=()) -> None:
        self._stores = list(stores)
        self.created = 0

    async def list_stores(self):
        return _Stores(list(self._stores))

    async def create_store(self, request):
        self.created += 1
        self._stores.append(_Store(f"created-{self.created}", request.name))
        return self._stores[-1]


async def test_an_existing_store_is_reused_and_nothing_is_created() -> None:
    client = _Client([_Store("s-1", STORE_NAME)])
    assert await _provider()._ensure_store(client) == "s-1"
    assert client.created == 0


async def test_the_first_process_creates_one_and_gets_it_back() -> None:
    client = _Client()
    store_id = await _provider()._ensure_store(client)
    assert client.created == 1
    assert store_id == "created-1"


async def test_processes_that_raced_converge_on_the_same_store() -> None:
    """The defect. Four processes created four stores; all four must now agree on one."""
    raced = [_Store(f"s-{n}", STORE_NAME) for n in ("3", "1", "4", "2")]
    chosen = {await _provider()._ensure_store(_Client(raced)) for _ in range(4)}
    assert chosen == {"s-1"}, f"processes disagreed: {chosen}"


async def test_stores_belonging_to_something_else_are_not_adopted() -> None:
    client = _Client([_Store("a-0", "someone-elses-store")])
    store_id = await _provider()._ensure_store(client)
    assert client.created == 1 and store_id == "created-1"
