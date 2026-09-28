"""The agent's view: a real service in ``api_key`` mode, driven only through the SDK.

A test holds keys the way a harness would - the bootstrap secret, an admin key, a service
key - and everything it learns, it learns through ``trellis.memory``. The two exceptions
are named where they happen: backdating rows to make retention due, and reading the
service's own configuration to know the default quota. PostgreSQL is real; search, cache and authorization are the in-process
stand-ins the hermetic suite uses, so a leak here is a leak in the service's own logic.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from memory_service.api.app import create_app
from tests.conftest import PG_AVAILABLE, _test_overrides
from tests.e2e.conftest import TABLES
from tests.support_real import reset_real_backends

BOOTSTRAP = "boot-secret-for-tests"


@pytest.fixture
def app(make_settings):
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        authentication={"mode": "api_key", "bootstrap_admin_key": BOOTSTRAP},
    )
    return create_app(settings, overrides=_test_overrides(tasks="inline"))


@pytest.fixture
def running(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        container = app.state.container

        async def _truncate() -> None:
            async with container.database.engine.begin() as conn:
                await conn.execute(
                    text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
                )
            await reset_real_backends(container)

        c.portal.call(_truncate)
        # The lifespan's background flusher lives in the portal's loop while SDK requests
        # run in the test's; stop it so the read audit is written only by the inline flush
        # a listing performs, in the test's loop, and never raced from another thread.
        c.portal.call(container.services["read_audit"].close)
        c.portal.call(container.services["tenant_registry"].close)
        yield c


def sdk(app, token: str):
    from trellis.memory import MemoryClient

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://memory.test")
    return MemoryClient("http://memory.test", api_key=token, http_client=http)
