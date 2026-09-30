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
from tests.agent import coverage
from tests.conftest import PG_AVAILABLE, _test_overrides
from tests.e2e.conftest import TABLES
from tests.support_real import reset_real_backends

BOOTSTRAP = "boot-secret-for-tests"


@pytest.fixture
def app(make_settings):
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        authentication={"bootstrap_admin_key": BOOTSTRAP},
    )
    return create_app(settings, overrides=_test_overrides(tasks="inline"))


@pytest.fixture
def running(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        container = app.state.container

        async def _truncate() -> None:
            async with container.database.engine.begin() as conn:
                # a reset, not a hot path: on a loaded host it may outlast the statement timeout
                await conn.execute(text("SET LOCAL statement_timeout = 0"))
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

    # Recording wrapper: every operation this client reaches is noted, which is how
    # tests/agent/test_api_coverage.py can fail on a route no test drives. Nothing else in
    # the suite is instrumented, so coverage is only ever earned through the SDK.
    transport = coverage.RecordingTransport(httpx.ASGITransport(app=app))
    http = httpx.AsyncClient(transport=transport, base_url="http://memory.test")
    return MemoryClient("http://memory.test", api_key=token, http_client=http)


# --------------------------------------------------------------------------- the coverage gate


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers(*operation_ids): this test drives these operations to a status under 400",
    )
    config.addinivalue_line(
        "markers",
        "covers_error(*operation_ids): this test drives these operations to an error status",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Record every claim at collection time, so the gate does not depend on test order."""
    coverage.CLAIMED.clear()
    coverage.CLAIMED_ERRORS.clear()
    for item in items:
        if "tests/agent/" not in item.nodeid and not item.nodeid.startswith("tests/agent"):
            continue
        for mark, claims in (
            ("covers", coverage.CLAIMED),
            ("covers_error", coverage.CLAIMED_ERRORS),
        ):
            for marker in item.iter_markers(mark):
                for operation in marker.args:
                    claims.setdefault(operation, []).append(item.nodeid)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    setattr(item, f"report_{report.when}", report)
    return report


@pytest.fixture(autouse=True)
def _verify_coverage_claims(request: pytest.FixtureRequest) -> Iterator[None]:
    """A claim a test did not actually make is a failure: the gate is only as good as this."""
    coverage.begin_test()
    yield
    report = getattr(request.node, "report_call", None)
    if report is None or not report.passed:
        return  # a failing test has one reason already; do not bury it under a second
    unknown = set(coverage.operations())
    for mark, errors in (("covers", False), ("covers_error", True)):
        claimed = {op for m in request.node.iter_markers(mark) for op in m.args}
        assert claimed <= unknown, f"{mark}: not operations in docs/openapi.json: " + str(
            sorted(claimed - unknown)
        )
        missing = claimed - coverage.reached(errors=errors)
        assert not missing, (
            f"{mark} claims {sorted(missing)} but this test never reached "
            f"{'an error status on' if errors else ''} them through the SDK; reached "
            f"{sorted(coverage.reached(errors=errors))}"
        )
