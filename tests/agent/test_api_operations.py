"""The four routes an operator reaches before any key exists: liveness, readiness, metrics,
version. An agent framework calls these to decide whether to send a turn at all, so they are
the one group that answers without a key - and the one group whose absence from the SDK meant
a caller had to reach around it for a scrape.
"""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk

pytestmark = pytest.mark.e2e


@pytest.mark.covers("operations.live", "operations.ready", "operations.version")
async def test_an_operator_reads_liveness_readiness_and_version(app, running) -> None:
    client = sdk(app, BOOTSTRAP)

    assert (await client.alive())["status"] == "ok"

    ready = await client.health()
    assert ready["status"] in ("ready", "degraded"), ready
    assert ready["dependencies"]["postgres"] == {"ok": True, "mandatory": True, "error": None}

    version = await client.version()
    assert version["api_version"] == "v1"
    assert version["environment"] == "test"
    # The providers block is what is *running*, not what was configured, which is the only way
    # a caller can tell a stand-in from the real store.
    assert set(version["providers"]) >= {"cache", "search", "authorization", "llm"}
    assert "secret" not in str(version).lower()


@pytest.mark.covers("operations.metrics")
async def test_a_scrape_reads_the_worker_s_own_registry(app, running) -> None:
    client = sdk(app, BOOTSTRAP)

    exposition = await client.metrics()

    assert isinstance(exposition, str) and exposition
    # The exposition format, not JSON: a scrape needs the type lines to parse it at all.
    assert "# HELP" in exposition and "# TYPE" in exposition


@pytest.mark.covers("operations.live")
async def test_liveness_answers_a_caller_holding_no_key_at_all(app, running) -> None:
    """Nothing in the probe path depends on authentication: a load balancer has no key."""
    anonymous = sdk(app, "")

    assert (await anonymous.alive())["status"] == "ok"
