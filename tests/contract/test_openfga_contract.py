"""Golden checks against a real OpenFGA server (needs Docker + registry access)."""

from __future__ import annotations

import pytest

from memory_service.ports.authorization import AccessCheck
from tests.unit.test_authz_provider import GOLDEN_CHECKS, GOLDEN_TUPLES

pytestmark = [pytest.mark.contract, pytest.mark.docker]


def _running_openfga() -> str | None:
    """The dev stack already runs one; prefer it over starting a second.

    ``testcontainers`` needs to bind-mount the Docker socket, which Docker Desktop on macOS
    refuses — so on a developer machine this test skipped even with OpenFGA healthy and
    listening. Ask the server directly instead.
    """
    import os

    import httpx

    url = os.environ.get("OPENFGA_API_URL", "http://localhost:8081")
    try:
        httpx.get(f"{url}/healthz", timeout=3)
    except Exception:
        return None
    return url


@pytest.fixture(scope="module")
def openfga_url():
    if running := _running_openfga():
        yield running
        return
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.waiting_utils import wait_for_logs
    except ImportError:
        pytest.skip("testcontainers not installed")
    try:
        container = (
            DockerContainer("openfga/openfga:v1.18.1").with_command("run").with_exposed_ports(8080)
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker/OpenFGA unavailable: {exc}")
    wait_for_logs(container, "starting HTTP server", timeout=60)
    try:
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(8080)}"
    finally:
        container.stop()


@pytest.fixture
async def openfga_store(openfga_url: str):
    """A store of this test's own.

    The provider reuses a store by name, so on the shared dev server the golden tuples would
    mix with everything every other test has written — ``list_objects`` then returns hundreds
    of threads instead of the two the fixture created. A per-test store keeps the assertions
    about the golden set exact.
    """
    import uuid

    import httpx

    async with httpx.AsyncClient(base_url=openfga_url, timeout=30) as http:
        response = await http.post("/stores", json={"name": f"contract-test-{uuid.uuid4().hex[:8]}"})
        response.raise_for_status()
        store_id = response.json()["id"]
        try:
            yield store_id
        finally:
            await http.delete(f"/stores/{store_id}")


async def test_openfga_agrees_with_reference_provider(openfga_url: str, openfga_store: str) -> None:
    from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider
    from memory_service.config.settings import AuthorizationSettings

    provider = OpenFGAAuthorizationProvider(
        AuthorizationSettings(
            provider="openfga", openfga_api_url=openfga_url, openfga_store_id=openfga_store
        )
    )
    await provider.write(GOLDEN_TUPLES)
    for user, relation, obj, expected in GOLDEN_CHECKS:
        if relation == "nonexistent" or obj.startswith("unknown:"):
            continue  # OpenFGA rejects unknown relations/types with 400 rather than False
        assert (
            await provider.check(AccessCheck(user=user, relation=relation, object=obj)) is expected
        ), (user, relation, obj)
    assert sorted(await provider.list_objects("user:admin1", "can_read", "thread")) == [
        "thread:acme/thr1",
        "thread:acme/thr2",
    ]
    await provider.close()
