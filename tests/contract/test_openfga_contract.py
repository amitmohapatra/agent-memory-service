"""Golden checks against a real OpenFGA server (needs Docker + registry access)."""

from __future__ import annotations

import pytest

from memory_service.ports.authorization import AccessCheck
from tests.unit.test_authz_provider import GOLDEN_CHECKS, GOLDEN_TUPLES

pytestmark = [pytest.mark.contract, pytest.mark.docker]


@pytest.fixture(scope="module")
def openfga_url():
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


async def test_openfga_agrees_with_reference_provider(openfga_url: str) -> None:
    from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider
    from memory_service.config.settings import AuthorizationSettings

    provider = OpenFGAAuthorizationProvider(
        AuthorizationSettings(provider="openfga", openfga_api_url=openfga_url)
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
