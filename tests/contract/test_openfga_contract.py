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
        response = await http.post(
            "/stores", json={"name": f"contract-test-{uuid.uuid4().hex[:8]}"}
        )
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


async def test_a_tuple_that_already_exists_does_not_discard_the_new_ones(
    openfga_url: str, openfga_store: str
) -> None:
    """Re-asserting a relation is not a failure, and must not lose its batch.

    OpenFGA's Write is transactional and rejects a batch containing a tuple it already holds.
    Because tuples outlive the rows they describe, a thread id reused after its rows were
    deleted hit that on every message write and became permanently unusable: "OpenFGA write
    failed: ValidationException", with the reason discarded.
    """
    from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider
    from memory_service.config.settings import AuthorizationSettings
    from memory_service.ports.authorization import AccessCheck
    from memory_service.ports.authorization import RelationTuple as R

    provider = OpenFGAAuthorizationProvider(
        AuthorizationSettings(
            provider="openfga", openfga_api_url=openfga_url, openfga_store_id=openfga_store
        )
    )
    existing = R(user="user:u1", relation="owner", object="thread:acme/reused")
    await provider.write([existing])

    # the same tuple again, batched with one that is genuinely new
    fresh = R(user="user:u2", relation="viewer", object="thread:acme/reused")
    await provider.write([existing, fresh])

    allowed = await provider.batch_check(
        [
            AccessCheck(user="user:u1", relation="owner", object="thread:acme/reused"),
            AccessCheck(user="user:u2", relation="viewer", object="thread:acme/reused"),
        ]
    )
    assert allowed == [True, True], "the new tuple must survive the redundant one"

    # deleting something that is not there is the same kind of no-op
    await provider.write([], [R(user="user:u9", relation="viewer", object="thread:acme/reused")])
    await provider.close()


async def test_a_real_write_failure_still_says_what_was_wrong(
    openfga_url: str, openfga_store: str
) -> None:
    """The idempotency path must not swallow genuine errors, or hide their reason."""
    from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider
    from memory_service.config.settings import AuthorizationSettings
    from memory_service.domain.errors import DependencyUnavailable
    from memory_service.ports.authorization import RelationTuple as R

    provider = OpenFGAAuthorizationProvider(
        AuthorizationSettings(
            provider="openfga", openfga_api_url=openfga_url, openfga_store_id=openfga_store
        )
    )
    with pytest.raises(DependencyUnavailable) as caught:
        await provider.write(
            [R(user="user:u1", relation="no_such_relation", object="thread:acme/x")]
        )
    message = str(caught.value)
    assert "no_such_relation" in message, f"the reason must survive, got: {message}"
    await provider.close()


async def test_a_subject_type_the_model_does_not_define_is_an_empty_scope(
    openfga_url: str, openfga_store: str
) -> None:
    """Not every caller is a type the model knows about.

    A request authenticated by API key alone carries no user and no agent, so scope
    resolution asked OpenFGA about ``service:...`` — a type the model does not define. That
    is not an outage: a subject type with no grants owns nothing. Raising turned every such
    request into a 500 from a perfectly healthy gateway.
    """
    from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider
    from memory_service.config.settings import AuthorizationSettings

    provider = OpenFGAAuthorizationProvider(
        AuthorizationSettings(
            provider="openfga", openfga_api_url=openfga_url, openfga_store_id=openfga_store
        )
    )
    assert await provider.list_objects("service:api-key", "can_read", "thread") == []

    # a type the model *does* define still resolves normally
    from memory_service.ports.authorization import RelationTuple as R

    await provider.write([R(user="user:u1", relation="owner", object="thread:acme/t1")])
    assert await provider.list_objects("user:u1", "can_read", "thread") == ["thread:acme/t1"]
    await provider.close()
