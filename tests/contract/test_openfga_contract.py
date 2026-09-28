"""Golden checks against a real OpenFGA server (needs Docker + registry access)."""

from __future__ import annotations

import json

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


async def test_workspace_revocation_is_immediate_for_groups_and_their_users(
    openfga_url: str, openfga_store: str
) -> None:
    """A user reads a workspace through a group; ending either link ends the access.

    ``modules/tenancy`` removes a group from a workspace (tuple ``group:<t>/<g>#member`` on
    ``workspace:<t>/<w>``) and removes a user from a group (``user:<u>`` member of
    ``group:<t>/<g>``). The real OpenFGA must agree with the reference provider that each
    removal alone is enough - ``viewer`` is the relation ``AuthorizedScope.workspace_ids``
    resolves, so a stale answer here is a cross-team read.
    """
    from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider
    from memory_service.config.settings import AuthorizationSettings
    from memory_service.ports.authorization import RelationTuple as R

    provider = OpenFGAAuthorizationProvider(
        AuthorizationSettings(
            provider="openfga", openfga_api_url=openfga_url, openfga_store_id=openfga_store
        )
    )
    admitted = R(user="group:acme/counsel#member", relation="member", object="workspace:acme/legal")
    in_group = R(user="user:lawyer1", relation="member", object="group:acme/counsel")
    direct = R(user="agent:bot", relation="member", object="workspace:acme/legal")
    await provider.write([admitted, in_group, direct])
    assert await provider.list_objects("user:lawyer1", "viewer", "workspace") == [
        "workspace:acme/legal"
    ], "a member reads as a viewer, through the group"
    assert await provider.list_objects("agent:bot", "viewer", "workspace") == [
        "workspace:acme/legal"
    ]

    await provider.write([], [admitted])  # the group leaves the workspace
    assert await provider.list_objects("user:lawyer1", "viewer", "workspace") == []
    assert await provider.list_objects("agent:bot", "viewer", "workspace") == [
        "workspace:acme/legal"
    ], "unrelated grants survive a revocation"

    await provider.write([admitted], [in_group])  # the group is back; the user has left it
    assert await provider.list_objects("user:lawyer1", "viewer", "workspace") == []
    assert await provider.list_objects("user:lawyer1", "member", "group") == []

    await provider.write([], [direct])
    assert await provider.list_objects("agent:bot", "viewer", "workspace") == []
    await provider.close()


async def test_the_authorization_model_rolls_forward_like_a_schema(
    openfga_url: str, openfga_store: str
) -> None:
    """An upgraded deployment must get the relations its code writes. The provider writes
    this build's model when the store's latest one differs, and leaves it alone when it does
    not - models are immutable and append-only, so writing is the safe direction."""
    from memory_service.adapters.authz.openfga_provider import (
        MODEL_PATH,
        OpenFGAAuthorizationProvider,
        _dsl_to_json,
        model_shape,
    )
    from memory_service.config.settings import AuthorizationSettings
    from memory_service.ports.authorization import RelationTuple as R

    settings = AuthorizationSettings(
        provider="openfga", openfga_api_url=openfga_url, openfga_store_id=openfga_store
    )
    current = _dsl_to_json(MODEL_PATH.read_text(encoding="utf-8"))
    # the pre-0021 model: workspaces admitted users and groups, not agents
    older = json.loads(json.dumps(current))
    for td in older["type_definitions"]:
        if td["type"] == "workspace":
            for rel in ("member", "viewer"):
                td["metadata"]["relations"][rel]["directly_related_user_types"] = [
                    d
                    for d in td["metadata"]["relations"][rel]["directly_related_user_types"]
                    if d["type"] != "agent"
                ]
    assert model_shape(older) != model_shape(current)
    seeded = OpenFGAAuthorizationProvider(settings, model_json=older)
    old_id = await seeded._ensure_model(await seeded._get_client())  # noqa: SLF001
    await seeded.close()

    provider = OpenFGAAuthorizationProvider(settings)
    client = await provider._get_client()  # noqa: SLF001
    new_id = client.get_authorization_model_id()
    assert new_id != old_id, "the store's model was older than this build's: rolled forward"
    models = await client.read_authorization_models()
    assert len(models.authorization_models) == 2
    assert model_shape(models.authorization_models[0].to_dict()) == model_shape(current)
    await provider.write([R(user="agent:bot", relation="member", object="workspace:acme/legal")])
    assert await provider.list_objects("agent:bot", "viewer", "workspace") == [
        "workspace:acme/legal"
    ]
    await provider.close()

    again = OpenFGAAuthorizationProvider(settings)
    assert (await again._get_client()).get_authorization_model_id() == new_id  # noqa: SLF001
    assert (
        len((await (await again._get_client()).read_authorization_models()).authorization_models)
        == 2
    ), (  # noqa: SLF001
        "the same model is not written twice"
    )
    await again.close()
