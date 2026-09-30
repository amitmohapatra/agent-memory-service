"""HTTP read policy with actual retrieval and a mocked, never paid, Bifrost adapter: a read
follows the tenant policy's read_assist unless it sets use_llm."""

import pytest

from memory_service.ports.credentials import tenant_identity
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "reader"}


def _read_assist(client, enabled: bool) -> None:
    container = client.app.state.container

    async def put() -> None:
        async with container.services["uow_factory"]() as uow:
            await container.services["model_policies"].set(
                uow,
                tenant_identity("acme"),
                uses=["query_expansion", "entity_resolution"],
                read_assist=enabled,
            )
            await uow.commit()

    client.portal.call(put)


def _configure(client, configured) -> None:
    """Point the service's assist at the mocked gateway, wired the way an enabled model is:
    with the service's policy resolution."""
    container = client.app.state.container
    assist = container.services["llm_assist"]
    assist.provider, assist.settings = configured.provider, configured.settings
    assist.policies = container.services["model_policies"]


@pytest.mark.parametrize("route", ["context", "recall", "verify"])
def test_a_read_follows_read_assist_and_an_opt_in_cannot_poison_the_native_cache(client, route):
    with mocked_gateway(
        [{"query_type": "GENERAL_SEMANTIC", "terms": ["telescope"], "identifiers": []}]
    ) as gw:
        configured = gw.assist(["query_expansion"])
        _configure(client, configured)
        _read_assist(client, False)
        payload = {"query": "astronomical instrumentation"}
        if route == "verify":
            payload["answer"] = "The observatory owns a telescope."
        try:
            native = client.post(f"/v1/{route}", headers=HEADERS, json=payload)
            assert native.status_code == 200, native.text
            assert gw.route.call_count == 0
            assisted = client.post(
                f"/v1/{route}", headers=HEADERS, json={**payload, "use_llm": True}
            )
            assert assisted.status_code == 200, assisted.text
            assert gw.route.call_count == 1
            repeat = client.post(f"/v1/{route}", headers=HEADERS, json=payload)
            assert repeat.status_code == 200, repeat.text
            assert gw.route.call_count == 1
            # the policy flips: the same read, with nothing set, is now assisted
            _read_assist(client, True)
            followed = client.post(
                f"/v1/{route}", headers=HEADERS, json={**payload, "query": "optical instruments"}
            )
            assert followed.status_code == 200, followed.text
            assert gw.route.call_count == 2
            refused = client.post(
                f"/v1/{route}",
                headers=HEADERS,
                json={**payload, "query": "radio instruments", "use_llm": False},
            )
            assert refused.status_code == 200, refused.text
            assert gw.route.call_count == 2, "an explicit use_llm=false overrides the policy"
        finally:
            client.portal.call(configured.provider.close)


def test_graph_entity_resolution_follows_the_read_policy(client):
    from memory_service.domain.context import MemoryExecutionContext
    from tests.integration.test_graph import _ingest

    container = client.app.state.container
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="reader")
    client.portal.call(_ingest, container, container.services["uow_factory"], ctx)
    with mocked_gateway(
        [{"matches": [{"query_name": "earnings metric", "entity_name": "Adjusted EBITDA"}]}]
    ) as gw:
        assist = gw.assist(["entity_resolution"])
        container.services["graph"].assist = assist
        _configure(client, assist)
        _read_assist(client, False)
        payload = {"entities": ["earnings metric"]}
        try:
            response = client.post("/v1/graph/query", headers=HEADERS, json=payload)
            assert response.status_code == 200, response.text
            assert gw.route.call_count == 0
            response = client.post(
                "/v1/graph/query", headers=HEADERS, json={**payload, "use_llm": True}
            )
            assert response.status_code == 200, response.text
            assert gw.route.call_count == 1
            assert response.json()["matched"][0]["canonical_name"] == "adjusted ebitda"
        finally:
            client.portal.call(assist.provider.close)
