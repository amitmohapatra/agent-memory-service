"""HTTP read policy with actual retrieval and a mocked, never paid, Bifrost adapter."""

import pytest

from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "reader"}


@pytest.mark.parametrize("route", ["context", "recall", "verify"])
def test_read_defaults_to_no_llm_and_opt_in_cannot_poison_native_cache(client, route):
    with mocked_gateway(
        [{"query_type": "GENERAL_SEMANTIC", "terms": ["telescope"], "identifiers": []}]
    ) as gw:
        configured = gw.assist(["query_expansion"])
        assist = client.app.state.container.services["llm_assist"]
        assist.provider, assist.settings = configured.provider, configured.settings
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
        finally:
            client.portal.call(configured.provider.close)


def test_graph_entity_resolution_is_opt_in(client):
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
