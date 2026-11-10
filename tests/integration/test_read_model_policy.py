"""HTTP read policy with actual retrieval and a mocked, never paid, Bifrost adapter: the read that
consults a model (graph entity resolution) follows the tenant policy's read_assist; context and
recall consult none."""

import pytest

from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "reader"}


def _read_assist(client, enabled: bool) -> None:
    container = client.app.state.container

    async def put() -> None:
        async with container.services["uow_factory"]() as uow:
            await container.services["model_policies"].set(
                uow,
                "acme",
                uses=["query_expansion", "entity_resolution"],
                read_assist=enabled,
                models={},
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


@pytest.mark.parametrize("route", ["context", "recall"])
def test_a_context_or_recall_read_calls_no_model_whatever_the_read_policy(client, route):
    """``query_expansion`` in a policy is accepted and ignored: with reads assisted or not, a
    context or recall read asks the model nothing. The graph route below is the read that
    follows read_assist."""
    with mocked_gateway(
        [{"query_type": "GENERAL_SEMANTIC", "terms": ["telescope"], "identifiers": []}]
    ) as gw:
        configured = gw.assist(["query_expansion"])
        _configure(client, configured)
        try:
            for enabled, query in (
                (False, "astronomical instrumentation"),
                (True, "optical instruments"),
            ):
                _read_assist(client, enabled)
                response = client.post(f"/v1/{route}", headers=HEADERS, json={"query": query})
                assert response.status_code == 200, response.text
            assert gw.route.call_count == 0
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
        params = {"q": "earnings metric"}
        try:
            response = client.get("/v1/graph/entities", headers=HEADERS, params=params)
            assert response.status_code == 200, response.text
            assert gw.route.call_count == 0
            _read_assist(client, True)
            response = client.get("/v1/graph/entities", headers=HEADERS, params=params)
            assert response.status_code == 200, response.text
            assert gw.route.call_count == 1
            assert response.json()["entities"][0]["canonical_name"] == "adjusted ebitda"
        finally:
            client.portal.call(assist.provider.close)
