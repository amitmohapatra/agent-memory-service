"""Real isolated SQL, mocked gateway: no provider credit is consumed."""

import asyncio

import pytest
from pydantic import SecretStr
from sqlalchemy import text

from memory_service.adapters.models.credential_cipher import AesCredentialCipher
from memory_service.config.settings import AgentCredentialSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.domain.ids import new_id
from memory_service.modules.llm.credentials import ModelCredentials, agent_identity
from memory_service.modules.llm.policy import current_model_identity
from memory_service.ports.credentials import ModelIdentity, tenant_identity
from tests.integration.test_memory import _observe
from tests.support_llm import bound_to, chat_response, mocked_gateway
from tests.unit.test_agent_credential_cipher import TEST_KEY
from tests.unit.test_narrative_memory import MESSAGE

pytestmark = pytest.mark.integration
ALICE = MemoryExecutionContext(tenant_id="acme", user_id="alice", agent_id="research")
BOB = ALICE.model_copy(update={"user_id": "bob"})


def service(uow_factory):
    return ModelCredentials(
        uow_factory,
        AesCredentialCipher(
            AgentCredentialSettings(active_key_id="test", encryption_keys={"test": TEST_KEY})
        ),
    )


async def save(credentials, uow_factory, ctx, key):
    async with uow_factory() as uow:
        record = await credentials.set(uow, ctx, SecretStr(key) if key else None)
        await uow.commit()
    return record


async def test_storage_rotation_revocation_and_tenant_owner_isolation(container, uow_factory):
    credentials = service(uow_factory)
    first = await save(credentials, uow_factory, ALICE, "vk-alice-test")
    assert first.revision == 1
    resolved_first = await credentials.resolve(agent_identity(ALICE))
    assert resolved_first is not None
    assert resolved_first.key.get_secret_value() == "vk-alice-test"
    assert resolved_first.identity == ModelIdentity("acme", ALICE.principal_id)
    assert await credentials.resolve(agent_identity(BOB)) is None
    assert (
        await credentials.resolve(agent_identity(ALICE.model_copy(update={"tenant_id": "rival"})))
        is None
    )
    async with container.database.engine.connect() as conn:
        stored = (await conn.execute(text("select ciphertext from agent_credentials"))).scalar_one()
    assert b"vk-alice-test" not in stored
    second = await save(credentials, uow_factory, ALICE, "vk-rotated-test")
    assert second.revision == 2
    with pytest.raises(ProviderNotConfigured):  # the call ran under the rotated-away key
        await credentials.confirm(agent_identity(ALICE), resolved_first)
    revoked = await save(credentials, uow_factory, ALICE, None)
    assert revoked.revision == 3 and revoked.ciphertext is None
    assert (await credentials.metadata(ALICE)).ciphertext is None
    with pytest.raises(ProviderNotConfigured, match="revoked"):
        await credentials.resolve(agent_identity(ALICE))
    restored = await save(credentials, uow_factory, ALICE, "vk-restored-test")
    assert restored.revision == 4


async def test_concurrent_agents_send_only_their_key_and_explicit_mcp_denial(
    container, uow_factory
):
    credentials = service(uow_factory)
    await save(credentials, uow_factory, ALICE, "vk-alice-test")
    await save(credentials, uow_factory, BOB, "vk-bob-test")
    with mocked_gateway(["accepted"]) as gateway:
        assist = gateway.assist(["reflection"])
        assist.provider.credentials = credentials

        async def call(ctx):
            with bound_to(ctx.tenant_id, ctx.principal_id):
                return await assist.complete(
                    "reflection", system="Service-owned instructions", user=ctx.user_id
                )

        try:
            assert await asyncio.gather(call(ALICE), call(BOB)) == ["accepted", "accepted"]
            assert current_model_identity() is None
            seen = {request.request.headers["x-bf-vk"] for request in gateway.route.calls}
            assert seen == {"vk-alice-test", "vk-bob-test"}
            for call_record in gateway.route.calls:
                headers = call_record.request.headers
                assert headers["authorization"] == f"Bearer {headers['x-bf-vk']}"
                assert headers["x-bf-mcp-include-clients"] == ""
                assert headers["x-bf-mcp-include-tools"] == ""
                assert headers["x-bf-disable-content-logging"] == "true"
            assert all(
                prompt["tool_choice"] == "none" and "tools" not in prompt
                for prompt in gateway.prompts()
            )
            await save(credentials, uow_factory, ALICE, None)
            assert await call(ALICE) is None
            assert gateway.route.call_count == 2  # no operator-key retry after revocation
        finally:
            await assist.provider.close()


async def test_rotated_inflight_response_is_discarded(container, uow_factory):
    credentials = service(uow_factory)
    await save(credentials, uow_factory, ALICE, "vk-before-test")
    with mocked_gateway(["must not be accepted"]) as gateway:
        assist = gateway.assist(["reflection"])
        assist.provider.credentials = credentials
        import httpx

        async def rotate_during_response(request):
            await save(credentials, uow_factory, ALICE, "vk-after-test")
            return httpx.Response(200, json=chat_response("must not be accepted"))

        gateway.route.mock(side_effect=rotate_during_response)
        try:
            with bound_to(ALICE.tenant_id, ALICE.principal_id):
                assert (
                    await assist.complete(
                        "reflection", system="Service instructions", user="source"
                    )
                    is None
                )
            assert gateway.route.call_count == 1
        finally:
            await assist.provider.close()


def test_agent_key_http_does_not_return_secrets_and_is_owner_scoped(client):
    container = client.app.state.container
    container.services["model_credentials"] = service(container.services["uow_factory"])
    headers = {
        "X-API-Key": "test-key",
        "X-Trellis-Tenant": new_id("request"),
        "X-Trellis-User": "alice",
    }
    body = {"scope": {"agent_id": "research"}, "virtual_key": "vk-http-test"}
    first = client.put(
        "/v1/agents/model-key", headers={**headers, "Idempotency-Key": "put-key"}, json=body
    )
    assert first.status_code == 200, first.text
    assert first.json()["revision"] == 1 and "vk-http-test" not in first.text
    replay = client.put(
        "/v1/agents/model-key", headers={**headers, "Idempotency-Key": "put-key"}, json=body
    )
    assert replay.json() == first.json()
    conflict = client.put(
        "/v1/agents/model-key",
        headers={**headers, "Idempotency-Key": "put-key"},
        json={**body, "virtual_key": "vk-different-test"},
    )
    assert conflict.status_code == 409 and "vk-different-test" not in conflict.text
    other = client.get(
        "/v1/agents/model-key?agent_id=research", headers={**headers, "X-Trellis-User": "bob"}
    )
    assert other.status_code == 200 and not other.json()["registered"]
    revoked = client.delete("/v1/agents/model-key?agent_id=research", headers=headers)
    assert revoked.status_code == 200 and revoked.json()["revoked"]
    invalid = client.put(
        "/v1/agents/model-key", headers=headers, json={**body, "virtual_key": "vk-secret\ninvalid"}
    )
    assert invalid.status_code == 422 and "vk-secret" not in invalid.text
    missing_agent = client.put("/v1/agents/model-key", headers=headers, json={**body, "scope": {}})
    assert missing_agent.status_code == 422


def test_http_read_policy_and_rotation_route_only_the_owners_key(client):
    container = client.app.state.container
    credentials = service(container.services["uow_factory"])
    container.services["model_credentials"] = credentials
    headers = {
        "X-API-Key": "test-key",
        "X-Trellis-Tenant": new_id("request"),
        "X-Trellis-User": "alice",
    }
    scope = {"agent_id": "research"}
    query = {"scope": scope, "query": "astronomical instrumentation"}
    with mocked_gateway(
        [{"query_type": "GENERAL_SEMANTIC", "terms": ["telescope"], "identifiers": []}]
    ) as gateway:
        configured = gateway.assist(["query_expansion"])
        configured.provider.credentials = credentials
        assist = container.services["llm_assist"]
        assist.provider, assist.settings = configured.provider, configured.settings
        assist.policies = container.services["model_policies"]

        async def reads_assisted() -> None:
            async with container.services["uow_factory"]() as uow:
                await container.services["model_policies"].set(
                    uow,
                    tenant_identity(headers["X-Trellis-Tenant"]),
                    uses=["query_expansion"],
                    read_assist=True,
                )
                await uow.commit()

        client.portal.call(reads_assisted)
        try:
            for index, key in enumerate(["vk-first-test", "vk-second-test"], start=1):
                assert (
                    client.put(
                        "/v1/agents/model-key",
                        headers=headers,
                        json={"scope": scope, "virtual_key": key},
                    ).status_code
                    == 200
                )
                # a new question each time: the assisted read is not served from the cache
                asked = {**query, "query": f"{query['query']} {index}"}
                response = client.post("/v1/context", headers=headers, json=asked)
                assert response.status_code == 200, response.text
                assert gateway.route.call_count == index
                assert gateway.route.calls.last.request.headers["x-bf-vk"] == key
            assert (
                client.delete("/v1/agents/model-key?agent_id=research", headers=headers).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/context", headers=headers, json={**query, "query": "radio telescopes"}
                ).status_code
                == 200
            )
            assert gateway.route.call_count == 2
        finally:
            client.portal.call(configured.provider.close)


@pytest.mark.parametrize("registered", [False, True])
async def test_background_ingestion_uses_owner_key_and_never_hindsight_operator_budget(
    container, uow_factory, registered
):
    from tests.support_hindsight import preview_server

    credentials = service(uow_factory)
    if registered:
        await save(credentials, uow_factory, ALICE, "vk-job-test")
    async with preview_server(facts=[]) as server:
        with mocked_gateway([{"units": [{"start": 0, "end": 1}]}]) as gateway:
            provider = container.services["memory_provider"]
            provider.assist = gateway.assist(["contextual_extraction"])
            provider.assist.provider.credentials = credentials
            provider.contextual_extractor = server.extractor
            try:
                ack = await _observe(container, uow_factory, ALICE, MESSAGE)
                assert ack.observation_id and gateway.route.call_count == 1
                assert not server.requests
                headers = gateway.route.calls.last.request.headers
                assert headers["authorization"] == (
                    "Bearer vk-job-test" if registered else "Bearer vk-test"
                )
                assert current_model_identity() is None
            finally:
                await provider.assist.provider.close()


async def test_document_indexing_uses_recorded_agent_owner_after_key_rotation(
    container, uow_factory
):
    from tests.integration.test_graph import _ingest

    credentials = service(uow_factory)
    await save(credentials, uow_factory, ALICE, "vk-upload-test")
    doc_id = await _ingest(container, uow_factory, ALICE)
    async with uow_factory() as uow:
        document = await uow.documents.get(ALICE.tenant_id, doc_id)
        assert document.model_principal == ALICE.principal_id
    await save(credentials, uow_factory, ALICE, "vk-index-current-test")
    with mocked_gateway([{"summary": "The report describes annual revenue and costs."}]) as gateway:
        assist = gateway.assist(["summaries"])
        assist.provider.credentials = credentials
        indexer = container.services["indexer"]
        indexer.assist = assist
        try:
            # Simulate a job running under an unrelated worker's ambient context.
            with bound_to(BOB.tenant_id, BOB.principal_id):
                await indexer.index_document(ALICE.tenant_id, doc_id, force=True)
                assert current_model_identity() == agent_identity(BOB)
            assert gateway.route.call_count > 0
            assert {call.request.headers["x-bf-vk"] for call in gateway.route.calls} == {
                "vk-index-current-test"
            }
        finally:
            await assist.provider.close()


async def test_user_scoped_reflection_preserves_bound_agent_and_current_key(container, uow_factory):
    from memory_service.modules.memory.reflection import ReflectionService
    from tests.integration.test_reflection import _memories
    from tests.integration.test_reflection import _observe as observe

    credentials = service(uow_factory)
    await save(credentials, uow_factory, ALICE, "vk-reflect-test")
    await observe(container, uow_factory, ALICE, "I prefer concise answers with code samples.")
    await observe(container, uow_factory, ALICE, "I prefer bullet points over long paragraphs.")
    sources = await _memories(container, uow_factory, ALICE)
    assert len(sources) == 2
    reply = {
        "insights": [
            {
                "content": "User prefers concise answers and bullet points.",
                "memory_type": "PREFERENCE",
                "source_memory_ids": [m.memory_id for m in sources],
            }
        ]
    }
    with mocked_gateway([reply]) as gateway:
        assist = gateway.assist(["reflection"])
        assist.provider.credentials = credentials
        try:
            created = await ReflectionService(uow_factory, assist=assist).reflect_all()
            assert len(created) == 1
            assert gateway.route.calls.last.request.headers["x-bf-vk"] == "vk-reflect-test"
            async with uow_factory() as uow:
                insight = await uow.memories.get(ALICE.tenant_id, created[0])
                assert insight.owner_principal == ALICE.principal_id
            assert current_model_identity() is None
        finally:
            await assist.provider.close()


@pytest.mark.parametrize("status", [429, 503])
async def test_revocation_after_transient_failure_prevents_the_next_http_attempt(
    container, uow_factory, status
):
    import httpx

    credentials = service(uow_factory)
    await save(credentials, uow_factory, ALICE, "vk-before-backoff-test")
    with mocked_gateway() as gateway:
        assist = gateway.assist(["reflection"], max_retries=2)
        assist.provider.credentials = credentials

        async def revoke_after_first_send(request):
            await save(credentials, uow_factory, ALICE, None)
            return httpx.Response(status, text="transient failure")

        gateway.route.mock(side_effect=revoke_after_first_send)
        try:
            with bound_to(ALICE.tenant_id, ALICE.principal_id):
                assert (
                    await assist.complete("reflection", system="Instructions", user="source")
                    is None
                )
            assert gateway.route.call_count == 1
        finally:
            await assist.provider.close()


@pytest.mark.parametrize("new_key", [None, "vk-registered-during-call-test"])
async def test_operator_fallback_stops_when_agent_policy_changes_during_backoff(
    container, uow_factory, new_key
):
    import httpx

    credentials = service(uow_factory)
    assert await credentials.resolve(agent_identity(ALICE)) is None
    with mocked_gateway() as gateway:
        assist = gateway.assist(["reflection"], max_retries=2)
        assist.provider.credentials = credentials

        async def change_policy(request):
            await save(credentials, uow_factory, ALICE, new_key)
            return httpx.Response(503, text="temporary outage")

        gateway.route.mock(side_effect=change_policy)
        try:
            with bound_to(ALICE.tenant_id, ALICE.principal_id):
                assert (
                    await assist.complete("reflection", system="Instructions", user="source")
                    is None
                )
            assert gateway.route.call_count == 1
        finally:
            await assist.provider.close()


async def test_auto_wiring_uses_registered_agent_key_without_model_or_use_configuration(
    container, uow_factory
):
    import respx

    from memory_service.adapters.wiring import _wire_llm, _wire_memory
    from memory_service.config.settings import LLMSettings
    from tests.integration.test_memory import _memories
    from tests.support_llm import BASE
    from tests.unit.test_narrative_memory import UNITS

    container.settings.agent_credentials = AgentCredentialSettings(
        active_key_id="test", encryption_keys={"test": TEST_KEY}
    )
    container.settings.models.llm = LLMSettings(base_url=BASE)
    _wire_llm(container)
    _wire_memory(container)
    credentials = container.services["model_credentials"]
    await save(credentials, uow_factory, ALICE, "vk-auto-ingest")
    assert container.services["memory_provider"].contextual_extractor is None
    with respx.mock(assert_all_called=False) as gateway:
        catalog = gateway.get(f"{BASE}/models").respond(
            200, json={"data": [{"id": "openai/gpt-4.1-mini"}]}
        )
        import json

        completion = gateway.post(f"{BASE}/chat/completions").respond(
            200, json=chat_response(json.dumps(UNITS))
        )
        await _observe(container, uow_factory, ALICE, MESSAGE)
        assert catalog.call_count == completion.call_count == 1
        memories = await _memories(uow_factory, ALICE, container)
        assert any("She did not restart the database." in memory.content for memory in memories)
        assert completion.calls.last.request.headers["x-bf-vk"] == "vk-auto-ingest"
        # A different owner and then the revoked owner retain sources without a model call.
        await _observe(container, uow_factory, BOB, MESSAGE)
        await save(credentials, uow_factory, ALICE, None)
        await _observe(container, uow_factory, ALICE, MESSAGE + " A new incident followed.")
        assert catalog.call_count == completion.call_count == 1
