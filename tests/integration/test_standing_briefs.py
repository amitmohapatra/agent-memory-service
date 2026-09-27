"""Persistent native/assisted briefs: real SQL and jobs, mocked model transport."""

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.briefs import BriefSpec
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import NotFound, ProviderNotConfigured
from memory_service.modules.jobs.registry import register_handlers
from tests.integration.test_reflection import _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration
CTX = MemoryExecutionContext(tenant_id="acme", user_id="brief-owner", agent_id="research")


async def create(container, uow_factory, *, kind="mental_model", use_llm=False, ctx=CTX):
    register_handlers(container)
    async with uow_factory() as uow:
        brief = await container.services["briefs"].create(
            uow,
            ctx,
            BriefSpec(
                title="Response preferences",
                question="What response format do I prefer?",
                kind=kind,
                use_llm=use_llm,
            ),
        )
        await uow.commit()
    return brief


@pytest.mark.parametrize("kind", ["mental_model", "knowledge_page"])
async def test_native_refresh_read_isolation_delete_and_no_read_models(
    container, uow_factory, kind
):
    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    service = container.services["briefs"]
    brief = await create(container, uow_factory, kind=kind)
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "pending" and stored.output is None
    await container.tasks.drain()
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "ready" and not stored.output.generated
    assert "concise answers" in stored.output.text and stored.output.sources
    with mocked_gateway(failing=True) as gw:
        service.assist = gw.assist(["briefs"])
        for _ in range(5):
            assert (await service.read(CTX, brief.brief_id))[1] == "ready"
        assert gw.route.call_count == 0
        await service.assist.provider.close()
    for changes in [
        {"tenant_id": "rival"},
        {"user_id": "other"},
        {"agent_id": "sibling"},
        {"thread_id": "another-thread"},
        {"agent_run_id": "another-run"},
    ]:
        with pytest.raises(NotFound):
            await service.read(CTX.model_copy(update=changes), brief.brief_id)
    async with uow_factory() as uow:
        await uow.briefs.delete(CTX.tenant_id, brief.brief_id)
        await uow.commit()
    assert not await service.refresh(CTX.tenant_id, brief.brief_id)
    with pytest.raises(NotFound):
        await service.read(CTX, brief.brief_id)


async def test_source_changes_hide_old_output_until_refresh(container, uow_factory):
    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    brief = await create(container, uow_factory)
    await container.tasks.drain()
    service = container.services["briefs"]
    assert (await service.read(CTX, brief.brief_id))[1] == "ready"
    await _observe(container, uow_factory, CTX, "I prefer bullet points for project reports.")
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "stale" and stored.output is None
    assert await service.schedule_due(now=datetime.now(UTC) + timedelta(hours=2)) == 1
    # The due lease prevents a concurrent scheduler from enqueueing another copy.
    assert await service.schedule_due(now=datetime.now(UTC) + timedelta(hours=2)) == 0
    await container.tasks.drain()
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "ready" and "bullet points" in stored.output.text


async def test_disabled_assistance_is_explicit_and_native_mode_works(container, uow_factory):
    with pytest.raises(ProviderNotConfigured):
        await create(container, uow_factory, use_llm=True)
    brief = await create(container, uow_factory)
    await container.tasks.drain()
    assert (await container.services["briefs"].read(CTX, brief.brief_id))[1] == "ready"


async def test_assisted_refresh_uses_citations_and_never_calls_model_on_read(
    container, uow_factory
):
    import json

    import httpx

    from tests.support_llm import chat_response

    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    service = container.services["briefs"]
    with mocked_gateway() as gw:
        assist = gw.assist(["briefs"])
        service.assist = assist

        def answer(request):
            payload = json.loads(request.content)
            evidence = json.loads(payload["messages"][1]["content"])["evidence"]
            sid = evidence[0]["source_id"]
            return httpx.Response(
                200,
                json=chat_response(
                    json.dumps(
                        {"text": f"Concise answers with code samples [{sid}].", "source_ids": [sid]}
                    )
                ),
            )

        gw.route.mock(side_effect=answer)
        try:
            brief = await create(container, uow_factory, use_llm=True)
            await container.tasks.drain()
            stored, status = await service.read(CTX, brief.brief_id)
            assert status == "ready" and stored.output.generated
            assert gw.route.call_count == 1
            assert (await service.read(CTX, brief.brief_id))[1] == "ready"
            assert gw.route.call_count == 1
        finally:
            await assist.provider.close()


def test_briefs_http_idempotency_owner_scoping_and_no_raw_context(client):
    headers = {
        "X-API-Key": "test-key",
        "X-Memory-Tenant": "brief-http",
        "X-Memory-User": "owner",
        "Idempotency-Key": "standing-question",
    }
    body = {
        "scope": {"agent_id": "research"},
        "spec": {"title": "Preferences", "question": "What format do I prefer?"},
    }
    first = client.post("/v1/briefs", headers=headers, json=body)
    assert first.status_code == 202, first.text
    assert "context" not in first.json()
    again = client.post("/v1/briefs", headers=headers, json=body)
    assert first.json() == again.json()
    bid = first.json()["brief_id"]
    read = client.get(f"/v1/briefs/{bid}", headers=headers, params={"agent_id": "research"})
    assert read.status_code == 200
    denied = client.get(
        f"/v1/briefs/{bid}",
        headers={**headers, "X-Memory-User": "other"},
        params={"agent_id": "research"},
    )
    assert denied.status_code == 404
    listing = client.get("/v1/briefs", headers=headers, params={"agent_id": "research"})
    assert [row["brief_id"] for row in listing.json()] == [bid]
    assert (
        client.delete(
            f"/v1/briefs/{bid}", headers=headers, params={"agent_id": "research"}
        ).status_code
        == 200
    )


async def test_forgetting_hides_copied_source_before_index_cleanup(container, uow_factory):
    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    brief = await create(container, uow_factory)
    await container.tasks.drain()
    service = container.services["briefs"]
    ready, _ = await service.read(CTX, brief.brief_id)
    source = ready.output.sources[0].item_id
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, CTX, source)
        await uow.commit()
    # The projection-removal job has deliberately not run yet.
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "stale" and stored.output is None
    assert await service.refresh(CTX.tenant_id, brief.brief_id)
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "ready" and stored.output.text == ""


async def test_memory_expiry_caps_brief_validity_without_waiting_for_expiry_job(
    container, uow_factory, monkeypatch
):
    from sqlalchemy import text

    from memory_service.modules.briefs import service as module

    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    expires = datetime.now(UTC) + timedelta(minutes=2)
    async with container.database.engine.begin() as conn:
        await conn.execute(text("UPDATE memories SET expires_at=:expires"), {"expires": expires})
    brief = await create(container, uow_factory)
    await container.tasks.drain()
    service = container.services["briefs"]
    stored, _ = await service.read(CTX, brief.brief_id)
    assert stored.output.valid_until == expires

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return expires + timedelta(seconds=1)

    monkeypatch.setattr(module, "datetime", Later)
    stored, status = await service.read(CTX, brief.brief_id)
    assert status == "stale" and stored.output is None


async def test_definition_changed_during_synthesis_cannot_restore_old_content(
    container, uow_factory
):
    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    service = container.services["briefs"]
    with mocked_gateway() as gw:
        assist = gw.assist(["briefs"])
        service.assist = assist
        try:
            brief = await create(container, uow_factory, use_llm=True)

            async def replace_while_synthesizing(old, sources):
                async with uow_factory() as uow:
                    await service.update(
                        uow,
                        CTX,
                        brief.brief_id,
                        BriefSpec(title="New question", question="What projects are active?"),
                    )
                    await uow.commit()
                return "Obsolete answer that must not be stored"

            service._synthesize = replace_while_synthesizing
            assert not await service.refresh(CTX.tenant_id, brief.brief_id)
            stored, status = await service.read(CTX, brief.brief_id)
            assert stored.generation == 2 and status == "pending" and stored.output is None
        finally:
            await assist.provider.close()


async def test_unknown_model_citations_are_rejected_without_native_fallback(container, uow_factory):
    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    service = container.services["briefs"]
    with mocked_gateway(
        [{"text": "A fabricated answer [mem_unknown].", "source_ids": ["mem_unknown"]}]
    ) as gw:
        assist = gw.assist(["briefs"])
        service.assist = assist
        try:
            brief = await create(container, uow_factory, use_llm=True)
            with pytest.raises(ProviderNotConfigured):
                await service.refresh(CTX.tenant_id, brief.brief_id)
            stored, status = await service.read(CTX, brief.brief_id)
            assert status == "pending" and stored.output is None
        finally:
            await assist.provider.close()


def test_session_bound_brief_round_trip(client):
    headers = {
        "X-API-Key": "test-key",
        "X-Memory-Tenant": "brief-session",
        "X-Memory-User": "owner",
    }
    scope = {"thread_id": "thread-1", "session_id": "session-1", "agent_id": "research"}
    created = client.post(
        "/v1/briefs",
        headers=headers,
        json={
            "scope": scope,
            "spec": {"title": "Session goals", "question": "What are the goals?"},
        },
    )
    assert created.status_code == 202, created.text
    bid = created.json()["brief_id"]
    assert client.get(f"/v1/briefs/{bid}", headers=headers, params=scope).status_code == 200
    assert (
        client.get(
            f"/v1/briefs/{bid}", headers=headers, params={**scope, "session_id": "different"}
        ).status_code
        == 404
    )
    assert client.get("/v1/briefs", headers=headers, params=scope).json()[0]["brief_id"] == bid
    assert client.delete(f"/v1/briefs/{bid}", headers=headers, params=scope).status_code == 200


async def test_brief_uses_earlier_temporal_validity_than_expiry(container, uow_factory):
    from sqlalchemy import text

    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    until = datetime.now(UTC) + timedelta(minutes=2)
    async with container.database.engine.begin() as conn:
        await conn.execute(
            text("UPDATE memories SET valid_to=:until, expires_at=:expires"),
            {"until": until, "expires": until + timedelta(hours=1)},
        )
    brief = await create(container, uow_factory)
    await container.tasks.drain()
    stored, _ = await container.services["briefs"].read(CTX, brief.brief_id)
    assert stored.output.valid_until == until


async def test_document_source_rebuild_does_not_repeat_paid_synthesis(
    container, uow_factory, monkeypatch
):
    import json

    import httpx

    from tests.integration.test_graph import _ingest
    from tests.support_llm import chat_response

    await _ingest(container, uow_factory, CTX)
    service = container.services["briefs"]
    with mocked_gateway() as gw:
        assist = gw.assist(["briefs"])
        service.assist = assist

        def answer(request):
            evidence = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
            sid = evidence[0]["source_id"]
            return httpx.Response(
                200,
                json=chat_response(
                    json.dumps({"text": f"Annual results [{sid}].", "source_ids": [sid]})
                ),
            )

        gw.route.mock(side_effect=answer)
        try:
            async with uow_factory() as uow:
                brief = await service.create(
                    uow,
                    CTX,
                    BriefSpec(
                        title="Results", question="What was Adjusted EBITDA in FY26?", use_llm=True
                    ),
                )
                await uow.commit()
            await container.tasks.drain()
            stored, status = await service.read(CTX, brief.brief_id)
            assert status == "ready" and stored.output.sources[0].document_id
            assert gw.route.call_count == 1
            original = service.builder.build

            async def rebuilt(*args, **kwargs):
                bundle = await original(*args, **kwargs)
                # Match the real fresh document path: timing and score metadata may
                # change while source identity and synthesis input remain identical.
                return bundle.model_copy(
                    update={
                        "knowledge": [
                            item.model_copy(
                                update={
                                    "score": item.score + 0.01,
                                    "evidence": [
                                        ref.model_copy(
                                            update={
                                                "observed_at": datetime.now(UTC) + timedelta(days=1)
                                            }
                                        )
                                        for ref in item.evidence
                                    ],
                                }
                            )
                            for item in bundle.knowledge
                        ]
                    }
                )

            monkeypatch.setattr(service.builder, "build", rebuilt)
            assert await service.refresh(CTX.tenant_id, brief.brief_id)
            assert gw.route.call_count == 1
        finally:
            await assist.provider.close()


@pytest.mark.parametrize("changed", ["model", "prompt"])
async def test_model_or_prompt_changes_rebuild_synthesis_with_unchanged_sources(
    container, uow_factory, monkeypatch, changed
):
    import json

    import httpx

    from memory_service.modules.briefs import service as module
    from tests.support_llm import chat_response

    await _observe(container, uow_factory, CTX, "I prefer concise answers with code samples.")
    service = container.services["briefs"]
    with mocked_gateway() as gw:
        first = gw.assist(["briefs"])
        second = gw.assist(["briefs"], model="test/new-strong")
        service.assist = first

        def answer(request):
            evidence = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
            sid = evidence[0]["source_id"]
            return httpx.Response(
                200,
                json=chat_response(
                    json.dumps({"text": f"Concise answers [{sid}].", "source_ids": [sid]})
                ),
            )

        gw.route.mock(side_effect=answer)
        try:
            brief = await create(container, uow_factory, use_llm=True)
            await container.tasks.drain()
            before, _ = await service.read(CTX, brief.brief_id)
            assert before.output.generation_profile
            assert await service.refresh(CTX.tenant_id, brief.brief_id)
            assert gw.route.call_count == 1  # unchanged inputs reuse the synthesis
            if changed == "model":
                service.assist = second
            else:
                monkeypatch.setattr(module, "_SYSTEM", module._SYSTEM + " Prefer short paragraphs.")
            assert await service.refresh(CTX.tenant_id, brief.brief_id)
            after, status = await service.read(CTX, brief.brief_id)
            assert status == "ready" and gw.route.call_count == 2
            assert after.output.generation_profile != before.output.generation_profile
        finally:
            await first.provider.close()
            await second.provider.close()
