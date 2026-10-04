import json
import re
from datetime import UTC, datetime

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import TracerProvider
from pydantic import ValidationError

from trellis.memory import (
    AuthorizationError,
    DependencyUnavailableError,
    DocumentsAPI,
    MemoryClient,
    MemoryContext,
    MemoryError,
    NotFoundError,
    TimeoutError,
    current_context,
)
from trellis.memory import transport as transport_module


@pytest.fixture
def client() -> MemoryClient:
    return MemoryClient("http://memory.test", api_key="k", max_retries=2)


def test_bind_builds_scope_and_child_contexts(client: MemoryClient) -> None:
    ctx = client.bind(
        tenant_id="acme", user_id="u1", thread_id="thr_1", session_id="ses_1", turn_id="trn_1"
    )
    assert isinstance(ctx, MemoryContext)
    child = ctx.agent("research", agent_run_id="run_1")
    assert child.scope.agent_id == "research" and child.scope.thread_id == "thr_1"
    assert child.scope.parent_agent_run_id is None
    grandchild = child.agent("writer")
    assert grandchild.scope.parent_agent_run_id == "run_1"
    assert grandchild.scope.agent_run_id and grandchild.scope.agent_run_id != "run_1"


@respx.mock
async def test_agent_key_methods_use_existing_scope_and_return_only_metadata(client):
    status = {"registered": True, "revoked": False, "revision": 1}
    put = respx.put("http://memory.test/v1/agents/model-key").respond(200, json=status)
    get = respx.get("http://memory.test/v1/agents/model-key").respond(200, json=status)
    delete = respx.delete("http://memory.test/v1/agents/model-key").respond(
        200, json={**status, "revoked": True, "revision": 2}
    )
    ctx = client.bind(tenant_id="acme", user_id="alice").agent("research")
    assert (
        await ctx.advanced.model_keys.set("vk-sdk-test", idempotency_key="rotate-1")
    ).revision == 1
    assert put.calls.last.request.headers["Idempotency-Key"] == "rotate-1"
    assert (await ctx.advanced.model_keys.status()).registered
    assert get.calls.last.request.url.params["agent_id"] == "research"
    assert (await ctx.advanced.model_keys.revoke()).revoked
    assert delete.calls.last.request.url.params["agent_id"] == "research"


async def test_context_manager_propagates_via_contextvars(client: MemoryClient) -> None:
    ctx = client.bind(tenant_id="acme")
    assert current_context() is None
    async with ctx:
        assert current_context() is ctx
    assert current_context() is None


_ACK = {
    "message_id": "msg_1",
    "thread_id": "thr_1",
    "session_id": "ses_1",
    "turn_id": "trn_1",
    "sequence": 1,
    "job_ids": ["job_1"],
}


@respx.mock
async def test_history_add_sends_scope_headers_and_idempotency_key(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/messages").mock(
        return_value=httpx.Response(202, json={"messages": [_ACK]})
    )
    ctx = client.bind(
        tenant_id="acme",
        user_id="u1",
        thread_id="thr_1",
        session_id="ses_1",
        turn_id="trn_1",
    )
    [ack] = await ctx.history.add([("USER", "hello")])
    assert ack.message_id == "msg_1" and ack.job_ids == ["job_1"]
    req = route.calls.last.request
    assert req.headers["X-Trellis-Tenant"] == "acme"
    assert req.headers["X-Trellis-User"] == "u1"
    assert req.headers["X-API-Key"] == "k"
    assert req.headers["Idempotency-Key"].startswith("msgs-")
    # same content + lineage => same key (safe retries)
    await ctx.history.add([("USER", "hello")])
    assert (
        route.calls[0].request.headers["Idempotency-Key"]
        == route.calls[1].request.headers["Idempotency-Key"]
    )
    await ctx.history.add([("USER", "different")])
    assert (
        route.calls[2].request.headers["Idempotency-Key"]
        != route.calls[0].request.headers["Idempotency-Key"]
    )


@respx.mock
async def test_a_0_1_error_envelope_still_maps_to_the_typed_exception(client: MemoryClient) -> None:
    respx.post("http://memory.test/v1/context").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "code": "SCOPE_DENIED",
                    "message": "denied",
                    "retryable": False,
                    "trace_id": "t1",
                }
            },
        )
    )
    ctx = client.bind(tenant_id="acme")
    with pytest.raises(AuthorizationError) as exc:
        await ctx.context("q")
    assert exc.value.code == "SCOPE_DENIED" and exc.value.trace_id == "t1"


@respx.mock
async def test_retryable_error_is_retried_for_idempotent_writes(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/memories").mock(
        side_effect=[
            httpx.Response(
                503,
                json={
                    "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "db", "retryable": True}
                },
            ),
            httpx.Response(201, json={"memory_id": "mem_1", "deduplicated": False}),
        ]
    )
    ctx = client.bind(tenant_id="acme")
    ack = await ctx.remember("something is true")
    assert ack.memory_id == "mem_1"
    assert route.call_count == 2


@respx.mock
async def test_retries_exhausted_raise(client: MemoryClient) -> None:
    respx.get("http://memory.test/v1/jobs/job_1").mock(
        return_value=httpx.Response(
            503,
            json={"error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "db", "retryable": True}},
        )
    )
    ctx = client.bind(tenant_id="acme")
    with pytest.raises(DependencyUnavailableError):
        await ctx.advanced.job("job_1")


@respx.mock
async def test_context_bundle_parses_and_flags_insufficient(client: MemoryClient) -> None:
    respx.post("http://memory.test/v1/context").mock(
        return_value=httpx.Response(
            200,
            json={
                "bundle_id": "b1",
                "evidence_status": "INSUFFICIENT",
                "token_estimate": 0,
                "missing_evidence": ["PAGE11"],
            },
        )
    )
    ctx = client.bind(tenant_id="acme")
    bundle = await ctx.context("q", format="full")
    assert bundle.insufficient and bundle.missing_evidence == ["PAGE11"]
    assert bundle.memories == [] and bundle.conversation is None


@respx.mock
async def test_administer_names_the_tenant_and_a_keyed_bind_sends_no_tenant_header() -> None:
    seen: list[str | None] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("X-Trellis-Tenant"))
        return httpx.Response(200, json=[] if request.method == "GET" else {"results": []})

    respx.get("http://memory.test/v1/keys").mock(side_effect=capture)
    respx.post("http://memory.test/v1/recall").mock(side_effect=capture)
    client = MemoryClient("http://memory.test", api_key="mk_k.s")
    await client.administer("globex").keys.list()
    await client.tenant.keys.list()
    await client.bind(user_id="u1").search("anything")
    assert seen == ["globex", None, None]


TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


@respx.mock
async def test_a_problem_maps_to_the_typed_exception(client: MemoryClient) -> None:
    respx.post("http://memory.test/v1/context").mock(
        return_value=httpx.Response(
            403,
            headers={"content-type": "application/problem+json"},
            json={
                "type": "urn:trellis:problem:scope-denied",
                "title": "Outside the caller's scope",
                "status": 403,
                "detail": "denied",
                "instance": "/v1/context",
                "code": "SCOPE_DENIED",
                "retryable": False,
                "trace_id": TRACE,
                "request_id": "req_1",
                "details": {"object": "thread:t1"},
            },
        )
    )
    with pytest.raises(AuthorizationError) as exc:
        await client.bind(tenant_id="acme").context("q")
    err = exc.value
    assert (err.code, err.status, err.message) == ("SCOPE_DENIED", 403, "denied")
    assert (err.trace_id, err.request_id, err.details) == (TRACE, "req_1", {"object": "thread:t1"})
    assert err.retryable is False


@respx.mock
async def test_every_call_carries_a_request_id_kept_across_its_retries(
    client: MemoryClient,
) -> None:
    route = respx.get("http://memory.test/v1/jobs/job_1").mock(
        side_effect=[
            httpx.Response(503, json={"code": "DEPENDENCY_UNAVAILABLE", "retryable": True}),
            httpx.Response(200, json={"job_id": "job_1", "status": "SUCCEEDED"}),
        ]
    )
    await client.bind(tenant_id="acme").advanced.job("job_1")
    ids = [call.request.headers["X-Request-ID"] for call in route.calls]
    assert len(ids) == 2 and ids[0] == ids[1] and re.fullmatch(r"[0-9a-f]{32}", ids[0])
    version = respx.get("http://memory.test/version").mock(
        return_value=httpx.Response(200, json={"version": "0.2.0"})
    )
    await client.transport.request("GET", "/version", headers={"X-Request-ID": "mine"})
    assert version.calls.last.request.headers["X-Request-ID"] == "mine"


@respx.mock
async def test_traceparent_is_built_from_a_w3c_scope_trace_id(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/recall").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await client.bind(tenant_id="acme", trace_id=TRACE.upper()).search("q")
    headers = route.calls.last.request.headers
    match = TRACEPARENT.match(headers["traceparent"])
    assert match and match.group(1) == TRACE and match.group(3) == "01"
    assert "X-Trace-ID" not in headers  # a response header; traceparent is the request's
    # an opaque id is not a trace the service would continue: it travels as the correlation id
    await client.bind(tenant_id="acme", trace_id="opaque-id").search("q")
    headers = route.calls.last.request.headers
    assert "traceparent" not in headers and "X-Trace-ID" not in headers
    assert headers["X-Correlation-ID"] == "opaque-id"
    await client.bind(tenant_id="acme", trace_id="opaque-id", correlation_id="corr-1").search("q")
    assert route.calls.last.request.headers["X-Correlation-ID"] == "corr-1"
    await client.bind(tenant_id="acme").search("q")
    assert "traceparent" not in route.calls.last.request.headers


@respx.mock
async def test_the_active_opentelemetry_span_wins_over_the_scope(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/recall").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    tracer = TracerProvider().get_tracer("agent")
    with tracer.start_as_current_span("turn") as span:
        await client.bind(tenant_id="acme", trace_id=TRACE).search("q")
    context = span.get_span_context()
    match = TRACEPARENT.match(route.calls.last.request.headers["traceparent"])
    assert match and match.group(1) == format(context.trace_id, "032x") != TRACE
    assert match.group(2) == format(context.span_id, "016x")


@respx.mock
async def test_without_the_otel_extra_the_scope_trace_id_is_used(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transport_module, "_PROPAGATOR", None)
    route = respx.post("http://memory.test/v1/recall").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    tracer = TracerProvider().get_tracer("agent")
    with tracer.start_as_current_span("turn"):
        await client.bind(tenant_id="acme", trace_id=TRACE).search("q")
    match = TRACEPARENT.match(route.calls.last.request.headers["traceparent"])
    assert match and match.group(1) == TRACE


@respx.mock
async def test_everything_past_the_verbs_is_under_advanced(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/documents").mock(
        return_value=httpx.Response(
            202,
            json={"document_id": "doc_1", "filename": "a.md", "checksum": "c", "size_bytes": 5},
        )
    )
    ctx = client.bind(tenant_id="acme", user_id="u1")
    assert isinstance(ctx.advanced.documents, DocumentsAPI)
    assert not hasattr(ctx, "documents") and not hasattr(ctx, "files")
    assert ctx.advanced.tenant is client.tenant and ctx.advanced.admin is client.admin
    assert not hasattr(ctx.advanced, "webhooks") and not hasattr(ctx.advanced, "briefs")
    handle = await ctx.advanced.documents.add(b"hello", filename="a.md", media_type="text/markdown")
    assert handle.document_id == "doc_1" and route.called


@respx.mock
async def test_tool_records_go_to_the_invocations_route(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/tools/invocations").mock(
        return_value=httpx.Response(
            202, json={"invocation_id": "tiv_1", "step": 0, "args_hash": "h", "recorded": True}
        )
    )
    result = await client.bind(tenant_id="acme", agent_id="bot").record_tool("search", {"q": "x"})
    assert result.invocation_id == "tiv_1" and route.called


@respx.mock
@pytest.mark.parametrize(
    ("response", "cls", "code", "retryable"),
    [
        (
            httpx.Response(
                502, text="<html>Bad Gateway</html>", headers={"content-type": "text/html"}
            ),
            DependencyUnavailableError,
            "DEPENDENCY_UNAVAILABLE",
            True,
        ),
        (httpx.Response(500, json=["not", "a", "problem"]), MemoryError, "INTERNAL", False),
        (httpx.Response(500, json={"error": "boom"}), MemoryError, "INTERNAL", False),
        (httpx.Response(404, json={"unexpected": "shape"}), NotFoundError, "NOT_FOUND", False),
    ],
)
async def test_an_unrecognised_error_body_is_classed_by_its_status(
    client: MemoryClient,
    response: httpx.Response,
    cls: type[MemoryError],
    code: str,
    retryable: bool,
) -> None:
    respx.get("http://memory.test/v1/jobs/job_1").mock(return_value=response)
    with pytest.raises(cls) as exc:
        await client.bind(tenant_id="acme").advanced.job("job_1")
    assert type(exc.value) is cls
    assert exc.value.code == code and exc.value.status == response.status_code
    assert exc.value.retryable is retryable
    assert exc.value.message == f"HTTP {response.status_code}"


@respx.mock
async def test_a_plain_rfc_9457_problem_keeps_its_words(client: MemoryClient) -> None:
    """A gateway in front of the service answers problems without the ``code`` extension."""
    respx.get("http://memory.test/v1/jobs/job_1").mock(
        return_value=httpx.Response(
            503,
            headers={"content-type": "application/problem+json"},
            json={"type": "about:blank", "title": "Service Unavailable", "status": 503},
        )
    )
    with pytest.raises(MemoryError) as exc:
        await client.bind(tenant_id="acme").advanced.job("job_1")
    assert isinstance(exc.value, DependencyUnavailableError)
    assert exc.value.code == "DEPENDENCY_UNAVAILABLE"
    assert exc.value.message == "Service Unavailable"
    assert exc.value.retryable is True  # from the status, so the read was retried


@respx.mock
@pytest.mark.parametrize("trace_id", ["0" * 32, "a" * 31, "a" * 33])
async def test_a_trace_id_the_service_would_reject_travels_as_correlation(
    client: MemoryClient, trace_id: str
) -> None:
    route = respx.post("http://memory.test/v1/recall").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await client.bind(tenant_id="acme", trace_id=trace_id).search("q")
    headers = route.calls.last.request.headers
    assert "traceparent" not in headers and "X-Trace-ID" not in headers
    assert headers["X-Correlation-ID"] == trace_id


def test_w3c_trace_id_rejects_what_the_service_rejects() -> None:
    assert transport_module.w3c_trace_id(TRACE.upper()) == TRACE
    assert transport_module.w3c_trace_id(TRACE + "\n") is None
    assert transport_module.w3c_trace_id("0" * 32) is None
    assert transport_module.w3c_trace_id(None) is None


@respx.mock
async def test_a_request_id_given_in_any_case_is_kept_alone(client: MemoryClient) -> None:
    route = respx.get("http://memory.test/version").mock(
        return_value=httpx.Response(200, json={"version": "0.2.0"})
    )
    await client.transport.request("GET", "/version", headers={"x-request-id": "mine"})
    sent = route.calls.last.request.headers
    assert sent.get_list("x-request-id") == ["mine"]


@respx.mock
async def test_a_gateway_body_keeps_its_message_and_request_id(client: MemoryClient) -> None:
    respx.get("http://memory.test/v1/jobs/job_1").mock(
        return_value=httpx.Response(403, json={"message": "Forbidden", "request_id": "gw-1"})
    )
    with pytest.raises(MemoryError) as exc:
        await client.bind(tenant_id="acme").advanced.job("job_1")
    assert exc.value.message == "Forbidden" and exc.value.request_id == "gw-1"
    assert isinstance(exc.value, AuthorizationError)
    assert exc.value.code == "AUTHORIZATION" and exc.value.status == 403


@respx.mock
async def test_a_caller_s_traceparent_is_sent_once_whatever_its_case(client: MemoryClient) -> None:
    from trellis.memory.models import Scope

    route = respx.get("http://memory.test/version").mock(return_value=httpx.Response(200, json={}))
    mine = f"00-{'b' * 32}-{'c' * 16}-01"
    await client.transport.request(
        "GET", "/version", scope=Scope(trace_id=TRACE), headers={"Traceparent": mine}
    )
    assert route.calls.last.request.headers.get_list("traceparent") == [mine]


@respx.mock
async def test_a_gateway_s_untyped_members_are_not_trusted(client: MemoryClient) -> None:
    route = respx.get("http://memory.test/v1/jobs/job_1").mock(
        return_value=httpx.Response(
            503,
            json={"message": "down", "retryable": "false", "details": ["x"], "request_id": 7},
        )
    )
    with pytest.raises(MemoryError) as exc:
        await client.bind(tenant_id="acme").advanced.job("job_1")
    # the status says retryable, the string does not count either way: the read was retried
    assert exc.value.retryable is True and route.call_count == 3
    assert exc.value.details == {} and exc.value.request_id is None
    respx.get("http://memory.test/v1/jobs/job_2").mock(
        return_value=httpx.Response(403, json={"message": "no", "retryable": "true"})
    )
    with pytest.raises(MemoryError) as exc:
        await client.bind(tenant_id="acme").advanced.job("job_2")
    assert exc.value.retryable is False


@respx.mock
async def test_the_body_scope_carries_no_trace_id(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/recall").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await client.bind(tenant_id="acme", trace_id=TRACE, correlation_id="corr-1").search("q")
    import json

    sent = json.loads(route.calls.last.request.content)
    assert "trace_id" not in sent["scope"] and sent["scope"]["correlation_id"] == "corr-1"


def test_every_documented_exception_is_exported() -> None:
    assert issubclass(TimeoutError, MemoryError)


@respx.mock
async def test_a_timeout_on_a_non_idempotent_write_is_not_retried(client: MemoryClient) -> None:
    """A read or write timeout means the service may have the request: a retry would duplicate
    a write that carries no idempotency key."""
    route = respx.post("http://memory.test/v1/admin/tenants").mock(
        side_effect=httpx.ReadTimeout("slow")
    )
    with pytest.raises(TimeoutError) as exc:
        await client.admin.create_tenant("Acme", tenant_id="acme")
    assert exc.value.code == "TIMEOUT" and exc.value.status == 0 and exc.value.retryable
    assert route.call_count == 1


@respx.mock
async def test_a_connection_that_never_opened_is_retried(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/admin/tenants").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(DependencyUnavailableError) as exc:
        await client.admin.create_tenant("Acme", tenant_id="acme")
    assert exc.value.code == "DEPENDENCY_UNAVAILABLE" and route.call_count == 3


@respx.mock
async def test_a_timeout_on_a_read_is_retried(client: MemoryClient) -> None:
    route = respx.get("http://memory.test/v1/jobs/job_1").mock(
        side_effect=httpx.ReadTimeout("slow")
    )
    with pytest.raises(TimeoutError):
        await client.bind(tenant_id="acme").advanced.job("job_1")
    assert route.call_count == 3


@respx.mock
async def test_the_upload_form_scope_carries_no_trace_id(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/documents").mock(
        return_value=httpx.Response(
            202,
            json={"document_id": "doc_1", "filename": "a.md", "checksum": "c", "size_bytes": 5},
        )
    )
    ctx = client.bind(tenant_id="acme", user_id="u1", trace_id=TRACE, correlation_id="corr-1")
    await ctx.advanced.documents.add(b"hello", filename="a.md", media_type="text/markdown")
    body = route.calls.last.request.content.decode(errors="replace")
    assert "corr-1" in body and TRACE not in body.split("traceparent")[0]
    assert '"trace_id"' not in body
    assert route.calls.last.request.headers["traceparent"].startswith(f"00-{TRACE}-")


def test_an_id_that_is_not_an_id_is_refused_when_the_scope_is_built(client: MemoryClient) -> None:
    """Header values must be ASCII and the service refuses anything outside its id grammar;
    the SDK says so at bind time instead of failing inside the HTTP client."""
    with pytest.raises(ValidationError, match="not an id"):
        client.bind(tenant_id="acme", trace_id="turn-\u00e9")
    with pytest.raises(ValidationError, match="not an id"):
        client.bind(tenant_id="acme", user_id="-starts-with-a-dash")
    assert client.bind(tenant_id="acme", correlation_id="turn-42").scope.correlation_id == "turn-42"


@respx.mock
async def test_graph_entity_search_and_profile(client: MemoryClient) -> None:
    entity = {
        "entity_id": "ent_acme",
        "name": "Acme",
        "canonical_name": "acme",
        "entity_type": "ORG",
        "mention_count": 3,
        "summary": "Acme (ORG): operates in Germany",
    }
    search = respx.get("http://memory.test/v1/graph/entities").respond(
        200, json={"entities": [entity]}
    )
    fact = {
        "relation_id": "rel_1",
        "subject": "Acme",
        "predicate": "operates_in",
        "object": "Germany",
        "status": "SUPERSEDED",
        "layer": "entity",
        "observed_at": "2026-09-15T00:00:00Z",
    }
    respx.get("http://memory.test/v1/graph/entities/ent_acme").respond(
        200,
        json={
            "entity": entity,
            "current": [
                {
                    "predicate": "operates_in",
                    "value": "France",
                    "relation_id": "rel_2",
                    "observed_at": "2026-09-16T00:00:00Z",
                }
            ],
            "relations": [],
            "history": [fact],
            "evidence": [],
        },
    )
    ctx = client.bind(tenant_id="acme", user_id="u1")
    [found] = await ctx.advanced.graph.entities("Ac", entity_type="ORG", limit=5)
    assert found.summary.startswith("Acme") and found.entity_id == "ent_acme"
    assert dict(search.calls.last.request.url.params) == {"q": "Ac", "type": "ORG", "limit": "5"}
    profile = await ctx.advanced.graph.entity("ent_acme")
    assert profile.current[0].value == "France" and profile.history[0].status == "SUPERSEDED"


@respx.mock
async def test_an_entity_s_traversal_sends_depth_layers_and_knowledge_time(
    client: MemoryClient,
) -> None:
    route = respx.get("http://memory.test/v1/graph/entities/ent_1").respond(
        200,
        json={
            "entity": {"entity_id": "ent_1", "name": "Acme", "canonical_name": "acme"},
            "neighborhood": {"entities": [], "facts": [], "visited": 3},
        },
    )
    ctx = client.bind(tenant_id="acme", user_id="u1")
    profile = await ctx.advanced.graph.entity(
        "ent_1", depth=2, layers=["causal"], valid_at=datetime(2026, 9, 1, tzinfo=UTC)
    )
    params = route.calls.last.request.url.params
    assert params["depth"] == "2" and params.get_list("layers") == ["causal"]
    assert params["valid_at"].startswith("2026-09-01")
    assert profile.neighborhood is not None and profile.neighborhood.visited == 3


@respx.mock
async def test_remember_states_a_memory_and_update_supersedes_it(client: MemoryClient) -> None:
    remember = respx.post("http://memory.test/v1/memories").respond(
        201, json={"memory_id": "mem_1", "deduplicated": False, "job_ids": ["obx_1"]}
    )
    update = respx.post("http://memory.test/v1/memories/mem_1/supersede").respond(
        200, json={"memory_id": "mem_2", "supersedes": "mem_1"}
    )
    ctx = client.bind(tenant_id="acme", user_id="u1", thread_id="thr_1")
    ack = await ctx.remember(
        "Prefers metric units.",
        memory_type="PREFERENCE",
        visibility="USER",
        entities=["metric system"],
        source="settings-page",
    )
    assert ack.memory_id == "mem_1" and not ack.deduplicated
    body = json.loads(remember.calls.last.request.content)
    assert body["content"] == "Prefers metric units." and body["memory_type"] == "PREFERENCE"
    assert body["visibility"] == "USER" and body["entities"] == ["metric system"]
    assert body["custom_metadata"] == {"source": "settings-page"}
    assert "hints" not in body, "a statement is not an observation with hints"
    first_key = remember.calls.last.request.headers["Idempotency-Key"]
    await ctx.remember("Prefers metric units.", memory_type="PREFERENCE", visibility="USER")
    assert remember.calls.last.request.headers["Idempotency-Key"] == first_key

    moved = await ctx.update("mem_1", "Prefers imperial units.", reason="corrected")
    assert (moved.memory_id, moved.supersedes) == ("mem_2", "mem_1")
    sent = json.loads(update.calls.last.request.content)
    assert sent["content"] == "Prefers imperial units." and sent["reason"] == "corrected"
    assert sent["scope"]["thread_id"] == "thr_1"


@respx.mock
async def test_a_message_without_a_turn_is_not_deduplicated_by_content(
    client: MemoryClient,
) -> None:
    route = respx.post("http://memory.test/v1/messages").respond(
        202,
        json={
            "messages": [
                {
                    "message_id": "msg_1",
                    "thread_id": "thr_1",
                    "session_id": "ses_derived",
                    "turn_id": "trn_derived",
                    "sequence": 1,
                }
            ]
        },
    )
    loose = client.bind(tenant_id="acme", user_id="u1", thread_id="thr_1")
    [ack] = await loose.history.add([("USER", "ok")])
    assert ack.turn_id == "trn_derived", "the service's derived ids come back"
    await loose.history.add([("USER", "ok")])
    keys = [call.request.headers["Idempotency-Key"] for call in route.calls]
    assert keys[0] != keys[1], "two identical messages without a turn are two messages"
    assert "session_id" not in json.loads(route.calls.last.request.content)["scope"]

    turned = client.bind(tenant_id="acme", user_id="u1", thread_id="thr_1", turn_id="trn_1")
    await turned.history.add([("USER", "ok")])
    await turned.history.add([("USER", "ok")])
    assert (
        route.calls[-1].request.headers["Idempotency-Key"]
        == (route.calls[-2].request.headers["Idempotency-Key"])
    ), "with a turn, lineage + content is the message's identity"
