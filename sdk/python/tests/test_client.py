import httpx
import pytest
import respx

from universal_memory import (
    AuthorizationError,
    DependencyUnavailableError,
    MemoryClient,
    MemoryContext,
    current_context,
)


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


async def test_context_manager_propagates_via_contextvars(client: MemoryClient) -> None:
    ctx = client.bind(tenant_id="acme")
    assert current_context() is None
    async with ctx:
        assert current_context() is ctx
    assert current_context() is None


@respx.mock
async def test_chat_user_sends_scope_headers_and_idempotency_key(client: MemoryClient) -> None:
    route = respx.post("http://memory.test/v1/messages").mock(
        return_value=httpx.Response(
            202,
            json={
                "message_id": "msg_1",
                "thread_id": "thr_1",
                "session_id": "ses_1",
                "turn_id": "trn_1",
                "sequence": 1,
                "job_ids": ["job_1"],
            },
        )
    )
    ctx = client.bind(
        tenant_id="acme",
        user_id="u1",
        group_ids=["g1"],
        thread_id="thr_1",
        session_id="ses_1",
        turn_id="trn_1",
    )
    ack = await ctx.chat.user("hello")
    assert ack.message_id == "msg_1" and ack.job_ids == ["job_1"]
    req = route.calls.last.request
    assert req.headers["X-Memory-Tenant"] == "acme"
    assert req.headers["X-Memory-User"] == "u1"
    assert req.headers["X-Memory-Groups"] == "g1"
    assert req.headers["X-API-Key"] == "k"
    assert req.headers["Idempotency-Key"].startswith("msg-")
    # same content + lineage => same key (safe retries)
    await ctx.chat.user("hello")
    assert (
        route.calls[0].request.headers["Idempotency-Key"]
        == route.calls[1].request.headers["Idempotency-Key"]
    )
    await ctx.chat.user("different")
    assert (
        route.calls[2].request.headers["Idempotency-Key"]
        != route.calls[0].request.headers["Idempotency-Key"]
    )


@respx.mock
async def test_error_envelope_maps_to_typed_exception(client: MemoryClient) -> None:
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
    route = respx.post("http://memory.test/v1/observations").mock(
        side_effect=[
            httpx.Response(
                503,
                json={
                    "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "db", "retryable": True}
                },
            ),
            httpx.Response(202, json={"observation_id": "obs_1", "job_ids": []}),
        ]
    )
    ctx = client.bind(tenant_id="acme")
    ack = await ctx.observe("something happened")
    assert ack.observation_id == "obs_1"
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
        await ctx.job("job_1")


@respx.mock
async def test_context_bundle_parses_and_flags_insufficient(client: MemoryClient) -> None:
    respx.post("http://memory.test/v1/context").mock(
        return_value=httpx.Response(
            200,
            json={
                "query": "q",
                "query_type": "GENERAL_SEMANTIC",
                "conversation": {"rendered": ""},
                "memories": [],
                "knowledge": [],
                "graph_facts": [],
                "summaries": [],
                "evidence": {"status": "INSUFFICIENT", "missing_groups": ["PAGE11"]},
                "token_budget": 100,
                "token_estimate": 0,
                "rendered": "",
            },
        )
    )
    ctx = client.bind(tenant_id="acme")
    bundle = await ctx.context("q")
    assert bundle.insufficient and bundle.evidence.missing_groups == ["PAGE11"]
