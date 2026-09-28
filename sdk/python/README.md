# trellis-memory

Python SDK for trellis-memory, the multi-agent memory service.

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://memory-service:8080", api_key="dev-key")

ctx = memory.bind(
    tenant_id="acme", user_id="u1", thread_id="thr_1", session_id="ses_1", turn_id="trn_1"
)

await ctx.chat.user("What changed in EBITDA?", attachments=["report.pdf"])
bundle = await ctx.context("What changed in EBITDA?")
answer = my_agent(bundle.rendered)
await ctx.chat.assistant(answer)
```

The SDK hides Qdrant, BM25, embeddings, rerankers, RRF, GCS compaction, graph enrichment,
dedup, memory types, TTLs, cache keys and task queues. Those are service configuration.

## Agent credentials and model-free reads

```python
agent = memory.bind(tenant_id="acme", user_id="u1", agent_id="research")
await agent.set_model_key(virtual_key, idempotency_key="research-key-v1")
status = await agent.model_key_status()  # status only; never the secret
bundle = await agent.context("What did we decide?", use_llm=False)
```

The service encrypts the virtual key and binds it to the tenant and agent owner. Registration
requires the operator's envelope-key configuration. Rotation and revocation also affect
background jobs and retries. `await agent.revoke_model_key()` prevents using that agent's
credential and does not switch it to the operator key. Memory-service model calls exclude MCP.

Ingestion model uses and read model uses are independent. Context, recall, verify and graph
query default to `use_llm=False`; `True` permits only the uses enabled by the operator.
Registering a key alone does not enable model calls.

## Standing questions and knowledge pages

```python
from trellis.memory import BriefSpec

brief = await agent.briefs.create(
    BriefSpec(
        title="Project decisions",
        question="Which project decisions are current?",
        kind="knowledge_page",
        use_llm=False,
    )
)
current = await agent.briefs.get(brief.brief_id)
if current.status == "ready":
    print(current.output.text)
```

Both `mental_model` and `knowledge_page` use a background refresh and a stored-content read.
Native refresh returns cited excerpts. Assisted refresh requires the `briefs` model use and
marks generated output; reading either kind makes no model call. Unchanged evidence and
synthesis configuration avoid repeat generation. Source or permission changes hide stale
output until refreshed. A brief is bound to the exact execution scope: omit run/thread/session
IDs when it should persist across runs of the same agent. Pages are maintained briefs, not
a full wiki with folders and history.

See the [implementation and validation handoff](../../docs/AGENT-CAPABILITIES-HANDOFF-20260927.md)
for configuration, isolation guarantees and measured limits.

## Closed sets are typed

Every closed vocabulary on the wire is a `Literal` in `trellis.memory.models`
(`MemoryType`, `Visibility`, `Lifetime`, `MessageRole`, `MessageKind`, `ObservationKind`,
`JobStatus`, `DocumentStatus`, `ArchiveStatus`, `QueryType`, `Representation`,
`TemporalStatus`, `EvidenceStatus`, `RecallKind`, `EvidenceKind`, `ToolSource`, `ToolStatus`,
`SideEffects`, `CacheScope`, ...). Client parameters such as `remember(memory_type=...)`,
`recall(kinds=...)` and `list_memories(memory_types=...)` take them, and response models
(`MemoryResult.memory_type`, `MessageInfo.role`, `DocumentInfo.status`, `JobHandle.status`,
`ContextBundle.query_type`, ...) carry them instead of `str`. A wrong value is a type error
in your editor and a 422 from the service naming the allowed values; it is never silently
dropped. The service keeps the SDK's Literals equal to its own enums with a test.

`recall(kinds=...)` accepts `chunk` (document passages), `memory` and `summary`. Earlier
docs listed a `fact` kind; the engine never served it (the call just returned nothing), so
it is gone from the type and the service now rejects it.

## Headers, tracing and errors

Every call carries the scope as `X-Trellis-Tenant` / `X-Trellis-Workspace` / `X-Trellis-User`,
an `X-Request-ID` (one per call, kept across retries) and, when the agent is tracing with
OpenTelemetry (`pip install trellis-memory[otel]`), the active span's `traceparent`; without
it, `bind(trace_id=<32 hex>)` produces one. The service continues the trace and answers
with `traceparent` and `X-Trace-ID`.

A `trace_id` that is not a 32-hex W3C id is sent as `X-Correlation-ID` (echoed, not traced);
an explicit `correlation_id` wins. The SDK does not send `X-Trace-ID`: the service names the
trace on the response. Every scope id follows the service's grammar (a letter or digit, then
letters, digits and `._:-`, at most 200 characters) and is checked when the scope is built.
An `http_client` you pass in is given the service credential as a default header, so give
the SDK a client of its own; an httpx client instrumented by OpenTelemetry injects its own
`traceparent` at send time, which then replaces the one built from `trace_id`.

Errors are RFC 9457 problems mapped to one exception per `code`: `AuthenticationError`,
`AuthorizationError`, `NotFoundError`, `ConflictError`, `ValidationError`,
`RateLimitedError`, `DependencyUnavailableError`, `TimeoutError`, `InsufficientEvidence`.
Each carries `status`, `retryable`, `trace_id`, `request_id` and `details`. A request that
got no response raises `TimeoutError` or `DependencyUnavailableError` with `status` 0.


## Feedback, webhooks and paging (0.2.1)

```python
from trellis.memory import MemoryClient
from trellis.memory.webhooks import verify_signature

async with MemoryClient(base_url, api_key=key) as client:
    async with client.bind(tenant_id="acme", user_id="u1", workspace_id="fin") as ctx:
        memories = await ctx.memories_page(limit=50)  # .items, .next_cursor
        async for memory in ctx.iter_memories():  # every page
            ...
        await ctx.feedback.submit("memory", memories.items[0].memory_id, "confirm", score=0.9)
        page = await ctx.feedback.page_for("memory", memories.items[0].memory_id)

    admin = client.administer("acme")
    hook = await admin.webhooks.create(
        "https://hooks.example.com/trellis", ["memory.created", "feedback.projected"]
    )
    secret = hook.secret  # shown once
    await admin.workspaces.set_model_key("fin", "vk-...")  # the team's Bifrost key

# in the receiver
if not verify_signature(secret, request.headers["X-Trellis-Signature"], raw_body):
    return 401
```
