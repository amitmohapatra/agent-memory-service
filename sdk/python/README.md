# trellis-memory

Python SDK for trellis-memory, the multi-agent memory service.

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://memory-service:8080", api_key="dev-key")

ctx = memory.bind(tenant_id="acme", user_id="u1", thread_id="thr_1")  # session/turn optional

await ctx.chat.user("What changed in EBITDA?", attachments=["report.pdf"])
bundle = await ctx.context("What changed in EBITDA?")
answer = my_agent(bundle.rendered)
await ctx.chat.assistant(answer)
```

The SDK hides Qdrant, BM25, embeddings, RRF, GCS compaction, graph enrichment,
dedup, memory types, TTLs, cache keys and task queues. Those are service configuration.

## The verbs

A bound `MemoryContext` carries the calls an agent makes every turn; everything else is
under `ctx.advanced`.

| Verb | What it does |
|---|---|
| `context(query, token_budget=, tools=, since_revision=)` | the pushed context: memories, knowledge, profile, thread summary, procedures, tool hints; `rendered` is prompt-ready |
| `remember(content, memory_type=, visibility=)` / `update(id, content, reason=)` / `forget(id)` | state, supersede or forget one memory |
| `search(query, kinds=, limit=)` | ranked evidence without bundle assembly |
| `history(limit=)` / `summary()` | the thread's latest messages / its durable summary |
| `observe(content)` | raw evidence the service learns from, asynchronously |
| `feedback(record)` or `feedback(kind, id, verdict)` | a judgement; `.list_for(kind, id)` reads it back |
| `record_tool(tool, args, output=, status=)` / `outcome(success=)` | what a run did and whether it worked |
| `tool_hints(task, available=, k=)` | which tool, the learned plan, the next step, prefilled and missing arguments |
| `agent_tools()` / `call_agent_tool(name, args)` | the memory tools an agent calls itself (pull mode) |
| `profile()` / `profile.set(block, text)` / `profile.edit(block, old, new)` | the pinned profile blocks |
| `verify(answer, bundle=)` | per-claim grounding of an answer |
| `chat.user(...)` / `chat.assistant(...)` | the transcript |

`ctx.advanced` holds `documents`, `graph`, `tools` (the catalog and approval
suggestions), `model_keys` (this agent's key), `memories` (the inventory), `job(id)`, and the
client's `tenant`, `admin` and `webhooks` administration objects.

## Agent credentials and model-free reads

```python
agent = memory.bind(tenant_id="acme", user_id="u1", agent_id="research")
await agent.advanced.model_keys.set(virtual_key, idempotency_key="research-key-v1")
status = await agent.advanced.model_keys.status()  # status only; never the secret
bundle = await agent.context("What did we decide?", use_llm=False)
```

The service encrypts the virtual key and binds it to the tenant and agent owner. Registration
requires the operator's envelope-key configuration. Rotation and revocation also affect
background jobs and retries. `await agent.advanced.model_keys.revoke()` prevents using that agent's
credential and does not switch it to the operator key. Memory-service model calls exclude MCP.

Ingestion model uses and read model uses are independent. Context, search, verify and graph
query follow the model policy's `read_assist` when `use_llm` is omitted; `True`/`False`
override it for one read, and permit only the uses the operator and the policy enable.

## Standing questions

A profile block can carry a standing question; the background job keeps its answer current and
every bundle pins it:

```python
await agent.profile.edit("user.decisions", source_query="Which project decisions are current?")
```

## Closed sets are typed

Every closed vocabulary on the wire is a `Literal` in `trellis.memory.models`
(`MemoryType`, `Visibility`, `Lifetime`, `MessageRole`, `MessageKind`, `ObservationKind`,
`JobStatus`, `DocumentStatus`, `ArchiveStatus`, `QueryType`, `Representation`,
`TemporalStatus`, `EvidenceStatus`, `RecallKind`, `EvidenceKind`, `ToolStatus`,
`SideEffects`, ...). Client parameters such as `remember(memory_type=...)`,
`search(kinds=...)` and `advanced.memories.list(memory_types=...)` take them, and response models
(`MemoryResult.memory_type`, `MessageInfo.role`, `DocumentInfo.status`, `JobHandle.status`,
`ContextBundle.query_type`, ...) carry them instead of `str`. A wrong value is a type error
in your editor and a 422 from the service naming the allowed values; it is never silently
dropped. The service keeps the SDK's Literals equal to its own enums with a test.

`search(kinds=...)` accepts `chunk` (document passages), `memory` and `summary`.

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


## Feedback, webhooks and paging

```python
from trellis.memory import MemoryClient
from trellis.memory.webhooks import verify_signature

async with MemoryClient(base_url, api_key=key) as client:
    async with client.bind(tenant_id="acme", user_id="u1", workspace_id="fin") as ctx:
        memories = await ctx.advanced.memories.page(limit=50)  # .items, .next_cursor
        async for memory in ctx.advanced.memories.iter():  # every page
            ...
        await ctx.feedback("memory", memories.items[0].memory_id, "confirm", score=0.9)
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
