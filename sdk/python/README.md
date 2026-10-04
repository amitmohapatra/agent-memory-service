# trellis-memory

Python SDK for trellis-memory, the multi-agent memory service.

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://memory-service:8080", api_key="dev-key")

ctx = memory.bind(tenant_id="acme", user_id="u1", thread_id="thr_1")  # session/turn optional

await ctx.history.add([("USER", "What changed in EBITDA?")])
bundle = await ctx.context("What changed in EBITDA?")
answer = my_agent(bundle.rendered)
await ctx.history.add([("ASSISTANT", answer)])
```

The SDK hides Qdrant, BM25, embeddings, RRF, GCS compaction, graph enrichment,
dedup, memory types, TTLs, cache keys and task queues. Those are service configuration.

## The verbs

A bound `MemoryContext` carries the calls an agent makes every turn; everything else is
under `ctx.advanced`.

| Verb | What it does |
|---|---|
| `context(query, token_budget=, tools=, window=)` | the pushed context: memories, knowledge, profile, thread summary, procedures, tool hints; `rendered` is prompt-ready, memories cited by bundle handle (`[m1]`) |
| `remember(content, memory_type=, visibility=)` / `update(id, content, reason=)` / `forget(id)` | state, supersede or forget one memory |
| `search(query, kinds=, limit=)` | ranked evidence without bundle assembly |
| `history(limit=)` / `history.add([...])` / `history.thread()` | the transcript: read it, append to it (`EVENT`: something that happened), the thread with its durable summary |
| `feedback(record)` or `feedback(kind, id, verdict)` | a judgement; `.list_for(kind, id)` reads it back; `.pending()` / `.approve(id, note=)` / `.dismiss(id, note=)` work the review queue (tenant admin key) |
| `record_tool(tool, args, output=, status=)` | what a run did; whether it worked is `feedback("run", run_id, verdict)` |
| `tool_hints(task, available=, k=)` | the tools that fit, best first, with confidence, success rate, the arguments found and the missing ones; the learned plan |
| `agent_tools()` / `call_agent_tool(name, args)` | the memory tools an agent calls itself (pull mode) |
| `profile()` / `profile.edit(block, new, old=, source_query=)` | the pinned profile blocks |
| `verify(answer, bundle_id=, run_id=)` | per-claim grounding of an answer against the context it was given (`bundle_id` from `context()`) |

`ctx.advanced` holds `documents`, `graph`, `tools` (the catalog and approval
suggestions), `model_keys` (this agent's key), `memories` (the inventory), `job(id)`, and the
client's `tenant` and `admin` administration objects.

## Agent credentials and model-free reads

```python
agent = memory.bind(tenant_id="acme", user_id="u1", agent_id="research")
await agent.advanced.model_keys.set(virtual_key, idempotency_key="research-key-v1")
status = await agent.advanced.model_keys.status()  # status only; never the secret
bundle = await agent.context("What did we decide?")
```

The service encrypts the virtual key and binds it to the tenant and agent owner. Registration
requires the operator's envelope-key configuration. Rotation and revocation also affect
background jobs and retries. `await agent.advanced.model_keys.revoke()` prevents using that agent's
credential and does not switch it to the operator key. Memory-service model calls exclude MCP.

Ingestion model uses and read model uses are independent. Whether context, search, verify and
graph queries consult a model is the tenant's model policy (`read_assist`, set with
`client.tenant.set_model_policy(uses, read_assist=...)` under the tenant admin key); a request
cannot override it, and only the uses the policy names run. There is no per-request `use_llm`.

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

`search(kinds=...)` accepts `memory`, `chunk` (document passages), `summary`, `episode`
(earlier conversations) and `message` (this thread's messages).

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
`RateLimitedError`, `DependencyUnavailableError`, `TimeoutError`. Insufficient evidence is not
an exception: `context()` returns `evidence_status` (`INSUFFICIENT` means say you do not know).
Each carries `status`, `retryable`, `trace_id`, `request_id` and `details`. A request that
got no response raises `TimeoutError` or `DependencyUnavailableError` with `status` 0.


## Feedback and paging

```python
from trellis.memory import MemoryClient

async with MemoryClient(base_url, api_key=key) as client:
    async with client.bind(tenant_id="acme", user_id="u1", workspace_id="fin") as ctx:
        memories = await ctx.advanced.memories.page(limit=50)  # .items, .next_cursor
        async for memory in ctx.advanced.memories.iter():  # every page
            ...
        await ctx.feedback("memory", memories.items[0].memory_id, "confirm", score=0.9)
        page = await ctx.feedback.page_for("memory", memories.items[0].memory_id)
        # a confirm by a user or an agent waits for review (page.items[0].review.state ==
        # "pending"); the tenant admin key works the queue:
        admin_ctx = MemoryClient(base_url, api_key=tenant_admin_key).bind()
        for vote in (await admin_ctx.feedback.pending()).items:
            await admin_ctx.feedback.approve(vote.feedback_id, note="checked")
```

See [`docs/USAGE.md`](../../docs/USAGE.md) for which call fits which scenario.
