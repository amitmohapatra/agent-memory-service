# trellis-memory

Python SDK for trellis-memory, the multi-agent memory service.

```python
from trellis.memory import MemoryClient

memory = MemoryClient()  # MEMORY_URL and TRELLIS_API_KEY from the environment

ctx = memory.bind(user_id="u1", thread_id="thr_1")  # the key names the tenant

await ctx.history.add([("USER", "What changed in EBITDA?")])
bundle = await ctx.context("What changed in EBITDA?")
answer = my_agent(bundle.rendered)
await ctx.history.add([("ASSISTANT", answer)])
```

The SDK hides Qdrant, BM25, embeddings, RRF, GCS compaction, graph enrichment,
dedup, memory types, TTLs, cache keys and task queues. Those are service configuration.

## Configuration

`MemoryClient()` reads the platform's shared names: `MEMORY_URL` for the service (the local
stack's `http://localhost:8080` when unset) and `TRELLIS_API_KEY` for the key. Arguments win:
`MemoryClient(url, api_key=...)`, or `bearer_token=` for a token (then no key is read from the
environment). Against the local stack's development key (`dev-key`) no tenant is needed
either: a development key acts in the service's development tenant, `default`, unless a call
names another (`bind(tenant_id=...)`).

| Argument | Default | What it does |
|---|---|---|
| `timeout` | `10.0` | seconds per attempt; connecting is bounded by 5 s of it. `context(..., timeout=)` and `search(..., timeout=)` take a per-call one |
| `max_retries` | `3` | how many times a retryable failure is sent again |
| `circuit_failure_threshold` | `5` | failed calls in a row that open the circuit; `0` disables the breaker |
| `circuit_open_seconds` | `30.0` | how long an open circuit fails fast before one call probes |
| `http_client` | a pooled `httpx.AsyncClient` | idle connections kept 30 s, at most 100 connections |

`async with MemoryClient() as memory:` closes the pool on exit (`await memory.aclose()`
otherwise). Use one client per process.

## Retries and the circuit breaker

A call is sent again when that cannot duplicate anything: GETs, writes carrying an
`Idempotency-Key` (the verbs that write send one), and the read-only POSTs - `context`,
`search` (`/v1/recall`), `verify` and `tool_hints`. They retry on a retryable error (`429`,
`502`, `503`, `504`, or a problem that says `retryable: true`), on timeouts and on dropped
connections. A connection that never opened is retried for every call, and so is a `429`,
which the service refuses before doing any work. A write without a key that may have reached
the service (a read timeout, a 503) is not retried, and no key is invented for it.

The wait is the service's `Retry-After` when it sent one (at most 30 s), otherwise full-jitter
exponential backoff: a uniform draw from 0 up to 0.5 s, 1 s, 2 s ... capped at 8 s.

After `circuit_failure_threshold` calls in a row fail for want of the service - no response,
or a 5xx, counted once per call however many attempts it made - the client stops sending:
every call raises `CircuitOpenError` at once for `circuit_open_seconds`. Then one call goes
through as the probe while the others keep failing fast; its success closes the circuit, its
failure opens it for another period. A 4xx is the service answering and a 429 is the service
asking for less: neither counts. The breaker is per client, so an agent's turn during an
outage degrades in microseconds instead of paying the timeout and every retry on each call:

```python
try:
    pushed = await ctx.context(question)
except DependencyUnavailableError:  # CircuitOpenError is one; retryable, with retry_after
    pushed = None  # answer without memory this turn
```

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
requires the operator's envelope-key configuration (a local `dev` service derives a
development one, so it works there with nothing set). Rotation and revocation also affect
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
`RateLimitedError`, `DependencyUnavailableError` (and its `CircuitOpenError`), `TimeoutError`.
Insufficient evidence is not an exception: `context()` returns `evidence_status`
(`INSUFFICIENT` means say you do not know). Each carries `status`, `retryable`,
`retry_after` (the `Retry-After` seconds, when sent), `trace_id`, `request_id` and `details`.
A body without a `code` - a gateway or proxy answered - is classed by its status: 400/413/422
`ValidationError`, 401 `AuthenticationError`, 403 `AuthorizationError`, 404 `NotFoundError`,
409 `ConflictError`, 429 `RateLimitedError`, 502/503 `DependencyUnavailableError`, 504
`TimeoutError` (those four retryable), anything else the base `MemoryError` (not retryable).
A request that got no response raises `TimeoutError` or `DependencyUnavailableError` with
`status` 0, both retryable.


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
