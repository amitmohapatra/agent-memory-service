# universal-memory

Python SDK for the Enterprise Multi-Agent Memory Service.

```python
from universal_memory import MemoryClient

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
from universal_memory import BriefSpec

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

Every closed vocabulary on the wire is a `Literal` in `universal_memory.models`
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
