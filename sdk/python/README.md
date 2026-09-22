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
