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
