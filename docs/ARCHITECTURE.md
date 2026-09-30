# Architecture

## Shape

Hexagonal architecture (ports and adapters), one microservice, horizontally scalable API
and worker processes sharing PostgreSQL.

```
                       +---------------------------+
   Plain Python  ----> |                           |
   REST client   ----> |   trellis-memory SDK      | ----> trellis-memory service (FastAPI)
   LangGraph adapter-> |                           |            |
                       +---------------------------+            v
                                                        application (use cases)
                                                                |
                                       +------------------------+-------------------------+
                                       |                        |                         |
                                    domain                   ports                    modules
                            (contracts we own)          (Protocols)           (conversation, ingestion,
                                                                |               retrieval, archive, ...)
                                                                v
                                                            adapters
                       PostgreSQL · Qdrant · Dragonfly · OpenFGA · Procrastinate · GCS/filesystem ·
                       Docling · ONNX/sentence-transformers · native memory/graph intelligence
                       Bifrost (the only LLM path: one HTTP adapter, no provider SDK anywhere)
```

Rules enforced by `tests/unit/test_architecture.py` and Ruff `banned-api`:

- `domain/`, `application/`, `modules/`, `ports/` and `api/` never import provider SDKs.
- LangGraph types never appear anywhere in this repository. Framework adapters are not a
  memory-service concern at all (ADR 0020); they live in `agent-harness`.
- No module reads another module's tables directly; it goes through that module's service.
- Every persistent write is idempotent; every derived memory keeps `EvidenceRef` provenance.

## Layers

| Layer | Contains | Depends on |
|---|---|---|
| `domain` | `MemoryExecutionContext`, `CanonicalMemory`, `Scope`, `Visibility`, `TemporalState`, `EvidenceRef`, conversation and document models, `ContextBundle`, errors | nothing |
| `ports` | `CacheProvider`, `SearchStore`, `BlobStore`, `TaskQueue`, `AuthorizationProvider`, `EmbeddingProvider`, `SparseEncoder`, `LLMProvider`, `MemoryIntelligenceProvider`, `GraphStore`, `GraphEnrichmentProvider`, `DocumentParser` | domain |
| `application` | composition root (`Container`), use-case orchestration | domain, ports |
| `modules/*` | feature slices: archive, audit, auth, authz, briefs, context, conversation, feedback, graph, grounding, idempotency, ingestion, jobs, llm, memory, rag, retrieval, tenancy, tools, webhooks, working_memory (the hot thread cache) | domain, ports |
| `adapters` | one package per provider; the only place SDKs are imported; `wiring.py` attaches configured providers | everything |
| `api` | FastAPI routers, typed schemas with examples, RFC 9457 problem details, middleware, OpenAPI customization | application |

## Data placement

```
HOT      Dragonfly      recent thread, ephemeral memories, caches (never source of truth)
WARM     PostgreSQL     threads/sessions/turns/messages, observations, canonical memories,
                        evidence refs, documents/nodes/chunks metadata, jobs, revisions,
                        archive manifests, graph, tools, feedback, model keys and policies
SEARCH   Qdrant         BM25 sparse + dense — rebuildable
ARCHIVE  GCS            raw chat segments (JSONL+zstd), raw files, imports, old versions
```

## Durability protocol

```
request -> validate trusted context -> idempotency check
       -> BEGIN
            persist staged payload + metadata + observation
            enqueue archive job + memory job (same transaction)
            bump revisions
          COMMIT
       -> 200/202
```

Archive worker: group -> compress -> checksum -> immutable upload -> verify generation +
checksum -> persist manifest -> mark ARCHIVED -> purge large staged payload after grace period.
Reconciler repairs ACKed-but-unarchived, manifest-missing, checksum-mismatch, stuck jobs,
and canonical/search drift.

## Retrieval pipeline

```
query -> rule-based route -> overlap encoder with authorized scope + graph prefetch
      -> exact lookup or dense/BM25 search -> native RRF + stable ties -> dedup -> bounded cut
      -> graph facts/evidence -> document companion expansion
      -> request-local evidence verification / bounded companion escalation
      -> ContextBuilder packs provenance and evidence groups under the token budget
```

Document selection is applied to exact hits and after every post-stage, before the next
stage verifies evidence. Tenant and visibility filtering remain independent requirements.
Companion group identities include their target nodes; verification metadata is local to a
request. Expansion batches target/sibling reads, and graph evidence hydration has an enforced
chunk cap. These bounds do not establish an end-to-end latency SLO.

Chunks are indexed with deterministic context (document, section path, page, entities)
prepended — Contextual Retrieval — while the original text is kept separately for display.
The Document Context Graph (PARENT/NEXT/PREVIOUS/ON_PAGE/IN_TABLE/FOOTNOTE/CROSS_REFERENCE/
MENTIONS/DEFINED_BY) is built without an LLM and is distinct from the semantic Knowledge Graph.

## Generative model access

The service never talks to an LLM provider. `LLMSettings.provider` is `disabled` or
`bifrost`; `adapters/models/llm.py` speaks the OpenAI-compatible HTTP API of a
[Bifrost](https://github.com/maximhq/bifrost) gateway that runs outside the service and
holds the provider keys. The service holds only a Bifrost *virtual key*
(`secrets.env` / `MEMORY__MODELS__LLM__API_KEY`). Calls are bounded (timeout, bounded
retries, circuit breaker), traced (`llm.chat` spans with model/tokens/latency), metered
(`memory_llm_*`) and logged without prompt text unless `service.log_source_text=true`.
Provider SDK imports are banned under `src/` by Ruff and `tests/unit/test_architecture.py`.

Every deterministic path stays complete on its own. `modules/llm/assist.py::LLMAssist` is
the single entry point modules use: a use is consulted only when its flag is in
`models.llm.uses` (contextual_extraction, relation_extraction,
entity_resolution, conflict_adjudication, summaries, reflection, memory_connections,
query_expansion, query_decomposition, chunk_context) and any failure returns `None`, so the
module continues with its native result. Mem0/LangMem/Graphiti/Cognee provider adapters were removed; comparisons belong
in benchmark code. The production wiring does not enable every implemented memory feature;
see [the capability audit](RESEARCH-RAG-2026-09-25.md).

## Caching

Cache-aside. Every sensitive key contains tenant, principal scope fingerprint, revision
fingerprint, provider/model version and the content/query hash. Invalidation is by revision
bump, never by scan.

## Observability

OpenTelemetry spans per stage, Prometheus metrics (`/metrics`), structured JSON logs with
tenant/thread/session/turn/agent-run/job/trace fields, OpenLineage events for processing
lineage, `EvidenceRef` for claim provenance. Source text is never logged by default.
