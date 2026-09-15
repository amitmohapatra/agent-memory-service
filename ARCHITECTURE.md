# Architecture

## Shape

Hexagonal architecture (ports and adapters), one microservice, horizontally scalable API
and worker processes sharing PostgreSQL.

```
                       +---------------------------+
   Plain Python  ----> |                           |
   REST client   ----> |   universal-memory SDK    | ----> Memory Service (FastAPI)
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
                       Docling · sentence-transformers/fastembed · Mem0/Cognee/LangMem · Graphiti · OPA
                       Bifrost (the only LLM path: one HTTP adapter, no provider SDK anywhere)
```

Rules enforced by `tests/unit/test_architecture.py` and Ruff `banned-api`:

- `domain/`, `application/`, `modules/`, `ports/` and `api/` never import provider SDKs.
- LangGraph types never appear in the core; `integrations/langgraph` is a separate package.
- No module reads another module's tables directly; it goes through that module's service.
- Every persistent write is idempotent; every derived memory keeps `EvidenceRef` provenance.

## Layers

| Layer | Contains | Depends on |
|---|---|---|
| `domain` | `MemoryExecutionContext`, `CanonicalMemory`, `Scope`, `Visibility`, `TemporalState`, `EvidenceRef`, conversation and document models, `ContextBundle`, errors | nothing |
| `ports` | `CacheProvider`, `SearchStore`, `BlobStore`, `TaskQueue`, `AuthorizationProvider`, `PolicyProvider`, `EmbeddingProvider`, `SparseEncoder`, `Reranker`, `LLMProvider`, `MemoryIntelligenceProvider`, `GraphStore`, `GraphEnrichmentProvider`, `DocumentParser` | domain |
| `application` | composition root (`Container`), use-case orchestration | domain, ports |
| `modules/*` | feature slices: conversation, working_memory, ingestion, classification, extraction, dedup, consolidation, temporal, metadata, rag, document_context, graph, retrieval, context, summarization, reflection, lifecycle, archive, evaluation, imports | domain, ports |
| `adapters` | one package per provider; the only place SDKs are imported; `wiring.py` attaches configured providers | everything |
| `api` | FastAPI routers, typed schemas with examples, error envelope, middleware, OpenAPI customization | application |

## Data placement

```
HOT      Dragonfly      recent thread, working memory, caches (never source of truth)
WARM     PostgreSQL     threads/sessions/turns/messages, observations, canonical memories,
                        evidence refs, documents/nodes/chunks metadata, jobs, revisions,
                        archive manifests, eval metadata
SEARCH   Qdrant         BM25 sparse + dense (+ optional SPLADE/ColBERT) — rebuildable
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
query -> authorized scope (OpenFGA, cached by revision) -> exact lookup -> QueryRouter (rules)
      -> [conversation | BM25 | dense | KG] -> RRF -> prune -> CPU rerank (bounded K)
      -> context/relationship expansion (bounded budget) -> evidence completeness verifier
      -> complete: ContextBuilder | incomplete: escalate (broaden K, graph, structured, optional
         PageIndex/sparse/ColBERT, filtered exact) -> still insufficient: ABSTAIN
```

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
`models.llm.uses` (ambiguous_extraction, ambiguous_worthiness, relation_extraction,
entity_resolution, conflict_adjudication, summaries, reflection, query_expansion,
chunk_context) and any failure returns `None`, so the module continues with its native
result. Third-party providers that need an LLM (Mem0, LangMem, Graphiti, Cognee) are
pointed at the same gateway through their OpenAI-compatible base-URL settings.

## Caching

Cache-aside. Every sensitive key contains tenant, principal scope fingerprint, revision
fingerprint, provider/model version and the content/query hash. Invalidation is by revision
bump, never by scan.

## Observability

OpenTelemetry spans per stage, Prometheus metrics (`/metrics`), structured JSON logs with
tenant/thread/session/turn/agent-run/job/trace fields, OpenLineage events for processing
lineage, `EvidenceRef` for claim provenance. Source text is never logged by default.
