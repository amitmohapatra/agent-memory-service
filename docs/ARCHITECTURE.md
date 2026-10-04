# Architecture

## Shape

Hexagonal architecture (ports and adapters), one microservice, horizontally scalable API
and worker processes sharing PostgreSQL.

### In the platform

```mermaid
flowchart LR
  subgraph clients[Callers]
    H["agent-harness<br/>(trellis-harness)"]
    SDK["trellis-memory SDK<br/>(sdk/python)"]
    REST[Any HTTP client]
  end
  H --> SDK
  SDK -->|/v1 + API key| API
  REST -->|/v1 + API key| API
  H -->|runs, interrupts| RUNS["agent-runs"]
  subgraph svc[trellis-memory service]
    API["API process<br/>(FastAPI)"]
    W["Worker process<br/>(Procrastinate jobs)"]
  end
  API --> PG[("PostgreSQL<br/>source of truth + job queue")]
  W --> PG
  API --> Q[("Qdrant<br/>search, rebuildable")]
  W --> Q
  API --> C[("Dragonfly / Redis<br/>hot cache")]
  W --> B[("GCS or filesystem<br/>archive")]
  API -. optional .-> FGA["OpenFGA"]
  API -->|virtual key| GW["Bifrost gateway"]
  W -->|virtual key| GW
  GW --> LLM["Model providers"]
  CT["trellis-contracts"] -. shared types .- H
  CT -. shared types .- RUNS
```

### Inside the service

```mermaid
flowchart TB
  api["api/<br/>routers, schemas, problem details"] --> app["application/<br/>Container, use cases"]
  app --> mods["modules/*<br/>feature slices"]
  app --> ports["ports/<br/>Protocols"]
  mods --> ports
  mods --> domain["domain/<br/>contracts we own"]
  ports --> domain
  adapters["adapters/<br/>the only place SDKs are imported"] -. implements .-> ports
  adapters --> ext["PostgreSQL · Qdrant · Dragonfly · OpenFGA · Procrastinate ·<br/>GCS/filesystem · Docling · ONNX/sentence-transformers · Bifrost (HTTP)"]
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
| `modules/*` | feature slices: archive, audit, auth, authz, context, conversation, feedback, graph, grounding, idempotency, ingestion, jobs, llm, memory, profile, rag, retrieval, tenancy, tools, agent_tools, working_memory (the hot thread cache) | domain, ports |
| `adapters` | one package per provider; the only place SDKs are imported; `wiring.py` attaches configured providers | everything |
| `api` | FastAPI routers, typed schemas with examples, RFC 9457 problem details, middleware, OpenAPI customization | application |

## Data placement

```mermaid
flowchart LR
  HOT["HOT · Dragonfly<br/>recent thread, ephemeral memories, caches<br/>(never the source of truth)"]
  WARM["WARM · PostgreSQL<br/>threads, sessions, turns, messages, observations,<br/>canonical memories, evidence refs, documents/nodes/chunks,<br/>jobs, revisions, archive manifests, graph, tool catalog/calls/<br/>statistics/procedures, approval patterns, agent-tool pulls,<br/>profile blocks, thread summaries, feedback, model keys and policies"]
  SEARCH["SEARCH · Qdrant<br/>BM25 sparse + dense: knowledge, memories, tools<br/>(rebuildable from WARM)"]
  ARCHIVE["ARCHIVE · GCS<br/>raw chat segments (JSONL+zstd), raw files,<br/>imports, old versions"]
  WARM -->|index jobs| SEARCH
  WARM -->|archive jobs| ARCHIVE
  WARM -->|cache-aside| HOT
```

### Core data model

Solid lines are foreign keys; dashed lines are references the service keeps without one
(evidence refs, graph pointers, feedback targets).

```mermaid
erDiagram
  threads ||--o{ sessions : has
  sessions ||--o{ turns : has
  turns ||--o{ messages : has
  messages ||--o{ message_attachments : has
  messages ||..o{ observations : "becomes"
  observations ||..o{ memories : "evidence for"
  memories ||--o{ memory_dependencies : "derived from"
  memories ||..o{ graph_relations : "linked by"
  graph_entities ||..o{ graph_relations : "subject / object"
  documents ||--o{ document_versions : has
  documents ||--o{ document_nodes : has
  documents ||--o{ chunks : has
  documents ||--o{ context_edges : has
  memories ||..o{ feedback : "target"
  agent_runs ||..o{ feedback : "target"
  tenants ||--o{ api_keys : issues
  tenants ||--o{ workspaces : has
  workspaces ||--o{ workspace_members : has
```

## Durability protocol

```mermaid
sequenceDiagram
  participant Cl as Client
  participant API as API
  participant PG as PostgreSQL
  participant W as Worker
  participant Bl as GCS / filesystem
  Cl->>API: write (Idempotency-Key)
  API->>API: validate the trusted context
  API->>PG: idempotency check
  rect rgb(240,240,240)
    API->>PG: BEGIN
    API->>PG: staged payload + metadata + observation
    API->>PG: archive job + memory job (job_outbox, same transaction)
    API->>PG: bump revisions
    API->>PG: COMMIT
  end
  API-->>Cl: 200 / 202
  W->>PG: claim jobs
  W->>Bl: group, compress, checksum, immutable upload
  W->>PG: manifest, ARCHIVED, later purge the staged payload
```

Archive worker: group -> compress -> checksum -> immutable upload -> verify generation +
checksum -> persist manifest -> mark ARCHIVED -> purge large staged payload after grace period.
Reconciler repairs ACKed-but-unarchived, manifest-missing, checksum-mismatch, stuck jobs,
and canonical/search drift.

## Retrieval pipeline

```mermaid
flowchart TB
  q[query] --> route["language + rule-based route<br/>(English cues only for English;<br/>any other language: GENERAL_SEMANTIC)"]
  route --> par{{"concurrently"}}
  par --> enc["encoders, with the authorized scope"]
  par --> gp["graph prefetch<br/>(one statement, stopped at the 150 ms budget)"]
  enc --> search["exact lookup or dense / BM25 search"]
  search --> mem["memories: weighted RRF of BM25, two dense spaces<br/>and ColBERT over two keys, then session / speaker /<br/>time / period rules (ADR 0026)"]
  search --> other["other items: RRF + stable ties"]
  mem --> cut[dedup, bounded cut]
  other --> cut
  gp --> gfx["graph facts and evidence"]
  cut --> gfx
  gfx --> comp["document companion expansion,<br/>request-local evidence verification"]
  comp --> sim["dense similarity of each item to the question"]
  sim --> pack["ContextBuilder: provenance + evidence groups<br/>under the token budget, relevance floor"]
  side["profile blocks, thread summary, procedures,<br/>prefetched memories, recent messages, tool hints<br/>(one indexed read each, in parallel)"] --> pack
  pack --> bundle[ContextBundle]
```

Memories are re-scored by standing (confidence and reinforcement, which feedback moves) with
a bounded factor (±15%) before the cut.

## Memory lifecycle

A memory's `temporal_status`, as the code moves it. Every state but CURRENT is out of
retrieval; nothing is hard-deleted by these transitions.

```mermaid
stateDiagram-v2
  [*] --> CURRENT: observation processed / memory written
  CURRENT --> SUPERSEDED: a newer fact or a correction replaces it (superseded_by)
  CURRENT --> RETRACTED: feedback reject, or DELETE /v1/memories/{id} (soft delete)
  CURRENT --> EXPIRED: expires_at passed (memory.expire, hourly)
  CURRENT --> ARCHIVED: forgetting policy (memory.forget, daily)
  ARCHIVED --> CURRENT: POST /v1/memories/{id}/restore
  SUPERSEDED --> [*]
  RETRACTED --> [*]
  EXPIRED --> [*]
```

`CONTRADICTED` is in the API's vocabulary but reserved: a conflict is resolved by
superseding, so no memory is set to it today.

## Learning (background)

```mermaid
flowchart LR
  rt["record_tool / outcome"] --> tl["tools.learn<br/>procedures per (audience, task pattern),<br/>procedural graph edges"]
  fb[feedback] --> rv["review (ADR 0028):<br/>a vote waits for the tenant admin"] --> fp["feedback.project<br/>memory standing, run outcomes, tool statistics,<br/>approval patterns, procedure rejection"]
  msg["every 20 messages"] --> sr["summary.refresh<br/>the thread's rolling summary"]
  up["USER / PREFERENCE memory"] --> pr["profile.refresh<br/>the user's pinned block"]
  pulls["agent-tool pulls"] --> pf["prefetch, every 5 min<br/>what the push pre-includes"]
  cat["catalog upsert"] --> ti["tools.index<br/>the tools search collection"]
  obs["observation (any language)"] --> mp["memory.process<br/>English rules; other languages:<br/>contextual_extraction (tenant model)"]
  md["memory / document"] --> ge["graph.enrich<br/>native entities and edges;<br/>relation_extraction (tenant model)"]
```

Where each model use runs, its tier and its fallback: [LLM-USES.md](LLM-USES.md).

The request path reads only what these jobs precompute (indexed, bounded).

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

The service never talks to an LLM provider. The model is available exactly when
`BIFROST_URL` is set (`LLMSettings.enabled`); `adapters/models/llm.py` speaks the
OpenAI-compatible HTTP API of a [Bifrost](https://github.com/maximhq/bifrost) gateway that
runs outside the service and holds the provider keys. The service holds only Bifrost
*virtual keys*: the operator's (`BIFROST_VIRTUAL_KEY`, e.g. in `secrets.env`) and the
agent- and tenant-level keys registered through the API, stored encrypted. Calls are bounded
(timeout, bounded retries, circuit breaker; the values are `constants.LLM` /
`constants.LLM_TRANSPORT`, not environment variables), traced (`llm.chat` spans with
model/tokens/latency), metered (`memory_llm_*`) and logged without prompt text unless the
`LOG_SOURCE_TEXT` constant is on.
Provider SDK imports are banned under `src/` by Ruff and `tests/unit/test_architecture.py`.

Every deterministic path stays complete on its own. `modules/llm/assist.py::LLMAssist` is
the single entry point modules use. A request or job first binds the identity that owns the
work (`LLMAssist.bound` / `reading`: one indexed read of the key and policy hierarchies); a use
is then consulted only when the gateway is configured, the resolved tenant policy allows it
(no policy row: every use except the opt-in `memory_restatement`) and a key can pay
(contextual_extraction, relation_extraction, entity_resolution, conflict_adjudication,
summaries, reflection, memory_connections, query_expansion, chunk_context,
memory_restatement, grounding_judge, procedure_abstraction; see `docs/LLM-USES.md`).
Any failure returns
`None`, so the module continues with its native result. Every successful call is counted in
`llm_usage_daily` (one upsert) and `memory_llm_tokens_total{tenant,use,direction}`. Mem0/LangMem/Graphiti/Cognee provider adapters were removed; comparisons belong
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
