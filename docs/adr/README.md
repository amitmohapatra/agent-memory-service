# Architecture decision records

Numbered in the order the decisions were taken. Each record states the context, the decision and
what it cost. When a decision is later reversed, narrowed or never carried out, the record is
**amended rather than rewritten** — a superseded ADR keeps its text and gains a note saying what
replaced it, because the reasoning that was wrong is the part worth keeping.

Status vocabulary: **accepted** (in force), **amended** (in force, changed by a later ADR),
**superseded** (replaced — the note says by what), **in progress** (the decision is taken, the work
is not finished).

| # | Decision | Status |
|---|---|---|
| [0001](0001-hexagonal-architecture.md) | Hexagonal architecture: every outbound dependency is a `Protocol` in `memory_service.ports`, adapters are registered in a provider registry, and the domain imports no provider SDK | accepted |
| [0002](0002-reuse-mature-oss.md) | Reuse mature OSS for infrastructure primitives (Qdrant, Dragonfly, OpenFGA, Procrastinate) rather than building them — with an outcome note recording that the OPA row was never built and was removed | accepted, with an outcome note |
| [0003](0003-build-environment-constraints.md) | Build-environment constraints: egress is limited, so tests must run without model weights or servers, and every number from that environment is labelled as a stand-in | accepted |
| [0004](0004-transactional-outbox.md) | A transactional outbox in front of Procrastinate, so a `2xx` means the record *and* its processing job are committed in one transaction | accepted |
| [0005](0005-authorization-and-visibility.md) | OpenFGA relationships plus visibility keys, filtered **store-side before** search — never "retrieve globally, filter in memory" | accepted; amended by 0021 (`WORKSPACE`) and 0022 (header names) |
| [0006](0006-archive-protocol.md) | The chat archive protocol: staged → immutable verified segment → purge, with content-addressed segments and verified uploads | accepted |
| [0007](0007-document-context-graph.md) | Docling parsing behind a port, natural chunking, and the Document Context Graph that expansion follows | accepted |
| [0008](0008-baseline-retrieval.md) | Baseline retrieval: one index, two named vectors — client-encoded BM25 with server-side IDF, fused with dense in a single query | accepted |
| [0009](0009-memory-intelligence.md) | Memory intelligence: native deterministic rules first (extract → classify → consolidate → persist → index), LLM providers as benchmarkable adapters | accepted |
| [0010](0010-knowledge-graph.md) | The knowledge graph lives in PostgreSQL next to the canonical rows, carries evidence, and is traversed under bounds | accepted |
| [0011](0011-context-preservation.md) | Context preservation: expansion follows the document graph rather than similarity, hierarchical summaries, and evidence verification | accepted |
| [0012](0012-advanced-retrieval.md) | Advanced retrieval strategies are benchmark-gated extra retrievers | **superseded by measurement (2026-09-20)**: seven strategies removed, `rerank` defaulted off |
| [0013](0013-multi-agent-semantics.md) | Multi-agent semantics: private versus shared memory, hand-off that flows down a run tree, no agent chatter in the user's history, conflicts surfaced rather than resolved | accepted |
| [0014](0014-langgraph-adapter.md) | The LangGraph adapter wraps nodes; it is not a checkpointer or a store | **superseded by 0020** (the decision held, the location did not) |
| [0015](0015-release-gates-and-hardening.md) | Release gates are produced by code, evaluated by code, and caveated honestly: a PASS says exactly what was measured | accepted; amended by 0022 (problem details) |
| [0016](0016-document-knowledge-graph.md) | The document knowledge graph is extracted by a deterministic business grammar — typed entities, resolved aliases — not by capitalisation | accepted |
| [0017](0017-real-component-validation.md) | Real-component validation: what proved itself against real servers and weights, and what was cut when it did not | in progress |
| [0018](0018-tool-memory.md) | Tool memory: record, learn, advise — **never execute** | accepted |
| [0019](0019-separate-model-tier.md) | The model tier is deployed separately, on CPU | **superseded (2026-09)**: the served tier was removed; models load in-process |
| [0020](0020-framework-adapters-live-in-the-consumer.md) | Framework adapters do not live in the memory service — a service that ships a LangGraph package depends on its own consumer | accepted |
| [0021](0021-tenants-keys-and-workspaces.md) | Tenants, API keys and workspaces are the service's own, with revocation that takes effect on the next request | accepted; amends 0005, amended by 0022 |
| [0022](0022-trellis-names-and-api-conventions.md) | Trellis names and API conventions: `trellis.memory`, `X-Trellis-*` headers, RFC 9457 problem details, `traceparent`, operation ids — with the old spellings kept as aliases for one release | accepted; amends 0005 and 0021 |
| [0023](0023-m3-lite-feedback-team-keys-pagination-webhooks.md) | M3-lite: `POST /v1/feedback` with a projector, team and workspace model keys, cursor pagination, and signed outbound webhooks | accepted; amends 0012, 0021, 0022 |
| 0024 | The multilingual runtime and its accuracy programme (Memory Service M2 + M4) | **landing with M2/M4** — the record and `docs/MULTILINGUAL-RUNTIME.md` arrive with that work |
| [0025](0025-late-interaction-two-keys-and-learned-memory-ranking.md) | Late interaction, two keys per memory, and a learned memory ranking | accepted; supersedes the `colbert` and `rerank` parts of 0012 |

## Where the numbers behind these decisions are

An ADR that claims a measurement names the artifact. The artifacts are
[`benchmark/results/`](../../benchmark/results) and the account of what each one measured — and in
which environment — is [`../MEASUREMENTS.md`](../MEASUREMENTS.md) and
[`../FINAL_REPORT.md`](../FINAL_REPORT.md). If an ADR and an artifact disagree, the artifact wins.
