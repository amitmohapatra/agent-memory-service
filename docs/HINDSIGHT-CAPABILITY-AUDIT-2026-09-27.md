# Hindsight capability audit — 2026-09-27

## Verdict and scope

**We do not have full Hindsight feature parity. Using Hindsight as the conversational memory backend is a reasonable option to evaluate.** Our existing permissions, tool-procedure memory, grounding verification, and document ingestion remain reasons to preserve selected service components; they do not establish that our memory engine is more accurate.

This is a documentation and static-code audit, not a new runtime, accuracy, latency, or security certification. Hindsight capabilities below are documented capabilities, not independently reproduced results. No paid model calls, database changes, implementation changes, or merges were made for this audit.

Compared versions:

- Merged service: `main`, `efaf008`.
- Unmerged retrieval stack: `codex/retrieval-stack-20260927`, `a121dc7`.
- Unmerged write stack: `codex/write-path-stack-20260927`, `f988c26`. This branch includes the retrieval stack in its ancestry; do not attribute its entire diff to ingestion.

Definitions: **present** means an implemented counterpart, not identical semantics or measured quality; **partial** means material differences; **missing** means no corresponding service capability found in the inspected API, wiring, or modules. Internal classes, enum values, historical database counts, and unmerged work are not treated as shipped features.

## Comparison covering the supplied pages

| Hindsight documentation | Current merged service | Assessment |
|---|---|---|
| [Overview](https://hindsight.vectorize.io/) | Observation ingestion, recall, context building, graph queries, verification, and tool memory exist. There is no equivalent complete retain/recall/reflect product surface. | Partial architecture overlap. |
| [Retain](https://hindsight.vectorize.io/developer/retain) | Native sentence/clause extraction, optional ambiguity assistance, evidence, temporal fields, reinforcement, merge, and supersession exist. Contextual narrative extraction from the write stack is unmerged. | Partial. Hindsight documents contextual fact extraction, several extraction modes, entity resolution and multiple link types. Its `chunks` mode skips LLM extraction; its richer default path uses it. |
| [Retrieval](https://hindsight.vectorize.io/developer/retrieval) | Dense + BM25 fusion, exact matching, graph expansion, and bounded multi-hop support exist. Graph queries can use `as_of` and validity. | Partial. No equivalent dedicated temporal retrieval/ranking pipeline was found. Cross-encoder reranking is disabled in shipped configuration. Hindsight documents four retrieval arms, reranking and recency/temporal/proof-count adjustments. |
| [Recall API](https://hindsight.vectorize.io/developer/api/recall) | `/recall` and `/context` expose limits, kinds, document restrictions, scoped evidence and token-budgeted context. | Partial. No matching Boolean tag-filter API, tag-match modes, or low/mid/high exploration-budget contract. |
| [Reflect](https://hindsight.vectorize.io/developer/reflect), [Reflect API](https://hindsight.vectorize.io/developer/api/reflect) | `/context` prepares evidence; an optional supplied answer can be verified. Background `ReflectionService` writes insights. | **Missing query-time equivalent.** Hindsight uses an LLM loop to search mental models, observations and facts, expand evidence, and produce a cited answer. It also supports mission/directive/disposition controls. Our job named reflection does not implement this contract. |
| [Observations](https://hindsight.vectorize.io/developer/observations) | `/observations` accepts raw input. A bounded, opt-in background LLM reflection job exists. `LandingReflection` exists but is not passed into the merged observation pipeline. | **Partial consolidation, not parity.** Hindsight documents incrementally reconciled, scoped observations with source support, evidence quotes, proof counts and changing beliefs. Our input endpoint has a different meaning. |
| [Mental models](https://hindsight.vectorize.io/developer/mental-models), [Mental-model API](https://hindsight.vectorize.io/developer/api/mental-models) | No corresponding public resource or service found. Document summaries are not standing answers to user-selected questions. | **Missing:** saved query-driven knowledge with scheduled/consolidation-triggered refresh, incremental updates, provenance and history. |
| [Knowledge pages](https://hindsight.vectorize.io/developer/knowledge-pages), [Knowledge-page API](https://hindsight.vectorize.io/developer/api/knowledge-pages) | No knowledge-base tree, living page resource or page refresh service found. | **Missing.** Hindsight pages are a configured use of mental models, presented as a living Markdown knowledge base. Do not count this as an entirely separate reasoning algorithm. |
| [Multilingual](https://hindsight.vectorize.io/developer/multilingual) | Current frozen embedding is `ibm-granite/granite-embedding-small-english-r2`; native extraction contains English-specific rules. Unicode storage is not multilingual retrieval validation. | **Not established.** Hindsight documents language-preserving LLM processing and configurable multilingual embeddings/reranking/tokenization. Its default embedding and reranker are also English-only, so multilingual quality requires configuration. |
| [Performance](https://hindsight.vectorize.io/developer/performance) | CPU inference, bounded work, async ingestion, caches and no generative calls by default are supported. | **No comparative verdict.** Their documented typical recall range is 100–600 ms, not a p99 guarantee on our hardware. Reflect adds LLM reasoning latency. We did not reproduce either system's latency in this audit. |
| [Storage](https://hindsight.vectorize.io/developer/storage) | PostgreSQL canonical data, Qdrant search, Redis-compatible cache, blob storage and configured OpenFGA authorization. | Present persistence, different operations tradeoff. Hindsight primarily uses PostgreSQL with vector/full-text/graph data; Oracle is an optional alternative. Their consolidation into one main database may reduce operational dependencies. |
| [RAG vs Hindsight](https://hindsight.vectorize.io/developer/rag-vs-hindsight) | Document parsing, hierarchy, chunks, summaries, hybrid retrieval, graph context, expansions and evidence checking already exist. | Our document RAG is not the semantic-only baseline in that page. Conversely, document hierarchy does not provide Hindsight's persistent mental models. Document Q&A and conversational memory need separate evaluations. |
| [Quickstart](https://hindsight.vectorize.io/developer/api/quickstart) | HTTP/OpenAPI and a Python SDK are present. | Partial client/deployment overlap. Hindsight documents Python, TypeScript, Go, CLI and server/control-plane setup. We do not have equivalent client breadth. |
| [Memories API](https://hindsight.vectorize.io/developer/api/memories) | List/get/forget; evidence, temporal status and supersession information are returned. | Partial. No equivalent public fact-edit/re-embed, invalidation/restore and change-history workflow. Internal supersession is not an editable memory-management API. |
| [Memory banks](https://hindsight.vectorize.io/developer/api/memory-banks) | Tenant/user/thread/run/agent-group/private scopes and authorization exist. | Partial isolation counterpart; **missing bank management/configuration resource**, per-bank missions, directives and disposition administration. A Hindsight bank is not a drop-in replacement for our run-tree permissions. |
| [Documents](https://hindsight.vectorize.io/developer/api/documents) | File upload, document lookup, stored document versions/chunks and source citations exist. | Partial. No equivalent public bank-scoped document/chunk listing and deletion lifecycle. Our rich-document parsing is a separate capability worth retaining. |
| [Operations](https://hindsight.vectorize.io/developer/api/operations) | Durable outbox, job processing, retries/reconciliation and `GET /jobs/{job_id}` exist. | Partial. No matching public operations list/cancel/retry administration. Hindsight also has durable background processing; it is not an exclusive advantage of ours. |
| [Webhooks](https://hindsight.vectorize.io/developer/api/webhooks) | No outbound event-subscription/delivery API found. | **Missing.** Hindsight documents per-bank HTTP notifications, transactional delivery, retries and optional signing. Our internal outbox is not this feature. |
| [Bank templates](https://hindsight.vectorize.io/developer/api/bank-templates) | No declarative bank provisioning/export/import resource found. | **Missing.** Environment/deployment profiles are not templates bundling bank configuration, directives and mental models. |
| [List-memory API reference](https://hindsight.vectorize.io/api-reference#tag/Memory/operation/list_memories) | `/memories` accepts type, superseded flag, scope anchors and a limit; no cursor/offset argument. | Partial. Hindsight's current reference calls this “List memory units” and includes pagination, text search, document/entity/state/tag filters and time windows. |

Additional integration gap: Hindsight has a [built-in MCP server](https://hindsight.vectorize.io/developer/mcp-server). No native MCP server implementation was found here. Recording a tool call whose source is `mcp` does not implement an MCP server.

## Code evidence and important qualifications

Paths below are relative to this repository unless explicitly identified as another branch.

- `src/memory_service/adapters/wiring.py:425`: `_wire_memory` constructs `ObservationPipeline` without `landing`. It does construct `ReflectionService`.
- `src/memory_service/modules/memory/reflection.py:69`: merged reflection defaults to a 24-hour lookback, at most 1,000 scanned records, 40 input memories per group and three insights. `modules/jobs/registry.py:234` schedules it every six hours only when the `reflection` LLM use is enabled. This is bounded recent-memory synthesis, not a durable continuous-consolidation backlog or query-time reasoning loop.
- `src/memory_service/api/routers/v1/memory.py:195`: observations ingestion; `:241`: list; `:289`: get; `:304`: forget. The full route inventory contains no mental-model, knowledge-page, bank-template or webhook resources.
- `src/memory_service/api/routers/v1/retrieval.py:56`: recall request contract; `:234` and `:267`: recall and context routes.
- `src/memory_service/modules/graph/retrieval.py:177`: temporal `as_of`; `:298`: graph-linked memory expansion. It would be incorrect to say the graph never contributes memory candidates in current main.
- `src/memory_service/modules/graph/service.py:183`: canonical-name/alias matching plus opt-in LLM entity resolution. `modules/graph/document_facts.py` has additional document-specific alias extraction. Entity canonicalization is partial, **not absent**; the missing equivalence is Hindsight's contextual ingestion-time resolution pipeline.
- `src/memory_service/config/constants.py:47`: English dense model; `:87`: current BM25 sparse encoder, **not a shipped SPLADE default**; `:454`: retrieval configuration. `hybrid_rrf_k=1` and `rerank=False` are the inspected defaults. Neither a model download nor an unused class proves runtime use.
- `src/memory_service/config/settings.py:194`: LLM defaults disabled, with an empty allowed-use list. Optional mechanisms cannot be counted as universally active.
- `src/memory_service/modules/authz/visibility.py`: server-side audience filtering, including distinct run/downward and child-report/upward namespaces. `api/routers/v1/tools.py:213` onward exposes tool records, planning, procedures and run outcomes.
- `src/memory_service/api/routers/v1/files.py:111` and `:230`: file upload and document lookup. `modules/ingestion`, `modules/rag`, `modules/context` and `modules/retrieval` implement the richer document path.
- `sdk/python` is the client SDK found in this tree. Test suites cover memory, graph, retrieval, visibility, reflection and tool flows; their presence is not a claim that they were rerun or that they prove competitor parity.

The unmerged write stack adds selectively enabled contextual narrative extraction, landing consolidation, derived-memory dependencies and lifecycle handling, and durable reflection progress through migrations 0009–0011. Its wiring gates landing on `contextual_extraction`; it does **not** imply one paid LLM call for every message. It does not add the missing mental-model, knowledge-page, query-time reflect, webhook or bank-management APIs. Merging it would therefore not establish full parity. No accuracy gain is asserted for that stack here.

“Continuous learning” in this comparison means continuously updating external memories and derived knowledge. It is not evidence of training or fine-tuning model weights. We have reinforcement/merge, supersession and learned tool procedures; comprehensive incremental knowledge maintenance is the larger gap.

## Should we use Hindsight directly?

There is no demonstrated reason to categorically reject it. The [official repository](https://github.com/vectorize-io/hindsight) publishes an MIT-licensed, self-hostable implementation with multiple hosted/local model options. Choosing it does not require Gemini specifically or a managed-cloud deployment. Adopting it could avoid rebuilding a substantial knowledge-management surface.

The reasons to avoid an immediate wholesale replacement are concrete compatibility questions:

1. Preserve our authorization semantics through every retrieval and derived-memory path. Hindsight provides [tenant/authentication and operation-validation extensions](https://hindsight.vectorize.io/developer/extensions); it would be wrong to claim it has no authorization. Our run-tree visibility still requires an explicit mapping and isolation tests. Caller-selected tags alone are not that mapping.
2. Preserve tool-procedure mining, run outcomes, claim verification and document ingestion where applications use those contracts. They can potentially remain services around a Hindsight backend; that adapter is not built or proven today.
3. Measure actual ingestion/consolidation cost and retrieval tail latency under our workload. “No LLM during recall” does not mean no LLM during ingestion, consolidation, reflection or benchmark answer generation. The no-extraction chunks mode should not be assumed to retain the quality of structured memory mode.
4. Verify answer quality under the same reader, judge, evidence budget and adversarial policy. A vendor headline is neither our measured accuracy nor a promise of 90%+ on our workload.

Recommendation: evaluate Hindsight as the conversational memory backend before funding another broad attempt at feature parity. Keep the current service baseline and isolate any evaluation database. Compare no-generative-query recall and agentic reflect separately, with ingestion settings recorded. Measure answer accuracy, multi-hop/list completeness, abstention, source correctness, p50/p95/p99 at stated concurrency, ingestion cost, deletion freshness and cross-thread/run isolation. Include general workloads beyond LoCoMo. Keep document RAG evaluation separate. Adoption should follow those results, not the amount of code already written.

## Verification record and handoff

All 23 supplied URLs were inspected. The quickstart failed repeatedly through the web reader but was successfully fetched directly with Python's standard HTTP client. The current list-memory endpoint was found under “List memory units” in the API reference despite the supplied fragment's older operation spelling.

The repository-required `browse` skill was attempted. Aside was absent; the bundled browser lacked its Chromium executable, and Playwright reported that its headless-shell package does not support macOS 12. Public pages were consequently read with the available web reader/direct HTTP fallback. This is a documentation-access limitation, not a tested Hindsight application failure.

Next AI: preserve the version distinctions above; do not convert documented capabilities into measured superiority, do not count unmerged code as production, and do not migrate a shared baseline database to run an adoption comparison. This audit authorizes no deployment and records no new benchmark result.
