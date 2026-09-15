# Milestones

> **Status: historical build log.** This records what each milestone delivered, in order,
> and is useful for understanding how the service came to be shaped this way. It is not a
> description of the current API — for that, see the [README](../README.md) — and the test
> counts are those at the time each milestone closed.

| Milestone | Scope | Status |
|---|---|---|
| M0 | repo, uv/venv, FastAPI + Swagger, domain contracts, ports, config, health/version, SDK skeleton | done (41 tests) |
| M1 | PostgreSQL, Alembic, repositories, Unit of Work, Procrastinate, idempotency, OTel | done (62 tests; real PostgreSQL 16 + Procrastinate 3.9) |
| M2 | execution context, OpenFGA, scope/visibility, tenant isolation tests | done (105 tests total; OpenFGA contract test docker-gated) |
| M3 | thread/session/turn/message, lineage, hot thread cache, SDK chat | done (114 tests; e2e via SDK) |
| M4 | GCS BlobStore, archive segments, compression/checksums, compaction, reconciler | done (filesystem+memory+GCS adapters; failure-injection tests; bench-storage) |
| M5 | Docling, hierarchy, Context Graph, natural chunks, contextual representations | done (147 tests; Docling DOCX verified offline, PDF needs HF models) |
| M6 | Qdrant, BM25, dense, RRF, reranker, ContextBuilder, benchmark | done (180 tests; critical Recall@20 = 1.00 and EGR = 1.00 on the golden set with hash embeddings — non-representative until real weights are benchmarked; retrieval isolation gate = 0 leaks) |
| M7 | native memory intelligence, dedup/consolidation/temporal, Mem0/Cognee/LangMem adapters | done (memories table, native extract/classify/consolidate, ObservationPipeline, memory index + exact lookup, /v1/observations /v1/memories, SDK observe/remember/get/forget, Mem0/LangMem/Cognee adapters; false-merge rate 0.00 on 35 labelled pairs) |
| M8 | GraphStore, native enrichment, Docling Graph, Graphiti, temporal facts, multi-hop | done (graph_entities/graph_relations in PostgreSQL, native enrichment from memories + documents, bounded temporal traversal with store-side visibility, GraphStage multi-hop evidence expansion, /v1/graph/query + SDK, Graphiti/Docling Graph adapters; 0 leaks in the graph isolation gate) |
| M9 | expansion, hierarchical summaries, evidence-group verification, multi-hop suite | done (ExpansionStage over context_edges, extractive hierarchical summaries indexed as kind=summary, rolling conversation summary, VerificationStage with required evidence groups + escalation + abstention, budget-aware evidence report, near-duplicate collapse; critical_evidence_complete_rate 1.00) |
| M10 | advanced retrieval benchmarks (gated) | done (extra-retriever fusion; PageIndex/RAPTOR/graph-PPR working model-free; SPLADE/miniCOIL/ColBERT/late-chunking adapters + Qdrant multivectors, loud failure without weights; benchmark/advanced.py with adoptable/rejected/skipped verdicts) |
| M11 | multi-agent semantics | done (Visibility.RUN lineage — hand-off context flows down one hop, never up or sideways; explicit sharing by hint; cross-agent corroboration counted via contributors; cross-principal conflicts kept as CONTRADICT with evidence notes; agent self-facts never become USER memories; isolation gates 0 leaks incl. run dimension; ADR 0013) |
| M12 | LangGraph adapter | done (`LangGraphMemory.wrap`: context from thread + checkpoint namespace, subgraphs = agent runs with stable task-derived run ids, bundle before / messages + observations after nodes, deterministic idempotency keys across checkpoint retries, evidence gating, writes never silent; real-graph tests; ADR 0014) |
| M13 | full eval / load / chaos / hardening / final report | done (gate producers: durability chaos run, security/failure-injection reducers, performance p95, tests.json plugin; failure suite with a real SIGKILLed worker, cache/blob/index/authz faults; reindex tool; per-tenant rate limit; two durability bugs fixed with regression tests; `make gates` + `make validate` -> RELEASE GATE: PASS with representativeness caveats; ADR 0015; docs/FINAL_REPORT.md) |

## Post-M13 verification pass (2026-09-15)

- Ingestion fidelity test (every line in exactly one chunk with the right page).
- Document knowledge graph rebuilt: typed entities, aliases, factual relations with attributes; KG gate (49 golden facts, 0 false, 0 noise, 8/8 questions) added to the release gate. ADR 0016.
- Retrieval golden set extended to 18 questions; R@20 = EGR = 1.00 on the gate corpus and on the 25-copy crowded corpus (graph facts de-duplicated per triple, relation-shaped questions routed to the graph, memories interleaved with document results).
- SDK completed (`memories()`, `chat.create/message/delete_thread`, `files.document/wait_ready`, `alive()`, `files.add(title=, visibility=)`, fact `attributes`, entity `aliases`); route-coverage test.
- `examples/`: live-server SDK tour (14/14) and LangGraph crew, both run over HTTP; four defects found by the tour fixed (hint precedence, idempotent forget, memory crowding, `valid_from` on supersede).
