# Milestones

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
| M10 | advanced retrieval benchmarks (gated) | pending |
| M11 | multi-agent semantics | pending |
| M12 | LangGraph adapter | pending |
| M13 | full eval / load / chaos / hardening / final report | pending |
