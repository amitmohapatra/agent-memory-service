# Milestones

| Milestone | Scope | Status |
|---|---|---|
| M0 | repo, uv/venv, FastAPI + Swagger, domain contracts, ports, config, health/version, SDK skeleton | done (41 tests) |
| M1 | PostgreSQL, Alembic, repositories, Unit of Work, Procrastinate, idempotency, OTel | done (62 tests; real PostgreSQL 16 + Procrastinate 3.9) |
| M2 | execution context, OpenFGA, scope/visibility, tenant isolation tests | done (105 tests total; OpenFGA contract test docker-gated) |
| M3 | thread/session/turn/message, lineage, hot thread cache, SDK chat | done (114 tests; e2e via SDK) |
| M4 | GCS BlobStore, archive segments, compression/checksums, compaction, reconciler | done (filesystem+memory+GCS adapters; failure-injection tests; bench-storage) |
| M5 | Docling, hierarchy, Context Graph, natural chunks, contextual representations | done (147 tests; Docling DOCX verified offline, PDF needs HF models) |
| M6 | Qdrant, BM25, dense, RRF, reranker, ContextBuilder, benchmark | pending |
| M7 | native memory intelligence, dedup/consolidation/temporal, Mem0/Cognee/LangMem adapters | pending |
| M8 | GraphStore, native enrichment, Docling Graph, Graphiti, temporal facts, multi-hop | pending |
| M9 | expansion, hierarchical summaries, evidence-group verification, multi-hop suite | pending |
| M10 | advanced retrieval benchmarks (gated) | pending |
| M11 | multi-agent semantics | pending |
| M12 | LangGraph adapter | pending |
| M13 | full eval / load / chaos / hardening / final report | pending |
