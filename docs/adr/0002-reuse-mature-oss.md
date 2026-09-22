# ADR 0002: Reuse mature OSS for infrastructure primitives

**Status:** accepted · **Date:** 2026-09-14

## Decision

> **Outcome, 2026-09-22.** Every row below was built except **OPA**. There was never a
> policy adapter in `src/`: `PolicySettings` was declared and read in exactly one place —
> `/version`, which reported the configured provider — while `container.policy` was never
> assigned and the `ProviderRegistry("policy")` never had anything registered in it. The
> compose service mounted `./deploy/opa`, a directory that does not exist, so it would have
> served no policies had anyone started it. Settings, registry slot, container field,
> `/version` key and compose service are all removed; this row stays to record that the
> decision was taken and not carried out.
| Problem | Chosen OSS | Version validated (PyPI, 2026-09-14) | License |
|---|---|---|---|
| Task queue with retries/locks/queues | Procrastinate | 3.9.0 | MIT |
| Fine-grained authorization | OpenFGA (+ `openfga-sdk`) | server v1.18.1 / sdk 0.10.4 | Apache-2.0 |
| Policy decisions | OPA (optional) — ~~chosen~~ **never built, dropped 2026-09-22** | image `openpolicyagent/opa:1` | Apache-2.0 |
| Vector + BM25 + RRF search | Qdrant (+ `qdrant-client`) | server v1.18.2 / client 1.19.0 | Apache-2.0 |
| Cache / working memory | Dragonfly (Redis protocol, `redis` client) | v1.40.1 / redis 8.1.0 | BSL-1.1 (approved for this deployment) |
| Document parsing | Docling | 2.127.0 | MIT |
| Document KG extraction | Docling Graph | 1.9.1 | MIT |
| Temporal KG enrichment | Graphiti (optional) | graphiti-core 0.30.2 | Apache-2.0 |
| Memory intelligence challengers | Mem0 / Cognee / LangMem | 2.0.20 / 1.5.4 / 0.0.30 | Apache-2.0 / Apache-2.0 / MIT |
| Evaluation | DeepEval (+ RAGAS optional) | 4.2.3 / 0.4.3 | Apache-2.0 |
| Lineage / tracing | OpenLineage / OpenTelemetry | 1.53.0 / 1.44.0 | Apache-2.0 |
| CPU embeddings | sentence-transformers / fastembed / onnxruntime / optimum | 6.0.1 / 0.8.0 / 1.30.0 / 2.3.0 | Apache-2.0 / Apache-2.0 / MIT / Apache-2.0 |

## Consequences
We own only the memory contract, scope/visibility semantics, execution context, evidence
provenance, Document Context Graph, QueryRouter, ContextBuilder, benchmark gates and the
archive policy. Everything else sits behind a port.

## Notes
- `cognee` pins `structlog<26`; the core therefore requires `structlog>=25.2` rather than 26.
- Heavy extras (`docling`, `models`, `cognee`, `graphiti`) are optional so the core
  installs in seconds.
