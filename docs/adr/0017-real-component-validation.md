# ADR 0017: Real-component validation — what proved itself, what was cut

**Status:** in progress · **Date:** 2026-09-15

## Context
Every number produced up to commit `465375f` came from a sandbox without model weights,
without Qdrant/Dragonfly/OpenFGA servers and without any LLM (ADR 0003, ADR 0015). The
target stack (`docs/TARGET_STACK.md`), the integrations plan and the tool-memory design
add 30 numbered changes, each of which must be measured with real components against the
baseline it replaces. This ADR records the environment, the decisions that do not depend on
measurement, and — as the runs complete — one row per keep/cut decision with numbers.

## Decisions that do not depend on measurement

- **One LLM path.** `LLMSettings.provider` is `disabled | bifrost`; `adapters/models/llm.py`
  is the only code that talks to a generative model, over Bifrost's OpenAI-compatible HTTP
  API with a virtual key from `secrets.env`. Provider SDK imports are banned under `src/`
  (Ruff `banned-api` + `tests/unit/test_architecture.py`). Mem0, LangMem, Graphiti and
  Cognee are pointed at the gateway through their OpenAI-compatible settings.
- **Claude behind the gateway, complexity-tiered.** Defaults: `anthropic/claude-sonnet-5`
  for complex uses (conflict adjudication, relation extraction on weak edges, reflection,
  grounding judge, entity summaries) and `anthropic/claude-haiku-4-5-20251001` for
  classification-sized uses (`fast_uses`: worthiness, query expansion, chunk context). The
  deterministic path is always complete on its own; `LLMAssist` consults the model only on
  an explicit ambiguity signal and falls back natively on any failure.
- **Linux is the validation and production target.** The host (macOS 12, Intel) cannot
  run `transformers` ≥ 5 or Docling; the gates run in the `memory-validate` compose
  service (`deploy/Dockerfile` target `validation`, repository bind-mounted, weights at
  `/models`). Linux resolves CPU-only PyTorch (`tool.uv.index pytorch-cpu`).
- **Test fixtures honour real providers only on request.** `MEMORY_TEST_PROVIDERS=env`
  merges the environment's models/search/cache/authorization/documents/retrieval settings
  into the hermetic test settings; fixtures then reset Qdrant collections and the cache
  between tests (`tests/support_real.py`). Default runs stay hermetic.
- **Sandbox artifacts are kept for comparison** under `benchmark/results/baseline-sandbox/`.
- **Golden sets are never edited to pass.** New questions were added (`public_pdfs.json`
  over four real PDFs); none were removed or weakened.

## Environment (recorded, not claimed)

| Component | Version |
|---|---|
| PostgreSQL | 16 (postgres:16-alpine) |
| Qdrant | v1.18.2 |
| Dragonfly | v1.40.1 |
| OpenFGA | v1.18.1 |
| Bifrost | v2.1.1 (`npx @maximhq/bifrost -port 8090`, outside the repository) |
| Python / uv | 3.12.14 / 0.12.14 |
| Docling | see `benchmark/results/*.json` → `provenance.packages.docling` |

Model weights (`models/MANIFEST.json`, git-ignored directory):

| Role | Model | Revision |
|---|---|---|
| Embedding (default candidate) | ibm-granite/granite-embedding-english-r2 | 47ea694b257b703fee9253d75c2b1f2985180498 |
| Embedding (low-latency tier) | ibm-granite/granite-embedding-small-english-r2 | 2ab6fa8ea2d674564defd37171ae19079b864b33 |
| Embedding challenger | Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |
| Reranker (current default) | cross-encoder/ms-marco-MiniLM-L6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a |
| Reranker (target default) | BAAI/bge-reranker-v2-m3 | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| Reranker (CPU-light) | ibm-granite/granite-embedding-reranker-english-r2 | d09d3d6971b689bf9c23839e45a470874d46e13a |
| Sparse (SPLADE) | Qdrant/Splade_PP_en_v1 (fastembed's own export) | latest |
| Late interaction | answerdotai/answerai-colbert-small-v1 | 934fa8bb4ce2284f4c2baa232d81aca4d076fa5e | *(removed 2026-09-20, see ADR 0012)* |
| NLI (grounding) | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 6f5cf0a2b59cabb106aca4c287eed12e357e90eb |
| Extraction tier | fastino/gliner2-base-v1 | 8437ba583a733d87f56ae902f3b197934eedd58e |

Not available: `google/embeddinggemma-300m` (gated repository), `knowledgator/gliner-relex`
(repository not found under that name) — recorded as untested, not as losers.

## Decision rule

A component survives only if, with real weights and real services, it beats the baseline it
replaces on Recall@20, evidence-group recall, KG fact recall, per-claim grounding rate or
false-merge rate, within the p95 budget of its tier, and ties or better on every critical
gate. Losers are removed from code, configuration, extras, lock file, compose and docs.

## What still has to be measured

Lifted from the `VALIDATION_PROMPT*.md` series when those five files were removed on
2026-09-22 — 876 lines of which 843 were duplicated text, addressed to an agent rather than
a reader, and naming an authoritative spec (`MASTER_MEMORY_SERVICE_IMPLEMENTATION_INSTRUCTION.md`)
that has never existed in this repository. The rules they stated are recorded above and in
../CONTRIBUTING.md; this ordering was the one thing they held that was not written down
anywhere else.

The tables below stay empty until each step is run with real weights and real services.

1. **Real weights, real gates.** `make gates` and `make validate` with a real embedding
   model and reranker; compare every number against the sandbox column. Then
   `make bench-embedding` and `make bench-reranker` to settle the defaults, and the public
   retrieval benchmarks so the numbers are comparable with the field and not only with
   ourselves. Docling against real PDFs with tables, footnotes and multi-column layout.
2. **Real infrastructure.** `pytest -m docker` against a real OpenFGA and Qdrant server,
   plus the whole `tests/security` isolation suite; `make failure-test` with real
   Procrastinate workers, Dragonfly and a Qdrant server; `make load-test` against a
   deployed api and worker, compared with the in-process figures; `make examples`.
3. **Advanced retrieval.** Measure before deciding — seven flags have already been removed
   this way (see `docs/CAPABILITY_COVERAGE.md` and ADR 0012).
4. **Providers, native versus third-party, all through Bifrost.** Memory intelligence
   (native / Mem0 / LangMem / Cognee), graph enrichment (native / Graphiti / Docling Graph /
   Cognee), and each `models.llm.uses` flag measured as a lift over the deterministic path
   it replaces.

The decision rule above then applies: anything that does not beat what it replaces is
removed from code, configuration, extras, lock file, compose and docs.

## Gate table (real run)

_Filled by the validation run; see `docs/FINAL_REPORT.md`._

| Gate | Threshold | Sandbox value | Real value | Result |
|---|---|---|---|---|
| Acknowledged data loss (in-process chaos) | 0 | 0 | | |
| Acknowledged data loss (network, real workers) | 0 | not measured | | |
| Unauthorized retrieval (real OpenFGA) | 0 / 0 / 0 | 0 / 0 / 0 (in-memory authz) | | |
| Critical Recall@20 (acme_fy26) | 1.00 | 1.00 (hash embedding) | | |
| Critical Evidence-Group Recall | 1.00 | 1.00 (hash embedding) | | |
| Critical Recall@20 / EGR (public PDFs, Docling) | 1.00 / 1.00 | not measured | | |
| KG fact recall / false facts / noise | 1.00 / 0 / 0 | 1.00 / 0 / 0 | | |
| False-merge rate | ≤ 0.01 | 0.00 | | |
| p95 budgets (in-process) | per settings | pass | | |
| p95 budgets (deployed, Locust) | per settings | not measured | | |
| Failure recovery | all pass | pass | | |
| Grounding (per-claim) | see gate file | not measured | | |
| All tests | 0 failed | 281 passed | | |

## Keep / cut table

_One row per component; filled from `benchmark/results/`._

| Component | Baseline | Quality delta | Latency delta | Cost | Verdict | Reason |
|---|---|---|---|---|---|---|
