# Prompt: real-component validation of the Memory Service — keep only what proves itself

You are working in the repository at the current directory (`agent-memory-service`): an
enterprise multi-agent memory service (FastAPI, `src/memory_service`), the Python SDK
`universal-memory` (`sdk/python`), the LangGraph adapter (`integrations/langgraph`), release
gates (`benchmark/`, `tests/`), and examples (`examples/`). The authoritative spec is
`MASTER_MEMORY_SERVICE_IMPLEMENTATION_INSTRUCTION.md`; read `docs/FINAL_REPORT.md`,
`docs/MILESTONES.md`, `ARCHITECTURE.md` and `docs/adr/` first.

Everything in the repo was built and gated in a sandbox WITHOUT model weights, without real
Qdrant/Dragonfly/OpenFGA servers, and without any LLM. Every retrieval-quality number
currently in `benchmark/results/` is therefore NOT representative. Your job is to re-run the
whole validation with real components, measure every optional strategy and provider against
its baseline, and then delete everything that does not earn its place. The result must be a
smaller, fully tested repository whose numbers are real.

**Read `docs/TARGET_STACK.md` first.** It fixes the defaults to adopt (Granite R2 + ONNX int8,
bge-reranker-v2-m3, BM25 sparse, ColBERT-small multivector stage, GLiNER2 extraction tier,
multi-layer temporal graph, observational memory + beliefs/entity summaries, the grounding
cascade with DeBERTa NLI, public benchmarks) and lists 25 numbered changes. Implement them in
that order inside the phases below, each one gated by measurement exactly like everything else.

Also read `docs/INTEGRATIONS_PLAN.md` (changes 26–29): after the retrieval/memory work, add the
Google ADK adapter (`integrations/adk`), the CrewAI adapter (`integrations/crewai`), the MCP
server (`integrations/mcp`) and the cross-adapter conformance suite, each with an example under
`examples/` that passes against the real stack, mirroring the LangGraph adapter.

## Hard rules

1. **All LLM calls go through Bifrost, and only Bifrost.** Bifrost (https://github.com/maximhq/bifrost)
   is an external LLM gateway; it is NOT part of the memory service and is never vendored or
   bundled into it. The service talks to Bifrost's OpenAI-compatible HTTP endpoint through one
   adapter behind the existing `LLMProvider` port (`src/memory_service/ports/models.py`). No
   provider SDK (`openai`, `anthropic`, `google-*`, `litellm`, …) may be imported anywhere under
   `src/` — add a ruff `banned-api` rule for them, like the existing `langgraph` ban. Third-party
   providers that need an LLM (Mem0, LangMem, Cognee, Graphiti) must be pointed at Bifrost via
   their OpenAI-compatible base-URL setting, never at a provider directly.
2. **Secrets never enter the repo or the transcript.** Provider keys live only in Bifrost's own
   config; the service only holds a Bifrost virtual key, read from `secrets.env` (git-ignored)
   or the environment `MEMORY__MODELS__LLM__API_KEY`. Add `secrets.env`, `models/`, `vendor/`
   and `.bifrost/` to `.gitignore` before anything else. Never print a key in logs or output.
3. **Hard release gates are unchanged** and must all pass before the final commit:
   acknowledged data loss = 0, unauthorized retrieval = 0, critical Recall@20 = 1.00,
   critical Evidence-Group Recall = 1.00, KG fact recall = 1.00 with 0 false facts,
   all tests pass, false-merge rate and p95 latency within the configured budgets,
   failure-recovery tests pass. Never claim production readiness while any gate fails.
4. **Commits are authored by the repository owner with no AI attribution** — no
   `Co-Authored-By`, no `Generated with`, no tool or session trailers of any kind, and no
   AI-tool config files (`.claude/`, `CLAUDE.md`, `.cursor/`, …) committed. Never create a
   nested `agent-memory-service/` directory inside the repo.
5. Work milestone by milestone, run the relevant tests after every change, keep `ruff` and
   `pyright` clean (`make lint typecheck`), and keep the public API and SDK backwards
   compatible unless a removal below requires a documented change.

## Phase 0 — environment

- `make setup` (uv, all extras). Start the real stack with `make dev-up` (`docker-compose.yml`:
  PostgreSQL 16, Qdrant, Dragonfly, OpenFGA, api, worker); run `make migrate`. Confirm the
  service uses the real servers (`MEMORY__SEARCH__PROVIDER=qdrant` with `QDRANT_URL`,
  `MEMORY__CACHE__PROVIDER=dragonfly`, `MEMORY__AUTHORIZATION__PROVIDER=openfga` with a
  written store/model id, `MEMORY__TASKS__PROVIDER=procrastinate`), not the embedded fallbacks.
- Download the model weights into `models/` (git-ignored) and point the settings at them:
  `ibm-granite/granite-embedding-english-r2` (also the `small` variant for a size/quality
  comparison) → `MEMORY__MODELS__EMBEDDING__MODEL_PATH`; `cross-encoder/ms-marco-MiniLM-L6-v2`
  → `MEMORY__MODELS__RERANKER__MODEL_PATH`; `prithivida/Splade_PP_en_v1` →
  `MEMORY__MODELS__SPARSE_MODEL_PATH`; `answerdotai/answerai-colbert-small-v1` →
  `MEMORY__MODELS__LATE_INTERACTION_MODEL_PATH`; Docling's PDF models for the `docling` parser.
  Verify each loads with `pytest -m models`.
- Install and run Bifrost **outside** the repo (its own directory or container; e.g.
  `npx @maximhq/bifrost` or the Docker image) on a port that does not collide with the
  service (`memory-api` uses 8080; run Bifrost on 8090). Configure the provider key inside
  Bifrost only, create a virtual key, enable at least one strong model and one cheap model,
  and smoke-test `POST /v1/chat/completions` with curl. Record the model names used.

## Phase 1 — Bifrost adapter (the only LLM path)

- Implement `src/memory_service/adapters/models/llm.py`: `BifrostLLM(LLMProvider)` using the
  project's existing HTTP client (httpx), `base_url` + virtual key + model from `LLMSettings`
  (add `"bifrost"` to the provider literal; remove `"openai"`/`"vertex"`/`"local"` from it —
  those would bypass the gateway). `complete()` maps to chat completions; `structured()` uses
  JSON-schema response formatting with validation and one bounded retry; timeouts, retries and
  circuit breaking are bounded; every call is traced (OTel span with model, tokens, latency)
  and metered (Prometheus counter/histogram); usage is logged without prompt text unless
  `log_source_text=true`. `ProviderNotConfigured` when disabled.
- Wire it in `adapters/wiring.py`; the service must still start and pass every gate with
  `MEMORY__MODELS__LLM__ENABLED=false`. Contract tests with a mocked Bifrost (respx) plus one
  `-m models`-style live test that hits the running Bifrost.
- Make every existing "requires_llm" path (`LLMSettings.uses`: ambiguous_extraction,
  ambiguous_worthiness, relation_extraction, entity_resolution, conflict_adjudication,
  summaries, reflection, query_expansion, chunk_context) actually call the port, each behind
  its own `uses` flag, each falling back to the native deterministic path on failure, and
  each covered by a test with the mocked gateway.

## Phase 2 — real weights, real gates

- With real embedding + reranker: `make gates` and `make validate`. Compare every number in
  `benchmark/results/*.json` with the sandbox run (keep the old files under
  `benchmark/results/baseline-sandbox/` for the report). The retrieval gate
  (`tests/eval/golden/acme_fy26.json`, 18 questions), KG gate (`tests/eval/golden/kg_facts.json`),
  memory gate and the 25-copy crowded benchmark (`benchmark/retrieval.py`) must reach the
  critical thresholds with real weights. If a critical question fails, fix retrieval —
  do not edit the golden set to pass.
- `make bench-embedding` and `make bench-reranker`: Granite R2 vs small R2 (and, as challengers,
  `google/embeddinggemma-300m` and `Qwen/Qwen3-Embedding-0.6B` if multilingual matters),
  sentence-transformers vs ONNX/OpenVINO runtimes; rerankers `cross-encoder/ms-marco-MiniLM-L6-v2`
  (current default, ~60% BEIR nDCG@10) vs `BAAI/bge-reranker-v2-m3` (~71.5% BEIR, ~12 ms/pair)
  vs `ibm-granite/granite-embedding-reranker-english-r2`, candidate_k 15/20/25 — pick the
  defaults by measured quality-per-millisecond on CPU within the p95 budget and write them
  into `settings.py` and `.env.example`.
- Public benchmarks, so the numbers are comparable with the field and not only with our own
  fixtures: add `benchmark/public.py` that runs (a) a BEIR subset (nfcorpus, scifact, fiqa)
  through the real retrieval engine and reports nDCG@10 / Recall@20 for baseline vs each
  advanced strategy, and (b) **LongMemEval** and **LoCoMo** through the memory pipeline
  (observe the conversations, answer the questions from `/v1/context` with the LLM via
  Bifrost, score with the benchmarks' own graders). Report native-only, native+Bifrost uses,
  and each third-party provider on the same table. These are the benchmarks the memory
  frameworks (Mem0, Zep/Graphiti, Letta, Mastra) publish against; without them no
  "state of the art" claim is allowed in the docs.
- Docling parser on real PDFs (add 3–5 real-world PDFs with tables, footnotes and multi-column
  layout to `tests/fixtures/`): `tests/integration/test_ingestion_fidelity.py` must hold
  (every line/row in exactly one chunk, tables intact, section paths correct). Extend the
  golden set with questions over those PDFs.

## Phase 3 — real infrastructure

- `pytest -m docker` (OpenFGA, Qdrant server) and the full `tests/security` isolation suite
  against real OpenFGA — unauthorized retrieval must be 0 with the real authorizer.
- `make failure-test` with real Procrastinate workers, Dragonfly and Qdrant server: worker
  SIGKILL mid-job, cache flush, blob outage, index rebuild (`tools.reindex`), authz outage —
  the durability benchmark (`benchmark/durability.py`) must report 0 acknowledged loss over
  the network, not in-process.
- `make load-test` (Locust) against the deployed api + worker; compare with the in-process
  `performance.json`; the p95 budgets in settings must be met by the deployed instance, or
  be re-justified in the report with the measured network/server hop.
- Run `make examples` (`examples/run_server.sh`, `examples/sdk_tour.py` 14/14,
  `examples/langgraph_crew/app.py`) against the real stack.

## Phase 4 — advanced retrieval: measure, then decide

`make bench-advanced` (`benchmark/advanced.py`) with real weights for every strategy:
PageIndex (summary-routed tree search), RAPTOR-style summary fusion, graph personalised
PageRank, SPLADE, miniCOIL, ColBERT (Qdrant multivector), late chunking. For each, record
Recall@20 / EGR on the golden and crowded sets, p95 latency, index size and ingest cost
versus the hybrid BM25+dense+rerank baseline. Also test them with LLM-assisted query
expansion through Bifrost, since some only pay off with it.

## Phase 5 — providers: native vs. third-party, all through Bifrost

- Memory intelligence: native vs Mem0 vs LangMem vs Cognee (`MEMORY__MEMORY__PROVIDER`), each
  configured to use Bifrost as its LLM. Measure on `benchmark/memory.py` and the memory gate:
  extraction precision/recall, false-merge rate on the 35 pairs, consolidation correctness,
  temporal supersession, latency and LLM token cost per observation.
- Graph enrichment: native vs Graphiti vs Docling Graph vs Cognee
  (`MEMORY__GRAPH_ENRICHMENT__PROVIDER`) on the KG gate: fact recall, false facts, noise
  entities, alias resolution, latency and cost.
- Native + Bifrost uses (Phase 1): for each `uses` flag, measure the lift over the pure
  deterministic path and its cost; keep a flag only if it improves a gate metric without
  breaking a critical one.
- GraphRAG community summaries through Bifrost: same treatment.

## Phase 6 — keep or cut

Decision rule: a component survives only if, with real weights and real services, it beats
the baseline it competes with on the benchmarks above (Recall@20, evidence-group recall, KG
fact recall, false-merge rate) without exceeding the p95 budget, and at least ties on every
critical gate. Everything else is removed completely — code, adapters, config literals,
settings, extras in `pyproject.toml`, `uv.lock` entries, docker-compose services, docs and
README mentions — not left behind a flag. For each decision write one row in a table: name,
baseline, metric deltas, latency delta, cost, verdict, reason. Put the table and the raw
JSON in `benchmark/results/` and the decisions in `docs/adr/0017-real-component-validation.md`.
After pruning: `make validate` must pass end to end, `make examples` must pass, and
`docs/FINAL_REPORT.md`, `README.md`, `ARCHITECTURE.md` and `.env.example` must describe
only what remains.

## What to report at the end

- The gate table (every hard gate, value, threshold, pass/fail) from the real run.
- The keep/cut table with numbers.
- Exact versions: models (name + revision hash), Bifrost version and models, Qdrant, Dragonfly,
  OpenFGA, PostgreSQL, Docling.
- Anything that still is not representative (say so plainly) and what would make it so.
- The list of commits, each authored by the repository owner, with no AI attribution.
