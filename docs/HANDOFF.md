# Handoff: where the memory-service programme stands

Written 2026-09-23 00:20 IST at commit `cb60bc8`, updated 00:45 at `7836657` so that whoever continues — a person or an
agent, after a model change or a fresh session — can pick up without the conversation
history. The plan of record is [ROADMAP-2026-09.md](ROADMAP-2026-09.md), the ranked list of
techniques worth borrowing is [GAPS-2026-09.md](GAPS-2026-09.md), and this file is the
state of execution against them. Update it whenever a phase step lands or a decision changes.

## Targets and how they are read

| Target | Reading | Status |
|---|---|---|
| Accuracy | best-in-class on the standard LoCoMo ruler (1,540 q, cat 5 excluded, Mem0's judge prompt, GPT-4.1-mini-class answerer+judge) with **zero LLM calls at query time**; strict and LoCoMo-Refined reported alongside. "> 94 %" is unreachable under any strict/human-aligned judge (answer-key noise caps at ~93.6; Mnemis 93.9 spends ~2.4 s of LLM per query) | 2-conversation set at HEAD: **0.833 strict / 0.893 lenient / 0.631 refined** (v5) |
| Retrieval latency | HTTP `POST /v1/context`, cache-miss arm, from a separate host, 20 rps for 5 min, remote Postgres/Qdrant/Dragonfly, on the 8 vCPU / 16 GB VM: **p99 < 300 ms** | not measurable on the 4-core no-AVX2 dev VM (encoder alone 178–360 ms); waits for the VM |
| Throughput | **20 rps** on that VM | never attempted with real models |
| Multilingual | encoder + sparse tokeniser + NLI + extraction regexes; gated on SciFact not dropping > 2 nDCG | Phase 3, not started |
| One package | models frozen in `config/constants.py`, env = URLs/credentials/topology only, `docker compose up` with no network | settings 206 → 39 done; image baking + compose rewrite pending (Phase 1 steps 10–11) |
| API | every closed set an enum, every free-form field bounded | done |
| "No Chinese mode" | **withdrawn by the user 2026-09-22** — no drift guard, no language pin, no provenance rule | — |

## Measured state (benchmark/results)

- `locomo_judged_v5.json` (+`_lenient`, `_refined`): HEAD code, 2 conversations, 304 q, deepseek-flash answerer and judge, depth 100, USES=grounding_judge only. Strict 0.833 (single 0.860, multi-hop 0.605, temporal 0.952, open 0.769; adversarial abstention 0.859), evidence recall 0.991 with every gold turn present, 39 wrong answerable answers of which 38 had the evidence in the bundle. Latency in this file is contaminated (agents shared the cores).
- `locomo_judged_v2.json`: the previous baseline (0.674 / 0.755). v3 = ingest LLM assists on (0.661, harmful). v4 was invalid (questioner not a workspace member — fixed in `7dce658`).
- Retrieval stage split on a quiet dev box: encoder 178–320 ms (torch fp32, no AVX2), Qdrant hybrid 40–90 ms, scope 35 ms, graph prefetched under the encoder, verify ~7 ms.
- Published comparators (same-harness reproductions, cat 5 excluded): Mem0 paper 66.9 / 68.4 (gpt-4o-mini), Zep 75.1, Letta 74.0, Omi 86.6, Hindsight 89.6, Mnemis 89.1 zero-LLM / 93.9 with System-2. Vendor 92.5 (Mem0) uses a GPT-5-class answerer+judge at top-200.

## Done (all on main)

- Batch 1–3 retrieval/answerer work: depth 100, answer prompt without status-abstention, dated chronological rendering, encoder before I/O, concurrent per-kind searches, GraphStage prefetch, access bump off the request path, `current` BOOL payload index, routing tightened (multi-hop 113 → 31 of 304), adapter repair round after the envelope fallback, harness retries on store hiccups and keeps partial results.
- **Phase 1**: observer, challengers (mem0/langmem/cognee/graphiti), served-model tier + litellm gateway, YAML source, dead Literals and ~140 dead functions removed; `config/constants.py` holds the frozen models and tuning; `Settings` = 39 env fields with SecretStr credentials; test/benchmark stand-ins go through `build_container(overrides=Overrides(...))` / `create_app(settings, overrides=...)`; `benchmark/env.py` (BenchEnv) replaces the Makefile's repeated env blocks; API enums + bounds + 422 mapping; OpenAPI and SDK regenerated. Every suite green: unit 403, contract 92, integration 144, e2e 24, eval 12, failure 6, security 10. Pyright: 0 errors (`7836657`).

## Running or pending when this was written

- ~~Technique-mining workflow~~ done: [GAPS-2026-09.md](GAPS-2026-09.md) holds the gap table, the ranked borrow list and the do-not list. Items 5-12 there are Phase 3/4 work.
- `python -m benchmark.failure_taxonomy <judged>.json` classifies a run's wrong answers into abstained / partial / wrong instance / evidence missing. On v5: 13 / 14 / 11 / 1.
- Phase 2 landed as four branches (p2-encoder, p2-wire, p2-datapath, p2-render), each adversarially reviewed and its blocking findings fixed. Merge order: encoder -> wire -> datapath -> render, regenerate docs/openapi.json, run every suite one at a time, then push.
- Merging the encoder branch alone lowers the throughput ceiling: every model is entered through a one-permit gate, so a single uvicorn process serves one encode at a time - about 12 rps at the measured torch p50 of 80.6 ms, about 37 rps at the ONNX 27.3 ms (docs/MEASUREMENTS.md section 7, on a 4-core box without AVX2). The 20 rps gate therefore depends on the wire branch's worker count landing as well; do not read a throughput number taken between the two merges as the system's.
- Docker: `bifrost-gateway`, `memory-service-postgres-1`, `memory-service-qdrant-1`.

## Next, in order

1. ~~Fix the 4 Pyright errors~~ done (`7836657`).
2. **Phase 2 (hot path)** — every gate is a VM measurement, but the code can land now:
   own `onnxruntime` runner for the encoder (tokenizer + session + CLS pooling + normalise; the sentence-transformers ONNX backend is *not* usable: `optimum-onnx` pins `optimum~=2.1`, incompatible with sentence-transformers 6), 2 intra-op threads, one executor + semaphore per worker, fingerprint includes the graph file; `WEB_CONCURRENCY=3`, `OMP_NUM_THREADS=2`; Qdrant gRPC + payload projection (`with_payload` include list under a unit test) + `on_disk_payload=False` for memories; one retrieval knob `final_k` (derived prefetch/fused = ceil(1.25×), gated on evidence recall ≥ 0.987); one serialisation pass + gzip; rate limit default 6000; `pool_pre_ping=False` + recycle; GIN index on `graph_entities.aliases` + graph wall budget via a shielded task; bulk access bumps; retry idempotent Qdrant reads once on connection errors (seen 4× in 304 queries through Docker's host gateway); pure-ASGI middleware; OTel gated. Gate on the VM: `uv run python -m benchmark.load.run --base-url http://<api-host>:8080 --api-key <key> -u 20 -r 20 -t 300s --arm cold` → rps ≥ 20, `/v1/context` p99 ≤ 300, no failures.
3. **Phase 1 leftovers**: bake weights into the image (int8 encoder, NLI, docling), compose → 3 services + `local-dbs` profile, `HF_HUB_OFFLINE=1`.
4. **Phase 3 (multilingual)**: `granite-embedding-97m-multilingual-r2` int8 (gated: SciFact ≥ 0.825), Unicode BM25 `bm25-v2` shared tokeniser (sparse.py, evidence.py, native.py, summaries.py, grounding/lexical.py, graph/service.py), `mDeBERTa-v3-xnli` NLI, `lang` tag per memory, MIRACL/MLDR benchmark. One reindex for Phase 2 + 3 together.
5. **Phase 4 (accuracy)**: `--judge-model` separate from the answerer; standard ruler on all 10 conversations with a GPT-4.1-mini-class model through Bifrost; answerer A/B; then the gated ingest experiments from the gap list (session-window verbatim chunks, dated per-entity lists via the dormant LandingReflection/BeliefService, relative-date resolution, graph-edge fusion, whole-session extraction — adopt only on +3 strict, false-merge 0.0, p50 unchanged).
6. **Phase 5**: pytest-xdist with a database per worker, models lane in CI, release gate that refuses artifacts not at HEAD, LongMemEval-S, SciFact full corpus, KG/tool public sets, archive stale results.

## How to run the things that matter

```bash
make bench-locomo-judged LOCOMO_ARGS="--conversations 2 --calls-per-minute 120 --out locomo_judged_vN.json"
make bench-locomo-rescore RESCORE_ARGS="benchmark/results/locomo_judged_vN.json --judge-ruler lenient"
uv run pytest tests/unit tests/contract -q            # hermetic
uv run pytest tests/integration -q                    # needs Postgres + Qdrant; one suite at a time
make typecheck && uv run ruff check src tests benchmark examples
```

Judged runs need the Bifrost gateway (`~/usage_data/gateway-bifrost/up.sh`, :8091) with a DeepSeek key; a 2-conversation run is ~40 min and ~$0.15. Never run two pytest suites concurrently (shared test database). Benchmark containers bind-mount the source; do not edit `src/` while one runs.

## Decisions taken (do not re-litigate without new evidence)

- No cross-encoder reranker (SciFact −5 nDCG, 21× latency). No query-time LLM. No dynamic batching / model server at 20 rps. No SPLADE. Ingest LLM assists per clause are off (measured harmful).
- deepseek-flash stays the default LLM through Bifrost (credit is paid); thinking disabled on its calls for cost/latency; `max_tokens` clamp to be decoupled from thinking budgets.
- Benchmarks run the shipped constants; the judged depth (100) is a benchmark constant in `benchmark/env.py`, never an env override.

## Blocked on the owner

- The 8 vCPU / 16 GB VM (every latency and throughput gate).
- A GPT-4.1-mini-class key in Bifrost for the comparable ruler.
