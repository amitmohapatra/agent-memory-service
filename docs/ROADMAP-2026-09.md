# Roadmap 2026-09: best-in-class memory at < 300 ms, 20 rps, multilingual, one package

Written 2026-09-22 from a seven-lens audit of this repository (hot path, ingestion, configuration and API surface, models, dead code and tests, load and wire) and of the 2026 memory-framework landscape read from primary sources (Mnemis, EverMemOS, Mem0, Zep, Letta, Hindsight, Omi, Nemori, Mastra, LoCoMo-Refined, the Penfield answer-key audit). Every claim below names a file:line or a source; every phase has a gate that is a command and a number. Nothing is credited without its gate.

## Targets, honestly assessed

STATE OF RECORD (HEAD 104fd01, working tree clean; the 'uncommitted' changes named in the task are all committed). LoCoMo v2 = 0.6738 strict / 0.755 lenient on 304 questions from 2 of 10 conversations (233 answerable + 71 adversarial), deepseek-flash as answerer AND judge, 6 judge fallbacks (benchmark/results/locomo_judged_v2.json caveats[0]); evidence_recall 0.9871 (max-over-turns); query p50 280.8 / p95 651.6 ms measured IN-PROCESS around builder.build (benchmark/locomo.py:509-510) with cache/authz/tasks = memory providers and NLI=lexical (Makefile:230-239) on a 4-core no-AVX2 Docker VM (docs/MEASUREMENTS.md:9,147-150). v3 (per-clause LLM ingest on) is worse on both axes (0.6609, p50 572.4 ms) and is not a baseline. No artifact records HTTP p99 with real models; nothing has been measured on AVX2 hardware or on the ONNX backend; 20 rps has never been attempted (load_test.json = 3 users, 15 s, 22 requests, 1.9 rps, hash embeddings).

TARGET accuracy > 94%: NOT reachable on any strict or human-aligned LoCoMo ruler, by us or by anyone published. Penfield's audit puts the theoretical maximum at ~93.6% (99/1,540 answer-key errors); under LoCoMo-Refined (86.33% human agreement; official judge Qwen3-14B) the best rescored system is 82.65% and the 93-class systems fall to 58-64%. Mnemis 93.9 (GPT-4.1-mini answerer+judge, category 5 excluded, k=30) needs System-2 LLM browsing at query time (3,637.65 s over 1,540 q = ~2.4 s/query) plus Qwen3-Reranker-8B; its zero-LLM System-1 is 89.1 and still carries an 8B reranker (35.8 s per 20 pairs on our CPU class). Our gap is not recall (0.987) but answerer + ruler + synthesis: lenient rescoring of the SAME v2 answers moves multi_hop 0.395 -> 0.732 and overall 0.674 -> 0.755; DeepSeek judges run ~8-12 pp stricter than GPT-class judges (arXiv 2607.21962), so 0.674 is not on the 89-93 ruler at all. Reachable and worth claiming in 4-6 weeks with zero query-time LLM calls and CPU retrieval under 300 ms: 82-88 on the standard ruler (1,540-question subset = LoCoMo minus adversarial, mem0's verbatim ACCURACY_PROMPT, GPT-4.1-mini-class answerer and judge), and >= 0.85 lenient on our 2-conversation set sooner; with deepseek-flash as answerer expect 6-8 points less (Nemori: full-context 0.723 gpt-4o-mini vs 0.806 gpt-4.1-mini). The only published >94 figures are vendor rulers (Mem0 LongMemEval 94.4 at top_200 / 6,956 tokens; Mastra 94.87 with a ~30k-token observation log in context) - we can report LongMemEval-S with a GPT-class judge and may land in the high 80s/low 90s there, but 'accuracy > 94%' must be reframed for leadership as 'best-in-class on the standard ruler at < 300 ms retrieval and zero query-time LLM calls, with LoCoMo-Refined reported alongside'. Anything else is a ruler artefact.

TARGET p99 < 300 ms: ruler must be defined first - HTTP POST /v1/context, cold arm (bundle-cache miss), from a separate host, 20 rps sustained 5 min, remote Postgres+Qdrant+Dragonfly. Today: single uvicorn process (src/memory_service/__main__.py:14-16), torch fp32 encoder 178 ms mean / 204 p95 with intra-op threads unbounded (embeddings.py:97, settings.py:195) and asyncio.to_thread on the default 12-thread pool (embeddings.py:125,128), Qdrant over REST with full payloads (qdrant_store.py:87,225-306), 4-6 serialisation passes on the HTTP path (builder.py:284-286,624-625; retrieval.py:252), sequential Redis GETs, pool_pre_ping (db/engine.py:31). Reachable on 8 vCPU with AVX2 IF the 2-thread int8 ONNX query encode measures <= ~60 ms on the target VM (estimate 20-45 ms; ZERO measurements exist) - then p50 ~80-150 ms and p99 ~200-280 ms is the modelled band after Phase 2. Not reachable on hardware like the dev boxes (encoder alone 178-636 ms), with any cross-encoder on the path, or with fused_k=200 full-payload REST reads on a remote on-disk-payload Qdrant.

TARGET 20 rps on 8 vCPU / 16 GB: never attempted. Today's shape cannot: 20 x 0.178 s = 3.6 encoder-core-seconds/s under thread oversubscription plus one GIL for all Python. With 3 workers x 2 ORT threads at ~40 ms encode (~0.8 cores) + ~1 core of Python + I/O it fits with headroom; also ServiceSettings.rate_limit_per_minute=1200 (settings.py:36) is exactly 20 rps per tenant+key and would 429 the load test, and pool_size 10 + max_overflow 10 (settings.py:47-48) with up to 3 checkouts per request sits at its cap at 20 rps x 0.3 s. Conditional on the same Phase-0 encoder measurement.

TARGET multilingual: not today - every default model is English-only (settings.py:188,214,239,331) and the BM25 tokenizer is [a-z0-9]+ (adapters/models/sparse.py:18) so a non-Latin query yields zero sparse terms and the hybrid silently degrades to English-only dense; native.py:41-42, summaries.py:25-26, grounding/lexical.py:8, graph/service.py:30, document_facts.py:225-226 are ASCII too (only evidence.py:41,48 and chunking.py:36 are Unicode/CJK-aware). Reachable with ibm-granite/granite-embedding-97m-multilingual-r2 (384-d, keeps the collection dimension, Apache-2.0, ships onnx/model_quint8_avx2.onnx 98 MB, MTEB multilingual retrieval 60.3 vs e5-small 50.9) at a published English cost (MTEB-v2 retrieval 50.1 vs 53.9 for small-english-r2) that must be gated on SciFact/golden, plus a Unicode BM25 tokenizer, mDeBERTa-v3-base-xnli NLI (MIT, int8 ONNX shipped) and language-pinned LLM prompts. Ruler: MIRACL dev subsets + MLDR through our ingestion path; no CPU-latency comparator exists publicly, so we publish ours.

TARGET 'no chinese mode': withdrawn by the user on 2026-09-22 ("not required; if later we want we can use a Chinese model"). No language-drift guard, no prompt language pin and no model-provenance rule are built; models are chosen on measured quality, cost and licence only. What survives from the analysis is plain LLM hygiene: the production max_tokens clamp (1024) must not be paired with a thinking model, and thinking is disabled on deepseek-flash calls for cost and latency, not for language.

TARGET one package / no open-ended configuration: reachable - settings.py exposes 206 leaf fields in 23 sections with 25 Literal branches, ~12 of which run in no compose target, benchmark or CI lane; a YAML config source with no file (settings.py:534,601); three retrieval tunings (defaults 100/100/50, .env.example 50/40, Makefile -e 200/200/100) so the shipped service and the benchmarked service are different artefacts; 10 services start on plain `docker compose up` and weights are downloaded at first up (docker-compose.yml:120-138; .dockerignore:22). After Phase 1: ~35 env fields (URLs, credentials, topology), 3 core services + a local-dbs profile, weights baked (int8 encoder ~98 MB + int8 NLI ~280 MB + docling 669 MB), zero network at up. Structural limit: four external stores (Postgres, Qdrant, Dragonfly, OpenFGA + its own database) mean ~10 URL/credential fields, and OpenFGA cannot be folded into the application image.

## Decisions taken on top of the audit

- **ONNX runtime, not the sentence-transformers ONNX backend.** That backend needs `optimum-onnx`, which pins `optimum~=2.1`, which cannot coexist with `sentence-transformers>=6` (uv resolution fails). The encoder and the NLI model run on a thin `onnxruntime` session of our own (tokenizer + session + pooling/softmax + normalise, ~60 lines each, no optimum at runtime); the Hub-shipped int8 graphs are used as files. Export for a model without a shipped graph happens once, off-box, and the file is baked.
- **LLM default stays `deepseek/deepseek-flash` through Bifrost**, with thinking disabled on every call for cost and latency (Bifrost passes `extra_body` through on the OpenAI-compatible DeepSeek path). `google/gemini-2.5-flash-lite` / `gemini-2.5-flash` are allowed alternatives. The 'no chinese mode' requirement was withdrawn on 2026-09-22: no drift guard, no language pin, no provenance rule; a Chinese-lab model may be chosen later on its measured merits.
- **LLM uses at ingest default to none** (`ambiguous_worthiness` / `ambiguous_extraction` measured harmful: 0.674 -> 0.661 with the query p50 doubling on a loaded box). The one admissible LLM-at-ingest design is a whole-session extraction job, off by default and gated (Phase 4, step 8).
- **The "> 94 %" target is reframed**: no system reaches it under a strict or human-aligned judge (theoretical maximum ~93.6 % after the answer-key audit; best human-aligned rescore 82.65 %). The number we publish is the standard ruler (1,540 questions, category 5 excluded, Mem0's verbatim judge prompt, GPT-4.1-mini-class answerer and judge) with the strict and LoCoMo-Refined rulers alongside, at < 300 ms retrieval and zero query-time LLM calls. The bar is: best-in-class on that ruler among systems that make no LLM call at query time.
- **The 300 ms ruler** is HTTP `POST /v1/context`, cold arm (bundle-cache miss), from a separate host, 20 rps sustained for 5 minutes, Postgres/Qdrant/Dragonfly remote, on the 8 vCPU / 16 GB VM. In-process LoCoMo timings are a stage split, not the headline.


## Amendments from the gap analysis (2026-09-23)

[GAPS-2026-09.md](GAPS-2026-09.md) ranked the borrowable techniques against a measured failure
shape and changed this plan in the following places. Where an amendment contradicts a phase step
below, the amendment wins.

**The failure shape the later phases are aimed at.** v5's 39 wrong answerable answers are 13
abstentions, 14 partial enumerations (correct under the lenient ruler), 11 wrong instances and
exactly **one** retrieval failure (`python -m benchmark.failure_taxonomy`). Enumeration
completeness and the answer protocol are two thirds of what is left; nothing in Phase 2 or 3
should be justified by accuracy.

**Four defects, fixed ahead of the rest** (they cost tokens and accuracy and nothing to fix):
the date and speaker are stamped into stored content *and* rendered again, so every line carries
both twice and every start-anchored ingest rule is blinded by the prefix; retrieval rank is
discarded before rendering, against a measured 75.8/53.8/63.2 positional effect; principal graph
nodes carry no bare-name alias while turn prefixes mint resolvable entities; the neighbourhood
LIMIT orders by a confidence every MENTIONS edge shares. These are folded into Phase 2's branch
set rather than waiting for Phase 4.

**Phase 3 gains four correctness items**, without which "multilingual" would be a claim about a
monolingual measurement: verbatim turns are dropped entirely for non-Latin scripts (the sentence
filter counts ASCII words), so the Phase 3 step 2 gate must assert verbatim count == message count
on a ja/zh/ar/ru fixture; NFKC + casefold and per-language negation lists must land *with* the
Unicode tokenizer, because that tokenizer is what first makes non-English merges possible and
nothing today blocks a negated duplicate; token estimation is `len // 4`, so CJK/Thai/Devanagari is
silently truncated at the 512-token window; and the gate needs a cross-lingual row (question
language != memory language) and a code-switched row, plus a Unicode-safe `_normalise` in the
harness, which today strips every answer to `[a-z0-9]`.

**Phase 4 step 1 (the ruler) is now explicit**: add Mem0's *verbatim* ACCURACY_PROMPT as its own
ruler (our "lenient" paraphrases a newer prompt that no comparator was graded with), use Mem0's
verbatim answer prompt for the standard column only, report category 5 as its own column, name the
judge model in the file and in every record (`--judge-model`, landed), and print the sampling error
bar - at n=233 it is about ±4.8 pp, wider than most of the gains below.

**Phase 4 step 3** takes its episode boundaries from embedding drift (the session's 35th-percentile
drift, capped at 25 turns) rather than a fixed 3-5 turn window; zero LLM either way.

**Phase 4 step 5** must emit *dated* aggregates ordered by `observed_at` - the dormant
`BeliefService` emits an undated, ingest-ordered list, which is the wrong shape for the 14 partial
enumerations it is meant to answer.

**Phase 4 step 8 gains a precondition and two constraints**: `consolidate()` only reaches dense
similarity when Jaccard >= 0.5, so an LLM paraphrase can never dedup and every outbox retry would
double the memory count - fix that first; each extracted fact carries the turn ids it came from;
and a zero-LLM grounding filter drops any fact whose cited turn shares no content token, number or
date with it.

**Phase 4 gains a fusion step** (after the instrumentation): weighted RRF or DBSF server-side -
Qdrant has supported weighted RRF since 1.17 and we run 1.18.2, so no upgrade is needed - fitted
offline from dumped per-leg gold ranks, and spent on halving retrieval depth rather than on recall
we already have. It requires the instrumentation named next.

**Instrumentation first**: the harness stores a bundle as four integers, so no fusion weight, depth
or salience decision can be evidenced. Dump per-item `[record_id, kind, retrievers, fused_rank]`
and the gold turns' rank in the dense-only, sparse-only and fused lists before tuning any of it.


## Phase 0 - Instrument and baseline at HEAD (measure before tuning; nothing below is credited without these artifacts)

**Goal.** Replace every stale, estimated or cross-machine number with one measured at HEAD 104fd01, and obtain the single decision number the whole latency/throughput plan hangs on: the 2-thread int8 ONNX query-encode cost on the target 8 vCPU VM.

**Gate.** All of the following exist under benchmark/results with provenance.git_commit == `git rev-parse HEAD`: locomo_judged_v4_head.json (answer_recall_at_k >= 0.674 strict, evidence_recall >= 0.987, every record carries latency_ms and timings_ms, judge_failures reported separately), locomo_judged_v4_head_lenient.json and _refined.json, model_throughput.json + embedding_onnx_vm.json + concurrency.json with provenance.cpu_count == 8 from the target VM, load_test_head.json with aggregated.requests >= 5000 and per-endpoint p50/p95/p99. Check: `for f in locomo_judged_v4_head model_throughput embedding_onnx_vm concurrency load_test_head; do python3 -c "import json;d=json.load(open('benchmark/results/$f.json'));p=d['provenance'];print('$f',p['git_commit'][:8],p['cpu_count'],p.get('platform'))"; done` must print HEAD's short sha on every line and 8 for the VM runs. Decision recorded in docs/MEASUREMENTS.md: E2 = 2-thread query p95 per (model, backend, quantisation) on the VM.

1. **Restore the judged targets to the v2 LLM configuration and make provenance honest: set MEMORY__MODELS__LLM__USES and FAST_USES back to ["grounding_judge"] in bench-locomo-judged and bench-longmemeval; pass -e GIT_COMMIT=$(shell git rev-parse HEAD) into every docker-run bench block and let benchmark/common.py _git_commit() read it.**
   - Where: Makefile:243-246 and :312-315 (USES/FAST_USES), Makefile:225-227 and :296-298 (docker run), benchmark/common.py:18 (_git_commit) and :81 (provenance git_commit)
   - Why: v3 turned on ambiguous_worthiness/ambiguous_extraction at ingest and lost accuracy and latency (0.674 -> 0.661, p50 281 -> 572 ms; locomo_judged_v3.json provenance.llm.uses); every locomo_*.json, scifact*, flag_off_* and model_throughput.json carries git_commit 'unknown' because the container has no .git.
   - Expected: A like-for-like baseline; artifacts that can be tied to a commit, which the release gate will later require.
   - Latency: none
   - Verify: `grep -n 'grounding_judge' Makefile` shows only the single-use lists; next artifact's provenance.git_commit equals `git rev-parse HEAD`.

2. **Fresh judged LoCoMo on the 2-conversation set at HEAD, and record the EFFECTIVE retrieval/context config (prefetch_k, fused_k, final_k, memories_max, token_budget as resolved by Settings) in provenance so the 40-memory cap is explained: v2 and v3 bundles were exactly 40 memories although the target passes -e FINAL_K=100 / MEMORIES_MAX=100, and settings sources rank env above dotenv, so the .env explanation in commit 66996f4 cannot be the whole story.**
   - Where: benchmark/locomo.py:625-626 (latency_ms/timings_ms already written per record), benchmark/common.py:81 (add resolved settings.retrieval/context to provenance), src/memory_service/config/settings.py:597-603 (source order: init, env, dotenv, yaml, secrets)
   - Why: No result file on disk has timings_ms or latency_ms even though HEAD emits them (engine.py:196, builder.py:231-260); every stage estimate in the seven lenses is unverifiable until one run records the split; any claim about depth 200 is about a configuration that has never produced a >40-memory bundle.
   - Expected: Per-stage p50/p95/p99 (scope/encode/search/graph/window) for the in-process path; the real depth cap identified before any knob is frozen.
   - Latency: none (measurement)
   - Verify: `make bench-locomo-judged LOCOMO_ARGS="--conversations 2 --calls-per-minute 60 --out locomo_judged_v4_head.json"` then a python one-liner over records[*].timings_ms printing p50/p95/p99 per stage and asserting bundle.memories == min(final_k, memories_max) on every record.

3. **Rescore v4 under the lenient and refined rulers (both exist; refined has never been run).**
   - Where: benchmark/locomo.py:288-320 (JUDGE_RULERS strict/refined/lenient), Makefile:256-290 (bench-locomo-rescore)
   - Why: Lenient rescoring of identical answers is the cheapest evidence that most of the strict multi_hop loss is the all-items rule (0.395 -> 0.732 on v2); the refined ruler is the human-aligned comparator (LoCoMo-Refined) and its number decides how the >94% target is framed.
   - Expected: Three rulers on one answer set; the ruler gap becomes a number instead of an argument.
   - Latency: none
   - Verify: `make bench-locomo-rescore RESCORE_ARGS='benchmark/results/locomo_judged_v4_head.json --judge-ruler lenient'` and the same with `refined`; by_category present in both outputs.

4. **Add evidence_all_hit (every gold turn present in the bundle) per record and per category, and report judge fallbacks as a separate count excluded from the headline.**
   - Where: benchmark/locomo.py:186 (_answer_present is max-over-turns), :587-626 (record fields), :650-716 (aggregation)
   - Why: '78 of 79 wrong answerable answers had the gold turn in the bundle' exists only in commit 104fd01's message; multi-hop needs every turn, which no artifact measures; v2's headline mixes 6 token-overlap fallbacks into a judged score (caveats[0]).
   - Expected: The 'retrieval is not the bottleneck' claim becomes reproducible; the accuracy work in Phase 4 is aimed by an artifact, not a commit message.
   - Latency: none
   - Verify: locomo_judged_v4_head.json has by_category[*].evidence_all_hit and top-level judge_failures; unit test in tests/unit/test_benchmark_locomo.py (new) for the all-hit rule.

5. **[TARGET-VM] Build the image on the 8 vCPU VM and run the model throughput benchmark there; add the multilingual candidate to it.**
   - Where: Makefile:418-428 (bench-model-throughput), benchmark/model_throughput.py:51-65 (hard-coded granite-small + MiniLM paths; add granite-embedding-97m-multilingual-r2, drop the reranker rows once Phase 1 removes it)
   - Why: Every latency number on disk comes from a 4-core no-AVX2 box (embedding.json, model_throughput.json provenance cpu_count 4; docs/MEASUREMENTS.md:9); the Makefile itself says sizing cannot be extrapolated (Makefile:419-421).
   - Expected: The first encoder number on deployable hardware: embed_query_ms at threads 1/2/4 for fp32 torch, the floor of the whole plan.
   - Latency: none (measurement)
   - Verify: `docker compose build memory-api && make bench-model-throughput THREADS="1 2 4"` on the VM; model_throughput.json provenance.cpu_count == 8.

6. **[TARGET-VM] ONNX smoke and candidate comparison: add runtime='onnx' catalogue entries for granite-embedding-small-english-r2 (export with optimum export_optimized_onnx_model O3 + export_dynamic_quantized_onnx_model avx2/avx512_vnni at build time) and for ibm-granite/granite-embedding-97m-multilingual-r2 (ships onnx/model_quint8_avx2.onnx); add both to benchmark.embedding MODELS; run the NON-quick mode (all BACKENDS) inside the image on the VM with threads pinned to 2.**
   - Where: src/memory_service/tools/download_models.py:30 (IGNORE excludes onnx/* for torch entries), :36-37 (ONNX_ALLOW), :64-70 (default entries); benchmark/embedding.py:48-56 (MODELS, BACKENDS, QUICK_BACKENDS); src/memory_service/adapters/models/embeddings.py:78-83 (backend map already accepts onnx)
   - Why: The ONNX/OpenVINO backends are wired but have literally zero measurements (embedding.json is mode.quick=true, torch only, 2 documents / 18 queries); models/granite-embedding-small-english-r2 contains no onnx/ directory; the 20 rps target and the p99 target both depend on this unknown.
   - Expected: E2 (2-thread query p95) per (model, backend, quantisation). Decision rule, fixed now: adopt the int8 file whose E2 <= 60 ms; if only fp32 ONNX O3 meets 60-100 ms, adopt it and re-plan workers; if nothing is <= 100 ms the 20 rps target is infeasible on this VM class - stop and report instead of tuning.
   - Latency: none (measurement); it decides the encoder stage budget for Phase 2
   - Verify: `docker run --rm --user root -v "$PWD":/app -v "$PWD/models":/models:ro -e PYTHONPATH=/app/src:/app -e VIRTUAL_ENV=/opt/venv -e MEMORY__MODELS__EMBEDDING__THREADS=2 --entrypoint sh memory-service-memory-api -c '/opt/venv/bin/python -m benchmark.embedding --out embedding_onnx_vm.json'`; candidates[*].embed_ms.query_p95 present for sentence_transformers/onnx/openvino per model; provenance.cpu_count == 8.

7. **[TARGET-VM] Run the concurrency benchmark once with threads pinned (peak RSS per process and sequential-vs-concurrent agreement on the shared torch/ORT module).**
   - Where: Makefile:350-371 (bench-concurrency), benchmark/concurrency.py:1-8
   - Why: No benchmark/results/concurrency.json exists; the RSS-per-worker figure that sizes 3-4 uvicorn workers in 16 GB is an estimate.
   - Expected: Measured per-process RSS and proof that concurrent entry returns the same vectors.
   - Latency: none
   - Verify: `make bench-concurrency BENCH_THREADS=2`; concurrency.json has peak_rss_mb and agreement == true.

8. **Fix the load scenario, then take the first real-model HTTP measurement from a separate host: locustfile wait_time -> constant_throughput(1.0) with 20 users (exactly 20 rps); on_start posts one message to the fresh thread before the history task can run (the 404 is 'thread not found', not a missing route); cold arm = unique salt per query so the bundle cache misses, warm arm = 50 repeated queries; task mix 70% POST /v1/context, 20% POST /v1/recall, 10% POST /v1/messages; seed 20 tenants x 5 users x 100 memories and drain the worker first; run.py gains --arm and samples `docker stats` CPU%/RSS per container.**
   - Where: benchmark/load/locustfile.py:42 (between(0.2,1.0)), :44-54 (on_start, random thread_id), :101-106 (history task); benchmark/load/run.py:110-141 (args); src/memory_service/api/routers/v1/conversation.py:145 (route exists)
   - Why: load_test.json is 3 users / 15 s / 22 requests / 1.9 rps against hash models with 2 x 404 from the ordering race; the shipped target `-u 20 -r 5 -t 60s` (Makefile:442-443) has never been run against real models; p99 has never been measured at all.
   - Expected: The baseline p50/p95/p99 per endpoint at 20 rps on the VM with today's code - the number Phase 2 must beat.
   - Latency: none (measurement)
   - Verify: `uv run python -m benchmark.load.run --base-url http://<api-host>:8080 --api-key <key> -u 20 -r 20 -t 300s --arm cold` from a second host; load_test_head.json aggregated.requests >= 5000, failures == [] (no 404s).

9. **Export the per-request stage timings to the existing memory_stage_seconds histogram with stage labels (scope, encode, search, graph, window, assemble) so p99 attribution is visible during the load run.**
   - Where: src/memory_service/observability/metrics.py:31-32 (stage_seconds histogram exists), src/memory_service/modules/retrieval/engine.py:196, src/memory_service/modules/context/builder.py:260
   - Why: Under load the in-process timings are the only way to tell encoder queueing from Qdrant/Postgres tails; today they exist only inside bundle.diagnostics.
   - Expected: Stage-level p99 during Phase 2 gates without re-running LoCoMo.
   - Latency: sub-ms (one histogram observe per stage)
   - Verify: `curl -s localhost:8080/metrics | grep memory_stage_seconds_bucket | grep stage=\"encode\"` non-empty after the load run.


## Phase 1 - Freeze the package: delete dead code, challengers and open-ended configuration; bake the models; type the API

**Goal.** One image, one frozen model set, ~35 env fields (URLs, credentials, topology only), every closed-set API string an enum with bounds, `docker compose up` with no network access to model hubs, and the benchmarks running the shipped constants.

**Gate.** `uv run pytest tests/unit tests/contract -q` green (including test_every_declared_provider_value_has_a_wiring_branch at tests/unit/test_architecture.py:223 and test_committed_schema_matches_generated at tests/contract/test_openapi_contract.py:86 after `make openapi`); `python3 -c "from memory_service.config.settings import Settings; import json; print(sum(1 for _ in json.dumps(Settings().redacted())))"` replaced by a unit test asserting the leaf-field count <= 40; `grep -rn -E 'ThreadObserver|search_features|PolicyProvider|_LATE_LICENSES|mem0_provider|langmem_provider|cognee_provider|graphiti_provider|docling_graph_provider|model_server|qdrant_local_path|sparse_model|observation_refinement|pageindex' src/ | wc -l` == 0; `docker compose config --services | wc -l` == 3 (memory-api, memory-worker, db-migrate) without profiles; `HF_HUB_OFFLINE=1 docker compose --profile local-dbs up -d && curl -sf localhost:8080/health/ready` returns 200 inside the (re-measured) start_period with the api container given no route to the internet.

1. **Delete the unwired observer: modules/memory/observer.py (437 lines), the memory.observe handler and registration, the per-message enqueue (which costs one list_thread SELECT per ingested message for a job that is a no-op), settings observer_hot_window_messages/observer_batch_messages/observer_max_notes and LLMUse 'observation_refinement'. Keep a no-op handler for one release (or purge queued rows) so old outbox rows do not fail dispatch with KeyError again.**
   - Where: src/memory_service/modules/memory/observer.py; modules/jobs/registry.py:151-177, :245; modules/memory/pipeline.py:245, :291-296; config/settings.py:375-377, :278; adapters/wiring.py:530
   - Why: Nothing constructs it (wiring.py:530 comment; registry.py:175 returns when 'thread_observer' is absent); tests mention it only in comments; the enqueue is on the write path.
   - Expected: ~500 lines and 3 settings + 1 LLM use gone; one SELECT fewer per ingested message.
   - Latency: ingest: -1 DB round trip per message; query: none
   - Verify: `grep -rn 'thread_observer\|memory.observe\|observation_refinement' src tests | wc -l` == 0 (except the transitional no-op handler); `uv run pytest tests/unit/test_llm_memory.py tests/unit/test_memory_native.py tests/integration -q`.

2. **Delete the provably dead definitions and stale M10 leftovers: native.py search_features (returns {}), api/deps.py get_authz, graph/postgres_store.py _alias, ports/intelligence.py PolicyDecision/PolicyProvider, advanced.py _LATE_LICENSES and its ColBERT/late-chunking docstring, indexer.py M10/multivector comments, the pageindex extra, README's phantom 'tool_reflection' use. Do NOT delete _layer_from_predicate (a pydantic before-validator), run_periodic or run_until_idle (test seams used by tests/integration and tests/failure).**
   - Where: src/memory_service/modules/memory/native.py:1119; src/memory_service/api/deps.py (get_authz); src/memory_service/adapters/graph/postgres_store.py:81; src/memory_service/ports/intelligence.py (PolicyDecision/PolicyProvider); src/memory_service/adapters/models/advanced.py:1-6,23; src/memory_service/modules/rag/indexer.py:72; pyproject.toml:64,69; README.md:525
   - Why: Zero call sites outside their definitions (verified by grep); the docstrings advertise strategies that were removed.
   - Expected: Dead surface gone without behaviour change; README stops documenting a use that does not exist.
   - Latency: none
   - Verify: `uv run pytest tests -m 'not docker and not models' -q` green; `grep -rn 'search_features\|get_authz\|PolicyProvider\|_LATE_LICENSES\|tool_reflection' src README.md | wc -l` == 0.

3. **Delete the challenger providers from the product: memory_intelligence.provider values mem0/langmem/cognee and adapters/intelligence/*, graph_enrichment.provider values graphiti/docling_graph/cognee with the five graphiti_* settings and both providers, the neo4j compose profile, the memory-providers/cognee/graphiti extras, and the process-global os.environ mutations they carry. If a comparison is ever needed again it lives under benchmark/, never in src/.**
   - Where: src/memory_service/config/settings.py:344, :390-394; adapters/wiring.py:486-495, :575-583; docker-compose.yml:418-425; pyproject.toml:57-60,69; src/memory_service/adapters/intelligence/*, adapters/graph/graphiti_provider.py, adapters/graph/docling_graph_provider.py
   - Why: pyproject labels them 'benchmark only; never the public contract'; benchmark/results/memory.json shows mem0/langmem skipped; they are the bulk of the 25 Literal branches that no compose target, benchmark or CI lane runs.
   - Expected: -8 settings, -2 providers, -1 profile, smaller image (no mem0ai/langmem/graphiti deps).
   - Latency: none
   - Verify: `grep -rn 'os.environ.update\|os.environ\[' src/memory_service/adapters | wc -l` == 0; architecture test passes with the reduced Literals.

4. **Delete the served-model tier and the litellm gateway: url/api_key/timeout_seconds/max_retries on Embedding/Reranker/NLI settings, models.sparse_url, tools/model_server.py, adapters/models/remote.py, the `memory-model` script, compose services memory-embed/rerank/nli/sparse and model-gateway, deploy/served-models.yml, deploy/model-gateway.yaml, and ProviderPolicySettings (licence/locality is a build-time fact once models are frozen).**
   - Where: src/memory_service/config/settings.py:200-207, :224-231, :246-253, :336, :504; docker-compose.yml:263-362; deploy/served-models.yml; deploy/model-gateway.yaml; pyproject.toml [project.scripts] memory-model; src/memory_service/tools/model_server.py; src/memory_service/adapters/models/remote.py
   - Why: 13 knobs and 5 services for a scale-out path an 8 vCPU box does not use; model_server.py has no batching and the same to_thread executor problem, so it only adds an HTTP hop.
   - Expected: -13 settings, -5 service definitions, one process to size; in-process embedding avoids a localhost hop per query.
   - Latency: neutral-to-positive on the query path
   - Verify: `grep -rn 'RemoteEmbedding\|RemoteReranker\|RemoteNLI\|sparse_url\|provider_policy' src | wc -l` == 0; docs/adr/0019 marked superseded.

5. **Freeze the remaining Literals to what one package ships and remove the rest: cache -> one redis-protocol implementation (drop valkey/redis/disabled spellings); auth -> trusted_dev|jwt (delete gcp_iam/mtls, gcp_id_token.py, jwt_hs256_secret); search provider 'memory' and qdrant_local_path become a build_container(overrides=...) test seam, not env; tasks 'inline'/'memory' likewise; fusion -> 'rrf' only (dbsf and none both fall into the manual-RRF branch); delete retrieval.contextual_chunks (no reader), database.echo, documents.fallback_parser (one-value Literal -> constant), EvaluationSettings and PerformanceBudgets (move with modules/evaluation to benchmark/), the YAML source and MEMORY_CONFIG_FILE, MEMORY_MODELS_DIR (set in three places, read nowhere), the duplicate HF_HOME (Dockerfile:104 vs compose:35).**
   - Where: src/memory_service/config/settings.py:80, :93, :105, :158-164, :403, :429, :456, :493, :519, :534-553, :601; src/memory_service/modules/retrieval/engine.py:507-527; adapters/wiring.py:131-136, :284-287; deploy/Dockerfile:104; docker-compose.yml:35, :396; Makefile:95,424; tests/conftest.py:23-71 (provider stand-ins via overrides)
   - Why: These branches are configuration nobody runs (reference counts in compose+deploy+Makefile: valkey 0, gcp_iam 0, mtls 0, dbsf 0, docling_graph 0, openvino 0); 'dbsf' is declared but unimplemented; the YAML file does not exist anywhere in the repo.
   - Expected: 25 Literal branches -> ~8; every remaining value is exercised by a test or a compose target.
   - Latency: none
   - Verify: `uv run pytest tests/unit/test_settings.py tests/unit/test_architecture.py tests/unit/test_deployment_profiles.py -q` green after the YAML/rrf_k env tests are removed.

6. **Create src/memory_service/config/constants.py with FrozenModels (dense: id, revision, onnx file name, dimension 384, max_seq_length 512, intra_op threads 2; sparse: bm25-v2; nli: id, revision, onnx file) and make the embedding fingerprint include the ONNX file/quantisation (today only backend + dimension), so switching int8 vs fp32 can never silently reuse a collection; embedding.dimension/model/model_path/provider/device/normalize/batch_size/max_tokens stop being env fields; EmbeddingSettings keeps only `threads` (frozen 2) and nothing else.**
   - Where: src/memory_service/config/constants.py (new); src/memory_service/adapters/models/embeddings.py:101 (max_seq_length), :130-132 (fingerprint); src/memory_service/config/settings.py:181-195; docker-compose.yml:24-26 (remove MODEL/MODEL_PATH/DIMENSION env)
   - Why: dimension is env-settable (settings.py:190; compose:26) and nothing ties it to the model - a wrong value means silent zero recall or a Qdrant dimension error at first upsert; the fingerprint names the collections (indexer.py:75-85).
   - Expected: The model is a fact of the build reported by /version, never a knob; collection names change exactly when vectors change.
   - Latency: none
   - Verify: Unit test: fingerprint differs between onnx/model.onnx and onnx/model_quint8_avx2.onnx; `grep -rn 'MODELS__EMBEDDING__DIMENSION' docker-compose.yml Makefile | wc -l` == 0.

7. **Collapse Settings to the env-only surface (see freeze.settings_kept_as_env), with SecretStr for database.url, cache.url, qdrant_api_key, openfga_api_token and llm.api_key, and one retrieval knob: final_k (default 50) from which prefetch_k = fused_k = ceil(final_k x 1.25) are derived; the judged benchmark depth (100) becomes a constant in benchmark/, not a Makefile -e. Split the host-side .env (settings.py:560 env_file) from the compose x-common-env explicitly: .env.example keeps only the env-only fields, and a unit test asserts that a real environment variable beats .env (settings.py:597-603).**
   - Where: src/memory_service/config/settings.py:26-527 (rewrite), :46, :81, :560, :597-603, :643 (redacted must mask every SecretStr); .env.example; docker-compose.yml:13-36; Makefile:248-250, :318-320 (delete -e PREFETCH_K/FUSED_K/FINAL_K/MEMORIES_MAX/TOKEN_BUDGET)
   - Why: Three value sets and three mechanisms (defaults, .env, Makefile -e) currently configure the same retriever; the 0.674 number was not produced under shipped defaults; database.url is a plain str that redacted() does not mask.
   - Expected: 206 -> ~35 fields; the benchmarked service and the shipped service become the same artefact.
   - Latency: none
   - Verify: `uv run pytest tests/unit/test_settings.py -q` includes test_env_beats_dotenv and test_secrets_are_masked; leaf-field count test <= 40.

8. **Type the request side and bound every free-form field (see freeze.api_enums): kinds, memory_type, tool status/source/visibility, VerifyItem.kind, files visibility, sub_calls; add an exception handler mapping pydantic ValidationError raised inside handlers to 422 (today Visibility(bad) and SubCall.model_validate fall to the generic 500 handler); clamp token_budget, max_visited, verify items/answer, custom_metadata/args/schema/output sizes.**
   - Where: src/memory_service/api/routers/v1/retrieval.py:48-54, :126-130, :213-216; routers/v1/tools.py:48-51, :79-89, :212; routers/v1/grounding.py:85-86, :103-108; routers/v1/memory.py:191; routers/v1/graph.py:36; routers/v1/files.py:97,160; api/deps.py:44; domain/context.py:90; api/errors.py:162-167; modules/tools/service.py:215
   - Why: Six closed sets are free strings (a phantom 'fact' kind is silently dropped; unknown kinds return empty results; tools.visibility returns 500); eight unbounded dict/Any fields; per-request knobs can 10x the work (token_budget up to 200k, max_visited up to 2000 vs server 200).
   - Expected: Invalid values become 422 with the allowed list; worst-case request cost bounded to ~2x the default path.
   - Latency: protects p99 (the request-path multipliers are max_visited and token_budget; answer/verify bounds protect /v1/grounding, not /v1/context)
   - Verify: `uv run pytest tests/contract tests/e2e -q`; new contract tests: POST /v1/tools with visibility='NOPE' -> 422; POST /v1/recall kinds=['fact'] -> 422.

9. **Type the response side: query_type -> QueryType, representation -> Representation, MemoryResponse enums, DocumentResponse.status -> DocumentStatus, JobResponse.status -> JobStatus, ClaimVerdictBody.verdict/method -> Literal, VerifyResponse.source -> Literal, FactOut.status -> TemporalStatus, RecallResponse.evidence -> EvidenceReport, ContextResponse lists -> typed ContextItem models and extra='forbid'; regenerate docs/openapi.json and move the SDK's str fields to Literal.**
   - Where: src/memory_service/api/routers/v1/retrieval.py:80, :108, :137-141, :183; routers/v1/memory.py:66; sdk/python/src/universal_memory/client.py:198,200,388,539; sdk/python/src/universal_memory/models.py:155,200; docs/openapi.json
   - Why: The OpenAPI contract leaks closed sets as str and ContextResponse allows extra keys; the SDK already assumes Literals for verdict and status.
   - Expected: The schema becomes a real contract the harness and SDK can rely on.
   - Latency: none measurable
   - Verify: `make openapi && uv run pytest tests/contract/test_openapi_contract.py sdk/python/tests -q` green; `git diff --exit-code docs/openapi.json` clean after the export.

10. **Bake weights into the image and stop mounting ./models: COPY the frozen set (encoder int8 ONNX + tokenizer, mDeBERTa int8 ONNX + tokenizer, docling artifacts) under /models in the runtime stage; export/quantise at build time via download_models as a build step; delete model-fetch and the ./models:ro mounts; drop models/ from .dockerignore for exactly those directories; set HF_HUB_OFFLINE=1, WEB_CONCURRENCY=3, OMP_NUM_THREADS=2 in the Dockerfile ENV; correct the stale 'three models / ~6 minutes' compose comment (the reranker has not been constructed since wiring.py:350-358) and re-measure start_period.**
   - Where: deploy/Dockerfile:69-98, :104; .dockerignore:22; docker-compose.yml:120-138, :170, :189-194, :211; src/memory_service/tools/download_models.py:62-135 (catalogue reduced to the frozen set)
   - Why: Today `docker compose up` downloads ~1 GB from HF on first start and the healthcheck waits 600 s on a measurement that predates the reranker fix; identical weights in every deployment is the definition of 'one package'.
   - Expected: Offline start; image + ~1.05 GB (98 MB + ~280 MB + 669 MB); reproducible model provenance (models/MANIFEST.json revision pinned).
   - Latency: cold start drops (two int8 models instead of fp32 torch + DeBERTa); no request-path effect
   - Verify: `HF_HUB_OFFLINE=1 docker compose --profile local-dbs up -d` with the api container on an internal-only network reaches /health/ready 200; `docker images` shows the size delta; `ls models/` on the host is no longer required.

11. **Rewrite docker-compose.yml to 3 core services (memory-api, memory-worker, db-migrate) + profile local-dbs (postgres, qdrant, dragonfly, openfga, openfga-migrate, openfga-db-init) + profile observability; delete memory-validate, model-fetch, neo4j, the served tier and gateway; worker gets deploy.resources.limits.cpus '2' and OMP_NUM_THREADS=1; relax the prod guard so blob.provider=filesystem is allowed with a persistent volume (today prod requires gcs while the shipped compose hard-sets filesystem, so the shipped stack cannot start in prod).**
   - Where: docker-compose.yml (rewrite; today 445 lines / 19 definitions / 10 started by default); src/memory_service/config/settings.py:612-613 (prod guard); tests/unit/test_deployment_profiles.py
   - Why: Operator supplies one .env with ~10 URL/credential fields and starts the stack; ingestion must not compete with query traffic for the same 8 cores (compose:238-240 documents the starvation).
   - Expected: One-file deployment for the VM with external stores; laptops keep a local-dbs profile.
   - Latency: worker CPU cap protects query p99 from ingest bursts
   - Verify: `docker compose config --services | wc -l` == 3; `docker compose --profile local-dbs config --services | wc -l` == 9; `MEMORY__SERVICE__ENVIRONMENT=prod uv run python -c 'from memory_service.config.settings import Settings; Settings()'` succeeds with filesystem blob.

12. **Move benchmark configuration out of the Makefile: one BenchEnv in benchmark/env.py (db urls, qdrant url, bifrost url, models, limits) replacing the seven repeated docker-run blocks; benchmarks run the shipped constants (no retrieval -e overrides); delete the unused BENCH_DB; keep the three command-line pass-through variables as they are (GNU make expands undeclared variables to empty - nothing to fix).**
   - Where: Makefile:121-172, :186-428; benchmark/env.py (new)
   - Why: Seven blocks of ~25 identical -e lines are where the shipped/benchmarked drift came from (Makefile:248-250 vs settings.py:431-432,455,476,487).
   - Expected: One place defines how a benchmark runs; -26 variables.
   - Latency: none
   - Verify: `grep -c 'MEMORY__RETRIEVAL__' Makefile` == 0; `make bench-locomo-judged LOCOMO_ARGS='--conversations 1 --sample 5 --out smoke.json'` still runs.

13. **Delete the benchmark-only weights and catalogue entries from the freeze: gliner2-base (806 MB, never imported), bge-reranker-v2-m3 (2.1 GB), qwen3-embedding-0.6b (1.1 GB), gte-multilingual-base (599 MB), granite-embedding-reranker-english-r2 (574 MB), bge-base/bge-small, granite-embedding-english-r2, the SPLADE weights (1.0 GB) and ms-marco-MiniLM-L6-v2 (566 MB incl. 479 MB of redundant exports) from models/MANIFEST.json and `make models-all`; keep a documented `benchmark/challengers.txt` for ad-hoc downloads.**
   - Where: src/memory_service/tools/download_models.py:86-135; models/MANIFEST.json; Makefile:56-60
   - Why: ~7 GB of weights on disk with no production code path; 'one package' means the catalogue equals the frozen set.
   - Expected: Disk and cognitive load; nothing else.
   - Latency: none
   - Verify: `uv run python -m memory_service.tools.download_models --list` prints exactly the frozen set.


## Phase 2 - Hot path: p99 < 300 ms and 20 rps on the target VM (every gate here is a TARGET-VM measurement)

**Goal.** Take the encoder from fp32 torch with unbounded threads to int8 ONNX with pinned threads and bounded concurrency, run 3 workers, cut the wire and serialisation cost, bound the tail, and prove it with the Phase-0 load scenario.

**Gate.** [TARGET-VM] `uv run python -m benchmark.load.run --base-url http://<api-host>:8080 --api-key <key> -u 20 -r 20 -t 300s --arm cold` from a separate host with Postgres/Qdrant/Dragonfly on their own hosts: aggregated.rps >= 20 sustained, endpoints['POST /v1/context'].p99_ms <= 300, failures == [], API container CPU <= 85% of 8 vCPU, RSS drift < 5% over the run; then `make bench-locomo-judged LOCOMO_ARGS="--conversations 2 --calls-per-minute 60 --out locomo_judged_v5_hotpath.json"` with evidence_recall >= 0.987 and strict >= (v4 strict - 0.01). Also run the 40 rps step (`-u 40`) once and record the knee in docs/MEASUREMENTS.md.

1. **Encoder: backend 'onnx' with the int8 file chosen by Phase-0 step 6, ORT SessionOptions intra_op_num_threads=2 passed through model_kwargs (settings.threads only calls torch.set_num_threads today, so the ORT session would otherwise use all cores), max_seq_length set explicitly to 512, a dedicated ThreadPoolExecutor(max_workers=1) + asyncio.Semaphore per process instead of asyncio.to_thread on the default 12-thread pool; same executor pattern for the NLI adapter. Reindex after the fingerprint change.**
   - Where: src/memory_service/adapters/models/embeddings.py:78-83, :97, :101, :125, :128; src/memory_service/adapters/models/nli.py:122; src/memory_service/config/settings.py:195
   - Why: The encoder is the floor of every request (178 ms mean today, engine.py:485); N concurrent encodes each fanning over all cores is the oversubscription benchmark/concurrency.py:3-6 names; bounded FIFO entry makes queue depth observable instead of collapsing CPU.
   - Expected: Encode stage from 178 ms (no-AVX2 fp32) to the measured E2 (expected 20-45 ms int8 on AVX2); p99 under load stops being thrash-bound.
   - Latency: p50 -130 to -160 ms in-process if E2 lands in the expected band; MUST be read from embedding_onnx_vm.json, not assumed
   - Verify: [TARGET-VM] per-stage encode p99 from memory_stage_seconds during the load run; `uv run python -m memory_service.tools.reindex --drop` completed; retrieval golden gate unchanged (`uv run pytest tests/eval/test_retrieval_gate.py -q`).

2. **Run 3 uvicorn workers via WEB_CONCURRENCY=3 (uvicorn.run already honours it when workers is None) with OMP_NUM_THREADS=2, per-worker Postgres pool 8+8, and readiness that lists only the components the API actually loads (embedding, nli) so a multi-worker start reports honestly.**
   - Where: src/memory_service/__main__.py:14-16; deploy/Dockerfile ENV (Phase 1 step 10); src/memory_service/config/settings.py:47-48; adapters/wiring.py:77 (components list)
   - Why: One process = one GIL for tokenisation, pydantic, JSON and qdrant-client parsing; at 20 rps that alone exceeds a core. Three processes also cut per-process pool pressure from ~18 to ~6 checkouts at the tail.
   - Expected: Throughput ceiling from ~10-15 rps (single GIL) to > 60 rps; p99 at 20 rps service-time-dominated instead of queue-dominated.
   - Latency: p99 -100s of ms at 20 rps (modelled; measured by the gate)
   - Verify: [TARGET-VM] `docker compose exec memory-api ps -o pid,rss,cmd` shows 3 workers each within the concurrency.json RSS band; load gate.

3. **Qdrant over gRPC (prefer_grpc=True, grpc_port from env, 6334 already published) and payload projection: with_payload = the include list of keys the engine and builder read (record_id, kind, text, text_hash, node_id, document_id, page, section_path, memory_type, subject, predicate, object, observed_at, contradicts, importance, confidence, representation, thread_id) - never tenant_id/visibility_keys/contributors/owner_principal/lifetime; keep the include list under a unit test that greps payload reads; on_disk_payload=False for the (small) memories collection; enable Qdrant scalar int8 quantisation for the dense vectors with rescoring as a measured option.**
   - Where: src/memory_service/adapters/search/qdrant_store.py:87, :126, :199 (include-list precedent), :206-213 (_hit), :225, :241, :259-263, :280, :292, :306; src/memory_service/ports/search.py:72; src/memory_service/modules/rag/indexer.py:325-349 (payload fields)
   - Why: Every hit returns the full payload including text[:2000] over REST JSON parsed under the GIL; on_disk_payload=True means a page-cache read per returned payload on the remote Qdrant host.
   - Expected: Smaller wire bytes and parse cost per request; fewer disk reads at the Qdrant tail (PLAUSIBLE, unmeasured - the search stage p99 from step 9 of Phase 0 is the before number).
   - Latency: search stage p50 -5 to -15 ms, p99 -20 to -40 ms (estimate; gate decides)
   - Verify: [TARGET-VM] memory_stage_seconds{stage="search"} p99 before/after in the load run; `uv run pytest tests/unit/test_retrieval_components.py tests/contract -q`.

4. **Depth from one knob: final_k=50 default with prefetch_k=fused_k=64 derived (RRF only reorders inside the prefetch union); benchmark depth 100/100 as a constant; verify with evidence_recall before freezing.**
   - Where: src/memory_service/config/settings.py:431-432, :455, :487; src/memory_service/modules/retrieval/engine.py:283, :349; benchmark/env.py
   - Why: v2 at an effective depth of 40 already reached evidence_recall 0.9871, so depth beyond ~50 buys <= 1.3% evidence recall while search payload and per-hit object churn scale linearly; 200/200 has never produced a 200-deep bundle.
   - Expected: Half the hit-object churn of 100/100 at the default; measured, not assumed.
   - Latency: search + assemble scale ~linearly with fused_k
   - Verify: locomo_judged_v5_hotpath.json evidence_recall >= 0.987 at the frozen default; lenient score not lower than v4 lenient - 0.01.

5. **Collapse the pre-retrieval round trips: fetch the revision keys once and pass the fingerprint into authz (drop the second get_many), cache _config_fingerprint at construction, issue the authz-scope GET + bundle GET + working-memory LRANGE as one Redis pipeline, and move the bundle cache SET into the tracked background task alongside the access bump.**
   - Where: src/memory_service/modules/context/builder.py:184-187, :203-215, :246, :262, :284-288, :291; src/memory_service/modules/authz/service.py:69, :74; src/memory_service/adapters/cache/redis_cache.py:78 (pipeline precedent)
   - Why: Two PG SELECTs on one connection and 3-4 sequential Redis commands before retrieval starts; a 30-80 KB SET awaited on the request path.
   - Expected: -1 PG round trip, -2 Redis round trips, -1 awaited SET per request.
   - Latency: p50 -3 to -6 ms on LAN, p99 -10 to -20 ms under pool contention
   - Verify: tests/integration/test_access_tracking.py drain() pattern extended to the cache SET; scope stage p99 in the load run.

6. **Serialise once: build the API dict with one bundle.model_dump(mode='json') + rendered, write those orjson bytes to the cache and return them via ORJSONResponse (response_model kept for OpenAPI through a typed model with the bytes returned directly); on a cache hit return the stored bytes without ContextBundle.model_validate_json + bundle_to_api; add GZipMiddleware(minimum_size=1024, compresslevel=5).**
   - Where: src/memory_service/modules/context/builder.py:246-251, :284-286, :624-625; src/memory_service/api/routers/v1/retrieval.py:252; src/memory_service/api/app.py:39, :49-54; pyproject.toml:30 (orjson already a dependency)
   - Why: The bundle is serialised/validated 4-6 times on a miss and parsed twice on a hit; responses are 30-80 KB with `rendered` duplicating every item's text (rendered_chars p50 ~9.1k, p95 11.3k).
   - Expected: HTTP-path CPU per request down (estimate 5-12 ms p50); wire bytes -70-80% for off-box callers. NOTE: this cannot move the in-process LoCoMo p50; it is measured only by the load gate.
   - Latency: HTTP p50 -5 to -12 ms, p99 -15 to -30 ms (estimate)
   - Verify: [TARGET-VM] load run p99 and `curl -H 'Accept-Encoding: gzip' -sD - ... | grep -i content-encoding`; contract tests keep the response shape.

7. **Rate limiter and Postgres plumbing for 20 rps: raise rate_limit_per_minute default to 6000 (env-bounded, per tenant+key), make the limiter one round trip (INCR+EXPIRE pipelined, or a per-worker local token bucket at limit/workers) and exempt /health; pool_pre_ping=False with pool_recycle=1800 and one retry on OperationalError; gate SQLAlchemyInstrumentor behind observability.otel_enabled and default it False (exporter is 'none' anyway).**
   - Where: src/memory_service/config/settings.py:36, :40, :499-500; src/memory_service/api/middleware.py:132-134; src/memory_service/adapters/db/engine.py:31, :42-44; src/memory_service/api/app.py:62
   - Why: 1200/min is exactly the 20 rps target on one key; the INCR is a Dragonfly RTT on every request; pre_ping is one extra round trip per checkout x up to 3 checkouts per request on a remote DB; the SQLAlchemy OTel wrapper runs on every statement regardless of the FastAPI hook.
   - Expected: No 429s at target load; -3 to -6 ms of round trips per request; less loop time per statement.
   - Latency: p50 -3 to -6 ms; p99 -10 to -20 ms
   - Verify: load run failures == [] with one tenant at 20 rps; `curl -s localhost:8080/metrics | grep http_requests_total | grep 429` absent.

8. **Bound the graph tail: add the missing GIN index on graph_entities.aliases (find_entities uses the JSONB ?| operator; only visibility_keys has a GIN today; the relation btree indexes already exist), give the prefetched traversal a wall budget implemented as wait_for on a shielded task that finishes in the background (never asyncio.timeout inside the unit of work), answer without graph facts on expiry and record it in diagnostics; derive the budget (100-150 ms) AFTER the encoder measurement, because the 3-hop traversal is hidden under the encoder only while the encoder is slow.**
   - Where: migrations/versions/0005_m8_graph.py:38,66,68 (existing indexes); new migration 0007 (GIN on graph_entities.aliases); src/memory_service/adapters/graph/postgres_store.py:302, :346-384; src/memory_service/modules/graph/retrieval.py:120, :151, :233; src/memory_service/modules/context/builder.py:291 (_track precedent)
   - Why: DOCUMENT_MULTI_HOP/ENTITY_RELATION routes run hops=3 with 3-5 sequential SELECTs; once encode drops to ~40 ms this becomes the critical path for those routes.
   - Expected: p99 on multi-hop/entity routes bounded at encode + search + budget; multi-hop accuracy must not regress (43 questions).
   - Latency: p50 unchanged; p99 -30 to -80 ms on graph routes
   - Verify: EXPLAIN on the find_entities query uses the GIN index; locomo_judged_v5_hotpath.json by_category.multi_hop >= v4 multi_hop - 0.02 and diagnostics count graph_budget_expired.

9. **Take the database writes off the read path's pool: coalesce the background access bumps per tenant (flush every 2 s or 200 rows) or route them through the task queue; cap the worker at cpus '2' with OMP_NUM_THREADS=1 and worker_concurrency frozen to 2; route consolidate()'s dense dedup through indexer.embed_cached and fetch candidates() once per turn instead of per candidate.**
   - Where: src/memory_service/modules/context/builder.py:301-332; src/memory_service/modules/memory/native.py:1036-1037; src/memory_service/modules/memory/pipeline.py:345-352; src/memory_service/modules/rag/indexer.py:100-103; src/memory_service/config/settings.py:94; docker-compose.yml memory-worker
   - Why: At 20 rps the bump issues ~20 UPDATE+COMMIT/s of up to 50 rows on the same remote Postgres and pool; dedup runs up to 20 uncached encoder passes per candidate on the shared 8 cores; the v2 -> v3 latency regression happened while ingest was draining on the same box.
   - Expected: Query-path pool and CPU protected from write-side work; ingest CPU per turn -10 to -400 ms worst case.
   - Latency: p99 protected during ingest bursts; no p50 change
   - Verify: load run with the 10% write mix keeps p99 <= 300; `uv run pytest tests/eval/test_memory_gate.py -q` (false_merge 0.0, dedup_recall 1.0) unchanged.

10. **Replace the two BaseHTTPMiddleware classes with pure ASGI middleware (same correlation and rate-limit semantics).**
   - Where: src/memory_service/api/middleware.py:34, :101; tests/contract (existing middleware tests)
   - Why: BaseHTTPMiddleware allocates a task group and buffers per request; small fixed loop cost at 20 rps x 3 workers.
   - Expected: ~0.5-1 ms of event-loop time per request.
   - Latency: p50 -0.5 to -1 ms
   - Verify: contract tests for X-Request-Id and 429 behaviour green; load run unchanged or better.


> **Superseded in part by [FREEZE-multilingual.md](FREEZE-multilingual.md) (2026-09-23).** An
> eleven-agent pass over primary sources corrected this phase in four places: the shipping graph is
> **ONNX fp32, not int8** (int8 measured slower and lossier here, and the graph file is inside the
> collection fingerprint, so the choice must be made on the VM *before* the one reindex); the
> fallback is **not** `multilingual-e5-small` (its only int8 graph targets an ISA this hardware
> lacks, it loses on both axes, and it needs query/passage prefixes this codebase never sends); the
> sparse tokenizer must be a **new** `text/` module, because both reference implementations this
> phase says to lift are themselves broken for Devanagari, Khmer, Myanmar and CJK sentence
> splitting; and the NLI swap carries a **published English regression** (MNLI 0.857 vs 0.903,
> ANLI 0.537 vs 0.579, FEVER 0.761 vs 0.777) that this phase never priced. Read that document
> first; the steps below remain useful as the task list.

## Phase 3 - Multilingual on the frozen set (encoder decision gated on Phase-0/2 numbers)

**Goal.** Non-Latin queries get dense AND sparse retrieval, extraction and grounding; the only generative path is language-pinned and guarded; the collection dimension stays 384.

**Gate.** `make bench-external` (SciFact-1000 subset on the server Qdrant): ndcg_at_10 >= 0.825 (rerank-off reference 0.8451, tolerance -2 points) and recall_at_10 >= 0.97; `make bench-multilingual` (new; MIRACL dev subsets ja/zh/ar/ru/de/hi/es via the ingestion path + MLDR 200 q/lang): recall@10 >= 0.60 per language and every non-Latin query yields >= 1 sparse term; `make bench-degenerate` non_latin case no longer INSUFFICIENT; `uv run pytest tests/eval -q` memory_gate false_merge_rate == 0.0; the Phase-2 load gate re-passed with the new encoder (E2 for the chosen model <= 60 ms).

1. **Freeze the dense encoder to ibm-granite/granite-embedding-97m-multilingual-r2 via onnx/model_quint8_avx2.onnx (fp32 onnx/model.onnx fallback), 384-d, no query/passage prefix (its README encodes queries and passages plainly; our unused prompt_name stays unused), IF Phase-0 step 6 measured E2 <= 60 ms and step 3 of this phase keeps SciFact within tolerance; otherwise keep granite-embedding-small-english-r2 int8 for English and escalate the multilingual/English trade to the user with both numbers (multilingual-e5-small with a self-exported quint8_avx2 graph is the fallback candidate).**
   - Where: src/memory_service/config/constants.py (FrozenModels); src/memory_service/adapters/models/embeddings.py:112 (prompt_name unused); src/memory_service/tools/download_models.py:64-70
   - Why: It is the only sub-100M multilingual retriever that keeps 384-d, is Apache-2.0 and ungated, and ships an AVX2 int8 graph; the published English cost (MTEB-v2 retrieval 50.1 vs 53.9) is why the freeze is gated, not assumed.
   - Expected: Retrieval for 52 enhanced / 200+ languages; LoCoMo neutral (evidence recall already 0.987).
   - Latency: same 12x384 compute as today's encoder plus a larger embedding table; E2 measured in Phase 0
   - Verify: embedding_onnx_vm.json E2 for the model; `make bench-external` numbers in the gate; `uv run python -m memory_service.tools.reindex --drop`.

2. **Unicode BM25 tokenizer bm25-v2: lift the Unicode word regex + CJK bigram logic from evidence.py into one shared memory_service/text module, keep the English suffix stemmer for Latin-script tokens only, split sentences on 。！？ as chunking.py already does, and use the shared module in sparse.py, grounding/lexical.py, memory/native.py, context/summaries.py, graph/service.py and graph/document_facts.py; bump the sparse fingerprint (forces new collections); add a ja/zh/ar/ru/de/hi fixture.**
   - Where: src/memory_service/adapters/models/sparse.py:18-26, :60-61, :99-100; src/memory_service/modules/context/evidence.py:41, :48, :56; src/memory_service/modules/ingestion/chunking.py:36; src/memory_service/modules/grounding/lexical.py:8-17, :51; src/memory_service/modules/memory/native.py:41-42, :124; src/memory_service/modules/context/summaries.py:25-26; src/memory_service/modules/graph/service.py:30; src/memory_service/modules/graph/document_facts.py:225-226; tests/unit/test_retrieval_components.py:112 (asserts bm25-v1)
   - Why: Seven ASCII regexes: a Japanese/Arabic/Cyrillic query has zero sparse terms so the hybrid degrades to dense-only (qdrant_store.py builds no sparse prefetch), the dedup/extraction tokenizer sees nothing, and CJK observations never split into sentences.
   - Expected: Sparse leg and lexical gates work for every script; ~300 duplicated lines removed; identical term semantics across dedup, evidence gate, BM25 and summaries.
   - Latency: neutral (regex cost)
   - Verify: `uv run pytest tests/unit -q` with the new multilingual fixture; `make bench-multilingual` sparse-term count per query > 0; memory_gate unchanged.

3. **NLI frozen to MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 loaded from its shipped onnx/model_quantized.onnx via optimum ORTModelForSequenceClassification (label order from config.id2label), eager-loaded in every API worker (keep /health honest - no lazy load); drop DeBERTa-v3-base-mnli-fever-anli; stop passing NLI provider=lexical in the benchmarks so grounding results become representative.**
   - Where: src/memory_service/adapters/models/nli.py:73-74, :122; src/memory_service/config/settings.py:238-264; Makefile:239 (NLI provider lexical) and the same line in bench-longmemeval/bench-external; tests/contract/test_nli_adapter.py; tests/eval/golden/grounding_claims.json
   - Why: The default NLI is English-only and fp32 (~700 MB); every grounding artifact on disk is representative=false because benchmarks run the lexical stand-in (grounding_gate.json nli_provider lexical-nli-v1).
   - Expected: Entailment for ~100 languages; grounding gate measured with the real model for the first time.
   - Latency: off the /v1/context path (cascade only runs with body.answer or on /v1/grounding); int8 halves the per-claim cost
   - Verify: `make model-test` with the new contract test; `uv run pytest tests/eval/test_grounding_gate.py -q` with representative=true in grounding_gate.json.

4. **LLM hygiene for thinking models: decouple `LLMSettings.max_tokens` (default 1024, clamp at llm.py) from any thinking budget by raising it to 4096; send `extra_body {thinking: {type: disabled}}` on deepseek-flash calls by default (cost and latency, not language); keep `model`/`fast_model` as a bounded Literal of models the gateway serves.**
   - Where: src/memory_service/config/settings.py (LLMSettings), src/memory_service/adapters/models/llm.py (clamp, extra_body), benchmark/locomo.py (MAX_TOKENS 16384 only needed while thinking is on)
   - Why: DeepSeek thinking is on by default and ignores temperature; a production deployment with the 1024-token clamp and a thinking model fails at the output budget (measured: 12% of assist calls came back empty at 200 tokens). The drift guard and language pin that were planned here were withdrawn by the user on 2026-09-22.
   - Expected: non-thinking answers return in hundreds of ms; the judged benchmark no longer needs a 16k budget.
   - Latency: ingest/judge calls faster; query path unaffected (LLM uses stay off at query time).
   - Verify: contract test that the DeepSeek request carries the thinking-disabled body; a 20-question judged smoke with MAX_TOKENS=4096 shows 0 exhausted outcomes.

5. **Tag every memory with lang (ISO 639-1, detected at ingest) and index it in the Qdrant payload.**
   - Where: src/memory_service/modules/rag/indexer.py:325-349 (payload); src/memory_service/adapters/search/qdrant_store.py:37-43 (_PAYLOAD_INDEXES); src/memory_service/modules/memory/native.py:391 (extract)
   - Why: Language-matched retrieval needs a per-record language fact; the payload has none today.
   - Expected: Filterable language; guard measurable per fact.
   - Latency: none (indexed keyword filter)
   - Verify: `make bench-multilingual` records lang per hit.

6. **New benchmark/multilingual.py reusing external_retrieval.py: MIRACL dev subsets (hi, ja, ar, zh, es, de; 213-2,896 queries per language) and MLDR (200 queries per language) through the ingestion path, A/B english-r2 vs the frozen multilingual encoder, query p95 on the VM; Makefile target bench-multilingual inside the runtime image.**
   - Where: benchmark/multilingual.py (new); benchmark/external_retrieval.py (ndcg/recall helpers); Makefile (new target next to bench-external:395)
   - Why: No multilingual measurement exists at all; no public CPU-latency comparator exists, so ours is the first.
   - Expected: The multilingual claim gets a number per language with the same harness as SciFact.
   - Latency: none (measurement)
   - Verify: `make bench-multilingual` writes benchmark/results/miracl.json with per-language recall@10/nDCG@10 and query_p95_ms, provenance.cpu_count == 8.


## Phase 4 - Accuracy: fix the ruler, then the answerer, then the bundle, then gated precomputation (zero LLM at query time)

**Goal.** Publish a number on the ruler the field uses (1,540 questions, category 5 excluded, mem0's verbatim judge prompt, GPT-4.1-mini-class answerer and judge) with public comparators, and lift the strict 2-conversation score through deterministic ingest/rendering changes first, LLM-at-ingest experiments last and only if they pass their own gates.

**Gate.** `make bench-locomo-judged LOCOMO_ARGS="--judge-ruler lenient --judge-model <gpt-4.1-mini-class via Bifrost> --calls-per-minute 60 --out locomo_judged_full_standard.json"` on all 10 conversations (new --judge-model flag; the 1,540-question subset reported separately from the adversarial category): pass = subset score >= 0.85 with in-process p50 within 10% of v5, reported next to Mnemis 93.9 / Hindsight 89.61 / Omi 86.6 / Zep 75.14 / Mem0 66.88 (same-ruler reproductions) and with the strict and refined rulers alongside; AND strict on the 2-conversation set >= 0.75 (`make bench-locomo-judged LOCOMO_ARGS="--conversations 2 --out locomo_judged_v6_accuracy.json"`). Each ingest experiment is adopted only if its own gated delta holds; anything below the bar is reported, never claimed.

1. **Harness: separate --judge-model from the answerer model (today BENCH_LLM_MODEL is both), report the standard ruler (1,540 subset, category 5 excluded, lenient = mem0's verbatim prompt) next to strict and refined, run all 10 conversations, and pace calls (429s documented in the Makefile).**
   - Where: benchmark/locomo.py:387-428 (_answer/_judge use one llm), :793-831 (CLI), :650-716 (aggregation); Makefile:165-170 (BENCH_LLM_MODEL)
   - Why: Every published 89-93 uses a GPT-4.1-mini/gpt-4o-mini judge with mem0's prompt on 1,540 questions; our 0.674 is a DeepSeek-judged strict number on 304 questions - the Mnemis paper itself scores Mem0 at 66.3 under its harness vs Mem0's self-reported 92.5, a 26-point swing from harness alone.
   - Expected: A defensible comparable number; the ruler gap becomes explicit for leadership.
   - Latency: none (retrieval untouched)
   - Verify: locomo_judged_full_standard.json has fields subset_1540_score, strict_score, refined_score, judge_model, answerer_model, comparators.

2. **Answerer A/B for the published number through Bifrost: gemini-2.5-flash-lite, a GPT-4.1-mini-class model, deepseek-flash with thinking disabled; same bundles, same judge.**
   - Where: benchmark/locomo.py:387-404; Makefile bench-locomo-judged (BENCH_LLM_MODEL / new BENCH_JUDGE_MODEL)
   - Why: The answerer alone is worth ~8 points (Nemori full-context 0.723 vs 0.806); 78/79 wrong answerable answers had gold in the bundle (to be confirmed by Phase-0 evidence_all_hit).
   - Expected: +5 to +10 apparent points on the comparable ruler with no system change; isolates how much of the remaining gap is the answerer.
   - Latency: none
   - Verify: three result files with identical bundle hashes and different answerer provenance; the best is the published one, all three are reported.

3. **Bundle rendering (deterministic): rank extracted facts above verbatim turns, collapse duplicate turn/window candidates in _dedup, adjacent-turn expansion by session_id/turn_seq payload fields (LoCoMo multi-hop gold is usually two adjacent turns), and index a session-window verbatim chunk (3-5 consecutive turns with the session date header) as a second memory kind.**
   - Where: src/memory_service/modules/context/builder.py:445 (memories cut), src/memory_service/domain/context_bundle.py (rendering order); src/memory_service/modules/retrieval/engine.py:569 (_dedup); src/memory_service/modules/rag/indexer.py:325-349 (session_id/turn_seq payload); benchmark/locomo.py:136-153 (session id/turn index as metadata)
   - Why: The answerer sees 40 raw first-person turns; evidence for replies and 'yesterday' references spans adjacent turns; Omi's biggest lever was verbatim windows (we already store per-turn verbatim, measured 0.163 -> 0.631 for verbatim on/off, so only the windowing residual is new and unmeasured).
   - Expected: Unmeasured; hypothesis +2 to +4 strict on single_hop/temporal; gate decides.
   - Latency: one more concurrent search kind (~+5-15 ms p50) - must stay inside the Phase-2 gate
   - Verify: locomo_judged_v6_accuracy.json vs v5 by_category; load gate re-run once.

4. **Absolute-date normalisation at ingest (deterministic): resolve 'yesterday', 'last Saturday', 'two weeks ago', 'four years ago' against observation.occurred_at, store both the phrase and the resolved date in the memory's temporal fields, render the resolved date in the bundle.**
   - Where: src/memory_service/modules/memory/native.py:102 (parse_date has no relative-date resolution), :201 (_EVENT_HINT detects only); src/memory_service/domain/context_bundle.py (date rendering)
   - Why: Removes the answerer's date arithmetic (deepseek-flash gets it wrong in recorded transcripts); temporal is 63/233 of answerable questions.
   - Expected: Unmeasured; temporal is already 0.81 strict / 0.85 lenient, so expect +1 to +2 at most.
   - Latency: none (ingest-time rules)
   - Verify: unit tests for 20 relative phrases; by_category.temporal in v6 >= v5.

5. **Route the existing 'summary' kind on person-shaped multi-hop and open-domain questions (today only GLOBAL_SUMMARY; 0 of 304 v2 bundles contained a summary) and wire LandingReflection + EntitySummaryService/BeliefService with a NEW dated, observed_at-ordered list formatter (today's derive_content is undated and created_at-ordered); gate on memory_gate false_merge 0.0 because landing supersedes single-valued slots.**
   - Where: src/memory_service/modules/retrieval/router.py:161 (needs_summaries); src/memory_service/modules/retrieval/engine.py:243-244; src/memory_service/modules/memory/derived.py:149-161, :234-248, :279; src/memory_service/modules/memory/landing.py:40-49, :135-136; src/memory_service/adapters/wiring.py:511; src/memory_service/config/settings.py:379-381
   - Why: 21 of 43 multi-hop questions are enumerations judged on completeness; a maintained per-(subject, predicate) dated list is the zero-query-cost version of the precomputed aggregates the 89+ systems rely on; the mechanism exists but is dead.
   - Expected: Unmeasured; hypothesis +2 to +4 strict on multi_hop; on LoCoMo landing only sees the ~0.36 rule facts per turn, so the ceiling is bounded until step 8 exists.
   - Latency: one more candidate kind in the existing concurrent gather (~+5-10 ms)
   - Verify: `uv run pytest tests/eval/test_memory_gate.py -q` false_merge 0.0; v6 by_category.multi_hop vs v5; bundles containing summaries > 0.

6. **Graph edge quality: stop emitting MENTIONS edges and entity rows from verbatim OBSERVATION memories, write entities + relations + the GRAPH revision bump of one index job in one transaction, remove the duplicate USER revision bump, then run the first judged with/without-graph ablation on LoCoMo.**
   - Where: src/memory_service/modules/graph/native.py:268-289; src/memory_service/modules/graph/service.py:126-131, :172-175; src/memory_service/modules/jobs/registry.py:75-85; src/memory_service/modules/memory/pipeline.py:500; benchmark/locomo.py --off graph
   - Why: Every capitalised phrase in a turn becomes a confidence <= 0.6 'mentions' edge competing for the 12 graph_fact slots; no judged graph ablation exists (flag_off_graph.json is the document golden set).
   - Expected: Neutral-to-positive accuracy; ingest -3 commits per turn; the ablation tells us what the graph is worth on conversations.
   - Latency: smaller neighbourhoods on graph routes
   - Verify: `make bench-locomo-judged LOCOMO_ARGS="--conversations 2 --off graph --out locomo_judged_v6_nograph.json"` compared with v6; tests/unit/test_graph_native.py updated for the verbatim exclusion.

7. **Permanently remove the per-clause ambiguous_worthiness/ambiguous_extraction LLM uses from every benchmark and from the default fast_uses (they stay in the LLMUse enum only if an operator opts in; better: delete them).**
   - Where: src/memory_service/config/settings.py:273-278, :308-309; src/memory_service/modules/memory/native.py:251-252, :452-461, :539; Makefile (Phase-0 step 1)
   - Why: Measured to hurt (v2 -> v3: 0.674 -> 0.661, p50 281 -> 572 ms) at ~1,900 would-be worthiness calls per 788 turns with no context.
   - Expected: Restores the v2 shape permanently; removes an ingest cost class.
   - Latency: ingest LLM calls per turn -> 0 by default
   - Verify: `grep -rn 'ambiguous_' src | wc -l` == 0 (or gated behind an explicit use with a test).

8. **EXPERIMENT (off by default, gated): one whole-SESSION extraction call through Bifrost as an async outbox job (not per turn): third-person dated episode narrative + atomic facts with turn-id citations, same language as input, schema-validated, deduped against rule candidates, verbatim turns kept (never replaced); adopt only if strict +3 on the 2-conversation set AND in-process p50 unchanged AND memory_gate false_merge 0.0 AND ingest CPU/tokens per session recorded.**
   - Where: src/memory_service/modules/memory/native.py:391 (extract), new _assist_session next to :539-619; src/memory_service/modules/jobs/registry.py (new job); src/memory_service/config/settings.py:273-278 (new LLMUse session_extraction)
   - Why: Every 89+ system precomputes narratives/atomic facts with an LLM at ingest; the controlled evidence says augment verbatim, never replace (arXiv 2601.00821: 43.9% vs 28.0%); our own per-clause attempt hurt, so the design must be whole-session, bounded, and judged before adoption.
   - Expected: Unproven locally; the largest remaining lever if it works (multi_hop evidence present 42/43, correct 16/43).
   - Latency: zero on the query path (one more candidate kind); ingest +1 LLM call per session, asynchronous, outbox-retried (dedup must handle paraphrased duplicates)
   - Verify: locomo_judged_v6_session.json vs v6; memory_gate with an LLM-facts fixture added to tests/eval/golden/memory_pairs.json.

9. **EXPERIMENT (gated): zero-LLM stand-in for Mnemis System-2 - embedding-based selection over precomputed per-session summaries (a 2-level hierarchy: session summary -> memories) added as an RRF list, no LLM at query time.**
   - Where: src/memory_service/modules/retrieval/engine.py:258-270 (gather), :349 (rrf); summaries from step 5/8
   - Why: System-2 is worth +4.2 to Mnemis but costs ~2.4 s of LLM per query; the only admissible version is one that fits the 300 ms budget.
   - Expected: Unknown; measured or dropped.
   - Latency: one more concurrent search (~+10 ms) - must pass the load gate
   - Verify: locomo_judged_v6_hier.json vs v6; load gate re-run.


## Phase 5 - Test suites that scale, benchmarks with public comparators, and a release gate that refuses stale artifacts

**Goal.** Every suite runs in parallel per worker with fresh artifacts tied to HEAD, and every headline (RAG, tools, facts, KG, grounding, memory, LoCoMo, LongMemEval, multilingual, load) has a same-ruler public comparator or an explicit 'none published' note.

**Gate.** `uv run pytest tests -n auto -m "not docker and not models" -q` completes with 0 failures in <= 5 minutes wall on the CI runner; `make validate` green, and src/memory_service/tools/release_gate.py refuses any artifact whose provenance.git_commit != `git rev-parse HEAD`; benchmark/results contains longmemeval_s.json, scifact_full.json, miracl.json, kg_multihop.json, tool_recall.json, load_test_head.json each with a `comparators` block naming the public number, its judge/answerer and its question set.

1. **pytest-xdist with one database per worker (DB_URL + worker_id, created and migrated once per worker), a session-scoped container per worker with the per-test TRUNCATE kept (the container build is the expensive part), Qdrant local ':memory:' collections reset per test, durations recorded via --durations and stored in tests.json; the architecture test relaxes to 'exactly one base URL'.**
   - Where: tests/conftest.py:42-43, :122-128, :141-159; tests/integration/conftest.py:95-108; tests/e2e/conftest.py:54-63; tests/unit/test_architecture.py:142; pyproject.toml:73-79 (dev deps), :237-246; benchmark/pytest_results.py:80; .github/workflows/ci.yml:50-57
   - Why: All DB-backed lanes (integration 101, e2e 24, eval 16, failure 6) build a full container and TRUNCATE 25 tables per test on one database, so nothing runs concurrently; three tables (message_versions, message_attachments, turn_run_links) have no tenant_id and TRUNCATE uses RESTART IDENTITY, so per-tenant isolation alone is not safe - per-worker databases are.
   - Expected: A 511-test run in the tens of seconds on 4 workers instead of ~2 minutes for 281 tests on a 2-vCPU box.
   - Latency: n/a
   - Verify: `uv run pytest tests -n 4 -m 'not docker and not models' -q --durations=20`; tests.json carries per-test durations and cpu_count.

2. **A models lane that runs inside the runtime image on every push (the 5 `models`-marked tests plus a new multilingual retrieval test with real weights), reusing the existing model-test docker form.**
   - Where: Makefile:89-100 (model-test); .github/workflows/ci.yml; tests/support_models.py:36-44
   - Why: torch/onnxruntime publish no macOS x86_64 wheels, so the real-weight tests only ever ran by hand; the hermetic suite runs hash/lexical stand-ins that prove nothing about the frozen models.
   - Expected: Real-model regressions caught in CI.
   - Latency: n/a
   - Verify: CI job 'models' green on a Linux runner; `make model-test` locally inside the image.

3. **Release gate freshness: refuse any artifact whose provenance.git_commit != HEAD or whose timestamp predates the newest change under src/; regenerate tests.json in CI.**
   - Where: src/memory_service/tools/release_gate.py:36 (_load), :214 (tests.json)
   - Why: tests.json is from 2026-09-15, commit 6293184 which is not in the current history, on a 2-vCPU box, with args naming a deleted directory; five gate files are a week old and the gate has no provenance check at all.
   - Expected: The gate cannot be satisfied by stale artifacts again.
   - Latency: n/a
   - Verify: `uv run python -m memory_service.tools.release_gate` fails on a deliberately stale artifact in a unit test.

4. **Merge the two conversational-memory harnesses: keep benchmark/locomo.py's ingestion/pacer/timings, move metrics and judge prompts from benchmark/public/{metrics,judge}.py into it, delete benchmark/public/beir.py in favour of external_retrieval.py, keep public/data.py for dataset loading; fix docs/TARGET_STACK.md's reference to a non-existent benchmark/public.py.**
   - Where: benchmark/public/*.py (1,305 lines) vs benchmark/locomo.py (855); benchmark/external_retrieval.py; docs/TARGET_STACK.md
   - Why: Two implementations of answer/judge/F1 and two nDCG implementations; LongMemEval must share LoCoMo's proven ingest path.
   - Expected: One judge, one F1, one ingest path for both conversational benchmarks.
   - Latency: n/a
   - Verify: `grep -rn 'benchmark.public' Makefile docs | wc -l` == 0 after the merge; LoCoMo smoke reproduces v5 within judge noise.

5. **LongMemEval-S judged run (never run) with the merged harness and the same judge as Phase 4; comparators: Mnemis 91.6 (GPT-4.1-mini), Hindsight 91.4 (Gemini-3), Mem0 94.4 (vendor ruler, top_200), Zep 71.2 (GPT-4o), Mastra 94.87 (gpt-5-mini answerer, 30k-token context, not retrieval).**
   - Where: Makefile:291-326 (bench-longmemeval); benchmark/public/data.py:26-27 (dataset repo)
   - Why: The second benchmark the field quotes; no result exists.
   - Expected: A second comparable number; likely the better one for us on a GPT-class judge.
   - Latency: n/a
   - Verify: `make bench-longmemeval LME_ARGS="--calls-per-minute 60"` writes longmemeval_s.json with per-type scores and comparators.

6. **RAG: rerun BEIR SciFact on the FULL 5,183-document corpus against the server Qdrant (the only comparable_to_published run scored 0.012 in local brute-force mode) and add NFCorpus; report nDCG@10 with comparable_to_published=true only; comparator: the frozen encoder's published BEIR figure and the granite-small card (50.9 avg).**
   - Where: Makefile:395-416 (bench-external); benchmark/external_retrieval.py; benchmark/results/scifact_rerank_on.json (invalid, retire)
   - Why: Every SciFact artifact on disk is either invalid or a subset flagged not comparable.
   - Expected: A real RAG number next to a published one.
   - Latency: n/a
   - Verify: `make bench-external BENCH_LIMIT=''` writes scifact_full.json with comparable_to_published=true and corpus 5183.

7. **KG multi-hop on a public set (HotpotQA distractor or MuSiQue subset through /v1/graph/query + /v1/context) and tool-memory recall on a public trajectory set (tau-bench / ToolBench traces) instead of the 2-document kg_facts golden and the fixture trajectories; facts/grounding gates re-run with the real NLI.**
   - Where: tests/eval/golden/kg_facts.json (2 documents); tests/fixtures/tool_trajectories.json; benchmark/kg_multihop.py and benchmark/tool_recall.py (new)
   - Why: Every self-authored gate reads 1.0 on 2 documents / 35 pairs / 40 cases; leadership numbers need an external ruler for KG and tools too.
   - Expected: First external KG and tool numbers; the 1.0 gates stay as regression checks.
   - Latency: n/a
   - Verify: kg_multihop.json and tool_recall.json with question counts >= 200 and a comparators block ('none published' where true).

8. **Retire superseded artifacts (scifact300/600/1000 ablations, flag_off_*, golden_off_*, locomo_smoke/judge_smoke/judge_baseline, locomo_rerank_on/off, degenerate.json, results/baseline-sandbox/) into benchmark/results/archive/ with a README; keep only artifacts the gate reads or the docs cite.**
   - Where: benchmark/results/ (60 top-level JSON + 12 in baseline-sandbox)
   - Why: Withdrawn or pre-fix numbers next to current ones invite the wrong citation.
   - Expected: The results directory equals the evidence.
   - Latency: n/a
   - Verify: `ls benchmark/results/*.json | wc -l` <= 25 and docs/MEASUREMENTS.md lists each remaining file with its ruler.


## The freeze

### Models

Dense encoder: ibm-granite/granite-embedding-97m-multilingual-r2 (ModernBERT 12 layers x 384 hidden, 384-d, Apache-2.0, ungated) on the sentence-transformers ONNX backend using the shipped onnx/model_quint8_avx2.onnx (fallback onnx/model.onnx fp32), max_seq_length 512, ORT intra_op 2 threads, one encoder executor + semaphore per worker, fingerprint includes the ONNX file name - CONDITIONAL on Phase-0 step 6 (2-thread query p95 <= 60 ms on the 8 vCPU VM) and Phase-3 gate (SciFact-1000 nDCG@10 >= 0.825; published English MTEB-v2 retrieval is 50.1 vs 53.9 for small-english-r2); if either fails, freeze granite-embedding-small-english-r2 int8 ONNX for English and escalate the multilingual trade with both numbers (fallback candidate multilingual-e5-small, MIT, self-exported quint8_avx2). Sparse: BM25 with Qdrant server-side IDF and the Unicode tokenizer bm25-v2 (Unicode words + CJK bigrams, English suffix stemmer for Latin script only); no SPLADE. Reranker: none (deleted; SciFact -5.2 nDCG, sign test p=0.012, 21x latency). NLI: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT, XNLI en .871 / zh .803) via onnx/model_quantized.onnx, eager-loaded, used only by /v1/grounding and /v1/context with body.answer. Parser: docling with baked artifacts. LLM only via Bifrost: fast_model google/gemini-2.5-flash-lite (grounding_judge, query_expansion, chunk_context, summaries, session_extraction if adopted), model google/gemini-2.5-flash; allowed override set as a Literal {google/gemini-2.5-flash-lite, google/gemini-2.5-flash, deepseek/deepseek-flash} where the DeepSeek entry sends extra_body thinking disabled (cost and latency); LLM uses default to none at query time. Image bakes: encoder int8 (~98 MB + tokenizer), NLI int8 (~280 MB), docling (669 MB); HF_HUB_OFFLINE=1; nothing downloaded at `docker compose up`.

### Settings kept as environment

Exactly the topology/credential surface, all bounded (~35 fields): service.environment (dev|staging|prod), service.port, service.log_level (Literal), service.log_json, service.rate_limit_per_minute (int, default 6000, per tenant+key), service.workers (WEB_CONCURRENCY, 1-8, default 3); database.url (SecretStr), database.pool_size, database.max_overflow; cache.url (SecretStr); tasks.worker_concurrency (1-4, default 2); authentication.mode (trusted_dev|jwt), jwt_issuer, jwt_audience, jwt_jwks_url, trusted_dev_api_keys (SecretStr list); authorization.openfga_api_url, openfga_store_id, openfga_model_id, openfga_api_token (SecretStr); blob.provider (filesystem|gcs), filesystem_root, chat_bucket, file_bucket, gcs_project; search.qdrant_url, search.qdrant_grpc_port, search.qdrant_api_key (SecretStr); models.llm.enabled, base_url, api_key (SecretStr), model (Literal set), fast_model (Literal set); observability.otel_exporter (none|otlp), otel_endpoint. Everything else is a constant.

### Settings frozen into constants

Moved to src/memory_service/config/constants.py (change = code change + `make reindex`, never an env edit): every model id/revision/file/dimension/max_seq_length/threads/batch_size/normalize/device; retrieval: exact/dense/bm25/graph on, fusion rrf, rrf_k, final_k 50 with prefetch_k = fused_k = ceil(1.25 x final_k) derived (benchmark depth 100 as a benchmark constant), rerank and splade removed; context: token_budget 8000, conversation_token_budget 2000, memories_max 50, graph_facts_max 12; graph: max_visited 200 (query-time 40), hops per route, prefetch wall budget (derived after the encoder measurement); memory_intelligence: dedup thresholds 0.92/0.90/20, landing_reflection_k 8, belief_min_support 2, entity_summary_min_facts 2; documents: chunking, max_chunk_tokens, contextual_chunks; NLI premises_per_claim / max_claims; LLM uses/fast_uses lists, max_tokens 4096, timeout, retries; cache TTLs; db/redis/qdrant timeouts and statement_timeout; header names; rate_limit_burst; max_body_bytes; on_disk_payload per collection (memories False); otel_enabled False.

### Settings and code removed

memory_intelligence.provider (mem0/langmem/cognee) + adapters/intelligence/* + extras memory-providers/cognee; graph_enrichment.provider values graphiti/docling_graph/cognee + graphiti_* (5) + both providers + neo4j profile + graphiti extra; served-model tier: models.embedding/reranker/nli.{url, api_key, timeout_seconds, max_retries} + models.sparse_url + tools/model_server.py + adapters/models/remote.py + memory-model script + compose memory-embed/rerank/nli/sparse + model-gateway (litellm) + deploy/served-models.yml + deploy/model-gateway.yaml; RerankerSettings entirely + reranker weights + memory-rerank; models.sparse_model + retrieval.splade + FastEmbedSparseEncoder + models/sparse; provider_policy; evaluation + performance_budgets (to benchmark/ with modules/evaluation); retrieval.contextual_chunks (no reader); fusion dbsf/none (unimplemented); search.provider 'memory' + qdrant_local_path and tasks 'inline'/'memory' as env (test seams via build_container overrides only); database.echo; authentication gcp_iam/mtls + gcp_id_token.py + jwt_hs256_secret; cache valkey/redis/disabled spellings; observer_hot_window_messages/observer_batch_messages/observer_max_notes + LLMUse observation_refinement + observer.py + memory.observe enqueue; LLMUse ambiguous_worthiness/ambiguous_extraction (measured harmful); YAML source + MEMORY_CONFIG_FILE; MEMORY_MODELS_DIR (unread) and the duplicate HF_HOME; embedding.model/model_path/provider/dimension/device/normalize/batch_size/max_tokens as env; documents.fallback_parser one-value Literal; pageindex extra; model-fetch service and ./models mounts; benchmark-only weights from the catalogue (gliner2-base, bge-reranker-v2-m3, qwen3-embedding-0.6b, gte-multilingual-base, granite-embedding-reranker-english-r2, bge-base/small, granite-embedding-english-r2, Splade_PP_en_v1, ms-marco-MiniLM-L6-v2).

### API: enums and bounds

Requests: RecallRequest.kinds -> list[Literal['chunk','memory','summary']] (min 1, max 3; the documented 'fact' kind does not exist and unknown kinds must 422, not return empty); GET /v1/memories memory_type -> list[MemoryType]; RecordRequest.status -> ToolStatus, DeclaredTool.source -> ToolSource, RecordRequest.visibility -> Visibility (today Visibility(bad) raises ValueError into the generic 500 handler), sub_calls -> list[SubCallIn] max 64 (SubCall.model_validate ValidationError also 500s today - add a handler mapping in-handler pydantic ValidationError to 422); VerifyItem.kind -> Literal over _SOURCE_RANK keys; /v1/files visibility Form -> Visibility. Bounds: ContextRequest.token_budget le=16_000; ContextRequest.answer and VerifyRequest.answer max 8_000 chars; GraphQueryRequest.max_visited le=500; VerifyRequest.items max 50 x 4_000 chars, unused max 20; RecallRequest/ContextRequest.document_ids max 100; custom_metadata (ScopeBody, observation, thread, message, files) max 32 keys / 8 KB serialised / depth 2; tools args, schema, output_schema max 64 KB serialised, output max 256 KB (blob spill exists at 4 KB); ProcessingHintsIn.custom_type max 64; source_system max 100 everywhere. Responses: RecallResponse.query_type -> QueryType, RecallItem.representation -> Representation, MemoryResponse memory_type/lifetime/visibility/scope_level/temporal_status -> enums, DocumentResponse.status -> DocumentStatus(STAGED|READY|FAILED) and archive_status enum, JobResponse.status -> JobStatus, ClaimVerdictBody.verdict/method -> Literal, VerifyResponse.source -> Literal['bundle','items','query'], FactOut.status -> TemporalStatus, ReadyResponse.status -> Literal['ready','degraded','not_ready'], RecallResponse.evidence -> EvidenceReport model, ContextResponse memories/knowledge/graph_facts/summaries/evidence -> typed ContextItem models with extra='forbid'. Every enum field carries a description (tests/contract/test_openapi_contract.py:95); docs/openapi.json regenerated with `make openapi`; SDK client str parameters and models moved to the same Literals.

### Secrets

database.url, cache.url, qdrant_api_key, openfga_api_token, llm.api_key and trusted_dev_api_keys become SecretStr and Settings.redacted() masks all of them (today it masks only trusted_dev_api_keys).


## Kill list (do not do these; each was measured or sourced)

- Do NOT adopt Mnemis System-2 hierarchical LLM browsing at query time: +4.2 points for ~2.4 s of GPT-4.1-mini per query (3,637.65 s / 1,540 questions) - it is the exact thing the < 300 ms zero-LLM retrieval budget forbids; only its precomputed structure is admissible, and only through Phase-4 step 9's zero-LLM stand-in with a gate.
- Do NOT adopt EverMemOS's sufficiency-check re-query (second retrieval round on 31% of questions), EMem-G's LLM recall filtering, Letta's agentic file search, MemoryOS's 4.9 LLM calls per response, or Mastra's ~30k-token observation log in context - every one of them puts an LLM or a second round on the request path.
- Do NOT put Qwen3-Embedding-0.6B/4B, bge-m3, bge-reranker-v2-m3 or Qwen3-Reranker-8B on the CPU path: measured 2,222 ms mean / 3,545 ms p95 per query and 35.8 s per 20 pairs on our 4-core class (benchmark/results/embedding.json, README.md:137-140).
- Do NOT turn cross-encoder reranking back on with any model: measured worse on SciFact-1000 (nDCG 79.33 vs 84.51, 95% CI [-0.0917,-0.0120], p=0.012) at 11,151 vs 533 ms p50 (settings.py:433-454, docs/MEASUREMENTS.md section 3e); keep at most a benchmark challenger under benchmark/, never in src/.
- Do NOT enable SPLADE or the OpenSearch multilingual neural-sparse encoder: a BERT-sized document pass at ingest (measured ~4 s/document class), English-only WordPiece for Splade_PP_en_v1, and a 15-language cap; the Unicode BM25 tokenizer with server-side IDF is the correct sparse leg.
- Do NOT re-enable per-clause ambiguous_worthiness/ambiguous_extraction LLM calls at ingest: the only LLM-ingest experiment on disk lost accuracy and doubled query latency (v2 -> v3: 0.674 -> 0.661, p50 281 -> 572 ms) at ~1,900 would-be worthiness calls per 788 turns.
- Do NOT build dynamic batching or a separate model server for query encodes at 20 rps: a 10 ms window collects a mean batch of ~1.2 (20 rps x 0.01 s), and tools/model_server.py only adds an HTTP hop on top of the same executor problem; batching matters for ingest embeddings only.
- Do NOT lazy-load the NLI (or any model) on first request: it contradicts the /health design (docker-compose.yml:189-194 keeps readiness red until weights are resident) and moves minutes of load into a user request; load eagerly, int8, in every worker.
- Do NOT delete run_periodic (inline_queue.py:46), run_until_idle (procrastinate_queue.py:224) or _layer_from_predicate (ports/intelligence.py, a pydantic before-validator) as 'dead code' - they are test seams and a validator; only search_features, get_authz, _alias, PolicyDecision/PolicyProvider, _LATE_LICENSES and observer.py are actually dead.
- Do NOT claim 'we beat Mnemis' (or anyone) from the 2-conversation, deepseek-judged, adversarial-inclusive strict number: different question set (304 vs 1,540), different judge (DeepSeek runs ~8-12 pp stricter), category 5 handling and top-k; the Mnemis paper scores Mem0 at 66.3 where Mem0 self-reports 92.5 - a 26-point harness swing. Only Phase-4's standard-ruler run may be compared.
- Do NOT promise > 94% on LoCoMo under any strict or human-aligned judge: the audited theoretical maximum is ~93.6% and the best human-aligned rescored system is 82.65%; > 94 figures exist only on vendor rulers (GPT-5-class judge, top_200, ~7k tokens/query, or 30k-token full context).
- Do NOT cite depth 200/200/100 for any latency or accuracy effect: no run has ever produced a bundle deeper than 40 memories (v2 and v3 records p50 = max = 40 despite the Makefile -e flags); measure the effective depth first (Phase-0 step 2).
- Do NOT use gzip level 9, compress bodies under 1 KB, or count gzip/orjson/middleware savings against the in-process LoCoMo p50 (benchmark/locomo.py:509-510 times builder.build only); they are HTTP-path changes measured solely by the load gate.
- Do NOT wrap a prefetched graph task in asyncio.timeout while it is inside `async with uow_factory()`: cancelling mid-query can leave the pooled connection aborted; use wait_for on a shielded task that completes in the background (builder._track pattern) and return partial facts.
- Do NOT run Locust on the same 8 vCPU box as the API, and do NOT run the load test against one tenant/key while rate_limit_per_minute is 1200 (= exactly 20 rps): both make the measurement wrong before it starts.
- Do NOT assume per-test tenant isolation is safe for parallel tests: message_versions, message_attachments and turn_run_links have no tenant_id and the TRUNCATE uses RESTART IDENTITY; use one database per xdist worker.
- Do NOT switch to gte-multilingual-base, EmbeddingGemma-300m, jina-embeddings-v3 or nomic-embed-text-v2-moe: 3x the compute of a 97M/384-d model at 20 rps on 8 vCPU, trust_remote_code or gated/non-commercial licences, and 768/1024-d collection rebuilds.
- Do NOT add LLM-authored facts at ingest (Phase-4 step 8) before the fresh baseline, evidence_all_hit and the memory-gate LLM-facts fixture exist; and never let them replace verbatim turns (arXiv 2601.00821: substituting artifacts for verbatim text forfeits 15-22 points).
- Do NOT keep three retrieval tunings (defaults, .env, Makefile -e) or any env override of prefetch/fused/final/memories_max/token_budget: the shipped service and the benchmarked service must be the same artefact.
- Do NOT delete provider=memory/inline stand-ins from the test suite - delete them from the env surface only; hermetic tests keep them through build_container(overrides=...).

## Risks

- The entire latency and throughput plan is conditional on one unmeasured number: the 2-thread int8 ONNX query-encode p95 on the target VM (Phase-0 step 6). No measurement exists on AVX2 hardware or on any ONNX backend; if E2 > 100 ms the 20 rps / p99 < 300 ms targets are infeasible on this VM class and must be reported as such rather than tuned around.
- English retrieval regression from the multilingual encoder: granite-embedding-97m-multilingual-r2 publishes MTEB-v2 English retrieval 50.1 vs 53.9 for small-english-r2; the Phase-3 gate (SciFact-1000 nDCG >= 0.825, golden recall unchanged, LoCoMo evidence_recall >= 0.987) can fail and force an explicit English-vs-multilingual decision with the user.
- int8 dynamic quantisation quality on ModernBERT (encoder) and mDeBERTa (NLI) is unmeasured here; keep the fp32 ONNX graphs as fallbacks in the same fingerprinting scheme and gate on golden/grounding sets before freezing.
- The ORT thread pin may not take effect through sentence-transformers model_kwargs; if intra_op threads stay at the default, N concurrent encodes re-create the oversubscription and p99 collapses under load - verify with the concurrency benchmark and the load run, not by reading the config.
- Cost and pacing of judged runs: v2 took 1,905 s for 304 questions (~6.3 s/question end-to-end at k=10); a full 1,986-question LoCoMo run is ~3.5 h of paid LLM calls per configuration, and the Makefile documents 429s and circuit-breaker refusals; Phase-4 needs a pacing budget and a GPT-class judge key in Bifrost.
- Ruler mismatch is the largest reputational risk: the honest comparable number requires a GPT-4.1-mini-class answerer and judge with mem0's verbatim prompt on the 1,540-question subset; any leadership claim made from the deepseek-judged 2-conversation strict number will not survive scrutiny.
- Reindex cost: the dense fingerprint (ONNX file), the sparse fingerprint (bm25-v2) and the payload additions each force a full re-embed of every chunk and memory; on the shared 8 vCPU VM this must run in the CPU-capped worker and is measured in hours for a large corpus.
- Removing the served-model tier, challengers and the litellm gateway gives up the documented scale-out path (ADR 0019) and every A/B comparison in src/; acceptable for a single VM with external stores, but the ADR must be marked superseded and benchmark/ must carry any future challenger.
- Multi-worker uvicorn triples eager model loads at startup; start_period (600 s today, based on a stale three-model measurement) must be re-measured with the int8 set, and readiness must stop listing components the API no longer loads (wiring.py:77).
- Graph wall budget can drop facts on multi-hop routes (43 questions, already the weakest category); the budget must be derived after the encoder measurement and the multi_hop by_category score gated, or the latency win becomes an accuracy loss.
- Ingest-time session extraction (Phase-4 step 8): hallucinated facts, non-deterministic output across outbox retries (paraphrased duplicates defeat normalized_hash dedup), token cost per session, and language drift; the gate (strict +3, p50 unchanged, false_merge 0.0, drift 0) is deliberately hard and the experiment may simply fail.
- Rate limiter and pool sizing at 20 rps are modelled, not measured: 1200/min equals the target per key, and pool 10+10 with up to 3 checkouts per request plus the background bump sits at its cap at the tail; the 5 s pool_timeout would show up as p99 spikes if the Phase-2 sizing is wrong.
- The effective 40-memory cap is unexplained: settings sources rank env above dotenv (settings.py:597-603), yet v3 bundles were 40 with -e FINAL_K=100 / MEMORIES_MAX=100; until Phase-0 step 2 records the resolved config, any depth or token-budget change may be a no-op.
- External facts pinned today can move: Gemini pricing/model availability, Bifrost's DeepSeek passthrough behaviour, Qdrant gRPC semantics for payload selectors, and HF-hosted ONNX files (revisions must be pinned in models/MANIFEST.json at freeze time).
- Test-suite parallelisation touches every DB-backed fixture; tests asserting on global counts or sequence ids will need per-worker databases and may surface latent ordering assumptions; budget a day of triage.
- The 300 ms target has no agreed ruler today; this plan defines it as HTTP POST /v1/context cold-arm p99 from a separate host at 20 rps for 5 minutes with remote stores - leadership must accept that definition, or the number will be argued about after it is measured.
