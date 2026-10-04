# Complex consolidation evaluation — 2026-09-26

Status: **stopped on provider availability; no new paired answer-accuracy result.** Do not
reuse the earlier ten-question synthetic diagnostic as the result of this experiment.

The clean replay completed 190/369 turns using 141 exact cached responses. Gemini 3.8
Flash returned HTTP 503 on the first missing extraction and on one delayed retry. Its
successful READY preflight did not establish availability for this workload. The pending
observation remains unprocessed; no native fallback was admitted for it. No reader or
judge calls were made, and no new retrieval/answer accuracy or latency is claimed.

Total this turn: **204 inference gateway requests**, including route checks: 183 HTTP 200,
11 HTTP 402, three HTTP 429, three HTTP 400, two HTTP 404 and two HTTP 503. The HTTP 400
route rejections were not forwarded model calls. Reported OpenRouter spend: **$0.0457248**;
Gemini spend was not reported. All experiment containers/jobs are stopped. Available
paid credit or restored provider capacity is needed to finish.

## Frozen comparison

`benchmark/complex_consolidation.py` selects the shortest LoCoMo conversation by turn
count, without using previous predictions: conversation index 1, all 369 turns across
19 sessions. Evaluate every multi-hop and temporal question: 11 and 26 respectively.
The complete history, including distractors and image captions, goes into each arm.
Gold answers and the existing strict ruler were frozen before any reader generation;
neither enters ingestion or reader prompts. Arm order alternates during retrieval and
answer generation. Query-time LLM assistance is disabled in the context builder.

The native arm uses current native extraction with landing consolidation disabled. The
assisted arm enables contextual extraction and landing consolidation, with explicit
background reflection. This measures the combined approach, not reflection in isolation.
The primary reader was planned as GPT-4o through OpenRouter, identically in both arms;
provider availability may require an explicitly recorded replacement before answers.

Use real PostgreSQL, Qdrant and ONNX embeddings, with in-process authorization/queue,
lexical NLI, disabled cache and the existing unthreaded LoCoMo ingestion convention.
Builder latency is one warm-up plus three measurements per question/arm, not HTTP/load
p99 or generation latency. All source-recall denominators use original annotations.

Three dataset caveats were recorded before reader calls, without changing primary gold:
q9 names Jean/John instead of Gina/Jon; q24 includes an unrelated evidence reference;
q31 says six months although its annotated dates imply five elapsed months. See
`codex_complex_consolidation_dataset_caveats.json` and `_source_audit.json`.

## Provider interruptions and clean recovery

The first attempt finished native ingestion but stopped assisted ingestion at session 11,
212 turns. It made 193 gateway requests: 182 HTTP 200 and 11 HTTP 402. Reported cost was
$0.0457248. GPT-4o reflection failed first; one delayed retry failed again. GPT-4o-mini
continued until its own credit rejection. Some reflection outputs also hit the existing
600-token cap and failed JSON validation despite HTTP 200.

OpenRouter account status showed no purchased credit and $0.19933365 total usage. The
key's $99.80066635 remaining limit is a spending ceiling, not available credit. The user
was asked asynchronously to add a small balance. Do not increase billing/key limits.

Route probes found configured Gemini 3.6 Flash quota-exhausted, 2.5 Flash deprecated,
3.6 Pro unavailable, and two OpenRouter free routes rate-limited. Google's deprecation
response recommended Gemini 3.8 Flash. Bifrost's old key allowlist excluded it. Attempts
to update the old key returned 404 for both its cache and persisted IDs. A new route
record, `gemini-current-model-route`, now enables **only Gemini 3.8 Flash using the same
existing environment credential reference**. No new account/key or billing change was
made. The route returned READY successfully. Registration details are in
`codex_complex_consolidation_gemini_route_update.json`; credentials are not exported.
[Bifrost's documented key API](https://docs.getbifrost.ai/api-reference/providers/update-a-key-for-a-provider)
was consulted. Actual quota is project-dependent; see [Google's rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

Because failed extraction had already admitted native fallback in the first attempt,
do not repair it by replaying processed observations or score it as complete assisted
ingestion. A **fresh assisted tenant** is rebuilding instead, retaining the complete
native control. Exact successful extraction responses were recovered from Bifrost logs:
141 records, all validated against the current schema. Inputs match on full prompt,
schema and use; no semantic guessing, manual extraction or gold-based selection.

`benchmark/recorded_ingestion.py` reuses these responses and stores new successful ones.
Its evaluation-only failure guard bypasses production's graceful fallback, leaving a
pending observation safely resumable before memory admission. Per-observation checkpoints
prevent duplicate submission. Production behavior is unchanged. A test verifies the
failure passes through LLMAssist without admitting fallback; another verifies resumption
does not insert the observation twice. The recovered seed cache is immutable; the live
cache appends new results.

The clean replay uses cached mini outputs and Gemini 3.8 Flash for missing extraction.
Reflection is deferred until historical bulk ingestion completes, rather than invoking
it after each historical session. This is a recorded experiment deviation, not an
all-GPT-4o or unchanged-model comparison. Stored LLM reflections are PRIVATE under the
current application policy, whereas the LoCoMo reader is a different shared-corpus user.
The interrupted first attempt contains 70 current PRIVATE reflections. Two of its
reflection responses reached the 600-token cap; they were followed by production fallback.
The final summary explicitly counts reflections returned to that reader; do not assume
stored insights help answers. Do not widen permissions to improve a benchmark score.

## Files and safe continuation

Database: `memory_complex_consolidation` at schema 0009. Qdrant suffix:
`__complex_consolidation`. Tenants: `complex-consolidation-native` (complete),
`complex-consolidation-assisted` (interrupted, preserve), and
`complex-consolidation-assisted-replay` (clean resumable attempt, stopped after 190 turns).
Its pending source is D11:1. No original benchmark
database or artifact was reset.

Artifacts under `benchmark/results/`:

- `codex_complex_consolidation_design.json`: frozen 37 questions, gold and ruler.
- `codex_complex_consolidation_contexts.json`: original interrupted attempt, model usage
  and complete native snapshot; `_interrupted_1.json`/`_interrupted_2.json` preserve earlier stops.
- `codex_complex_consolidation_replay_contexts.json`: active progress, pending observation,
  source map, subsequent model responses and (when finished) paired contexts.
- `codex_complex_consolidation_ingestion_cache_seed.json`: 141 recovered responses.
- `codex_complex_consolidation_ingestion_cache.json`: seed plus future responses.
- Route manifests/results, credit status, settings and deviations use the same prefix.
- `benchmark/complex_consolidation_eval.py` prepares blinded judge jobs and reports paired
  wins/losses, missing judgments, source coverage, timing and costs. Never score failures
  as wrong answers. Manual audits must remain separate from raw judge results.

Resume the replay with the model-capable Docker image, not host hash embeddings:

```bash
docker run --rm --name complex-consolidation-replay --entrypoint python \
  -v /Users/ricky/usage_data/agent-memory-service:/app -w /app \
  -e PYTHONPATH=/app/src:/app \
  -e CONSOLIDATION_DB_HOST=host.docker.internal \
  -e MEMORY__MODELS__LLM__BASE_URL=http://host.docker.internal:8091/v1 \
  -e MEMORY__SEARCH__QDRANT_URL=http://host.docker.internal:6333 \
  -e HF_HUB_OFFLINE=1 memory-service-memory-api:latest \
  -u -m benchmark.complex_consolidation --resume \
  --prefix codex_complex_consolidation_replay \
  --cache benchmark/results/codex_complex_consolidation_ingestion_cache.json \
  --extraction-model gemini/gemini-3.8-flash \
  --reflection-model gemini/gemini-3.8-flash --rpm 4
```

Check whether this container is already running before resuming. Do not launch a second
writer. Stop on quota failures; never silently change models or replay paid calls when
the exact response is already cached. After completed ingestion, answer generation must
use the same model/prompt/cap within every pair and record any change from planned GPT-4o.
The reader manifest contains no gold. Keep interrupted artifacts for accounting.

For a funded mini-based continuation, replace both model arguments with
`openrouter/openai/gpt-4o-mini`; the existing cache remains valid and original route/model
provenance is preserved. Record the model change. The manifest's original planned reader
is GPT-4o; choose and record a reader affordable for **all 74 answers**, consistently in
both arms, before generation. Do not silently mix reader models or spend a small top-up
on a larger reader that cannot finish. The judge helper accepts `--prefix
codex_complex_consolidation_replay` for this clean attempt.

Final focused validation: 24 tests pass; Ruff and Pyright pass. No application code,
deployment, original broad accuracy artifact or original p99 result changed in this turn.
This one-conversation comparison, even when completed, cannot establish whole-LoCoMo
accuracy or generalization beyond the benchmark.
