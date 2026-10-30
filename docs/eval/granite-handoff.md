# Granite 2B evaluation: handoff

How to run the memory service's evaluations with a local Granite 3.3 2B model (llama.cpp
behind the Bifrost gateway). Use it to start a test in a new chat or on a new machine. Every number below
was measured without a model; what you measure with Granite is new.

## Branches

| branch | what it holds |
|---|---|
| `main` | Point 9 (worker reliability), point 7 (cross-conversation replacement and answer coverage; ADR 0038, 0040) and the read path (ADR 0041); PR #34. |
| `claude/cool-maxwell-w5avo8-followup` | The follow-up fixes to the read path (multi-hop chains, restricted instructions, lists under the relevance floor) on top of `main`. |
| `claude/cool-maxwell-w5avo8-evaldet` | The harness fixes an LLM-mode run needs (below). Merge it, or run the harness from it, before an LLM-mode capability eval. |

## The model and the gateway

Run llama.cpp's server with a context it can hold on a 16 GB, 4-CPU host and without a
prompt cache (the cache grew until the kernel killed the server twice):

```bash
llama-server -m models/llm/granite-3.3-2b-q4km.gguf --alias granite-3.3-2b --jinja \
  -c 8192 --cache-ram 0 -n 512 --temp 0.1 -t 4 --host 127.0.0.1 --port 8099
```

Put Bifrost in front of it on `127.0.0.1:8080`, with a custom provider `granite` that points at
`http://127.0.0.1:8099`. Create a virtual key, then export `LLM_BASE_URL=http://127.0.0.1:8080/v1`,
`LLM_GATEWAY=bifrost` and `LLM_API_KEY`. Load one model stack at a time: an API + worker pair takes about 2.3 GB.

## The evaldet fixes (an LLM-mode run is not valid without them)

* **Per-request timeout 600 s.** A CPU model answers one call at a time, and the harness's
  earlier default dropped answers that were on their way.
* **`CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS=60000`.** It gives query expansion a deadline Granite can
  meet. An expansion that misses its deadline is counted as a fallback.
* **The model-reach gate (`benchmark/capabilities/gate.py`, `UNREACHED_MAX = 0.01`).** A run
  fails when more than 1% of its model calls never reached the model (timeouts, open circuit,
  gateway errors). Without this gate, such a run measures the no-model fallback and calls it a
  model result.

## Commands

The capability eval runs on a store of its own, dropped at the end. Use `SET=core` for no LLM, and
`MODE=llm SET=llm_core` for the model:

```bash
CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS=60000 make eval-capabilities-isolated MODE=llm SET=llm_core \
  CAPEVAL_ARGS="--timeout 600"
make eval-capabilities-isolated MODE=nollm SET=core
```

LoCoMo source recall (all 1,986 questions). It needs a private Qdrant and a fresh database
that `alembic upgrade head` has migrated:

```bash
make bench-locomo-prepare
BENCH_SEARCH=qdrant BENCH_EMBEDDING=frozen BENCH_DENSE=ensemble \
MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@localhost:5432/<private db> \
MEMORY__SEARCH__QDRANT_URL=http://localhost:16333 MEMORY__SEARCH__QDRANT_GRPC_PORT=16334 \
  python -m benchmark.native_source_retrieval --data benchmark/data/locomo10.json --output locomo.json
```

LongMemEval-S retrieval, using the same environment. `--questions 18` is the stratified sample (seed 7) that
main's baseline used; leave it out to run the full set:

```bash
python -m benchmark.longmemeval_retrieval \
  --data benchmark/data/longmemeval/longmemeval_s_cleaned.json --output lme.json --questions 18
```

## Main's no-LLM baselines (70198fa)

| evaluation | main |
|---|---|
| capability eval, core set, macro accuracy (two runs, run date 2026-10-11) | 0.4789 / 0.4804 |
| capability eval, dev / held-out split | 0.51 / 0.45 |
| abstention, new chat: correct refusals / false refusals | 0.31 / 0.35 |
| abstention, same chat: correct refusals / false refusals | 0.15 / 0.25 |
| context p95 / search p95 (capability gate) | 65.2 / 44.1 ms |
| LoCoMo source recall @10, all answerable (two runs) | 0.8073 / 0.8065 |
| LoCoMo @10: single_hop / multi_hop / temporal / open_domain | 0.8964 / 0.5659 / 0.8705 / 0.5124 |
| LongMemEval-S turn recall @10, 18 questions (session recall in the same file) | 0.8611 |
| stress_accumulated pass rate, 1,410 questions | 0.170 |
