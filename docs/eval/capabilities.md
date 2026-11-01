# The capability eval suite

> The measuring instrument for the memory capabilities an agent needs: eleven of them, each
> asked across five domains and five languages, through the real service. It was built before
> any capability work so that every later change has to show it raises accuracy without
> lowering any cell or slowing the context down. It changes nothing in the service.

**Up:** [Documentation](../README.md) · **See also:** [12 · Testing and gates](../guide/12-testing-and-gates.md)

---

## What it measures

Each capability is a question a user asks long after telling the memory something. What is
scored is what the service hands the agent for that question, and, in harness mode, the answer
the agent's model writes from it.

| # | Capability | Asserted on the context (and, in harness mode, on the answer) |
|---|---|---|
| 1 | Single-fact recall | the stated value (a code, an id, a name) is there |
| 2 | Multi-hop across sessions | each fact the answer joins, from different sessions, is there |
| 3 | Temporal / recency | the newest state is there; the superseded one is not served as current |
| 4 | Update & contradiction | the replacement is there; the replaced value is not served as current |
| 5 | Abstention | nothing answers it: the context says `INSUFFICIENT`; an answer declines and invents nothing |
| 6 | Forgetting / decay | the current items are there; dense records about a closed or removed thing are not |
| 7 | Performance at scale | over a three-year tenant: every record an aggregate needs is there (partial credit says how much), the right keys in the answer, and the latency |
| 8 | Procedural / skill memory | the standing instruction (or the learned tool procedure) reaches the matching request; the answer follows its format |
| 9 | Test-time learning | the corrected procedure is there; the wrong one is not served as current |
| 10 | Implicit personalization | the evidence of the trait (past interactions) reaches the request it should shape |
| 11 | Explicit personalization | the stated rule reaches the request, its exception included; the answer obeys it |

Capability 12 (multimodal) is out of scope.

## The dataset

`benchmark/capabilities/data/`, one JSON file per capability (`NN_<capability>.json`), plus the
scale specs (`scale_*.json`), the scorer's refusal phrases (`_languages.json`) and the modes and
samples (`_modes.json`).

| | |
|---|---|
| Scenarios | 105: per capability 6 retail/warehouse (the InventorySmart case plus five more) and one each for personal assistant, IT support, finance/HR and field service; capability 7 is one retail tenant with six questions plus one tenant per other domain |
| Queries | 111 (one per scenario; two for the InventorySmart explicit-rule case, with and without its exception; six on the retail scale tenant) |
| Phrasings | 10 per query, one of each kind: `direct`, `indirect`, `paraphrase`, `typo`, `long_winded`, `implicit`, `negated`, `later`, `terse`, `imperative` - 1,110 per language |
| Languages | English, German, Spanish, Arabic and Hindi: every setup turn and every phrasing translated (identifiers, codes and numbers unchanged); plus 44 code-switched phrasings (`mixed`: Hinglish, Spanish-English, German-English, Arabic-English) on one scenario per capability |
| Time | sessions days to months apart (`day` relative to the run date); generated logs `every_hours` apart; the scale tenant three years of weekly records (`--scale large`: four years daily, ~1.05M tokens per language, opt-in) |
| Distractors | near-duplicates next to every target (the Warehouse 2 code next to Warehouse 3's, `CC-4400` next to `CC-4471`, Forklift #3 next to #4, a different lot, another store) |
| Split | dev / held-out by query, deterministic: within a capability, queries ordered by the hash of their id alternate between the halves (5/5, 6/5 for capability 11). **The held-out half is never used to tune anything**: a change is developed against `--split dev`, and the held-out number is read only when the change is final |

A scenario is **sessions of turns** (`USER`, `ASSISTANT`, `EVENT`), written through
`history.add` with their `occurred_at`, one thread per session; a session may instead
**repeat** a templated log line (`repeat`: count, interval, seeded value ranges) or **record
tool runs** (`tool_runs`: `record_tool` as the agent, then each run labelled by a person
through the tenant's administrator credential, how an agent learns a procedure: a run's own
`system` report of success teaches nothing, and with only a service key the label waits for
review, so the procedure scenarios need the administrator path; a
query about a learned procedure asserts its steps in order with a regex, which only the learned
skill's item meets — a small toolbox's tool hints name every tool on their own). A
scale scenario names a **corpus** spec; its aggregate questions carry the computation
(`aggregate`: record kind, calendar period relative to the run date, group, value, top-k) and
the ground truth is computed from what was generated (`generate.py`).

An **assertion** names anchors: an invariant string (`"8492"`), a regex (`{"re": ...}`) or a
language map (`{"en": "repaired", "de": "repariert", ...}`):

```json
{"contains": ["PackagingCorp"],
 "absent": [{"text": "Uline", "unless": ["PackagingCorp"]}],
 "answer": {"contains": ["PackagingCorp"], "absent": ["Uline"]}}
```

`contains` groups must each be met by some item; an `absent` anchor is a **stale leak** when a
current item carries it without any of its `unless` anchors - "we dropped Uline, PackagingCorp
supplies us now" is not a leak, "our supplier is Uline" served as current is; an item the
service marks as ended (a fact with `valid_to`, a superseded memory) never leaks. `abstain`
means the right answer is "I do not know". `table_columns` checks an answer's markdown table.
`answer` overrides for the answer surface. Matching is on normalised text (NFKC, case-folded,
every script's digits as ASCII, thousands separators dropped, Arabic marks folded); a number
never matches inside a longer number.

`python -m benchmark.capabilities.validate` (and the eval suite) checks that every `contains`
and `absent` anchor occurs in what its scenario writes, in every language - an assertion no
text can meet would read as a capability failure forever - and that every scenario has all
five languages, all ten phrasing kinds and a past date. The scoring code holds no domain word;
retail lives only in the data.

## How a run works

`benchmark/capabilities/runner.py` drives the running service through the SDK, nothing else:

1. **Determinism and isolation.** What a run writes depends on the dataset, the selection
   and the run date alone. Each scenario and language is one user whose id is a stable,
   opaque hash of the two (`identity`); its threads are named from that user and the
   scenario's session or question names. Nothing is drawn per run: the service indexes a
   memory's subject with its text, and a random user id moved every cosine by 0.04-0.05,
   enough to flip 10-28 cells near the relevance floor between two runs of the same code.
   The run date (`--run-date`, `CAPEVAL_RUN_DATE`; default the UTC date the run starts) is
   recorded as `meta.run_date`, the naming scheme as `meta.identities`. Because the names
   repeat, isolation comes from the store: the isolated stack is a new database, Qdrant and
   Redis db per run, and a run refuses an eval tenant (`capeval-<mode>`, or `--tenant`) that
   already exists (`--reuse` aside). On a shared stack the Makefile names a new tenant per
   run (`CAPEVAL_TENANT`). The mode's model policy is set on the tenant, and a service key is
   issued for the run and revoked after it.
2. **Writing.** Sessions in time order; each session is written only after the previous one's
   processing jobs finished, as a conversation days apart would be.
3. **Readiness - the service's own signals, never a sleep.** Every write's processing job
   reaches a terminal state (`GET /v1/jobs/{id}`; archive staging, scheduled a minute out by
   design, excluded); the outbox is drained (`memory_outbox_backlog`); with
   `MEMORY_WORKER_METRICS_URL`, two consecutive worker samples taken after the last write show
   no job waiting on the indexing queues (`chat-fast`, `memory-extract`, `embedding`); the
   enrichment queues (`graph`, `summary`) are then given up to 30 minutes to drain, and what
   still waits on them when the questions start is recorded (`queued_at_ask`): graph
   enrichment is throttled per tenant and can trail indexing by hours, so a run that waited
   for it would measure the queue, not the service. Last, each scenario's last statement is
   findable through `search` in its own thread. The wait is recorded as the ingestion lag. In
   harness mode the model's own readiness is awaited too: a one-token completion through the
   gateway must succeed (with backoff, up to 15 minutes) before the first question; a run in
   which any model call still failed is scored `error`, written as `harness.failed.json`
   beside the result, and is never a baseline.
4. **Asking.** Every phrasing on three surfaces: the pushed context in a **new session** (a
   fresh thread - the user comes back another day; the primary number), the pushed context in
   the thread of the scenario's **most recent session** (`same_session`) with `window=False`
   (the framework keeps the transcript; memory must carry the rest), and the agent's **pull** (`search(kinds=[memory, episode, message])`, new
   session). The context is asked as agent-harness asks it: `format="full"`, a 2,000-token
   budget. Questions are asked one at a time so the latency is the request's own.
5. **Scoring and the record.** Every phrasing on every surface is a record (verdict, reason,
   what was missing or leaked, the first items that came back on a failure, the latency); the
   raw records are saved before anything summarises them (`<mode>.raw.json.gz`).

The host's load average is sampled while the questions are asked; above 1.0 per core the
latency is flagged as an upper bound (`latency_trustworthy: false`) and the gate refuses it.
Run every baseline on a quiet machine, alone - `HEAVY=` wraps it in a machine-wide lock.

## Modes

| Mode | Tenant policy | What answers | Run |
|---|---|---|---|
| `nollm` | no model use at all, reads deterministic | the context itself is scored | the full dataset, every language |
| `llm` | the live policy: every use but `memory_restatement`, `read_assist` on, one gateway model (`CAPEVAL_LLM_MODEL`, required) | the context is scored | sample `llm` |
| `harness` | as `nollm` | an agent wrapped by agent-harness: a function agent whose prompt is the pushed context and whose model (`CAPEVAL_ANSWER_MODEL`) answers; the answer is scored | sample `harness` |

The samples are deterministic (`_modes.json`): per capability the InventorySmart scenario first,
then the first scenario of each other domain in id-hash order. `llm`: English, three phrasing
kinds (`direct`, `paraphrase`, `implicit`), every capability but scale (extracting a
1,100-message tenant with a CPU model takes hours; scale is measured in the other two modes).
`harness`: English, the `direct` phrasing, every capability. Each question is asked as a user
of its own, its scenario written again for that user (users named `question-language/v1`), so
what its run records — the transcript, the outcome — lands in that user's memory and no answer
becomes another question's memory: a user's memories are read for that user only, and an
agent's own notes are bound to the user it runs for. A scenario with several questions is
therefore written once per question. Grounding sampling is off (`without={"grounding"}`).

## Metrics

| | |
|---|---|
| accuracy | the share of phrasings that pass, per capability x language x domain x split x phrasing kind x surface x placement |
| macro accuracy | the mean of the capabilities' accuracies (each capability weighs the same) |
| retail track | the macro accuracy over the retail scenarios only |
| origin cases | accuracy on the eleven InventorySmart scenarios |
| partial credit | `contains` groups met / required (for scale: how much of the evidence came back) |
| abstention | precision of `INSUFFICIENT`, its recall on the unknowable questions, the false-refusal rate on the answerable ones |
| stale leak rate | phrasings with a stale value served as current / phrasings that have one |
| latency | context p50/p95/max, search p50/p95, answer p50/p95 (harness), per capability p95; scale: share within 5 s and 10 s |
| ingestion | turns, jobs, failed jobs, write seconds, the readiness signals and their wait; every worker queue's depth when the run started (`queues_at_start`) and the enrichment backlog when the questions started (`queued_at_ask`) - a backlog the run inherited is recorded, never waited out or worked around |

## Files and the gate

| File | What it is |
|---|---|
| `benchmark/results/capabilities/<mode>.json` | the run: summary, per capability, cells, failure reasons with examples, provenance (commit, host load, packages), the service's `/version` and the tenant policy |
| `benchmark/results/capabilities/<mode>.md` | the same as a readable summary |
| `benchmark/results/capabilities/<mode>.raw.json.gz` | every record as measured, saved before anything summarises it; `python -m benchmark.capabilities.report <mode>.raw.json.gz` rebuilds the summary |
| `benchmark/results/capability_gate.json` | the thresholds: every cell's recorded accuracy and the context p95, per mode, for one dataset hash |

The gate is a **ratchet: accuracy may only go up**. Every cell's threshold is its best
recorded accuracy. `gate record` raises a threshold whenever it measures a cell higher, and
keeps it for any cell whose questions did not change, even across a dataset change. A cell
measured lower is refused. The ratchet starts at main `420bbf3`. `targets` sit beside the
thresholds: goals that are reported and never failed. A cell the baseline has and
the run lacks fails, **context p95 may rise at most 10 %**, a latency measured on a
loaded machine fails, and a run on another dataset (`dataset.sha256`: the capability files,
the scale specs, `_languages.json`, the scorer and the generator) or another selection is refused - a dataset change
re-records the baseline in the same change, where both numbers are reviewed. A threshold is
never lowered to let a change through; an improvement is recorded by re-recording.

More rules:

- **Noisy cells.** Repeat runs recorded together give each cell its lowest value; a cell
  they disagree on is recorded as noisy, with the range they spanned. A drop beyond that
  range that is confined to noisy cells exits with code 3 (`RERUN`) instead of failing. One rerun decides it: `make eval-capabilities-gate RERUN=<second run>.json`
  (`gate check --rerun`). A noisy cell fails only if both runs are below its threshold less its range.
- **Empty runs.** A run that measured nothing is refused as a baseline and fails as a check.
- **New cells.** A cell the run measured but the baseline does not hold yet is listed as a new
  cell; it is gated once the baseline is re-recorded.
- **Noise allowance, once.** A threshold that is already the lower of the repeats gets no
  further allowance. Only a carried best value above what was measured may be undercut, and
  then by no more than the cell's own spread. Below that floor a noisy cell asks for one
  rerun, and the rerun's own value decides. The rerun must be of the same dataset,
  selection, mode and code, or it is refused.
- **What resets thresholds.** Any byte change to `scoring.py`, `generate.py` or the language
  data (`_languages.json`) changes the dataset hash and resets every threshold. A change to
  one capability's file resets only that capability's cells.
- **What the hash covers.** The dataset hash includes the scorer (`scoring.py`) and the
  generator (`generate.py`), so a change to what a verdict means re-records the baseline in
  the same change.

### The gate runs on a store of its own

`make eval-capabilities-isolated MODE=nollm SET=core` runs the eval against a memory service
of its own (`benchmark/capabilities/isolated_stack.sh`). The stack is:

- a PostgreSQL database (`capeval`);
- a Qdrant instance on its own storage path, since the service has no collection-prefix
  setting;
- an in-memory OpenFGA;
- a Redis database number;
- a blob directory;
- `memory-api` and `memory-worker` from this checkout, on their own ports, with no model
  gateway (`CAPEVAL_GATEWAY=1` adds one, for `MODE=llm`).

In llm mode the service is started through `benchmark/capabilities/stack_service.py`: its own
entry points with one frozen-tuning override (`Overrides.llm`), the model call's timeout,
`CAPEVAL_LLM_TIMEOUT_SECONDS` (default 600; recorded as `meta.stack`). The shipped 30 s
(`constants.LLM`, two retries, a breaker that opens after five consecutive failures) is sized
for a hosted model. The eval's gateway serves Granite 3.3 2B on the measuring host's CPU at
about 8 tokens a second alone and 4 per request with four requests at once: a 160-token summary
takes 19-29 s alone and 47-64 s beside three others. At 30 s nearly every long call timed out
three times and fell back, and the breaker then failed the rest fast: main's llm_core baseline
logged over 1,000 `llm.assist.fallback`. On the English llm_core subset (122 turns) the same
stack with the shipped timeout used the model 85 times and fell back 58 times (19 exhausted, 39
circuit open); with 600 s it used it 190 times and fell back 0 times (wall 956 s → 1,078 s).
Key, model name, llama-server slots and prompt length were all checked and were not the cause.

The read path had the same mismatch, unseen: model-assisted query expansion waits
`query_expansion_timeout_ms` (250 ms) and then searches the query as written. Granite needs
seconds, so every expansion was abandoned - llama-server cancelled 1,007 requests in one llm_core
run - and the cancellation was counted nowhere, so the API reported no model call at all while
each context carried the 250 ms wait (llm_core context p95 312 ms against nollm's 63 ms). The
service now counts a missed deadline as the use's fallback, and the llm stack sets the deadline
too (`CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS`, default 60,000; recorded as `meta.stack`). The llm
read path then really asks the model, and its context latency is the CPU model's: recorded,
not gated (`latency_gated: false` for llm_core).
`meta.llm_calls` records each process's calls by use and outcome, so a run that measured the
model-free path says so, and the gate holds it to that: an llm run in which more than 1 % of the
model calls never reached the model (`exhausted`, `circuit_open`, an HTTP error; the model's own
`invalid_output` or `empty_output` is measured, not excused) fails `gate check` and is never
recorded as a baseline (`model_unreached`).

That guard came from the next llm_core run, with the timeout fixed: 1,499 calls answered over
2.3 hours of indexing, then 1,135 fallbacks (api 910, worker 225; 6 + 8 exhausted, the rest
circuit open), every one after the moment the model server died. The kernel's memory-cgroup OOM killer had
killed the gateway's llama-server (3.6 GB resident) as the indexing wait ended: the host's one
14 GB cgroup held it, the eval's memory-api and -worker (2.3 GB each, the encoders), and another
memory service. The model server is part of the measuring host, not of this checkout: its
footprint is set where it is started. `-c 8192 --cache-ram 0` (the longest prompt plus answer
measured was 1,063 tokens, so four slots share 8,192 with room) instead of 16,384 and a 512 MB
prompt cache take its anonymous memory from 3.6 GB to 2.1 GB (plus 1.5 GB of mapped model file).
Even so, the llm stack (model server about 3.6 GB resident, API and worker about 2.3 GB each)
does not fit beside a second memory service in a 13.7-14.3 GB cgroup: the isolated stack's
memory wait now refuses to start it there rather than let the OOM killer end it hours later.

Before it creates anything, the script refuses to start if any of these holds:

- the database name does not match `^capeval`;
- a Qdrant port is the live one (6333/6334);
- a port already answers;
- WORK does not resolve under `/var/tmp/capeval*`;
- the Redis database is 0, a reserved one (`CAPEVAL_REDIS_RESERVED`) or not empty;
- `PGHOST` is not local, unless `CAPEVAL_ALLOW_REMOTE_PG=1`;
- `CAPEVAL_LOGS` resolves inside (or to) the WORK of any isolated run, this one's included.

Every path it may delete (WORK, LOGS) is resolved with `realpath -m` before these checks, and
numbers are read as numbers (`00` is database 0, `06333` is port 6333).

One stack runs at a time: an exclusive lock sits beside WORK. The service starts with a clean
environment (`env -i`) and from WORK, so no `.env` is read. Both database URLs, including the
`DIRECT_URL` that migrations, the job queue and graph sessions use, point at the eval
database. Everything the script creates is written to a manifest. Teardown removes only what
the manifest names, and signals a process only if its command line is still the one started
here. A process that ignores SIGTERM (a worker draining a job) is sent SIGKILL after
`CAPEVAL_KILL_WAIT` seconds (default 20), so it cannot outlive its stack and hold the next
run's port. When the command ends, the stack is dropped, whether the run passed, failed or was
interrupted. The script deletes the Qdrant collections through the API, stops the instance,
removes its path, drops the database, flushes the Redis database and removes the blob
directory. It records the disk before the stack starts and after it is gone; a gate run
leaves nothing. Machine paths and ports are environment variables (`QDRANT_BIN`,
`OPENFGA_BIN`, `CAPEVAL_DB`, `CAPEVAL_API_PORT`, ...). It waits (`CAPEVAL_MEM_WAITS` x 10 s)
for 5 GB of free memory (`CAPEVAL_MIN_MEM_MB`) before starting the two processes that load the
encoders. Free means the smaller of the host's `MemAvailable` and the room left in every memory
cgroup above the script (its limit less anonymous memory, shmem and mapped files - what
reclaim cannot give back; v1 or v2): in a container the
cgroup is the limit the OOM killer enforces, and it can be gigabytes below the host's figure.
Measured here: the host reported 6.2 GB available while the 14 GB cgroup had under 2 GB, and
the encoders' start was killed.

The `llm` and `harness` modes need the model gateway, so they still run against a shared
stack. There a run's own purge, below, is what cleans up.

### A run leaves the service as it found it

Every mode forgets what it wrote when it ends, a failed run too
(`benchmark/capabilities/cleanup.py`). It goes through the service's own API, as a tenant
administrator would:

- every memory of each thread written is forgotten (`DELETE /v1/memories/{id}`);
- every thread written is deleted (`DELETE /v1/threads/{id}`).

The users and threads a run wrote are kept in its raw records (`meta.written`). The purge's
counts are in `meta.cleanup`. `--keep` leaves the writes in place for a later `--reuse`, and
a `--reuse` run never purges. A kept run is purged later with
`python -m benchmark.capabilities.cleanup <run>.raw.json.gz`.

The service has no tenant purge. Both deletes are soft: the vector index points go (an index
job removes them, and Qdrant reclaims the space as it vacuums segments), but the PostgreSQL
rows, and the graph entities and relations built from the writes, stay.

What one full `nollm` run leaves behind on the live stack, without the purge (measured
2026-10-08 from the eval tenants' share of each store):

- Qdrant: about 70,000 memory points, about 1.3 GB.
- PostgreSQL: about 350 MB. That is memories about 115 MB, graph relations about 165 MB,
  graph entities about 35 MB, messages and observations about 25 MB, and read records about
  12 MB.
- The run's result files: about 0.6 MB.

The purge returns the Qdrant share. The PostgreSQL share stays.

The purge is slow. Forgets commit at about 45 a second, but each one queues its own
`memory.index` job, and the worker clears those at about 9 a second. Freeing a full run's
points therefore takes about 2 hours of worker time. The next run's readiness wait (the
`embedding` queue) waits for it.

### Two gate sets: the PR gate and the full run

A full `nollm` run takes about two hours of a quiet machine. Most of that is capability 7:
8,938 turns to write and extract, and an enrichment backlog to wait on. So a change is gated
on a fast set, and the full set runs nightly:

| Set | What it runs | Baseline | When |
|---|---|---|---|
| `SET=core` | every capability but scale; all five languages and the code-switched phrasings; every phrasing kind and every surface. Questions start once the run's own queued work, summaries and graph enrichment included, has drained (bounded at 15 minutes; about 4 in practice). A run takes about 17 minutes, the isolated stack included | its own, `nollm/core` (cells and context p95), ratcheted: each cell's threshold is its best recorded value (two runs recorded together count as their lower value). A run fails below the cell's floor. The floor is the threshold itself, or the threshold less the cell's own spread when the threshold is a carried best value. A noisy cell gets one rerun | every PR that touches capability work |
| `SET=llm_core` (`MODE=llm`) | a fixed stratified subset of core: per capability (scale aside), the InventorySmart origin scenario and the other-domain scenario with the fewest turns. All five domains, every language and the code-switched phrasings, three phrasing kinds: 20 scenarios, 697 turns, 326 phrasings per surface (`_modes.json` lists them). It runs on the isolated store through the live gateway and the CPU model (`CAPEVAL_GATEWAY=1`, `CAPEVAL_LLM_MODEL`) | its own, `llm/llm_core`, ratcheted like `nollm/core` | every PR that touches capability work |
| `SET=scale` | capability 7 alone, every language | held to the full baseline's scale cells; its latency is not compared | the PR of a change that targets scale, and nightly |
| `SET=full` (default) | everything | **no baseline yet**: record it on main first (the 2026-10-08 full runs are pre-B1 and unreproducible) | nightly, once recorded |

`make eval-capabilities-isolated MODE=nollm SET=core` writes `nollm_core.json`, and
`make eval-capabilities-gate MODE=nollm SET=core` checks it. A set's baseline is keyed
`mode/set` in `capability_gate.json`. A mode's own sample, `llm` in llm mode or `harness` in
harness mode, stays keyed by the mode. Sets are defined in `data/_modes.json`. They narrow
only by capability, which keeps the dataset hash and every cell's definition unchanged.

```bash
make eval-capabilities MODE=nollm                 # the full set, every language
make eval-capabilities MODE=llm                   # sample llm, through the model policy
make eval-capabilities MODE=harness               # sample harness, agent-harness + its model
make eval-capabilities-scale                      # opt-in: the ~1M-token tenant (capability 7)
make eval-capabilities-isolated MODE=nollm SET=core  # the PR gate set (no scale), on a store of its own
make eval-capabilities MODE=nollm SET=scale       # capability 7 alone, held to the full baseline
make eval-capabilities-gate MODE=nollm SET=core   # fail on any drop against the baseline
make eval-capabilities-baseline                   # record the measured runs as the thresholds
make eval-capabilities-validate                   # every assertion satisfiable in every language
```

Environment: `MEMORY_URL`, `CAPEVAL_ADMIN_KEY` (a credential that may onboard a tenant - the
platform key, or a development key on a laptop), `MEMORY_WORKER_METRICS_URL` (the worker's
metrics port), `CAPEVAL_LLM_MODEL` / `CAPEVAL_ANSWER_MODEL`, and for harness mode the
agent-harness environment (`LLM_BASE_URL`, `LLM_API_KEY`; `HARNESS_PY` is its Python).
`CAPEVAL_ARGS` passes runner options: `--languages`, `--split dev`, `--capabilities`,
`--domains`, `--kinds`, `--scenarios`, `--placements`, `--surfaces`, `--scale`.

The fast slice runs with the eval suite (`make eval`): `tests/eval/test_capability_eval.py`
validates the dataset, pins the scorer, the generator, the split and the gate, and drives the
runner end to end against an in-process service through the SDK (hermetic stand-ins; it pins
the wiring, not the quality). The live, model and harness runs are the targets above and are
not part of CI.

## The baseline

The recorded numbers, with what produced them, are in the result files and their summaries:
[`nollm.md`](../../benchmark/results/capabilities/nollm.md),
[`llm.md`](../../benchmark/results/capabilities/llm.md),
[`harness.md`](../../benchmark/results/capabilities/harness.md) - all three **pre-B1**
(2026-10-08, `a06c2c9` plus uncommitted work), kept as history. Read them with their dates and
`provenance`; the summary table below is copied from them.

Recorded on 2026-10-08 on a 4-core laptop. The service ran `a06c2c9` plus uncommitted ingestion work, with the dataset `600b1e5d758f`. Accuracy is the share of phrasings whose assertions held. Reading the columns:

- **new session**: the pushed context in a fresh thread. This is the primary number.
- **same session**: the pushed context in the thread of the scenario's last session.
- **search**: the agent's pull.
- **nollm** is the full set: the 1,110 phrasings of 105 scenarios in each of 5 languages, plus 44 code-switched phrasings, on three surfaces (search skips the abstention questions), for 16,278 records.
- **llm** is the `llm` sample: English, one scenario per domain, three phrasing kinds, every capability except scale. That is 153 phrasings per surface.
- **harness** is the `harness` sample: English, one scenario per domain, the `direct` phrasing, every capability. That is 61 answers.

| # | Capability | nollm context, new session | nollm context, same session | nollm search | llm context, new / same / search | harness answer |
|---|---|---|---|---|---|---|
| 1 | single_fact_recall | 0.02 | 0.31 | 1.00 | 0.00 / 0.20 / 1.00 | 0.00 |
| 2 | multi_hop | 0.00 | 0.02 | 0.60 | 0.00 / 0.20 / 0.80 | 0.00 |
| 3 | temporal_recency | 0.00 | 0.95 | 0.02 | 0.00 / 0.93 / 0.00 | 0.00 |
| 4 | update_contradiction | 0.02 | 0.91 | 0.00 | 0.20 / 0.93 / 0.00 | 0.20 |
| 5 | abstention | 1.00 | 0.16 | - | 1.00 / 0.20 / - | 1.00 |
| 6 | forgetting_decay | 0.00 | 0.19 | 0.50 | 0.00 / 0.40 / 0.40 | 0.00 |
| 7 | scale | 0.00 | 0.00 | 0.00 | - / - / - | 0.00 |
| 8 | procedural_skill | 0.10 | 0.25 | 0.50 | 0.20 / 0.20 / 0.60 | 0.00 |
| 9 | test_time_learning | 0.00 | 0.00 | 0.21 | 0.00 / 0.00 / 0.07 | 0.00 |
| 10 | implicit_personalization | 0.00 | 0.42 | 1.00 | 0.00 / 0.40 / 1.00 | 0.00 |
| 11 | explicit_personalization | 0.05 | 0.50 | 0.27 | 0.17 / 1.00 / 0.17 | 0.00 |
| | macro | **0.11** | **0.34** | **0.41** | **0.16** / **0.45** / **0.45** | **0.11** |

The table shows four things:

- **The model policy changes nothing measured.** On the `llm` sample's 153 phrasings, `nollm` gives the same verdicts as `llm` on every record of every surface. With a small model on the CPU, `query_expansion` times out on the read path (≈250 ms added). The LLM enrichment jobs were still queued behind the graph backlog when the questions started (`queued_at_ask`).
- **The harness answer follows the context.** On the harness sample's 61 phrasings, the pushed context in a new session reached 0.14 and the answers 0.11. The agent can only answer what the context brings. Where the context brought a rule but not the facts it applies to, the model either refused or made the facts up.
- **Languages.** Primary macro by language: en 0.14; de, es, ar and hi 0.10 each; code-switched 0.09. Same session: en 0.40, de 0.35, es 0.35, ar 0.30, hi 0.29, code-switched 0.36. Search is flat at 0.40-0.42. The English lead on the primary surface is one capability: an explicit rule reaches a new session only through the profile block, which keeps English statements (en 0.27; every other language 0).
- **Determinism.** Two full `nollm` runs gave identical verdicts in 191 of 192 gate cells. The one difference was same-session abstention in Hindi, 0.16 against 0.19. The gate keeps the lower of the two (`repeat_runs: 2`).

| | nollm | llm sample | harness sample |
|---|---|---|---|
| context p50 / p95 | 37.07 / 52.28 ms; quiet re-measure 39.93 / 53.51 ms (gated) | 33.77 / 280.62 ms | - |
| search p50 / p95 | 31.95 / 42.84 ms | 29.44 / 280.06 ms | - |
| answer p50 / p95 | - | - | 5,566 / 10,702 ms (the CPU model; load 1.31 per core, set by the model itself: not baseline-grade, not gated) |
| abstention: precision / recall / false refusal | 0.07 / 0.58 / 0.71 | 0.09 / 0.60 / 0.70 | 0.09 / 1.00 / 0.88 |
| stale-value leak: context new / same / search | 0.00 / 0.001 / 0.59 | 0.00 / 0.00 / 0.63 | answer 0.00 |
| ingestion | 8,938 turns, 10,948 jobs, 0 failed; written in 703 s; indexed 2,589 s after the last write; graph 11,541 queued at the questions | 284 turns, 383 jobs; indexed 1,832 s after; graph 3,958 queued | 1,427 turns, 1,711 jobs; indexed 1,899 s after; graph 2,499 at the start, 1,788 at the questions |

The quiet re-measure re-asked the same writes with `--reuse`, starting only once the
worker's queues were empty and the load was under 0.5 per core. It gave context p50/p95 of
39.93/53.51 ms and search p50/p95 of 33.93/42.59 ms, at a peak of 0.7 per core. That agrees
with the first run's 37.07/52.28 ms, taken at 0.9 per core. The gate holds the quietest
trustworthy measurement, **53.51 ms**, so a PR's full run may reach 58.86 ms. Its verdicts
matched the first run's on 16,274 of 16,278 records, the same 4 same-session phrasings that
differ between repeats.

An earlier re-measure, taken while the shared worker enriched other tenants' graphs through
the CPU model, gave 49.46/70.13 ms at 1.63 per core. It was flagged and not recorded (see
the known gaps).

Most failures have the same few causes:

- **A new session sees nothing, in every capability (`false_abstain`).** A memory taken from a conversation is THREAD-visible, so a new thread's context is empty unless the profile carries the statement. For example, "What's the gate code at Riverside pumping station?" gets nothing back, while search, which reads the user's other threads, finds `5512`.
- **The same session over-answers (`no_abstain`).** The context always returns a nearest neighbour, so an unknowable question gets a confident near-miss. For example, "What's the serial number of the pump at Hilltop?" gets back the Riverside serials `R-1182`/`R-1183`.
- **Search leaks stale values (59 %).** The old turn comes back next to the new one, unmarked. For example, "Who drives van #7?" gets back an episode that still names `Ana Silva` beside `Peter Brandt`.
- **Multi-hop gets only one hop.** The context holds the first fact but not the second. For example, the seal kit `SK-19` for the pump `R-1182` is missing.
- **Corrections stay where they were made (`test_time_learning`, 0 in context).** The corrected value (`55 Nm`, `0.67`) lives in the thread where the user corrected it, and no later context carries it.
- **Forgetting.** In a same session, the context keeps the removal and drops the replacement. For example, it carries "GEN-T1 has been removed" but not `UPS-4`.
- **Scale is 0 on every surface.** The monthly records are multi-line, and extraction stores single-line memories rather than the records, so no period label (evidence) comes back. Writing the scale tenant made about 70,000 memories out of 8,938 turns. Graph enrichment then drains at about 30 jobs per minute per tenant and was still hours behind when the questions started. That backlog sits on the shared worker's `graph` queue. The eval records it and does not wait it out.

## Known gaps

These are what the baseline measured, recorded for the blocks that will close them. Each
number is from the `nollm` full run unless the line names another.

- **A new session sees nothing.** In every capability but abstention, nearly every
  phrasing's context in a new thread is `false_abstain`: 448 to 504 phrasings per capability.
  A memory taken from a conversation is THREAD-visible, so another thread of the same user
  never sees it. Primary macro is 0.11; the same session reaches 0.34 and search 0.41.
  **B1** (user visibility).
- **Stale values and corrections.** Search returns the superseded turn unmarked beside the
  new one: a stale-value leak rate of 0.59, which is 1,190 of 2,016 phrasings (0.63 in the
  llm sample). `update_contradiction` search is 0.00 (477 `stale_leak`) and
  `temporal_recency` search is 0.02. A correction stays in the thread where it was made:
  `test_time_learning` is 0.00 in context, in both a new and the same session.
  **B4 / P4**.
- **The second hop is missing.** `multi_hop` same session is 0.02 (415 `missing_evidence`).
  The context holds the first fact but not the one it leads to (the seal kit `SK-19` for
  pump `R-1182`). **P4**.
- **Abstention precision is 0.07.** The service says INSUFFICIENT mostly when the answer
  was there: false refusal is 0.71. In the same session it says too little: abstention there
  is 0.16 (425 `no_abstain`, with a near neighbour returned confidently). **P4** coverage
  gate.
- **Scale is 0 everywhere.**
  - Monthly multi-line records are flattened into single-line memories, so no period label
    (the evidence an aggregate answer needs) comes back: search is 504 of 504
    `not_retrieved`.
  - 8,938 turns became about 70,000 memories. Graph enrichment drains at about 30 jobs per
    minute per tenant, and 11,541 jobs were still queued when the questions started.
- **LLM mode: `query_expansion` times out on the read path. This is a latency regression.**
  With a small model on the CPU as the policy's model, context p95 is 280.62 ms in llm mode against 52.28 ms in
  nollm, and search p95 is 280.06 ms. The read-time `query_expansion` call waits for its
  budget and gives up. It adds about 250 ms to p95 and changes no verdict: on the llm
  sample's 444 records, llm and nollm give identical verdicts. It is to be fixed before llm
  mode is the default.
- **LLM graph enrichment inflates read latency on a shared worker.** A `nollm` re-measure
  that re-asked the same writes ran while the worker enriched other tenants' graphs through
  the CPU model, which used three of the four cores. It measured context p50/p95 of
  49.46/70.13 ms at 1.63 load per core: +34 % on p95. Background model work and the read
  path share the host.

- **No hard delete and no tenant purge: an erasure gap, and the disk grows.** The only
  supported deletes are `DELETE /v1/memories/{id}` and `DELETE /v1/threads/{id}`, and both
  are soft. A forgotten memory stays as a PostgreSQL row. The graph entities and relations
  built from it are never removed: `graph_relations` (378 MB) and `graph_entities` (87 MB)
  were the largest tables of the live database, and almost all of it was eval data. Nothing
  can erase a tenant or a user. To be scheduled as its own block.
  About 99 % of the live database's rows were eval data, roughly 820 MB, and no supported
  call removes them.
- **Forget frees the vector only when the worker runs.** The index point is removed by a
  `memory.index` job. While the worker was down (it died when the disk filled), forgets
  committed at once but every point stayed in Qdrant, still marked `current`.

### Since B1, B2 and B3 (main `420bbf3`, core set)

On the core set, context in a new session went from 0.1193 to 0.4787 macro, and in the same
session from 0.37 to 0.63. Statements now follow the user: single-fact recall in a new
session rose from 0.02 to 0.99, and test-time learning from 0.00 to 0.93.

Three cells moved the other way. The superseded value now reaches the context unmarked:

- `update_contradiction` in the same session fell from 0.90 to 0.05 (444 `stale_leak`).
  "Draft a reorder email to our pallet supplier" still gets "Our primary supplier ... is
  Uline."
- `temporal_recency` in the same session fell from 0.95 to 0.39 (282 `stale_leak`).
- `abstention` in a new session fell from 1.00 to 0.31 (347 `no_abstain`): a nearest
  neighbour now always comes back.

`capability_gate.json` keeps those 18 cells at their pre-B1 values as `targets`. A run below a
target is reported ("below target (not failed)"), never failed: the old values came from
thread isolation hiding the superseded statement, not from replacement logic. They are what
B4 / P4 must reach with statements visible across threads. The thresholds are the ratchet's,
from 420bbf3 on.

These are the gaps listed above, now visible: stale leaks for **B4 / P4**, abstention for
**P4**. Search on `update_contradiction` cannot pass until the message, summary and profile
items it returns carry history marking, as memories and facts already do. That work is
tracked for **P4**.

The core baseline is re-recorded on main `70198fa` with the deterministic runner, from two
runs on the isolated store on one run date, 2026-10-13 (`nollm_core.json` and
`nollm_core_repeat.json`, regenerable):

- macro 0.4813 and 0.4813;
- wall time 1,127 s and 1,106 s;
- context p95 62.7 and 63.0 ms (search p95 41.9 and 42.1 ms), at 0.81 and 0.74 load per core.

The two runs agree on all 174 cells, so no cell is noisy and none has a tolerance. The earlier
thresholds were not carried: they were measured with other user ids, so other index text, and
the ratchet had kept each cell's best of several random draws. Against them, 30 cells are
higher and 17 lower, by at most 0.04. Every cell's threshold is now its measured value.

**The full baselines are not reproducible.** The full `nollm`, `llm` and `harness` runs of
2026-10-08 were measured on `a06c2c9` plus uncommitted work, and on an earlier dataset hash.
They are not in `capability_gate.json`; their summaries are kept as history. The plan is to
re-record them on main with the nightly full run: `make eval-capabilities MODE=...`, then
`make eval-capabilities-baseline`.

### The llm_core baseline (main `420bbf3`)

Two isolated runs through the live gateway and the CPU model (`llm_llm_core.json` and
`llm_llm_core_repeat.json`, regenerable):

| | run 1 | run 2 |
|---|---|---|
| macro, context in a new session | 0.6133 | 0.61 |
| wall time | 4,616 s | 2,765 s |
| writes | 692 turns in 2,938 s (run 1) | |
| context p50 / p95 | 291 / 309 ms | 289 / 308 ms |
| load during the questions | 1.42 per core | 1.43 per core |

The two runs differ in 6 of 174 cells. Their ranges reach 0.17, because these cells are
small. The latency is recorded but not gated: the read path calls the model on the same
host, so the quiet-machine rule cannot hold. It is still the llm-mode regression in the
known gaps, `query_expansion` waiting out its budget. The disk was left as found.

### Run-to-run noise: the user ids (fixed)

The 41 cells two core runs of one commit disagreed on came from the runner, not the service.
The users were named `ce{run_id}-{i}-{lang}` with a random run id, and the service indexes a
memory's subject with its text (`memory_index_text`: `[date] <subject>: <type>: <content>`).
With the frozen multilingual encoder (relevance floor 0.2), twelve random ids moved one
memory's cosine to its question by 0.023-0.054; the English encoder by 0.006-0.010. Paired runs
of two commits that pinned the run id and date agreed on every one of 14,766 records; two
unpinned runs of one commit differed in 129 verdicts.

The runner now names users and threads from scenario and language and pins the run date
(`--run-date`, recorded). Two isolated core runs of one commit on one run date agree on every
verdict, every score field and every cell; only the order of equally scored graph facts in
some failure previews differs (24 of 14,766 records), which nothing scores. The baseline
re-recorded on 2026-10-18 holds it again: three isolated core runs on run date 2026-10-18 (one
on 8cf9dbb, two on fe995bb, whose service code is main's) agree on all 14,766 verdicts and
scored fields and all 174 cells (macro 0.4806 each; 18-36 previews reordered).

The run date is not a constant, on purpose. Sessions are dated relative to it and the service
reads them against its own clock: named periods in a question ("last month") and the
forgetting half-life use the moment of asking. A fixed date would drift a day further from that
clock every day. The date does enter the index text: a one-day shift moves a cosine by at most
0.004 (0.017 for one Arabic case), about a tenth of the id's effect. A gate run on another day
than its baseline is noted by `gate check`; pin `--run-date` to compare two commits on one day.
Measured: the same code on run dates 2026-10-13 and 2026-10-18 (each on its own day) differs in
42 of 14,766 verdicts and 28 of 174 cells, each by at most 0.02 (macro 0.4813 and 0.4806); this
is why a re-recorded baseline starts its thresholds afresh rather than ratcheting across dates.

Weighed separately, and not changed here: in production the indexed subject is the caller's
user id, which is usually opaque. It costs a few hundredths of cosine at random per user, so
whether a memory near the floor is packed depends on how the user's id happens to tokenise,
and the graph's facts carry it too ("user:ce952f851b8651 mentions ..."). It is worth a
follow-up: index the subject only when it is a name.

## Corrections to the instrument (2026-11-01)

Each failing English item of the `direct` and `later` phrasings was traced to its cause on
main 7d8202a. These were the instrument's, not the service's, and are corrected here; the
gate's dataset hash changes with them, so the baselines are re-recorded on the measuring host.

| Item | What was wrong | Correction |
|---|---|---|
| `forget-it-decommissioned-db` | 30 daily readings from day -30 ran to the day after "legacy-db-02 has been decommissioned": a reading dated after the decommission was served as current on every surface | 28 readings; the validator now refuses a repeated log that runs past the start of a later session (`validate.overlapping`) |
| every answer `absent` anchor | an answer is prose, judged as one item: "The OfficeHub contract ended" named the old vendor to say it is gone and was scored as serving it | the answer's sentences and table rows are judged one by one (`scoring.sentences`), so an `unless` anchor in the same sentence clears a mention, as it does for an item of the context |
| forgetting (all ten), `recall-finance-cost-centre`, `recall-retail-hazmat-code`, `temporal-personal-reading`, `update-field-van-assignment`, `abstain-personal-sister`, `abstain-field-hilltop-serial`, `explicit-retail-zero-stock` (q1), `explicit-it-trading-window`, `explicit-personal-no-cilantro`, `explicit-field-no-north-fridays` | the answer's `absent` anchors had no `unless`: "J-11 and J-13 are north of the river, so you can't take them", "your brother Tom ... nothing about a sister", "R-1182 and R-1183 belong to the Riverside pumps", "Don't confuse it with Warehouse 2's 3317", "an earlier note said a third of the way, since replaced", "Ana Silva drives van #9 from now on" failed - and judged per sentence, a sentence naming the old value had to carry its own `unless` | the context's own `unless` for the forgetting items; a negation or an earlier-value word, the brother, Riverside, Warehouse 2, van #9 or the regular cost centre for the others |
| `temporal-it-vpn-incident` | the answer had to say "resolved"; "VPN access was restored" (the event's own word) failed | the answer may say resolved or restored |
| `procedure-it-mfa-skill` on the search surface | the agent's pull searches memories, episodes and messages; a learned tool procedure is none of them, so its step regex could never be met there | a query that names its tools is scored on the pushed context (its skills section) and the answer, not on the pull (`runner.pulled`) |
| harness mode's prompt | "using only the memory context" told the model not to bring general know-how, so a pancake recipe in grams, a VPN install on a Mac team or ideas for an afternoon in London were refused with the user's trait in front of it | the context for everything about the user; the model's own knowledge for general know-how, shaped by the context |
| harness mode's `MAX_TOKENS` | 400: 20 of 107 answers were cut mid-way (a service report lost the signature line its standing rule ends it with) | 1,024 |

## What it does not measure

- Answer quality beyond the assertions: an answer that names the right value in a wrong
  sentence passes; the context surfaces measure whether the evidence reached the agent, not
  what a model makes of it (harness mode does, on a sample).
- Anchors are substrings: a context that mentions the right code next to the wrong door still
  passes `contains`; the near-duplicate distractors and `absent` anchors are what keep that
  honest where it matters.
- The scale tenant is synthetic, and its standard size (~110k tokens per language) is what a
  laptop ingests in a run; the million-token size is opt-in.
