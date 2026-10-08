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
tool runs** (`tool_runs`: `record_tool` + run feedback, how an agent learns a procedure). A
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

1. **Isolation.** One tenant per mode (`capeval-nollm`, `capeval-llm`, `capeval-harness`),
   created with the admin credential when missing, the mode's model policy set on it, a
   service key issued for the run and revoked after it. Each scenario and language is a fresh
   user, so nothing one run writes is read by another, or by anything else on the stack.
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
| `llm` | the live policy: every use but `memory_restatement`, `read_assist` on, one gateway model (`CAPEVAL_LLM_MODEL`, default Granite 3.3 2B) | the context is scored | sample `llm` |
| `harness` | as `nollm` | an agent wrapped by agent-harness: a function agent whose prompt is the pushed context and whose model (`CAPEVAL_ANSWER_MODEL`) answers; the answer is scored | sample `harness` |

The samples are deterministic (`_modes.json`): per capability the InventorySmart scenario first,
then the first scenario of each other domain in id-hash order. `llm`: English, three phrasing
kinds (`direct`, `paraphrase`, `implicit`), every capability but scale (extracting a
1,100-message tenant with a CPU model takes hours; scale is measured in the other two modes).
`harness`: English, the `direct` phrasing, every capability. Harness runs record nothing back
(`without={"records"}`), so no answer becomes the next question's memory.

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

The gate: **every cell must reach its recorded accuracy** (no tolerance), a cell the baseline
has and the run lacks fails, **context p95 may rise at most 10 %**, a latency measured on a
loaded machine fails, and a run on another dataset (`dataset.sha256`: the capability files,
the scale specs and `_languages.json`) or another selection is refused - a dataset change
re-records the baseline in the same change, where both numbers are reviewed. A threshold is
never lowered to let a change through; an improvement is recorded by re-recording.

```bash
make eval-capabilities MODE=nollm                 # the full set, every language
make eval-capabilities MODE=llm                   # sample llm, through the model policy
make eval-capabilities MODE=harness               # sample harness, agent-harness + its model
make eval-capabilities-scale                      # opt-in: the ~1M-token tenant (capability 7)
make eval-capabilities-gate MODE=nollm            # fail on any drop against the baseline
make eval-capabilities-baseline                   # record the measured runs as the thresholds
make eval-capabilities-validate                   # every assertion satisfiable in every language
```

Environment: `MEMORY_URL`, `CAPEVAL_ADMIN_KEY` (a credential that may onboard a tenant - the
platform key, or a development key on a laptop), `MEMORY_WORKER_METRICS_URL` (the worker's
metrics port), `CAPEVAL_LLM_MODEL` / `CAPEVAL_ANSWER_MODEL`, and for harness mode the
agent-harness environment (`BIFROST_URL`, `BIFROST_VIRTUAL_KEY`; `HARNESS_PY` is its Python).
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
[`harness.md`](../../benchmark/results/capabilities/harness.md). Read them with their dates and
`provenance`; the summary table below is copied from them.

Recorded on 2026-10-08 on a 4-core laptop. The service ran the `claude/cool-maxwell-w5avo8` tree at `a06c2c9`, plus uncommitted ingestion work, with the dataset `600b1e5d758f`. Accuracy is the share of phrasings whose assertions held. Reading the columns:

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

- **The model policy changes nothing measured.** On the `llm` sample's 153 phrasings, `nollm` gives the same verdicts as `llm` on every record of every surface. With a CPU Granite, `query_expansion` times out on the read path (≈250 ms added). The LLM enrichment jobs were still queued behind the graph backlog when the questions started (`queued_at_ask`).
- **The harness answer follows the context.** On the harness sample's 61 phrasings, the pushed context in a new session reached 0.14 and the answers 0.11. The agent can only answer what the context brings. Where the context brought a rule but not the facts it applies to, the model either refused or made the facts up.
- **Languages.** Primary macro by language: en 0.14; de, es, ar and hi 0.10 each; code-switched 0.09. Same session: en 0.40, de 0.35, es 0.35, ar 0.30, hi 0.29, code-switched 0.36. Search is flat at 0.40-0.42. The English lead on the primary surface is one capability: an explicit rule reaches a new session only through the profile block, which keeps English statements (en 0.27; every other language 0).
- **Determinism.** Two full `nollm` runs gave identical verdicts in 191 of 192 gate cells. The one difference was same-session abstention in Hindi, 0.16 against 0.19. The gate keeps the lower of the two (`repeat_runs: 2`).

| | nollm | llm sample | harness sample |
|---|---|---|---|
| context p50 / p95 | 37.07 / 52.28 ms | 33.77 / 280.62 ms | - |
| search p50 / p95 | 31.95 / 42.84 ms | 29.44 / 280.06 ms | - |
| answer p50 / p95 | - | - | 5,566 / 10,702 ms (the CPU model; load 1.31 per core, set by the model itself: not baseline-grade, not gated) |
| abstention: precision / recall / false refusal | 0.07 / 0.58 / 0.71 | 0.09 / 0.60 / 0.70 | 0.09 / 1.00 / 0.88 |
| stale-value leak: context new / same / search | 0.00 / 0.001 / 0.59 | 0.00 / 0.00 / 0.63 | answer 0.00 |
| ingestion | 8,938 turns, 10,948 jobs, 0 failed; written in 703 s; indexed 2,589 s after the last write; graph 11,541 queued at the questions | 284 turns, 383 jobs; indexed 1,832 s after; graph 3,958 queued | 1,427 turns, 1,711 jobs; indexed 1,899 s after; graph 2,499 at the start, 1,788 at the questions |

The `nollm` latency was taken at a peak load of 0.9 per core, with a mean of 0.57 per core: trustworthy, but near the bar. A re-measure that re-asked the same writes was taken while the shared worker was working off LLM-assisted graph enrichment for other tenants. The CPU model used three of the four cores, and the re-measure gave context p50/p95 of 49.46/70.13 ms at 1.63 per core. That is +34 % on p95, from the service's own background enrichment. It was flagged and not recorded. A quiet re-measure replaces the latency in the gate when it is trustworthy. It re-asks the same writes with `--reuse`, started once the worker's queues are empty and the load is under 0.5 per core. The gate keeps the quietest trustworthy measurement.

Most failures have the same few causes:

- **A new session sees nothing, in every capability (`false_abstain`).** A memory taken from a conversation is THREAD-visible, so a new thread's context is empty unless the profile carries the statement. For example, "What's the gate code at Riverside pumping station?" gets nothing back, while search, which reads the user's other threads, finds `5512`.
- **The same session over-answers (`no_abstain`).** The context always returns a nearest neighbour, so an unknowable question gets a confident near-miss. For example, "What's the serial number of the pump at Hilltop?" gets back the Riverside serials `R-1182`/`R-1183`.
- **Search leaks stale values (59 %).** The old turn comes back next to the new one, unmarked. For example, "Who drives van #7?" gets back an episode that still names `Ana Silva` beside `Peter Brandt`.
- **Multi-hop gets only one hop.** The context holds the first fact but not the second. For example, the seal kit `SK-19` for the pump `R-1182` is missing.
- **Corrections stay where they were made (`test_time_learning`, 0 in context).** The corrected value (`55 Nm`, `0.67`) lives in the thread where the user corrected it, and no later context carries it.
- **Forgetting.** In a same session, the context keeps the removal and drops the replacement. For example, it carries "GEN-T1 has been removed" but not `UPS-4`.
- **Scale is 0 on every surface.** The monthly records are multi-line, and extraction stores single-line memories rather than the records, so no period label (evidence) comes back. Writing the scale tenant made about 70,000 memories out of 8,938 turns. Graph enrichment then drains at about 30 jobs per minute per tenant and was still hours behind when the questions started. That backlog sits on the shared worker's `graph` queue. The eval records it and does not wait it out.

## What it does not measure

- Answer quality beyond the assertions: an answer that names the right value in a wrong
  sentence passes; the context surfaces measure whether the evidence reached the agent, not
  what a model makes of it (harness mode does, on a sample).
- Anchors are substrings: a context that mentions the right code next to the wrong door still
  passes `contains`; the near-duplicate distractors and `absent` anchors are what keep that
  honest where it matters.
- The scale tenant is synthetic, and its standard size (~110k tokens per language) is what a
  laptop ingests in a run; the million-token size is opt-in.
