# Phase 7 (memory M2 + M4): what was measured, and what was not

Date: 2026-09-28. Branch: `worktree-agent-a6eafbc99e7c1aa35`, based on `main` `a9e5aab`.
Decisions: ADR 0024. Runtime: `docs/MULTILINGUAL-RUNTIME.md`. Plan of record:
`docs/PLAN-MULTILINGUAL-PLATFORM-2026-09-28.md` (D2, D6, D8, D9, §4 rows M2 and M4, §5).

**No production-readiness claim is made from this record.** The M2 container gates and the
M4 judged arms were **not run** in this session; the reason is recorded below with the host
state that caused it. Every gate threshold is quoted from the plan and none was edited. An
unrun gate is unrun, not passed: the same rule the measurement protocol applies to a failed
model call ("unmeasured, never wrong") applies to a gate nobody executed.

## Host

4 cores, 8 GB. One other agent worked on this machine throughout. Load average at the close
of the session 3.73 (1 min) / 6.11 (5 min) / 11.41 (15 min); free physical memory during the
gate decision was ~66 MB of 8 GB, with ~3.1 GB active and ~3.1 GB inactive. Every test run
below waited for the 1-minute load average to fall under 8 first, and ran alone.

## What passed

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| Unit suite (incl. SDK) | green | 1548 passed, 2 skipped | **pass** |
| Integration suite, isolated `p7_integration` | green | 244 passed, 1 skipped, 4 deselected | **pass** |
| Contract suite | green | 124 passed, 11 skipped, 1 failed (below) | **pass except an external outage** |
| e2e + agent (SDK) + security | green | 100 passed | **pass** |
| `test_architecture.py` | not raised | passes unchanged | **pass** |
| `test_complexity_budget.py` | not raised: at most 43 functions over cc 15 | within the budget, which is still 43 | **pass** |
| `test_model_provenance.py` | no Chinese-origin model anywhere | passes | **pass** |
| Ruff lint + format | clean | `All checks passed`, 626 files formatted | **pass** |
| Committed OpenAPI schema matches generated | equal | equal after regeneration | **pass** |
| 12-language SDK recall suite | passes hermetically | included in the 100 above | **pass** |
| Budget tool reads the governing key | reads `current_usage` | $0.006426 of $10 | **pass** |
| Budget guard refuses an unaffordable arm | nonzero exit | projected $0.14 → exit 0; projected $7.50 → exit 1, "exceeds the phase cap 6.00 USD" | **pass** |
| Judged reader/judge model reachable | a call succeeds | `openrouter/openai/gpt-4.1-nano` HTTP 200 | **pass** |

The one contract failure is `tests/contract/test_bifrost_llm.py::test_live_bifrost_roundtrip`:
the gateway returned **503 `UNAVAILABLE`, "This model is currently experiencing high demand"**
for `gemini/gemini-3.8-flash`, which is what `.env` names. The gateway itself is healthy
(`/metrics` 200, providers configured) and `openrouter/openai/gpt-4.1-nano` answered 200
through the same key in the same minute, so this is an upstream provider outage, not a defect
in this branch. A failed call is unmeasured.

## Measured (this section is generated from the artifacts, 2026-09-29)

Host: 4 cores, 8 GB, Docker VM taking most of it; the D8 target assumes 8 vCPU.
Every arm below ran with `paid_llm_calls: 0` -- these are retrieval numbers on the
model-free floor, not answer accuracy.

### SciFact through the runtime store path (M2 gate) -- MISSES

| metric | threshold | measured (ensemble) | verdict |
|---|---|---|---|
| ndcg@10 | 0.7557 | 0.7533 | **misses by 0.0024** |
| recall@10 | 0.8926 | 0.8926 | meets |

English-only arm: nDCG@10 0.7415, recall@10 0.8912.
5,183 documents, 300 queries, indexing 7302 s.
Artifact: `benchmark/results/phase7/runtime_scifact.json`.

### XQuAD, 12 languages (M2 gate) -- PASSES

Batch a: mean same-language recall@10 0.9948 against 0.98, pass=True.
Batch b: mean same-language recall@10 0.9963 against 0.98, pass=True.

| language | english-only | ensemble | cross-language, english-only | cross-language, ensemble |
|---|---|---|---|---|
| ar | 0.9303 | 0.9916 | 0.0933 | 0.9462 |
| de | 0.9815 | 0.9975 | 0.8008 | 0.9857 |
| el | 0.9454 | 0.9941 | 0.2748 | 0.9496 |
| es | 0.9891 | 0.9975 | 0.8697 | 0.9933 |
| hi | 0.9723 | 0.9950 | 0.1328 | 0.9748 |
| ro | 0.9790 | 0.9933 | 0.8202 | 0.9748 |
| ru | 0.9235 | 0.9975 | 0.1555 | 0.9891 |
| th | 0.9840 | 0.9958 | 0.1613 | 0.9664 |
| tr | 0.9462 | 0.9899 | 0.6042 | 0.9370 |
| vi | 0.9798 | 0.9992 | 0.5395 | 0.9378 |
| zh | 0.9941 | 0.9992 | 0.1454 | 0.9866 |

Mean over the 11 languages with both arms: english-only 0.9659, ensemble 0.9955.
Cross-language is where a single English space fails outright, which is what the second
vector space was added for. Artifacts: `runtime_xquad_a.json`, `runtime_xquad_b.json`.

### LoCoMo source arms, exact annotated source-turn recall -- both complete

1,986 questions each, 10 conversations, no LLM. `english` is the control (Granite only);
`ensemble` is the shipped runtime (dense_en + dense_ml + bm25, script-pruned prefetch).

| category | n | control @10 | ensemble @10 | delta | control @50 | ensemble @50 | delta |
|---|---|---|---|---|---|---|---|
| all_answerable | 1536 | 0.6015 | 0.6484 | +0.0469 | 0.7682 | 0.8113 | +0.0431 |
| single_hop | 841 | 0.6740 | 0.7208 | +0.0468 | 0.8373 | 0.8751 | +0.0379 |
| multi_hop | 282 | 0.3364 | 0.3977 | +0.0612 | 0.5774 | 0.6338 | +0.0564 |
| temporal | 321 | 0.7105 | 0.7510 | +0.0405 | 0.8281 | 0.8741 | +0.0460 |
| open_domain | 92 | 0.3699 | 0.3965 | +0.0265 | 0.5123 | 0.5532 | +0.0409 |

| latency (ms) | p50 | p95 | p99 | max |
|---|---|---|---|---|
| control | 369.77 | 833.34 | 1622.52 | 5593.82 |
| ensemble | 321.36 | 732.02 | 1523.17 | 4096.34 |

The ensemble is both better and faster: script-aware pruning skips the English encode
for non-Latin queries. Artifacts: `locomo_source_english.json`, `locomo_source_ensemble.json`.

### Fast-path p99 (D8 target) -- FAILS

Target p99 < 300 ms on 8 vCPU. Measured 1523.17 ms on 4 cores with other work on the
host. Not met, and not claimed. The depth-halving gate in D6 step 2 is the intended route
to it, and it is unrun.

### The finding that shapes the next phase

multi_hop recall is 0.3977 at depth 10 and 0.6338 at depth 50. The supporting turns are
retrieved and ranked too low, so the honest ceiling for reordering at depth 10 is that
depth-50 number. A cross-encoder is not the answer here: measured full scale
(`benchmark/results/full_rerank_ettin17m.json`, 1,986 questions, ettin-17m) it costs p50
649 ms, p95 1322 ms, p99 2018 ms. Its *quality* on this corpus is unestablished -- the
on/off pair that suggested it hurts is 2 conversations, 120 questions, 16 multi-hop
questions, which is noise, and that claim should not be repeated.

### Grounding gate on the real weights -- FAILS, and the gate itself was broken

This gate had never run: `MODEL_TESTS` named `tests/contract/test_advanced_adapters.py`,
deleted in `24f229f`, so pytest exited 4 on a missing path and no real-weight test executed.
Fixed in `c519d26`; what follows is its first honest result.

| check | result |
|---|---|
| `test_nli_adapter.py` + `test_model_adapters.py` against the frozen weights | 6 passed, 2 failed |
| the mDeBERTa golden (`tests/eval/golden/grounding_claims.json`) | 1 passed, 1 failed |

Two of the three failures were defects in the adapters or the test, and are fixed:

* **The licence was not reported.** Both the ONNX NLI adapter and the embedding adapters
  answered `"see model card"`, or guessed `Apache-2.0` from the string `granite` in the id,
  while the frozen specs already declare `license` (`DenseModel.license = "Apache-2.0"`,
  `NLIModel.license = "MIT"`). The licence column *is* the provenance record under D2, so a
  generic placeholder made the one field that matters unusable. The adapters now read the spec.
* **A fingerprint assertion pinned a literal.** `ce-tiny-ce` no longer matches because the
  fingerprint carries a 12-character digest of the whole spec, which is the better behaviour
  (a tuning change produces a different fingerprint). The test now pins the shape.

The third is a real quality finding and is **not** fixed, because fixing it would mean editing
a golden set:

* **mDeBERTa turns an unsupported citation into a contradiction.** Case `citation-03` answers
  "Revenue decreased to EUR 400 million [1]" while citation [1] resolves to E3, the Nordwind
  acquisition, not E2, the revenue sentence. The claim's content is true of E2 but the cited
  evidence does not support it, so the golden expects `unsupported`. mDeBERTa returns
  `contradicted`. E3 and the claim are unrelated, not opposed, so `contradicted` is wrong, and
  it is wrong in the dangerous direction: a false contradiction is what retracts a correct
  memory, whereas an abstention only declines to help. It also qualifies the NLIModel
  docstring's claim that the FP32 graph scores "the same as the English DeBERTa it replaces":
  on this golden it does not.

## What was not run when this document was first written

Everything below was run afterwards, on 2026-09-29, and its numbers are in the section above.
The judged arms (B / A0 / A1 and the mini rerun) remain unrun; they are the only part that
spends budget, and they are the first item of the next phase that needs it.

### The original note



The M2 container gates and the M4 judged arms require the `memory-service-memory-api` image
with both real encoders loaded, over 5,183 SciFact documents, 14,280 XQuAD questions and ten
LoCoMo conversations; the spec's own estimate is ~17 hours of wall time for the set. They were
not run for two reasons, both recorded rather than worked around:

1. The session was instructed not to start, stop, build or restart a Docker container. Every
   `make bench-*` target is a `docker run` of that image.
2. The host had ~66 MB of free physical memory and a 15-minute load average of 12.95 while a
   second agent was working. Two ONNX encoders plus a Postgres and Qdrant client inside a
   container, for hours, would have swapped and would have degraded the other agent's work.
   A latency gate (p99 < 300 ms) measured under those conditions would also be worthless.

| Gate (unrun) | Threshold from the plan | Command that runs it |
|---|---|---|
| SciFact through the runtime store path | nDCG@10 ≥ 0.7557 and R@10 ≥ 0.8926, 5,183 docs / 300 queries | `make bench-runtime-retrieval RUNTIME_ARGS="--suite scifact …"` |
| XQuAD, 12 languages | mean same-language R@10 ≥ 0.98, 14,280 questions | `make bench-runtime-retrieval RUNTIME_ARGS="--suite xquad …"` |
| LoCoMo source arms | Granite-only and ensemble complete on 10 conversations, `paid_llm_calls: 0` | `make bench-locomo-source BENCH_DENSE=english\|ensemble` |
| Fast-path p99 (D8) | p99 `ContextBuilder.build()` < 300 ms, load recorded beside it | the ensemble source arm above |
| Grounding golden on mDeBERTa | ≥ 36/40 | `make model-test` (models marker) |
| D6 step 2, weighted RRF at halved depth | `evidence_all_recall ≥ 0.99`, p99 down | `python -m benchmark.fit_rrf_weights <dump>` then a no-judge arm |
| D6 step 3, entity routing | strict multi-hop not lower; ID-keyed `complete@50` not lower | judged A0 vs B |
| D6 step 4, dated aggregates | strict multi-hop ≥ +3, `false_merge_rate == 0.0` | judged A0 vs B, and `make eval` |
| D6 step 5, rank-anchored rendering | no change to `query_p50_ms` | judged A0 vs B |
| D6 step 6, reader re-ask | abstain-with-evidence down, category-5 abstention not lower | judged A0 vs B |
| Judged arms B / A0 / A1 + one mini rerun | the published table, both rulers, category 5 separate | `make bench-locomo-judged LOCOMO_DB=… LOCOMO_ARGS="--judge …"` |

Because the judged arms did not run, `RetrievalSettings.hybrid_weights` stays `None` (the
unweighted RRF that is already measured), `RetrievalSettings.entity_prefetch` stays `False`,
and `MOST_RELEVANT_SHARE` stays `0.0`. Each is the conservative setting, and each is the
setting its own arm was supposed to promote. **Nothing was promoted on an unmeasured
argument.** The read-side aggregate of D6 step 4 is on, because it is the render's own
correctness (several values of one slot are one fact, not several) and it is covered by unit
tests; its accuracy contribution is unmeasured and is not claimed.

## Budget

| Point | `current_usage` on `memory-accuracy-usd10` |
|---|---|
| Before this session | **$0.005593** of $10 |
| After this session | **$0.006426** of $10 |
| Spent in this session | **$0.000833** |
| Phase 7 cap | $6.00, floor $2.00 — neither approached |

The entire spend is model-reachability probing: one-token calls to
`openrouter/openai/gpt-4.1-nano` (200) and `gemini/gemini-3.8-flash` (503, unbilled). No
judged arm ran, so no arm cost is reported. The ledger is
`benchmark/results/phase7/budget.json`. The key's token was never printed and lives only in
the git-ignored `.env`.

## Two hazards found while reviewing, both older than this phase

**A harness could reset a store it did not own.** `benchmark/common.py:reset_store` issues
`TRUNCATE` over 25 tables and deleted the tenant's vectors, against whatever database the
container was pointed at. `native_source_retrieval.py` guarded the database name;
`locomo.py` did not. Meanwhile the benchmark default URL in `benchmark/retrieval.py` and the
checked-in `.env` both name the shared **`memory`** database, and `BENCH_QDRANT_URL` pointed
at the shared Qdrant on **6333** rather than the isolated store on 16333.
`tests/unit/test_benchmark_modules.py::test_embedding_benchmark_stand_in_end_to_end` drives
that path with no override, so a plain `make unit` on a machine with the dev stack reachable
truncated the development database. The same class of bug is already recorded in
`tests/conftest.py` for the test suite itself ("the suite truncated it on every run"); it had
recurred in the benchmark path.

Fixed by putting the check where the damage happens — `reset_store` refuses any database not
named for a benchmark — by pointing `BENCH_QDRANT_URL`/`BENCH_QDRANT_GRPC_PORT` at
16333/16334, by giving Phase 7 its own `p7_*` databases, and by pointing that test at the
suite's own database. `tests/unit/test_benchmark_store_guard.py` pins the refusals.

**An anchored prefetch could widen its own audience.** The entity prefetch merged its filter
into the query's with a dict update, so a key collision would *replace* the base filter's
`must_any` rather than add to it. `must_any` is what carries `visibility_keys`, so an anchor
keyed `visibility_keys` would have widened that arm inside the store, which is exactly where
the tenant boundary is meant to be unconditional. Only `entities` is used today, so nothing
was exposed; the collision is now refused, and two tests in
`tests/unit/test_named_vectors_wire.py` pin both the narrowing and the refusal.

## Smaller corrections

- The documented flagship example of temporal normalisation, `last Tuesday (2023-05-02)`, is
  not something the resolver produces: `relative-time` covers named and counted offsets, not
  weekday phrases. Reaching those needs `absolute-time`, which on this base resolves the
  weekday correctly **and** reads the word "we" as a Wednesday and annotates the absolute
  dates the step exists to leave alone. The parser stays narrow and the module docstring, the
  ADR, `MULTILINGUAL-RUNTIME.md` and the tests now promise only what it does. Measured cost:
  0.84 ms when the cue pre-check rejects a text, 6-27 ms when it parses, ingest path only.
- `CollectionSpec` now forbids unknown fields. `dense_dim=4` outlived the field it named, and
  two store tests were silently asking for a collection with no dense vector at all.
- `script_of` is cached per character: a 2,000-character chunk costs 0.28 ms instead of
  1.20 ms, paid once per indexed record.
- The dead `TransformersNLI` adapter is deleted (D9: dead paths go). Nothing referenced it.
- Predicate cardinality was a private of `modules/memory/native.py`, with its rule duplicated
  in `landing.py`. It is one domain vocabulary now (`domain/predicates.py`), which is also
  what lets the renderer group a bundle without the domain importing the write path.

## How to finish this phase

Run the unrun gates in the order of the table above, one at a time, on a quiet host, and fill
their numbers in. Read `python -m benchmark.budget read --checkpoint "before <arm>"` before
each judged arm and again after it; run the 20-question smoke
(`--conversations 1 --sample 20 --judge`) before any 2,000-call arm and project from its cost
per question. `LOCOMO_DB` selects the corpus: `BENCH_DB_P7_CONV` for B and A0,
`BENCH_DB_P7_LLM` for A1 (`--llm-ingestion`, which adds only `contextual_extraction`). Do not
promote `hybrid_weights`, `entity_prefetch` or `MOST_RELEVANT_SHARE` until the arm that
measures each one says so.
