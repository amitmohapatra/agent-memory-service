# Handoff: where the memory-service programme stands

## Funded final investigation underway — 2026-09-26

Latest status and active jobs: [FINAL-RESULTS-2026-09-26.md](FINAL-RESULTS-2026-09-26.md).
Primary-source comparison: [FINAL-RESEARCH-2026-09-26.md](FINAL-RESEARCH-2026-09-26.md).
The provider-blocked status below is historical. The frozen 37-question comparison has
now completed, but shows no reliable assisted-ingestion gain. Raw automated judging is
lenient about some incomplete lists; keep the separate manual completeness audit visible.
Full current native retrieval and answer evaluation are complete: source recall 83.85%,
complete-source coverage 77.47%; fixed mini reader/judge answerable accuracy 71.88%
(1,107/1,540), all-question accuracy 68.53% (1,361/1,986). These are different metrics.
The isolated background-consolidation and actor/topic answer comparisons remain in progress.
No new whole-LoCoMo 90% claim. Read the live handoff before resuming jobs
or spending credit; it records budgets, copied databases, checkpoints and source revisions.

## Real multi-hop/temporal test interrupted by providers — 2026-09-26

Read [COMPLEX-CONSOLIDATION-2026-09-26.md](COMPLEX-CONSOLIDATION-2026-09-26.md) first.
Frozen test: full 369-turn LoCoMo conversation 1; all **11 multi-hop + 26 temporal**
questions. **No new accuracy or latency result**: no reader/judge calls occurred.
Native ingestion complete. Original assisted attempt reached 212 turns before credit
failure; preserve it but do not score it as complete. A clean assisted tenant replayed
141 exact successful extractions and completed 190 turns; Gemini 3.8 Flash then returned
503 on the missing extraction and one delayed retry. Safe pending checkpoint at D11:1.

This turn: **204 inference gateway requests**, reported OpenRouter spend **$0.0457248**.
OpenRouter has no purchased credit; its remaining key limit is not funds. User was asked
to top up. Gemini 3.8 Flash is now registered in Bifrost under a new route record referencing
the **same existing environment credential**, after Google's 2.5 deprecation response.
Its READY preflight succeeded but actual extraction failed under high demand. No billing
limits changed. Other checked routes returned quota/model-availability errors.

Useful findings: 70 current reflections in the interrupted arm are PRIVATE; current
600-token reflection cap caused two truncated responses. No permissions or application
code changed. New evaluation-only response cache/failure guard prevents native fallback
from being silently scored as assisted ingestion and resumes without duplicate observation
submission. **24 tests pass**, Ruff/Pyright clean. All benchmark jobs stopped. Resume
commands, experiment deviations and exact artifacts are in the linked handoff.

## Paired consolidation answer diagnostic — 2026-09-26

Read [CONSOLIDATION-ANSWER-EVAL-2026-09-26.md](CONSOLIDATION-ANSWER-EVAL-2026-09-26.md)
first for the latest measurement. Ten synthetic development questions, same GPT-4o-mini
reader: required-fact completeness **9/10 → 9/10**; strict manual grounding **8/10 → 9/10**.
The only strict gain depends on a debatable inference, so **no reliable overall uplift or
new full-LoCoMo accuracy claim**. Tiny-corpus builder median **76.03 → 98.74 ms**, not p99.
Fresh isolated indexes excluded all six earlier speculative reflections. Gemini completed
two pairs (both arms correct) before 503 then quota 429. Mini completed all twenty answers;
the earlier 402 did not recur in these calls. Twenty-eight total requests, twenty-six
successes, reported OpenRouter cost $0.0042609. No new ingestion/model grading calls.
Focused validation **19 passed**, Ruff/Pyright clean. Original broad artifacts unchanged;
no application code changes or deployment this turn. All benchmark jobs stopped.

## LLM ingestion and consolidation wired and locally enabled — 2026-09-26

Read [CONSOLIDATION-2026-09-26.md](CONSOLIDATION-2026-09-26.md) first. This supersedes
historical OFF notes below. The ignored local `.env` now selects GPT-4o-mini for contextual
extraction and GPT-4o for background reflection through Bifrost. Query-time LLM uses remain
disabled. Migration 0009 is applied locally; no deployed app/image restart occurred.

Landing consolidation is constructed, source dependencies and audience checks are enforced,
source changes recursively invalidate derived memories, and retrieval canonically validates
stale derived hits and can fetch six original source companions. Model calls are outside
admission locks. Fixed the strict reflection schema that caused real provider HTTP 400s.

Ten metered live-probe requests, eight successful, reported cost $0.00972875. A later
live contract roundtrip hit OpenRouter HTTP 402; native fallback remains available. The final factual
prompt/model stored five observations on software/gardening sources, but one still overstates
causality. Both original retrieval arms already had complete source coverage in this tiny
sample. **No demonstrated overall accuracy gain; no new full LoCoMo score or production p99.**
Final affected-path rerun: **79 passed**; Ruff/Pyright pass. The broad run passed 1,147,
with its complexity failure subsequently fixed and its external HTTP 402 still unresolved.
The original benchmark artifacts are unchanged. See the detailed handoff for tests, exact
limitations, probe contamination, migration semantics and safe reproduction.


## Model rotation completed; one targeted retrieval win and a narrative guard — 2026-09-26

The user authorized new OpenRouter/Gemini calls to finish the targeted work. This supersedes
the older exhausted-allowance notes for this turn. **32 requests: 28 successes, one 402,
two 429s, one 503; no external judges.** Reported OpenRouter cost $0.0633092. All jobs stopped.
Read [MODEL-ROTATION-2026-09-26.md](MODEL-ROTATION-2026-09-26.md) before resuming anything.

Completed same-model retrieval comparisons with GPT-4o, direct Gemini 3.6 Flash, OpenRouter
Gemini 2.5 Flash and GPT-4o-mini. With Gemini 3.6, Audrey's answer improves **4/5 → 5/5**
activities when retrieval restores grooming. Tim remains **6/7 → 6/7**: the newly retrieved
Alchemist appears, but Hobbit is omitted despite being present. No new whole-LoCoMo score.

A same-model evidence-ledger prompt was tested on these two examples plus four preselected
additional failures. It recovered Audrey's missing item but also misattributed other people's
books/desserts and invented quotations under valid memory IDs. Do not promote this format.
Across six pairs, median reader request time excluding pacing was 2.08 s → 4.09 s; not p99
or `/context` latency. Original DeepSeek and full native context artifacts are unchanged.

Live non-LoCoMo ingestion exposed a detached-pronoun unit. `narrative.py` now rejects common
English third-person/demonstrative openings instead of guessing a referent; raw turns stay.
The exact live failure is a regression test. The feature remains OFF; this guard is not a
general coreference solution or demonstrated recall improvement. Final focused checks:
**89 passed**, Ruff/Pyright pass; separate adapter/harness selection 74 passed, one optional
live-contract skip. New resumable model-specific runner: `benchmark/targeted_reader.py`.

## OpenRouter key persisted in Bifrost; targeted GPT-4o diagnostic — 2026-09-26

Latest authorization: store and use the supplied OpenRouter key in Bifrost. Provider
`openrouter`, key name `openrouter-user`, route **`openrouter/openai/gpt-4o`** at
`http://localhost:8091/v1`. Provider/key persistence verified in the mounted Bifrost DB;
three real inference calls succeeded. No secret is in repository artifacts. The installed
API requires adding the provider and key separately. Application defaults are unchanged.

Read [OPENROUTER-TARGETED-2026-09-26.md](OPENROUTER-TARGETED-2026-09-26.md).
Same two targeted failures and contexts as Gemini: Tim's book answers cover 3/7 expected
titles in both arms despite source coverage increasing 6/7 → 7/7. Audrey's experimental
answer covers 2/5 activities despite full source coverage; its control request returned
HTTP 402. **No accuracy gain established; no new full-LoCoMo score.**

Six requests reached OpenRouter: three answers, two initial output-budget 402s and one
final credit-reservation 402. Three earlier requests were rejected locally before key
registration. Successful usage metadata totals $0.0762. Final output cap is 1,024 tokens,
zero retries; no external judge calls. No benchmark process remains active. The final
402 mentions unsettled in-flight credits, not proof of a permanently empty account.
Original DeepSeek/native artifacts verified unchanged. No application code changes or
new application regression run. Gemini's earlier four-call allowance remains exhausted.

## Gemini available; user narrowed evaluation to four multi-hop calls — 2026-09-26

**Latest instruction: test only previously failing complex multi-hop questions, using four
Gemini calls.** The targeted run used all four; do not automatically resume the broad reader
or make further calls under this allowance. No Gemini benchmark process remains running.
Read [GEMINI-TARGETED-2026-09-26.md](GEMINI-TARGETED-2026-09-26.md).

Gemini 3.6 Flash works through Bifrost (2.5 Flash returned 404), so DeepSeek billing is no
longer the only available route. However the targeted test had **three HTTP 503 high-demand
errors and one successful answer**, yielding zero complete before/after pairs. Saved source
coverage improves Tim's books 6/7 → 7/7 and Audrey's classes 4/5 → 5/5. The successful Gemini
baseline answer still omits grooming. No answer-accuracy improvement is established.
Artifact: `codex_gemini_four_call_multihop.json`, with exact request count and manual review.

Earlier unrestricted probes and the now-stopped incomplete broad Gemini checkpoint are
separate from these four targeted calls. The original DeepSeek checkpoint is unchanged.
Live narrative smoke testing before the restriction exposed a bad passage selection in one
of two completed non-LoCoMo cases. The prompt now specifies explicit zero-based sentence IDs;
this revision has offline tests but no new live-model validation. The feature remains OFF.

New quota harness: 37 checks passed, one optional live contract skipped. Final focused tests
after prompt clarification: **101 passed**, including durable ingestion and complexity gates.
See `codex_gemini_validation.json` for source hashes, artifacts and limitations. No deployment.

## Generalization and selective ingestion — 2026-09-26

The user explicitly confirmed **generalize beyond LoCoMo** and permits latency above the
previously accepted 551 ms when justified by accuracy. No benchmark-specific runtime rules.
Read [GENERALIZATION-AND-INGESTION-2026-09-26.md](GENERALIZATION-AND-INGESTION-2026-09-26.md)
for primary research, a 20-row manual diagnostic, the implemented opt-in
`contextual_extraction` ingestion use, measured eligibility and remaining validation.

The new use groups source sentences within one complex user message into bounded narrative
quotations, through the existing LLMAssist/job pipeline. It cannot write generated facts,
identities, timestamps or access scopes. It consumes one logical assist call rather than
falling through to repeated sentence assists; default remains OFF. It is an extractive
precursor, **not cross-message fact synthesis or completed observation consolidation**.
Gate audit: 3633/5882 turns qualify (61.76%); all empty-output fallback candidates match the
native path. This high rate needs benefit/cost validation before enabling it.

Hindsight's 89.61% LoCoMo uses Gemini for answers and its retain path uses LLM extraction;
89.0% OSS-120B in the paper is LongMemEval. “LLM-free recall” is not LLM-free end to end.
Manual review found reader omissions, lost temporal clauses, speaker/gold defects and
missing visual detail. Its judgments are separate, unblinded diagnostics; original scores
and checkpoints are unchanged. Provider HTTP 402 still blocks the unfinished full reader
and live-model evaluation of enriched ingestion. Do not claim new accuracy or p99 here.

Validation: **858 passed / 0 failed / 0 skipped**, including 30 narrative cases and five new
actor-renaming/opaque-ID cases; changed-file Ruff and targeted Pyright pass. Details:
`benchmark/results/codex_narrative_validation.json`; full regression
XML and preserved initial failures alongside it. Local DB port is **5432**, Qdrant **6333**;
these are not 15432/16333. No deployment, paid model calls, schema changes or commits.

## Actor/topic retrieval implemented and measured — 2026-09-26

The user asked for concrete intelligence improvements, not another explanation of gaps.
Implemented bounded actor/topic query decomposition for aggregate memory queries. It finds
up to two named canonical subjects in authorized candidates, performs subject-filtered
secondary searches, and fuses them with the original query. It uses no LLM, keeps the same
100-memory / 8000-token limits, and falls back after 200 ms. Exact lookups, explicit limits,
document selectors and mixed/document pools bypass it. Tests cover store-level isolation,
lineage, cancellation, timeouts and bypasses. `memory_entity_search` remains **OFF**.

Full paired evaluation completed on 1986 questions, with development conversations 0–2 and
validation 3–9. Overall source recall **83.9047 → 84.2410%**, multi-hop recall
**68.3416 → 69.9924%**, complete multi-hop sources **42.9078 → 45.3901%**. Paired p99
**526.6 → 605.3 ms**, p95 353.8 → 418.9 ms. Validation has 17 improved / 1 regressed
answerable questions. The source audit verifies all 5882 turns and 6712 selected memories;
canonical and indexed evidence reproduce every score. This is a small evidence gain,
not 85%/90% answer accuracy. It remains experimental because answer/abstention validation
is still blocked by provider HTTP 402 and the latency trade-off is not yet justified.

The current saved reader diagnostic is now updated: 836/937 (89.22%) correct with complete
annotated sources versus 83/260 (31.92%) without them, on the partial completed cohort.
This is an association, not an oracle ceiling. Repeated source IDs were uncommon, so no
source-diversity feature was added. A nearby-source diagnostic is not a runtime result.

Read [ENTITY-TOPIC-RETRIEVAL-2026-09-26.md](ENTITY-TOPIC-RETRIEVAL-2026-09-26.md) for protocol,
confidence intervals, hashes, exact-source audit, and artifact names. A final exact-lookup
bypass cannot change any of the 1986 benchmark inputs; its source parity is recorded.
`codex_entity_topic_reader_reuse.json` identifies 2873 reusable arm judgments / 1355 complete
pairs; 696 distinct uncached context evaluations remain. It does not perform LLM calls.
The original full-reader checkpoint and its 475 unfinished questions remain untouched.
Do not overwrite that checkpoint when testing this separate experimental configuration.

Final validation: **814 passed / 0 failed / 0 skipped**, including 19 new feature cases.
Changed-file Ruff and targeted Pyright checks pass. Artifacts:
`codex_entity_topic_validation.json` and `codex_entity_topic_tests.xml`. Temporary experiment
containers were removed after successful completion and log capture. No deployment or
new LLM calls; the source tree remains uncommitted.

## Wider memory recall promoted — 2026-09-26

The user explicitly accepted the measured **551 ms retrieval p99**. The working-tree
defaults now select the full-LoCoMo candidate configuration: `memory_recall_k=100`,
`memories_max=100`, unchanged `token_budget=8000`. Document/mixed final pools remain at 50;
explicit caller limits are preserved. Memory hybrid search fetches up to 200 candidates.
The existing context-cache fingerprint includes both settings, so old narrow bundles are
not reused by the new configuration. A running deployment needs rebuilt/restarted workers;
this source change is not evidence of a production rollout.

The measured candidate achieved **83.90% source recall**, **68.34% multi-hop source recall**,
and **550.6 ms retrieval p99**. Those are complete retrieval results, not a new benchmark
after promotion. Partial DeepSeek Flash answer accuracy is **76.73%** on 1199 answerable
questions; 312 adversarial questions give **84.29%** abstention. Another 475 question pairs
remain blocked by provider HTTP 402. No full answer score is available.

Historical tables below and saved artifacts retain their original control/candidate labels:
their "default" means the previous narrow configuration, and their "wider" is now the
source default. The archived orchestrator refuses changed source hashes. The depth probe
pins control memory recall to zero but inherits context defaults; reproducing the old control requires explicit
`memory_recall_k=0` and `memories_max=50`. Saved reader contexts and judgments remain valid;
resuming the reader needs no retrieval or ingestion calls. Its retrieval provenance stays
at the original source hash; new reader-session provenance can have the promotion hash.

Promotion validation: **795 distinct tests passed** (782 unit cases, a 65-case focused
rerun including one new context-budget case, and 12 API/integration cases). Ruff lint and
format checks pass. All retrieval/context defaults exactly match the saved full-run
candidate configuration. No new LLM calls or latency benchmark were needed. Validation:
`benchmark/results/codex_wider_default_validation.json`. Source hash after promotion:
`ad3767614a9dc52191a5cf7ac0b27a5d63451a054c8698b2aefbe5a9b6139dbc`.

## Full LoCoMo evaluation requested — 2026-09-26

The user requested all ten conversations with LLM-free metrics and LLM reader/judge
results reported separately. Full LLM-free evaluation is complete: source recall
77.40 → 83.90%, complete multi-hop source coverage 32.62 → 42.91%, and p99
439.7 → 550.6 ms (previous default → now-promoted wider recall). All 1986 rows and 5882 source turns
were audited. The reader is **blocked by provider HTTP 402 (payment required)** after
1511 valid pairs; 475 remain incomplete or untested. Valid pairs and the interrupted
checkpoint are preserved. The wrapper now stops immediately on 402; six harness tests
pass. Restore provider access before resuming the stopped `memory-full-current-reader`
container. Do not re-ingest or rerun retrieval. See
[FULL-LOCOMO-2026-09-26.md](FULL-LOCOMO-2026-09-26.md) for its source hash, protocols,
checkpoint artifacts and recovery commands. Do not claim that the historical 72.92%
or the subset scores below are the full-current-code result.

## Accuracy experiments — 2026-09-26

Read [ACCURACY-EXPERIMENTS-2026-09-26.md](ACCURACY-EXPERIMENTS-2026-09-26.md) for the latest
paired evidence/reader measurements, controls and reproduction. Broader memory recall is
originally implemented as an opt-in bounded setting, with explicit caller/document limits preserved.
Across conversations 1/2, depth 100 improves strict answer accuracy 80.69 → 82.83% and
multi-hop 57.14 → 66.67%, but abstention falls 89.23 → 86.15%; promotion was deferred then.
The isolated `memory_bench_accuracy` database / `bench_accuracy_conv2` tenant holds the new
663-turn conversation 2 corpus. Existing benchmark databases were not reset.

Native dense/sparse fusion now has an explicit `hybrid_rrf_k`; default 1 translates to
Qdrant's historical zero-based k=2. Existing `rrf_k=60` is outer strategy fusion. Native
inner-60 reduced source recall and remains rejected. The 43-question gold-turn reader
diagnostic improves 48.84 → 69.77%, but annotations are incomplete and it is not a ceiling.
The selective-ingestion/consolidation work described below remains unfinished.

Final-code replay reproduces all 193 held-out control/candidate contexts exactly, so the
paired held-out reader score applies to the current inputs: answer 80.26 → 82.24%,
multi-hop 58.06 → 67.74%, source recall 83.33 → 87.94%, abstention 87.80% unchanged.
Final warm serial builder p50 is 126.6 → 170.7 ms and p99 337.7 → 519.1 ms, excluding
answer generation. Neither arm meets 300 ms p99. Full SciFact retains all 300 rankings,
nDCG@10 0.7436 / true recall@10 0.8731 / hit@10 0.8867; final repeat p99 values are
407.5 / 511.2 ms. Canonical `external_retrieval.json` now uses
`codex_accuracy_release_rag_baseline.json`, superseding the canonical-file statements
in older sections below. See `codex_accuracy_final_parity.json` for source identity.

**Latest validation:** 859 passed / 2 skipped / 0 failures in the final-source regression
run. Real DeBERTa and Docling PDF lanes were skipped because their host dependencies are
unavailable. Ruff lint/format passes; Pyright has 0 errors and 21 optional-dependency
warnings. Per-case results and fresh gates are in `codex_accuracy_validation.json` and
`codex_accuracy_gate_results.json`. Historical gate files were restored and temporary
benchmark containers removed. Changes remain uncommitted; no deployment was performed.

## Current follow-up — 2026-09-25

Earlier user preference: a small latency increase is acceptable for better accuracy.
The user has since accepted measured retrieval p99 551 ms (see promotion above). Report paired
quality and p50/p95/p99 deltas; do not treat this as permission for unbounded query-time
LLM loops or silently declare a new SLO. Prioritize a gold-evidence reader diagnostic,
bounded complementary retrieval, selective contextual ingestion, safe consolidation,
and temporal retrieval. Source-turn promotion already failed its paired accuracy trial;
do not re-enable it merely under a new diversity label.

The feature audit's historical failure diagnosis now records conditional answer accuracy:
90.20% with complete candidate evidence versus 29.23% without, using the old overlap
diagnostic and valid judges. Multi-hop complete candidate coverage was only 34.04%.
These are saved-run associations, not current-code results or proof of a reader ceiling.

Read [FEATURE-AUDIT-2026-09-25.md](FEATURE-AUDIT-2026-09-25.md) first. It checks the newly
supplied feature inventory against actual wiring, corrects the SPLADE/ColBERT and invalid
contextual-chunk ablation claims, and documents the remaining reflection lifecycle work.
The user's latest instruction permits **selective LLM ingestion for complex cases**; it
does not request enabling every LLM use or adding LLM work to the recall path.

New working-tree changes:

- Graph relations now hydrate both direct memory IDs and explicit memory evidence through
  one bounded canonical batch read. Access, deletion, expiry, current/as-of validity and
  source lineage are checked. Exact and graph reads share `memory_candidate`.
- A real-model probe found a second cut in context assembly: full primary-memory lists
  discarded every graph companion. Graph companions now have their own already-bounded
  allowance and share the hard token budget, matching document companion semantics.
  `memories_max` limits primary memories; total memories can exceed it by up to six graph
  companions under the shipped graph budget. Cache format is `rag-evidence-v5`.
- O(n) soft document diversity is implemented and tested but remains OFF by default.
  The full-corpus cap-1 result improved only three queries, with an interval including zero
  and no latency improvement established. Cap 2 did not improve quality.
- `benchmark.rag_ablation` runs sequential query-only full-corpus arms on an audited index.
  `benchmark.external_retrieval` rejects invalid ablation names/types, including the
  ingestion-only `contextual_chunks` flag. That flag requires a separately rebuilt index.
- `benchmark.graph_memory` provides paired, alternating on/off graph-source measurements
  without re-ingestion or LLM calls. It verifies the dataset's source bodies exist first.

The eight initial full SciFact arms are in `codex_rag_gap_*.json`. Their control and repeat
have identical 300 query rankings, nDCG 0.7436 / recall 0.8731 and p99 450.8 / 395.6 ms.
Those runs precede the **memory-only packing fix**; final-source controls are recorded
separately. `codex_graph_memory_probe_before_packing.json` preserves the informative null:
six memories fetched per eligible query but zero graph memories packed. Do not quote it as
proof of the finished graph feature or as an answer-accuracy measurement.

The final-source SciFact controls (`codex_rag_gap_final_baseline*.json`) retain identical
rankings and the same quality, with p99 **337.5 / 405.6 ms**. Canonical
`external_retrieval.json` now uses the first of these. The corrected graph probe verifies
369 source turns and performs 420 timed requests over 105 questions (two passes per arm).
Graph companions reach 36 questions per pass; annotated-source recall is unchanged. Its
cap-0/cap-6 p99 is **542.5 / 485.9 ms**, with treatment p95 worse. This is functional
coverage evidence, not an accuracy or speed improvement. All target claims remain open.

**Still unfinished:** safe continuous belief/summary consolidation and invalidation,
conversation hierarchy/global selection, independent temporal retrieval/ranking, grounded
multi-actor extraction, fielded BM25, measured late interaction and an MCP facade.
Landing reflection remains unwired because code inspection found audience, deletion,
concurrency and bounded-window correctness gaps. Do not enable it by adding the constructor
argument alone. No new LoCoMo answer accuracy or production HTTP p99 achievement is claimed.

Keep the isolated databases, frozen-source benchmark discipline and provenance rules below.
Final follow-up measurement and validation details are in the feature audit.

**Follow-up validation:** 845 passes / 2 skips in the broader run plus one passing golden
definition ablation: **846 passed / 2 skipped / 0 failures** across the two final runs.
See `codex_feature_validation.json` for per-case results and
`codex_feature_gate_results.json` for generated gates. Ruff passes; Pyright has 0 errors
and 21 optional-dependency warnings. The skipped lanes are real DeBERTa and Docling PDFs.
The golden test confirms definition expansion contributes evidence that verification can
independently recover when the flag is off. Do not interpret flat final scores as dead code.

## Previous handoff — 2026-09-25 (before the feature follow-up above)

Working tree is based on `05a45b7`; changes are not committed. Read
[LoCoMo and ONNX measurements](OPTIMIZATION-2026-09-25.md) and
[competitor research, capability audit and RAG validation](RESEARCH-RAG-2026-09-25.md)
before changing defaults. The user's priorities are measured accuracy, p99 <300 ms,
feature coverage grounded in actual wiring, reusable architecture, regression tests and
an accurate handoff.

**Do not claim the targets achieved.** Full historical DeepSeek LoCoMo is 72.92%; the
304-question paired source-turn experiment is a development subset. Among 233 answerable
questions, control is 189/233 (81.12%) and promotion 188/233 (80.69%). Promotion remains OFF.
All 304 final pairs are valid after retrying a failed whole pair. ONNX spin-off preserved
100/100 query vectors bit-for-bit and measured warm serial p99 231–260 ms versus 325–385 ms;
this is not the production HTTP/load SLO. See artifacts and limitations in the linked report.

**Implemented in this working tree:** canonical source references survive memory index,
exact reads and bundle construction; deterministic source grouping; ONNX spinning off;
request-local RAG evidence metadata; source-specific companion identities; correct parent
summary attribution; document selection across exact/post-stage retrieval; batched expansion
reads and position maps; a hard graph evidence-chunk cap; deterministic native fusion
ties (score, then record ID); truthful external RAG recall,
index readiness audit and regression gates. Cache format changed. Legacy memory indexes
need rebuilding to acquire the source-reference payload; legacy reads retain their fallback.

**Research corrections:** Hindsight's original 89.0 is LongMemEval with OSS-120B; 89.61 is
LoCoMo with Gemini-3. LLM-free recall is not LLM-free ingestion/answering. Its v0.4.19 report
has 92.0 LoCoMo and 94.6 LongMemEval. Mnemis System-1 reaches 89.1 with RRF alone; the old
8B-reranker requirement was false. Mem0's current 92.5 headline is managed-platform LoCoMo,
not an OSS or our-reader reproduction. There is no universal strict-score ceiling established
here; disregard the historical claim below that >94% is categorically unreachable.

**Final-code RAG measurement:** all 5183 SciFact documents / 6814 current indexed chunks,
300 queries, real ONNX + PostgreSQL + Qdrant. Both `codex_rag_stable*.json` runs have
nDCG@10 **0.7436**, true recall@10 **0.8731**, hit rate@10 **0.8867**, and identical top-10
rankings across all 300 queries. Serial p99 is **445.1 / 315.6 ms**, still above 300 ms.
These are retrieval metrics, not LoCoMo answer accuracy. The earlier baseline's true recall
recomputes to 0.8731; its reported 0.8867 was hit rate. Canonical `external_retrieval.json`
uses the first final-code run. No statistically established relevance improvement is claimed.

**Production gaps:** graph evidence hydration follows chunk IDs, not conversational memory
IDs. LandingReflection/derived belief services exist but are not injected by `_wire_memory`;
optional LLM reflection is not enabled by the LLM-free baseline. Independent temporal
candidate retrieval and refreshable mental-model workflows are not demonstrated. Do not
mark these complete based on classes or enum values. Do not enable dormant ingestion code
without replay, provenance, scope and quality tests.

**Run discipline:** use `memory_bench_docs` for SciFact, `memory_bench_conv` for LoCoMo,
`memory_tests` for pytest. Never reset the application database. Avoid concurrent CPU-heavy
jobs during latency measurement and freeze `src/` and `benchmark/*.py` while a bind-mounted
benchmark runs. Model files are available to the Linux Docker image; host macOS x86_64
cannot run this model stack. Real quality runs must specify `BENCH_EMBEDDING=frozen`,
`BENCH_SEARCH=qdrant`, `BENCH_DEPTH=shipped` and the isolated database URL.

**Validation:** latest per-case union **1093 passed / 19 skipped / 0 unresolved failures**,
not a single uninterrupted invocation. The broad run had one stale `workspaces=` test-fixture
argument, repaired and rerun. Four real OpenFGA contracts passed with the correct Docker
socket setting. After the final tie-break, 72 focused tests and the final evaluation/e2e/
security run (18 passed, 2 optional skips) passed, including all six distractor-corpus cases.
Ruff/format/diff checks pass; Pyright has 0 errors and 21 optional-dependency warnings.
Full breakdown: `benchmark/results/codex_rag_validation.json`; remaining skips cover optional
GCS/model/Docling/Bifrost/worker lanes. Never count those skips as provider validation.
Temporary benchmark containers were removed; PostgreSQL/Qdrant/Bifrost remain running.

**Artifacts / reproduction:**

- `benchmark/results/codex_source_reader.json`, `codex_comparison.json`, `codex_spin_*.json`,
  `codex_scheduling_parity.json`: previous LoCoMo/ONNX evidence.
- `benchmark/results/codex_rag_before.json`: full SciFact baseline; legacy `recall_at_10`
  means hit rate. Recompute true recall from saved rankings/qrels.
- `python -m benchmark.external_retrieval --reuse-index --out codex_rag_stable.json`: full existing
  SciFact index, with metric version 2 and corpus audit. Pre-tie-break runs are retained as
  `codex_rag_after*.json`; fixed-vector diagnostics confirmed equal-score ordering variance.
  Canonical `external_retrieval.json`
  is refreshed from a valid v2 run, preserving the historical baseline separately.
- Regression commands: `.venv/bin/python -m pytest tests/unit tests/contract tests/integration
  tests/e2e tests/security tests/failure sdk/python/tests -q`; then `tests/eval` with the new
  external artifact. The external gate checks its source hash: rerun when runtime/benchmark
  code changes. Ruff check/format and Pyright are also required. Use local services;
  skipped model/provider lanes are not evidence that those providers passed.

Next work should follow the ranked experiments in the research report: fixed full-set
reader/judge ruler, error partition, evidence-grounded session synthesis, bounded graph-to-
memory retrieval, temporal candidate arm, target-hardware HTTP load, and answer-level RAG
validation. No new ingestion/ranking experiment has earned default enablement from this
correctness patch. Update this section when results or decisions change.

---

## Historical handoff (2026-09-23; conflicting statements above take precedence)


> **Updated 2026-09-23 afternoon at `c866390`.** What changed since the morning entry below,
> in the order it matters:
>
> - **The judged benchmark never had the settings it documents.** `_settings` used
>   `model_dump(exclude_unset=True)` to find env-set values; on a pydantic-settings object
>   that returns *everything*, so the benchmark's own config was overwritten wholesale.
>   Measured: `max_tokens` 16384 -> **1024**, `timeout` 120 -> 30, `max_retries` 0 -> **2**,
>   plus `environment` and `log_level`. The 1024 is the root cause of v6's 19 "output budget
>   exhausted" judge failures - the 16384 the code passes never arrived. Fixed; every judged
>   result before this carries the wrong ceiling and three wire requests per logical call.
> - **The encoder is 61% of query p99** and ONNX fp32 is 2.5-2.8x faster with **bit-identical
>   vectors** (cosine 1.00000, two independent runs). int8 is *slower* here and the only
>   variant whose vectors move - no AVX-512 VNNI on this CPU. See
>   [LATENCY-LAYERS-2026-09.md](LATENCY-LAYERS-2026-09.md).
> - **The gate arithmetic in that doc was corrected**: three worker processes are three
>   `SerialRunner`s, so the aggregate ceiling is ~46/s on torch, not 9/s. The gate was never
>   what blocked 20 rps; CPU is. The flip is still the largest lever, for a different reason.
> - **An audit found 50 confirmed findings** (13 high), each verified by an agent told to
>   refute it: [AUDIT-2026-09-23.md](AUDIT-2026-09-23.md). Six are fixed - two authorization
>   holes (body-supplied `group_ids` became read keys; `GET /v1/jobs/{id}` leaked across
>   tenants), the dedup encoder batch, the encoder thread setting that had no effect, the
>   OpenFGA store race, and the staging guard. One is **disputed and must not be "fixed" the
>   obvious way** - see the note on the internal-message finding.
> - **The load test never reached docling**, so every capacity number so far describes traffic
>   with no documents in it. `--docs pdf` now exercises it.
> - **`make reindex` cannot run on this host** (no macOS x86_64 torch wheels); `reindex-image`
>   is the one to use. And the encoder flip needs **no reindex and no test edits** - the
>   LoCoMo harness resets and re-ingests on every run.
>
> Targets remain **unmet and unmeasured on the target hardware**: the only real throughput
> number is still 0.66 rps on a 4-core box, and p99 is 668 ms measured / 381 ms projected
> under ONNX. Nothing here changes the table below until the VM run exists.


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
   own `onnxruntime` runner for the encoder (tokenizer + session + CLS pooling + normalise; the sentence-transformers ONNX backend is *not* usable: `optimum-onnx` pins `optimum~=2.1`, incompatible with sentence-transformers 6), 2 intra-op threads, one executor + semaphore per worker, fingerprint includes the graph file; `WEB_CONCURRENCY=3`, `OMP_NUM_THREADS=2`; Qdrant gRPC + payload projection (`with_payload` include list under a unit test) + `on_disk_payload=False` for memories; one retrieval knob `final_k` (prefetch/fused derived from it; landed at the ratio every artifact on disk was produced at — 2.0, i.e. the shipped 100/100/50 and the judged 200/200/100 are unchanged. The roadmap's 1.25× reduction to 63/63/50 is **not** landed: it is a retrieval-quality change and stays gated on a judged run reporting evidence recall ≥ 0.987 plus `tests/eval/test_retrieval_gate.py`, neither of which has been run. When it passes, `DEPTH_RATIO` in `config/constants.py` is the only line that moves); one serialisation pass + gzip; rate limit default 6000; `pool_pre_ping=False` + recycle; GIN index on `graph_entities.aliases` + graph wall budget via a shielded task; bulk access bumps; retry idempotent Qdrant reads once on connection errors (seen 4× in 304 queries through Docker's host gateway); pure-ASGI middleware; OTel gated. Gate on the VM: `uv run python -m benchmark.load.run --base-url http://<api-host>:8080 --api-key <key> -u 20 -r 20 -t 300s --arm cold` → rps ≥ 20, `/v1/context` p99 ≤ 300, no failures.
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
