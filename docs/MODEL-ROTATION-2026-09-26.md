# Model rotation and grounded multi-hop diagnosis — 2026-09-26

The user authorized switching between OpenRouter and Gemini to finish the outstanding
targeted work. This supersedes the earlier exhausted Gemini allowance for this turn.
It does not authorize treating mixed-model outputs as one benchmark score. All inference
went through the existing Bifrost gateway; no production model defaults changed.

## Completed work

**32 requests, 28 successful responses, one HTTP 402, two HTTP 429s, one HTTP 503.**
No external judge calls. Reported OpenRouter response costs total **$0.0633092** for this
turn, separate from the previous $0.0762 experiment. These are response metadata, not a
complete provider billing audit. All processes stopped; successful outputs are preserved.

- Completed the missing GPT-4o Audrey control, preserving its earlier successful candidate.
- Completed both retrieval arms on both development questions with direct Gemini 3.6 Flash,
  OpenRouter Gemini 2.5 Flash, and GPT-4o-mini. Same original answer prompt and saved contexts.
- Tried Qwen3.8 27B and Gemma 4 31B free routes; each returned 429, then its circuit stopped.
- Tested a separate evidence-ledger answer format on the two development examples and four
  additional failures from distinct conversations, with both prompt arms on GPT-4o-mini.
- Completed three live non-LoCoMo narrative-extraction cases, fixed a reproduced unsafe
  span boundary, and replayed the outputs without more LLM calls.

OpenRouter's initial key-status endpoint reported a $100 key cap and $99.9238 remaining
under that cap. This is **not the account balance**. The first OpenRouter Gemini request
with an 8,192-token output allowance returned 402 (maximum affordable: 7,813); a separate
4,096-token configuration then succeeded. Gemini 3.6 completed its four original-prompt
requests but returned 503 on the next experimental-format call, so that format was tested
on GPT-4o-mini instead. All attempts are saved; failures are unscored.
The provider documents distinguish key caps, balance and temporary in-flight reservations:
[OpenRouter limits](https://openrouter.ai/docs/api_reference/limits).

## Retrieval comparison

These numbers are **expected-item coverage**, not answer accuracy. Both questions were
selected earlier because DeepSeek failed and experimental retrieval restored missing
annotated sources. This is not a random or representative benchmark sample.

| Reader | Tim: control → experimental, out of 7 books | Audrey: control → experimental, out of 5 activities |
| --- | --- | --- |
| GPT-4o | 3 → 3 | 2 → 2 |
| Gemini 3.6 Flash, direct | 6 → 6 | **4 → 5** |
| Gemini 2.5 Flash, OpenRouter | 4 → 5 | 4 → 3 |
| GPT-4o-mini | 4 → 5 | 3 → 4 |

For Gemini 3.6, Audrey's restored grooming evidence produces a complete, supported answer.
Tim's experimental context restores The Alchemist, and the reader includes it, but now
omits The Hobbit, which remains in the context. His source coverage is 7/7 while answer
coverage remains 6/7. The weaker readers also show omissions and cross-speaker mistakes.
The Gemini 2.5 control incorrectly includes Dune, which John was reading, in Tim's list.
Coverage alone must never be reported as successful grounded answering.

All original-source hashes and the prior DeepSeek checkpoint remain unchanged. GPT-4o's
completed pair combines its three earlier successes with one new, matching control;
other model outputs are kept in their own records. No new full-LoCoMo answer score exists.

## Evidence-ledger experiment

The alternative format requests one row per answer item, memory IDs and a short source
quotation before the final answer. It contains no benchmark names, gold answers or item
hints. The four additional questions were fixed before generation: rows 19, 208, 327 and
507, the first prior complete-source multi-hop failure from each of conversations 0–3.
Both arms use GPT-4o-mini, identical contexts and a 2,048-token output cap.

| Case | Finding |
| --- | --- |
| Tim's books | Coverage 5/7 → 7/7, but ledger adds John's Dune and misquotes evidence. Still incorrect. |
| Audrey's classes | 4/5 → 5/5; the missing training course is restored. Final answer supported. |
| Melanie's children | Both omit dinosaurs from image-caption evidence. |
| Shared city | Original question says Jean/John; actual speakers are Gina/Jon. Baseline abstains; ledger incorrectly claims shared Paris despite Gina's explicit denial. |
| Maria's desserts | Ledger incorrectly adds John's apple pie and attributes its quote to Maria. Gold also confuses enjoying a sundae with making it. |
| Joanna's emotions | Baseline already conveys all five emotions, including hopes; ledger final answer does too, with some misassigned quotations. |

All ledger citation IDs exist in the supplied context. **ID existence does not establish
that the cited memory contains the quotation or supports the speaker/relation claimed.**
The format is not promoted: it recovers some missing items but introduces unsafe assertions.
Original strict scores are not rewritten to accommodate annotation defects. These local,
unblinded manual findings are recorded separately in `codex_model_rotation_manual_review.json`.

Request timing now excludes deliberate quota pacing. Across these six same-model pairs,
median baseline reader time is **2.08 s**, versus **4.09 s** for the ledger; observed maxima
are 3.61 s and 5.40 s. This is a tiny diagnostic sample, not p99 or end-to-end service latency.
No new `/context` latency or load benchmark ran; the retrieval path was not changed.

## Narrative ingestion: actual fix and limits

GPT-4o-mini returned valid structured selections on three synthetic domains. Conservation
and gardening kept both intended two-sentence units. Software selected a standalone
"She did not restart the database" without the preceding Maya sentence, plus a correct
Omar unit. Explicit zero-based indexing alone did not solve semantic span quality.

`modules/memory/narrative.py` now rejects units starting with common English third-person
or demonstrative references. It does not guess a referent or attach an arbitrary preceding
topic. First-person and named openings remain allowed. This is a bounded, conservative
boundary check, not multilingual coreference resolution or proof of semantic correctness.
It may reject context-independent uses such as "It is raining"; the raw turn remains.

The captured software failure is now a regression test. Offline replay rejects that
detached unit and preserves the correct Omar unit; both other cases retain their two units.
This does **not** recover the lost Maya representation or establish an accuracy gain.
Raw turns and application-owned source IDs survived all live cases. `contextual_extraction`
remains opt-in/OFF; the earlier 61.76% eligibility rate still needs benefit/cost work.
No live corpus re-ingestion or new retrieval benchmark was run for this guard.

## Validation, artifacts and next work

Final focused regression: **89 passed, zero skipped**, covering narrative/native extraction,
durable ingestion, complexity bounds and quota/rotation harnesses. The broader harness and
Bifrost contract selection passed **74 with one optional live-contract skip**; real requests
above exercised the gateway independently. Ruff and targeted Pyright pass. A preliminary
test command named a nonexistent unit-test path; the corrected contract path ran successfully.

`benchmark/targeted_reader.py` is the reusable bounded runner. It uses the existing adapter
and wire pacer, checkpoints every attempt, refuses changed manifests on resume, reuses
successful answers, and stops a failed model for that invocation while allowing other
models to continue. It never silently relabels a fallback as the original model. Manifests
contain only model inputs, not gold labels. `WirePacer` records request latency after its
quota wait and preserves finish reasons for truncation audits.

Primary artifacts under `benchmark/results/`:

- `codex_model_rotation_{manifest,answers}.json` and
  `codex_model_rotation_fallback_{manifest,answers}.json`: route-specific comparisons.
- `codex_evidence_ledger_{selection,manifest,answers}.json`: selection and failed direct
  Gemini format attempt; `codex_evidence_ledger_mini_{manifest,answers}.json`: 12 successes.
- `codex_model_rotation_narrative_smoke.json` and
  `codex_model_rotation_narrative_guard_replay.json`: live units and offline guard replay.
- `codex_model_rotation_manual_review.json`, `codex_model_rotation_validation.json`,
  test XMLs and source snapshots: assessment, hashes, request counts, timing and checks.

The next accuracy work needs speaker-bound, source-validated consolidation and structured
evidence aggregation, with quotation-to-source validation and relation checking. Additional
model rotation or requesting longer lists alone does not resolve the observed failures.
The optional actor/topic retriever and narrative feature remain OFF pending broader paired
quality validation. No 85%/90% claim, production deployment or commit was made.
