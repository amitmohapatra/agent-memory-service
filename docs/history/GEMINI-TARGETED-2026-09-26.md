# Gemini: restricted multi-hop diagnostic — 2026-09-26

The user authorized the public Gemini provider through Bifrost, then narrowed the work:
**only previously failing complex multi-hop questions, with four Gemini calls**. This was
interpreted conservatively as four total requests for the targeted experiment. Do not resume
the broad Gemini run or issue more model calls under that exhausted allowance.

## Four-request result

Selected two earlier DeepSeek failures, each evaluated with current and experimental
actor/topic contexts using the **same Gemini 3.6 Flash reader and answer prompt**. No external
judge calls, no retries, concurrency one, at most four requests/minute and four requests total.
The model did not receive the gold answers. Selection was fixed before Gemini outputs:
among prior multi-hop failures with exact baseline-context parity and incomplete baseline /
complete experimental source coverage, choose the two with most annotated sources.
This deliberately selected diagnostic cannot estimate overall benchmark accuracy.

| Question | Current sources | Experimental sources | Restored evidence | Gemini result |
| --- | --- | --- | --- | --- |
| What books has Tim read? | 6/7 | 7/7 | The Alchemist | Both requests returned HTTP 503 |
| What classes/groups has Audrey joined for her dogs? | 4/5 | 5/5 | Dog grooming course | Experimental request returned 503; current answer still omitted grooming |

Exactly **four targeted requests**, **one successful answer**, **three provider 503 errors**,
**zero complete before/after answer pairs**. Error message: model experiencing high demand.
These were not 429 rate-limit responses. No additional requests were made after the cap.
The successful baseline answer was reviewed locally against gold and source evidence: it
lists four of five required activities and therefore fails the strict completeness rule.
Errors are unscored, never counted as incorrect answers.

The missing source facts are demonstrably present in experimental retrieval. The experiment
**does not establish an answer-accuracy gain**. Both arms use native ingestion and the same
100-memory/8000-token settings. This tests the existing optional actor/topic retrieval,
not the new contextual-extraction feature. No product defaults were changed or deployed.

Artifacts:

- `benchmark/results/codex_gemini_four_call_multihop.json`: inputs/hashes, fixed selection,
  per-arm output/error, wire responses and separate manual assessment.
- `benchmark/results/codex_gemini_four_call_probe.py.txt`: exact four-request runner; refuses
  to overwrite an existing artifact to prevent accidental repetition.
- Existing `codex_full_current_llm.json` and `codex_full_current_nollm.json` remain unchanged.

## Earlier activity, before the four-request restriction

Bifrost `/v1/models` listed Gemini 2.5 Flash, 3.6 Flash and 3.6 Pro. A real 2.5 Flash call
returned 404 (unavailable to new users); a 3.6 Flash call successfully answered saved row 0.
No gateway credentials or production model settings were changed.

A separate broad reader checkpoint was started before the restriction. It has one completed
adversarial judgment and the initial row-0 answer awaiting judgment. A subsequent answer
request returned 503. This checkpoint is **incomplete and stopped**, not a full Gemini score.
It is kept separately in `codex_full_current_gemini.json`; DeepSeek results were not merged.
An initial custom-client wiring error made no successful provider request; it was corrected
and covered by a real HTTP-mock integration test.

Two synthetic non-LoCoMo extraction smoke cases also completed before the restriction:
conservation selected the two intended passages; software returned a bad passage crossing
speaker/topic boundaries. The third case was canceled while waiting for pacing, before its
request was sent. Original text/provenance survived, but semantic selection was not reliable.
The raw output is saved in `codex_gemini_narrative_smoke.json`.

The narrative prompt now explicitly supplies zero-based sentence IDs and states the indexing
convention. This addresses an ambiguity in the earlier unnumbered input array; it is **not a
verified fix for all semantic selection failures**. No further live call tested that revision.
Contextual extraction remains OFF by default. Do not call this a demonstrated accuracy gain.

## Harness and validation

`benchmark/quota_reader.py` provides request-level pacing, a hard request budget, atomic
checkpoints after generation and judgment, fixed model/context/prompt validation on resume,
and a fixed balanced-category schedule. It evaluates one saved-context arm. Adapter repairs
and transport retries pass through the same request hook, so they consume the request budget.
The general harness permits bounded transient retries; a 429 is intercepted and pauses it.
The four-request diagnostic explicitly disabled retries instead.

Quota response details and token usage are saved without request credentials. Per-project
limits may depend on requests, tokens and daily quotas; four RPM is our conservative pacing,
not a verified statement of this project's entitlement.
[Google rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

Validation: **37 passed / 1 skipped** across harness, adapter retry/budget and Bifrost contract
checks. The skipped live contract test requires model-enabled environment settings; explicit
live probes above independently exercised the configured gateway. After the sentence-ID
prompt clarification, **101 focused tests passed**, covering native extraction, narrative
extraction, the durable pipeline, complexity limits and the quota harness. Ruff checks pass;
targeted Pyright passes. Results are in `codex_gemini_harness_tests.xml`,
`codex_gemini_final_tests.xml`, and `codex_gemini_validation.json`.

There are no running Gemini benchmark processes. Under any future separately authorized
allowance, retain successful answers and request only missing experimental/control answers;
keep the model and prompts fixed. Do not claim 85%/90% or extrapolate these selected failures.
