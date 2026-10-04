# Ingestion, consolidation and source-backed retrieval — 2026-09-26

This supersedes the earlier “contextual extraction remains OFF” local-configuration notes.
The user explicitly authorized ingestion-time LLMs and available OpenRouter/Gemini routes.
No new overall LoCoMo answer score has been established. Do not present these smoke tests
as 85%/90% accuracy, or equate source coverage with answer correctness.

## What is now wired

The local ignored `.env` enables `contextual_extraction` and `reflection`, using
`openrouter/openai/gpt-4o-mini` for extraction and `openrouter/openai/gpt-4o` for background
reflection through Bifrost at localhost:8091/v1. Provider credentials remain in Bifrost.
Retries are zero, timeout 60 seconds. No query-time LLM use is enabled. This configuration
applies on the next application/worker start; no deployed service/image was restarted or
rebuilt. The currently running infrastructure is PostgreSQL, Qdrant and Bifrost.

Extraction consults the model for eligible multi-sentence messages, preserving exact source
spans, identities, negation and raw text. A failed model call retains native ingestion. The
eligibility gate is not a proven rare-call gate: the previous corpus audit estimated 61.76%
of LoCoMo turns eligible. Budget a full re-ingestion accordingly; old persisted memories are
not automatically enriched just because the new configuration is enabled.

`LandingReflection` is now constructed for the contextual-extraction profile. It creates
bounded extractive BELIEF/ENTITY_SUMMARY records after admission. Native admission remains
the only owner of source reinforcement and supersession: landing no longer reinforces twice
or resolves conflicts by arrival order. It also rebuilds after reinforcement changes a source
revision. Whole source statements fit a 1,800-character budget; no 120-character truncation
that could remove a late negation. At most 64 current sources are considered, within the
same author, scope, subject and exact stored audience. This is a recent source synopsis,
not a comprehensive lifetime entity profile or semantic generalization engine.

Background `ReflectionService` now uses complete source text within a 12,000-character prompt,
requires at least two cited sources, rejects unknown/unseen citations, and rechecks revisions,
expiry and audiences when writing. Other derived memories do not feed back into its source
pool. The existing periodic schedule is every six hours, with a 24-hour/1,000-record scan;
this is bounded background consolidation, not model-weight learning or a complete historical
reprocessing scheduler. GPT-4o-mini first produced unsupported preferences/motives; the revised
prompt asks for factual synthesis and explicitly prohibits those inferences. Reflection
records are tagged as LLM-produced, with a prompt version. They still require source checking:
valid citation IDs alone do not establish entailment.

A live provider defect was found and fixed: the strict reflection JSON schema lacked
`additionalProperties: false` on both object levels, causing OpenRouter/OpenAI HTTP 400.
Both corrected live reflection calls succeeded. The schema shape has a regression assertion.

## Source lifecycle and retrieval

Migration `0009_memory_dependencies` adds indexed source dependencies with captured revisions
and a unique live derived slot. Same-source admission locks are acquired before source writes;
slot locks plus the unique index prevent competing current versions. Model calls stay outside
write transactions/locks. Generated insights cannot become reinforcement targets for new
source facts.

Changing, superseding, forgetting or expiring a source retracts descendants recursively in
PostgreSQL. Removal jobs enter the same transactional outbox. Insert-time source row locks
prevent accepting an insight derived from a snapshot changed while the model was working.
Access uses actual source-key intersections, not a total ordering of RUN/THREAD/USER/GROUP.
Reflection remains private where the source audience permits the owner principal; it cannot
invent broader access for run-only sources. Derived expiry cannot exceed the first source
expiry.

Search hits marked as derived, including legacy memory-source references, receive a canonical
batch check before use. Invalidated vector entries cannot remain usable while the outbox
catches up. Bundles containing derived memories bypass the bundle cache; the new cache-key
version avoids pre-change cached bundles. Queries without derived candidates incur no new
canonical check. The cost of lost bundle-cache hits must be measured under production load.

Retrieval can fetch up to six missing original sources in one batch, with current-state and
visibility checks. There is no recursive query-time traversal or model call. These companions
share the existing token budget and survive a full primary-memory count cap, like existing
graph companions. The token budget can still omit evidence; this is not a guarantee that all
supporting memories fit. All source IDs remain available as provenance.

The migration retracts legacy beliefs/summaries/reflections because their source revisions
were not captured; originals remain. Reconciliation removes old index entries, and canonical
checks reject them meanwhile. Upgrade/downgrade/upgrade was tested on a cloned disposable
probe database: zero legacy current derived records and 18 current originals afterward.
The local configured `memory` database was at migration 0008 with zero memories; it is now
at 0009. The live probe uses separate database `memory_consolidation_probe` and Qdrant
collections ending `__consolidation_probe`; regression tests use `memory_tests`.

## Live evidence and limitations

The fixed probe covers two messages each in software operations and gardening. It uses
real PostgreSQL, server Qdrant, the frozen ONNX encoder and native BM25, plus real Bifrost
calls. Authorization/queue are in-process, cache disabled, NLI lexical. It does not exercise
HTTP load, production ReBAC latency, document RAG, or an answer reader/judge.

| Artifact | Requests | Finding |
| --- | ---: | --- |
| `codex_consolidation_live.json` | 6 | Four extraction successes, two reflection HTTP 400s; derived extractive records indexed and retrieved. |
| `codex_consolidation_reflection_live.json` | 2 | Fixed schema; six GPT-4o-mini insights stored, but unsupported generalizations appeared. |
| `codex_consolidation_factual_live.json` | 2 | Revised factual prompt plus GPT-4o; five observations stored. Actor attribution substantially better on these examples. |

Metered probes: 10 attempts, eight successful responses, reported OpenRouter cost
**$0.00972875**. The later broad suite also invoked the live gateway contract (a preflight
and a roundtrip); its roundtrip was rejected with HTTP 402. Those requests were not captured
by the probe cost recorder. Do not interpret the probe sum as total account spending.
The final prompt/model were changed together, so their effects are not isolated. The last
probe reused the same isolated corpus; earlier experimental insights remain in that database
and therefore its later retrieval timings include those records. These are debug artifacts,
not an uncontaminated final quality comparison.

Both initial arms had **2/2 source turns present for all four queries**. No measured source
coverage gain on this small corpus. Across 20 warm, cache-disabled builds per arm: native
median 86.95 ms/max 148.28 ms; assisted extractive median 120.21 ms/max 315.21 ms. After final
reflection, 20 builds had median 94.72 ms/max 338.43 ms; different sequential runs under changing
load, not a speedup comparison or p99 estimate. Ingestion includes network/pacing and grew
into seconds. Four extraction provider-request times were 1.23–1.61 seconds; final two GPT-4o
reflection requests were 3.35 and 3.81 seconds, outside the retrieval path.

Manual review of the final output: software actor/action/negation synthesis is supported,
including Maya not restarting the database. Gardening still says covering seedlings “helped”
them survive, which asserts causality stronger than the explicit sequence of events. Thus
citation validation and this prompt are not sufficient semantic verification. Do not label
these generated observations as verified facts or claim better quality than Hindsight.

## Validation and remaining work

New PostgreSQL tests cover concurrent admission, repeated ingestion, single live summaries,
source revision changes, deletion, recursive descendants, expiry, rollback, outbox removals,
stale vector hits, mixed PRIVATE/USER audiences, evidence expansion, minimum source count,
unknown citations and a source deleted during a model call. Deployment-profile tests build
an LLM-ingestion container, so a missing constructor argument cannot silently disable landing.
Context tests require source companions to obey tokens while surviving the primary count cap.

The initial broad suite passed 1,142 tests with 21 skips and three failures. Two outdated
reflection doubles/contracts were repaired; the complexity ratchet prompted refactoring,
without relaxing its limits. Focused real-DB/profile/adapter checks then passed 76 tests.
The latest broad run passed 1,147 tests, skipped 20, and failed the already-loaded complexity
check plus the now-enabled live Bifrost contract. The complexity check passes after refactoring.
The external contract failed with OpenRouter HTTP 402 at a 1,024-output-token request: an
upstream credit/reservation constraint, not a passing test. Successful 600-token reflection
calls earlier in the run do not establish that every later request is affordable. The local
profile remains enabled; provider failures use native fallback. No more provider calls were
made after this refusal. Final affected-path checks and exact counts are in
`benchmark/results/codex_consolidation_validation.json` and the associated XML files.
Final affected-path rerun: **79 passed**, including run-only audience isolation. Ruff and
Pyright pass, and the complexity budget was not raised. Some tests require optional external
services/explicit live flags and remain skipped. The live HTTP 402 remains unresolved.

Reproduction:

```bash
.venv/bin/pytest -q tests/unit tests/integration tests/security tests/contract
.venv/bin/ruff check src/memory_service/modules/memory benchmark/consolidation_smoke.py
# Use the model-capable image: the host Intel macOS venv lacks compatible ONNX 1.30 wheels.
docker run --rm --entrypoint python -v "$PWD:/app" -w /app \
  -e PYTHONPATH=/app/src:/app -e CONSOLIDATION_DB_HOST=host.docker.internal \
  -e MEMORY__MODELS__LLM__BASE_URL=http://host.docker.internal:8091/v1 \
  -e MEMORY__SEARCH__QDRANT_URL=http://host.docker.internal:6333 \
  -e HF_HUB_OFFLINE=1 memory-service-memory-api:latest -m benchmark.consolidation_smoke
```

The probe refuses an existing output or reused source tenant. `--reflect-existing` is a
bounded two-call diagnostic, and `CONSOLIDATION_OUTPUT` selects a distinct artifact;
`CONSOLIDATION_MODEL` selects the route. Do not remove artifacts to blindly repeat spending.

Next accuracy work: independent holdout scenarios with named actors, conflicting updates,
negation and multi-session chronology; source-entailment acceptance for generated observations;
then a fixed-reader, paired full LoCoMo re-ingestion/answer run with request/cost accounting.
Also measure full-corpus retrieval and loaded HTTP p99 with derived cache bypass. Actor-level
canonicalization, causal relations, comprehensive historical retrieval and multi-level topic
routing are not completed by this patch. Document RAG/SciFact has not been rerun here.

The original DeepSeek/full-native results remain unchanged:

- DeepSeek checkpoint SHA256: `449506f05efd00aa0c61e34c5c84f144a524996914f80708363832090f90a779`.
- Full native context SHA256: `82e20f985e1aba28d22ecc6f703e5e46c770deb559227d020d49a58b8a5c49b5`.
- Last known broad answer accuracy: 76.78% on 1,206 graded answerable questions; 78.33%
  including 312 graded adversarial questions. Incomplete 1,986-question answer run.
- Earlier full native source recall: 83.90%; retrieval-context p99 around 551 ms in its
  serial benchmark. Those measurements predate this ingestion profile.
