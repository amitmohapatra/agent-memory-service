# Generalization, research and selective narrative ingestion — 2026-09-26

The user confirmed that improvements must generalize beyond LoCoMo and explicitly permits
selective ingestion-time LLM use and higher retrieval latency when accuracy justifies it.
551 ms is an accepted previous result, **not a newly imposed upper bound**. Do not add
benchmark-question branches, hard-coded answers, speaker-specific extraction or gold labels
to runtime decisions. No deployment or full new answer benchmark was performed in this step.

## What the competing results mean

Hindsight's paper reports **89.0% on LongMemEval with OSS-120B**, and **89.61% on LoCoMo
with Gemini-3 as answer generator**. It explicitly uses LLM-based narrative extraction.
Those are not end-to-end LLM-free results. Its architecture combines entity/semantic/causal
links, temporal retrieval, reranking and asynchronous entity summaries. “No LLM at recall”
is a narrower claim than “no LLM in the system.”
[Primary paper, especially sections 4 and 7](https://arxiv.org/html/2512.12818v1).

The later Hindsight v0.4.19 report gives **92.0% LoCoMo / 94.6% LongMemEval**, in single-query
mode. The authors identify observations, improved retain and retrieval as likely major
contributors, while explicitly saying attribution across changes is not clean. Treat this
as their reported result, not an isolated observation-layer ablation or our reproduced score.
[Primary report](https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark).

Mem0's April update describes context-aware, single-pass extraction, secondary facts,
agent-generated facts and entity matching alongside semantic/keyword retrieval. It states
that managed-platform scores include proprietary optimizations absent from the OSS SDK.
Its remaining weaknesses at long context lengths include temporal and cross-session
reasoning. Do not assume a single optional component explains its whole score increase.
[Primary report](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm).

Mnemis combines similarity retrieval with top-down traversal of hierarchical memory;
HyperMem explicitly organizes topics, episodes and facts, with LLM-driven episode detection.
These motivate testing structured context, rather than indefinitely increasing top-k.
Neither is evidence that hierarchy can reach its headline score on our current CPU/reader
configuration without measurement.
[Mnemis paper](https://arxiv.org/abs/2602.15313),
[HyperMem implementation](https://github.com/EverMind-AI/HyperMem).

In these mechanisms, ongoing memory synthesis and revision are distinct from training model
weights. We already reinforce repeated evidence and maintain supersession; that does not
supply missing event relationships or a functioning observation/summary maintenance loop.

## Implemented, with conservative boundaries

New `modules/memory/narrative.py` is called by `NativeMemoryIntelligence.extract` through the
existing LLMAssist and background observation pipeline. Enable by adding
`contextual_extraction` to `models.llm.uses` with LLM enabled. It uses the configured fast
model by default and remains disabled unless explicitly selected. No new dependency,
endpoint, database schema or query-time model call was introduced.

- Eligible only for user-authored MESSAGE observations with at least two substantive
  sentences containing clauses that native rules cannot extract. Known simple facts,
  questions, greetings and agent working notes do not take this path.
- One logical structured consultation per eligible message; it consumes the sentence-assist
  budget even on empty/failed output. Existing transport retry settings still apply, so
  one logical consultation is not a guarantee of one billed wire request.
- The model receives at most 16 sentences / 6000 source characters. Oversized messages
  bypass enrichment rather than hiding a later correction through truncation.
- It selects at most six contiguous passages, each at most four sentences / 2000 characters.
  Only original sentence text is accepted. Model-authored content, subjects, scopes,
  dates and source IDs are rejected/ignored, never persisted from the response.
- Passages are OBSERVATION quotations, not asserted semantic triples. Their source identity,
  author and observation time come from the application. Native facts remain, as does the
  raw turn; duplicate selections, including selections equal to the raw turn, add nothing.
- Empty/invalid responses and provider failure preserve native results. Cancellation
  propagates. Selection may still omit necessary context; verbatim text is not a semantic
  correctness guarantee, so quality validation remains required before default enablement.

This is a **bounded extractive precursor**, not completed Hindsight-style narrative fact
synthesis. It does not yet resolve references across separate messages, normalize relative
event dates, generate typed causal triples or maintain topic/entity summaries. The prompt
asks for antecedents and corrections; tests verify output boundaries and mocked behaviors,
not whether a live model reliably selects them.

The default path remains unchanged apart from a cheap disabled-use check. An enabled path
can increase stored representations and indexing cost; do not infer unchanged retrieval
latency after corpus enrichment merely because there is no query-time LLM call.

## Manual diagnostic, separate from the benchmark judge

Selected four completed candidate answers per category using seed 20260926, before reviewing
that selected sample. Saved selection and checkpoint hashes, questions, answers, original
judgments and separate manual notes in:

- `benchmark/results/codex_manual_review_selection.json`
- `benchmark/results/codex_manual_review.json`

This is a **20-row unblinded assistant diagnostic**. Core answers and relevant retrieved
passages were inspected; disputed examples were checked against raw dataset turns. It is
not an exhaustive claim audit, a substitute model run, or a population accuracy estimate.
The original Flash scores were not changed. No external model calls were made.

Concrete findings:

| Row | Observation | Implication |
| --- | --- | --- |
| 798 | Retrieved context contains Tim's plot twist, but the answer omits it | Reader completeness error; simply increasing recall does not fix it |
| 1158 | Raw bowling turn contains “yesterday”; retrieved candidate only says “I love bowling” | A retrieved source ID does not prove the needed temporal clause survived |
| 616 | Question names Nate; annotated source D11:13 is Joanna speaking | Gold/question speaker mismatch; do not train incorrect attribution |
| 352 | Source dated July 31 supports July 30; malformed gold was judged as July 2 | Flag adjudication separately; no automatic score rewrite |
| 26 | Named book title is absent from supplied text/caption | Text-only evidence is insufficient to verify the title; image itself was not inspected |
| 30 | Gold infers an identity from lack of disclosure | Preserve defensible abstention rather than inventing identity attributes |
| 1335 | Adversarial label requires abstention despite partly relevant facts about James | Ambiguous annotation and some cross-speaker phrasing; no blanket rule to answer adversarial questions |

## Tests and gate audit

Added non-LoCoMo software-incident, conservation, gardening and telescope examples; actor
renaming checks include multi-word, accented and non-Latin names. These are contract and
metamorphic tests, not evidence of multilingual or cross-domain benchmark accuracy. The
actor/topic planner still relies on name-bearing canonical subjects and English routing;
opaque user IDs require a real alias/entity-resolution layer rather than name guessing.

The gate audit processes all **5882 turns**, with a local empty-output stub and zero external
model calls. **3633 turns (61.76%) qualify**. Native fallback outputs match on every turn.
This is not yet a rare-case gate or demonstrated cost saving; actual useful-unit yield,
false attribution, indexing amplification, retrieval accuracy and cost per correct answer
must determine whether to tighten it or enable it. CPU-only extraction timing excludes
model, network, indexing and retrieval, and is not service p99.
Artifact: `benchmark/results/codex_narrative_gate_audit.json`; archived audit script alongside it.

Final validation: **858 passed / 0 failed / 0 skipped**, including 30 narrative cases and
five new actor-plan generalization cases. Seven changed Python files pass Ruff lint and
format; narrative/native/settings pass Pyright with zero errors or warnings.
Regression results and source hashes are recorded in `codex_narrative_validation.json` and
`codex_narrative_regression_tests.xml`. The initial run used the wrong local DB port and
was interrupted; its XML is preserved. A subsequent complexity-budget failure prompted
parser refactoring, with the existing threshold unchanged. Integration test expectations
were corrected to use persisted category metadata and to assert absence of the forgotten
memory (other authorized memories can still be returned by fallback retrieval).

## Remaining measured work

1. Run the same reader/ruler on separately ingested native and enriched corpora; keep reader
   choice fixed, persist extraction responses, and report model tokens/calls, source coverage,
   answer/abstention, per-category scores and retrieval p50/p95/p99. Do not mix this into the
   existing partial full-reader checkpoint. Provider billing is still unresolved (HTTP 402).
2. Validate cross-message narrative extraction with authorized bounded history and exact
   multi-source provenance; distinguish event time from observation time. No gold-derived
   trigger or query-conditioned ingestion.
3. Make derived summaries safe before wiring LandingReflection: indexed unique scope slots,
   source dependencies, invalidation on revocation/deletion/supersession, and audience
   intersection. Do not enable the existing unsafe implementation merely to fill BELIEF rows.
4. Evaluate temporal/event ranking and typed graph links, then hierarchical topic/episode
   selection and conversation-specific reranking. Retain document RAG as a separate gate;
   the old SciFact reranker regression does not establish that every conversation reranker
   fails. Test wider latency budgets on fixed corpora rather than promising 90% beforehand.
5. Use held-out domains and production-shaped threaded conversations, renamed entities,
   paraphrases, corrections and permission changes. Add LongMemEval or another independently
   held-out conversation dataset before claiming cross-benchmark accuracy.

Latest established full native source recall remains **83.90%**, multi-hop **68.34%**;
partial Flash answer accuracy remains **76.73%**. The previous actor/topic experiment gives
**84.24% source recall / 69.99% multi-hop**, paired p99 **605.3 ms**. These are different
metrics and configurations, not 90% answer accuracy and not results for the new ingestion
path. Full answer evaluation still has 475 incomplete/untested pairs due to provider HTTP 402.
