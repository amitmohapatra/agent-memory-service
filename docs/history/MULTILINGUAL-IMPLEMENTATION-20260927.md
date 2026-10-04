# Multilingual implementation and model selection

Status: **in progress**, following the user's explicit multilingual goal on September 27.
Worktree: `ams-agent-capabilities`, branch `codex/agent-capabilities-20260927`.
This work builds on uncommitted agent-capability changes. No production deployment or
shared corpus migration is authorized by a benchmark result. No paid model calls.

## September 28 constraints and queue correction

The user excludes Chinese-developed models, including derivatives with Chinese base
models. This is a model-provenance restriction, not a restriction on Chinese-language
support. The queued Qwen retry was cancelled before execution; the failed earlier artifact
remains historical evidence, not an eligible candidate. Do not resume it or its sampling arm.
Bekko remains eligible: hotchpotch's Japanese-published model uses Johns Hopkins mmBERT /
ModernBERT lineage. Publisher nationality alone is insufficient for other derivatives.

Target: **20 sustained retrieval/context requests per second on 8 vCPU / 16 GB RAM**.
Measure cache misses separately from cache hits and model-assisted reads. Current host
screens cannot certify that target. Projection must use CPU seconds per request and a
memory budget, with explicit headroom and separate DB/search/ingestion overhead. Do not
infer throughput from sequential p99 or a closed-loop client's requested arrival rate.

Owned queues after cancellation (check command identity before using any PID):
- Bekko queue 51377 completed both SciFact and all 12 XQuAD languages.
- New capacity/final-regression/LoCoMo queue 1908 waits on Bekko 51377.
- New OCR queue 1909 waits on 1908. OCR image build finished successfully.
- Old waiting queues 88054, 74863, 72163 were terminated in dependency order.

Production OCR now explicitly selects local CPU Tesseract with per-page script detection,
for both PDF and image input. The deployment image installs all Tesseract language/script
packs. RapidOCR/PaddleOCR are not selected. Model-free unit tests pass; the real 12-language
PNG/scanned-PDF screen remains queued. Upstream OSD can fall back to English on uncertain
script detection; coverage must be measured, not assumed from installed packs.

Agent VK automatic activation is now implemented and under regression: default auto mode
requires the acting agent's credential (or an operator key), discovers recognized text
model families through authenticated `/models`, caches by owner/revision, and permits only
bounded ingestion/default uses. Read calls retain their independent explicit permission.
False is still a deployment-wide prohibition. No paid model evaluation is authorized.

## Acceptance and measurement

The target is multilingual retrieval, ingestion, query understanding and grounding with
CPU inference and improved English LoCoMo/SciFact quality. No finite benchmark certifies
every language or document. Report tested languages separately from model-card coverage.
Never substitute source recall, fixture success or a vendor leaderboard for answer accuracy.

Baseline artifacts are the completed September 27 LoCoMo source and full SciFact screens:

- Current English Granite source R@50 0.7741966, complete@50 0.7096354;
  observed sequential context p99 327.78 ms, not a production SLO.
- E5 source R@50 0.7969606, complete@50 0.7317708, p99 489.84 ms.
- Full SciFact, 5,183 documents / 300 queries: English Granite hybrid RRF k=1
  nDCG@10 0.7409, recall@10 0.8912. E5 0.7055 / 0.8482.
- XQuAD paragraph retrieval, 12 languages: measured in the earlier multilingual report.

Models and routing policies must be selected on distinct development data or predeclared
hypotheses. Do not repeatedly tune against the LoCoMo/SciFact test labels. Keep all failed
arms and category regressions. Fixed corpus/order, source IDs and retrieval depths remain
comparable. Accuracy and latency results need model, dataset and source fingerprints.

## Sequence

1. Audit English-only gates: sentence/word parsing, chunk budgets, citations, negation,
   entity extraction, time expressions, query routing, cached context and OCR.
2. Benchmark compact multilingual encoders through the existing CPU adapter. Start with
   already-downloaded Granite 97M multilingual; compare newer compact candidates against
   the existing E5/Granite results. Do not mix vector spaces or silently reindex production.
3. Add CPU ONNX NLI so multilingual verification does not require Torch. Evaluate real
   weights on English golden claims and public multilingual NLI, plus cross-language
   citations. A missing explicitly configured ONNX model must fail instead of silently
   substituting a lexical scorer.
4. Evaluate a small local generative model for bounded ingestion extraction on source-backed
   fixtures, including secondary facts, negation, temporal claims and multilingual spans.
   Compare against native retention and extraction. Keep generation independently controlled
   from retrieval and maintain the existing Bifrost/credential/no-MCP boundary.
5. Address remaining query/extraction limitations with measured, reusable components,
   rather than claiming that changing the embedding alone makes the product multilingual.
6. Run full English LoCoMo source retrieval, SciFact and multilingual retrieval after model
   selection; test false merges, abstention, citations, source deletion, scope isolation and
   context budgets. Real-model grounding is a separate timing path from ordinary recall.
7. Record reproducible bootstrap/reindex instructions and the exact language/task coverage.
   Promote defaults only with the measurement evidence, and disclose unresolved regressions.

## Research recorded this pass

- [Granite 97M multilingual R2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2):
  full SciFact CPU run completed: hybrid nDCG@10 **0.7153**, recall@10 **0.8462**.
  It regresses the English baseline and is not promoted. Query encoder p99 was 62.43 ms;
  that is component timing on a shared host, not the context endpoint SLO.
- [mDeBERTa multilingual NLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7):
  revision `b5113eb38ab63efdd7f280f8c144ea8b13f978ce`, upstream quantized ONNX graph
  338,679,133 bytes. Model card includes cross-language premise/hypothesis training;
  published quality varies by language. Downloaded locally, not yet promoted.
- [Bekko a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m): revision
  `c721113d59a1d91b447450324f51c4b3332c924a`; compact multilingual candidate with
  384-dimensional mean pooling and no prefixes. Publisher recommends OpenVINO for native
  CPU over ONNX. That is a research hypothesis, not a measured speed claim on our host.
  Full SciFact ONNX screen completed: **0.7256 nDCG@10 / 0.8626 recall@10**, query encoder
  p50 **8.14 ms**, p99 **30.88 ms**. It is faster than the tested multilingual Granite on
  this host, but remains below the English Granite quality baseline. Full XQuAD finished:
  14,280 queries across 12 languages, mean same-language dense paragraph R@10 **0.988305**;
  cross-language **0.972549** (includes English control). Dense+BM25 means were **0.985504**
  and **0.973599** respectively. Per-language warm query encoder p99 ranged **28.53–59.95 ms**.
  These are retrieval component results, not generated-answer accuracy or endpoint timings.
- [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): local generative extraction
  candidate, Apache 2.0. Downloaded the public Unsloth Q8_0 GGUF at revision
  `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`. Historical failed screen: 0/24 exact source-unit sets; now excluded. The CPU llama-server
  image is pinned to `sha256:00bd6c289c590e576948cb3639b8195e4d3e6dd6ba1d761f0e2b871fcd4df24f`.
  Its agent/tool/MCP features must remain disabled; only the model directory is mounted.

A predeclared equal-weight RRF k=1 combination of the existing Granite dense, E5 dense
and BM25 candidate lists scored **0.7529 nDCG@10 / 0.8922 recall@10** on all 300 SciFact
queries (`scifact_granite_e5_fusion_screen.json`). This is offline list fusion, with no
runtime implementation or latency result. The small recall difference is not evidence of
answer-accuracy improvement. Do not promote a dual-encoder deployment from this alone.
Paired bootstrap (10,000 query resamples, fixed seed) gives nDCG delta 0.0120 with 95%
interval **[-0.00256, 0.02676]**; recall delta 0.0010 with **[-0.01667, 0.01933]**.
Both include zero. The input hashes and win/loss counts are in
`benchmark/results/scifact_fusion_paired_screen.json`; the baseline metrics were reproduced
before comparison. The reusable input preparation and comparison scripts must be preserved.

The old `FREEZE-multilingual.md` predates implemented E5 prefixes and newer measurements.
Its rejected-candidate claims and small-sample promotion thresholds must not override the
newer full-corpus evidence or this explicit user goal.

## Changes and tests so far

Full regression before the later semantic-graph, source-selection, short-claim and structural
chunk edits: **1,496 passed, 19 skipped, 12 deselected** in
641.07 seconds. Repository-wide Ruff passed. Real-model, Bifrost and Docker markers were
excluded from this suite; the model experiments are separate. Exact source hashes, JUnit
identity and completed benchmark artifact hashes are recorded in
`benchmark/results/multilingual_validation_20260927.json`. No default encoder was promoted.

- Grounding now uses shared Unicode tokenization/sentence boundaries. Trained NLI can judge
  evidence without token overlap, including cross-language citations; unresolved citations
  and blanked unverified generated sources still fail. Uncited premise work remains bounded.
- Initial grounding regression: **36 passed** (routing fixtures, not model-quality scores).
- CPU ONNX NLI adapter added with shared batched scoring orchestration, explicit label mapping,
  dynamic pair padding, stable softmax and local-only model loading. Model remains a challenger.
- Multilingual chunks now split unspaced long input and account for join spaces and overlap.
  The estimate conservatively budgets non-ASCII UTF-8 bytes; it is not a tokenizer count.
  Context cache fingerprints changed with packing semantics. Oversized tables/code and
  contextual headers still need separate tokenizer-limit validation.
- Graph lookup preserves NFC entity names, Unicode quoted entities and bounded CJK n-grams.
  This repairs lookup; it does not replace English-only typed-relation extraction or routing.
- Focused combined grounding/chunking/context regressions: **86 passed**. Graph, ingestion,
  sparse, document-benchmark and local-extraction harness checks: **54 passed**. Additional
  separator/overlap plus architecture/complexity checks: **43 passed, 1 skipped**. Counts
  overlap; they must not be summed into a unique-test total.
- `benchmark.multilingual_nli` uses a label-balanced, hash-selected XNLI screen: 90 aligned
  examples, 15 same-language and 14 cross-language settings, 2,610 pairs. Selection and
  source hashes are recorded before inference. English golden grounding is scored separately.
  The quantized mDeBERTa run completed: English **62/90 (68.89%)**, same-language scores
  range **53.33–68.89%**, and English golden grounding **35/40**. It is not promoted.
  The full-precision diagnostic on the identical 90 English pairs scored **77/90 (85.56%)**,
  versus quantized **62/90 (68.89%)**. Reject this quantized graph/runtime combination;
  the entire model family cannot be rejected from its result. FP32 golden grounding passed
  **36/40**, with p99 **905.34 ms** (verification endpoint component, not normal recall).
  Full FP32 multilingual evaluation finished: same-language **1,062/1,350 (78.67%)**;
  cross-language **1,013/1,260 (80.40%)**. Same-language results range **72.22–85.56%**.
  These are three-way XNLI classification scores; no default promotion yet.
  `benchmark.prepare_xnli` reproduces the original input byte-for-byte from the pinned
  Parquet file; it checks label ordering and keeps all language pairs aligned.
- Numeric guards now share Unicode decimal normalization, including Arabic decimal/grouping
  separators and full-width digits. Existing comma-as-thousands behavior is preserved;
  ambiguous locale-specific decimal commas remain unsupported. Quantity/native-memory/
  grounding regression checks: **67 passed** (overlaps earlier counts).
- `benchmark.local_narrative` uses the production prompt/source-span selector, tools disabled,
  loopback-only endpoint and no credentials. Its 24 cases across 12 languages are synthetic,
  not native-speaker-reviewed, and supply sentence boundaries. It bypasses Bifrost for the
  local component screen and does not establish production or LoCoMo answer accuracy.

### Later source changes requiring a fresh full-suite manifest

- Unclassified semantic queries can now use the same bounded three-hop graph traversal as
  English multi-hop routes. `RetrievalSettings.semantic_graph` permits a paired ablation.
  Seven non-English integration questions reach a third-hop fact; cross-user and cross-tenant
  tests remain blocked. Mixed CJK/ASCII queries retain whole names such as Acme before the
  bounded n-gram budget is consumed. Focused validation: **42 passed**. This is a controlled
  mechanism test, not a LoCoMo improvement measurement.
- The broader guard suite exposed graph document evidence in memory-only requests. A shared
  source-selection filter now covers exact, search, graph/expansion and derived-source paths.
  Summaries belong to documents; source-linked relations accompany the selected source kind.
  Relations without lineage only accompany unrestricted source queries. Regression checks,
  including grounding and memory retention: **119 passed, 1 skipped**.
- Removed English word-count floors that discarded short factual statements before retention
  or grounding. Known English acknowledgements share one exclusion; unknown short statements
  remain source-backed observations and reach verification. Short-retention tests: **133 passed**
  (overlapping suite counts). This changes the write-path corpus: final LoCoMo comparisons need
  fresh isolated ingestion, not reuse of the old indexed corpus.
- Tables/code no longer bypass the chunk budget by up to four times. A shared linear packer
  handles oversized rows and code blocks; bounded table headers repeat, oversized headers
  remain source-preserving fragments with node lineage. Removed the two overflow-enabling
  frozen flags rather than retaining inert controls. Tiny merges include separator cost.
  Document parsing/chunking/fidelity checks: **39 passed, 1 skipped**. Model-tokenizer input
  limits still need broader validation; the packing estimate alone does not certify that
  an encoder never truncates.
- Actual Bekko a25m tokenizer audit found **48/96** synthetic chunks over 512 tokens when
  document titles were long (maximum **1,567**). Bounded the deterministic indexing header
  to 96 planning tokens; full title/section metadata and source bodies are preserved.
  Repeating the identical 12-language, short/long-title stress cases produced **0/96**
  over-limit chunks (maximum **499**). Artifacts: `multilingual_input_budget_audit.json`
  and `multilingual_input_budget_bounded.json`; reproduction `.bench_data/input-budget-audit.py`.
  Paragraphs, tables/code, LLM context insertion and local-model harness retest:
  **49 passed**. The LLM context fixture now uses a table that actually fits its 100-token
  budget; oversized table behavior is covered separately. Generated situating context and
  arbitrary model tokenization remain outside this deterministic-header guarantee.

`multilingual_validation_20260927.json` predates these edits and must not be presented as
their full-suite result. Refresh it only after a new complete regression run.

## Isolation / resumption

Use `.bench_data/test-env.sh` before pytest. Tests can auto-migrate even when labelled unit.
Only the owned databases and Qdrant 16333/16334 / Redis 16379 may be used. The current queue
PIDs are recorded at the top of this document; older queues are completed or cancelled.
Verify command identity before signalling a PID. One real model benchmark runs at a time.

| Component | Role | Current status |
| --- | --- | --- |
| English Granite | Dense retrieval | Existing default; not multilingual |
| Unicode BM25 | Lexical retrieval | Default; cross-script tokenization tested, no translation |
| Bekko a8m | Multilingual CPU dense candidate | Full SciFact/XQuAD done; new LoCoMo arm queued |
| Bekko a25m | Larger candidate | Full SciFact done, XQuAD running; no default promotion |
| E5 / multilingual Granite | Earlier dense challengers | Completed; retain English regressions |
| mDeBERTa FP32 | Multilingual entailment | XNLI screen done; not promoted; int8 failed |
| Qwen | Historical local generative experiment | Excluded; no retry or production integration |
| Docling + Tesseract | Structure parsing and OCR | CPU OCR wired; real multilingual OCR queued |
| Neural reranker | Candidate reordering | Off; no demonstrated multilingual quality/latency win |

Remaining semantic gaps include typed relations, relative dates and multilingual
negation/coreference. Unicode preservation does not establish those capabilities.
No new LoCoMo answer-accuracy score has been produced.

Host runs use ONNX Runtime 1.19.2 and tokenizers 0.22.2 from the isolated benchmark
environment. Deployment requires a newer runtime, so host screens alone do not certify
production compatibility or latency. Historical validation manifests predate later edits.

Current CPU sequence: finish a25m XQuAD; run two 20-RPS encoder capacity screens; run the
full non-model regression; run fresh Granite graph-off / graph-on and Bekko **a8m** graph-on
LoCoMo source-retrieval arms; run the real multilingual OCR screen. The LoCoMo script refuses
existing arm databases. Expected new databases are `memory_hi_ml_granite_off_20260927`,
`memory_hi_ml_granite_on_20260927` and `memory_hi_ml_bekko8_on_20260927`.
Outputs: `benchmark/results/locomo_multilingual_current_{granite_off,granite_on,bekko8_on}.json`.
No baseline corpus is migrated/replaced. Graph/derived evidence consumes the same fixed-depth
positions. These are source-coverage measurements, not reader-answer accuracy.

Logs: `.bench_data/bekko-a25m-queue.log`, `.bench_data/current-locomo-queue.log`,
`.bench_data/ocr-queue.log`. New regression JUnit: `.bench_data/multilingual-final-regression.xml`.
Avoid source/fixture edits once the final regression begins so following arms use that snapshot.

### Completed local generative investigation (historical, model now excluded)

The first host health probe failed because Docker Desktop's isolated network could not be
reached through a host-published port. A separate client sharing the server network namespace
resolved transport; 24/24 responses parsed and owned containers/network were cleaned up.
The investigate skill session is closed. No application transport workaround was added.

Quality failed independently: **0/24 exact source-unit sets**, with out-of-range indices
rejected by the source validator. Request p50 14.83 s / p99 38.39 s on the shared host.
The preserved artifact is `local_narrative_qwen35_08b.json`. The production narrative schema
now bounds indices to the supplied message, and source-payload/schema code is shared with
the generic local harness. This is output validation, not a measured accuracy improvement.
The Qwen bounded-schema/sampling retry was cancelled; its executable sampling option removed.

### Other completed screens

Fixed title-term weight two slightly raised nDCG but reduced Granite/Bekko recall; not
promoted. `scifact_title_weight_screen.json` records it. This is weighting before BM25
saturation, not full field normalization: [BM25F discussion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sp0237-svore.pdf).
Harrier 270M was researched but not downloaded or tested. Its publisher name does not
establish eligible backbone provenance. Do not select it without lineage verification.

### September 28: eligible challenger and capacity harness

Bekko a25m full SciFact completed: hybrid k=1 nDCG@10 **0.7213**, recall@10
**0.8546** (5,183 documents / 300 queries), below Bekko a8m and English Granite.
Artifact: `benchmark/results/scifact_dense_bekko_a25m.json`. Query encoder p99
384.49 ms is a shared-host observation; Docker build and other work overlapped parts
of the run, so it is not a controlled CPU comparison. No default promotion.

`benchmark.cpu_capacity` adds fixed-arrival-rate encoder screening: bounded pending
requests, scheduled-arrival latency, deadline/error/overload accounting, process CPU
seconds, peak process RSS, and a qualified 8-core projection with 30% headroom.
Seven focused tests pass; Ruff and Pyright pass. It is explicitly an encoder screen,
not an HTTP/context capacity claim. Two 1,200-request, 20-RPS arms (a8m/a25m) are
queued before the full regression.

OCR image preflight: Pillow RAQM available and 275 Noto font files. Tesseract reports
161 entries; package names must be checked against Docling's script mapping before
claiming all script packs are usable. No OCR quality measurement yet.

The predeclared new LoCoMo challenger is now **Bekko a8m**, alongside Granite graph-off
and graph-on controls. The choice was made before seeing its LoCoMo results: it beat a25m
on the completed full SciFact screen and is smaller/faster. The a25m multilingual screen
continues as evidence; it is not the selected LoCoMo arm. Earlier queued a25m LoCoMo
descriptions are superseded. No arm databases have been created yet.

Removed the unrun Qwen-specific sampling option from the local generative benchmark;
the generic greedy local harness remains reusable for eligible models. Historical failed
Qwen output remains retained. Latest focused capacity/local-harness/OCR tests: **24 passed**. Ruff passes across src/tests/benchmark/SDK; Pyright passes for the new capacity and revised local harness.

The OCR dependency inspection confirmed script packs are installed at the tessdata root
(e.g. Arabic, Devanagari, HanS, Hangul); they are not prefixed with `script/`. The zero
`script/` count in the initial preflight was a naming assumption, not missing packages.

Heron is published/trained by IBM Research Switzerland. Its paper identifies RT-DETRv2
and pretrained ResNet backbones; it does not name the exact initialization checkpoint.
Architecture origin and trained-model provenance are different facts. Exact upstream
backbone lineage remains to be pinned before a strict all-components origin guarantee.
Source: https://arxiv.org/html/2509.11720v1 .

### Automatic VK activation, September 28

Settings default to `enabled=auto`, `model=auto`, `fast_model=auto`. An omitted use list
selects contextual extraction, reflection, briefs and query expansion; explicit `uses=[]`
disables all. Explicit `enabled=true` retains the existing required use-list contract.
With no key, no gateway request is sent by the auto adapter. Auto mode does not construct
a Hindsight client or probe the gateway with an unowned startup credential. Agent and
operator keys remain separate; revoked agent credentials cannot select the operator key.

Model discovery uses the same owner key and MCP denial headers as completion. Its cache
is bounded to 128 owner/revision entries for five minutes, at most 32 in-flight lookups,
and a five-second GET timeout. Concurrent requests share discovery, with cleanup after
cancelled callers. Only recognized original GPT/Claude/Gemini families are automatically
selected; opaque aliases and excluded model families are not guessed. Compact models are
preferred for extraction/expansion, full models for synthesis; this deterministic policy
is not a measured best-model claim. Gateway credit/quota availability is not inferred.

Validation: **74 focused tests passed, one live test deselected**, then the complete native
ingestion flow passed with a registered key and no configured model/uses. That integration
also proves another owner and a revoked key send no further gateway calls. These feature
HTTP responses are mocked; they establish activation/isolation, not answer-quality gains.
Pyright reports zero errors and three unavailable optional-import warnings on the host.

Test-guard incident: the first focused command accidentally included the existing live
Bifrost marker. Changing the default to the truthy string `auto` exposed a guard that only
checked truthiness. Its existing unauthenticated preflight attempted a POST with model
`auto`; the actual adapter completion was denied before sending because no key existed.
The probe response was not retained, so no claim is made about upstream execution/cost.
The live guard now requires explicit true, a key and a concrete model. The redundant raw
HTTP preflight has been removed. All subsequent commands exclude live/model/Docker markers.

Source edits are complete for this activation slice; refresh the full validation manifest
only after the queued complete suite succeeds. Existing deployment configurations with
explicit false remain disabled; `.env.example` and README describe the new auto default.

Follow-up validation: **76 passed, one skipped, one live test deselected** across catalog,
settings, Bifrost contract, wiring, read policy, architecture and source-benchmark tests.
The separate real-SQL/native-ingestion auto-VK integration passed. The local gateway on
8090 has zero provider/key/environment-key/routing-rule rows in its read-only configuration
DB, and an empty request-log table; no upstream execution is recorded for the guard incident.
No further live requests were sent. The browser research retry encountered a 404, so it
established nothing about upstream `/models` filtering. Auto selection uses the key-bound
catalog response and still relies on the gateway to enforce model permissions/budgets.

### Completed measurements after automatic-mode changes

Bekko a25m XQuAD completed all 14,280 questions / 12 languages. Mean dense R@10 is
0.990546 same-language and 0.983964 cross-language (including the English control).
Hybrid means are 0.987255 and 0.983894. This modestly improves on a8m multilingual
retrieval, while a25m regressed full English SciFact and required more CPU work.

Both fixed-arrival capacity screens completed 1,200/1,200 encoder requests at 20 RPS,
zero errors/deadlines/overload, across interleaved queries from all 12 XQuAD languages:

| Encoder | p50 ms | p99 ms | CPU seconds/request | peak process RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Bekko a8m | 10.68 | 108.84 | 0.013445 | 753.53 |
| Bekko a25m | 39.65 | 1046.88 | 0.042415 | 796.89 |

Artifacts: `cpu_capacity_bekko-a8m.json`, `cpu_capacity_bekko-a25m.json`. At 20 RPS,
encoder CPU demand alone is about 0.27 or 0.85 core-equivalents on this host. The 8-vCPU
projection reserves 30% headroom, leaving 5.33 or 4.75 core-equivalents for other work.
These are necessary CPU-budget estimates, not endpoint capacity or p99 guarantees.
The computed encoder-only theoretical ceilings must not be reported as service RPS;
they omit serialization, queues, CPU differences, database, search, authorization and OCR.
Peak RSS is per process. Multiworker replication and full deployment RAM are not measured.

A fixed cached-list fusion hypothesis combined English Granite dense + Bekko a8m dense +
BM25 at equal-weight RRF k=1. Full SciFact nDCG@10 improved **0.7409 -> 0.7570**;
recall@10 was **0.8912 -> 0.8926**. Nominal paired bootstrap nDCG delta interval is
[0.00141, 0.03158]; recall interval includes zero. This is one exploratory test-set screen
within a larger model-selection programme, not multiple-comparison-adjusted or held-out
promotion evidence. It is not answer accuracy, production RAG, or runtime latency.

Artifact: `scifact_granite_bekko_fusion_screen.json`; reproducible script:
`.bench_data/screen-granite-bekko-fusion.py`. No inference/provider calls were required.
A useful next implementation hypothesis is independent named dense vectors with one
Qdrant fusion query and identical ACL filters, retaining Bekko as the multilingual base.
Concatenating embeddings is NOT equivalent to this ranking-list fusion and must not be
presented as reproducing the measured gain. English/multilingual ensemble regressions and
application latency still require measurement before selecting a default.

The final regression started at 00:51:46 local time. Source/fixture edits are now frozen
through its following LoCoMo arms; documentation/result analysis can continue.

### Follow-up validation and real Qdrant fusion screen (28 September)

The initial regression stopped at 405 passed, 18 skipped and one failed assertion:
`test_decisions_are_thread_scoped_and_shared_in_thread`. Native ingestion now retains the
short setup source `kick-off`, so the old assertion allowing only the decision was stale.
The revised test requires the decision, verifies every returned memory belongs to the
same thread with THREAD visibility, and still requires an unrelated principal to get
zero candidates. Its focused run with multilingual memory tests passed all 14 cases.

A resumed suite covered 1,568 collected `tests/` nodes across two segments: 1,550 passed,
17 skipped and one failed static payload-contract assertion. That AST scanner interpreted
`memory_id` on locally produced graph facts as a Qdrant payload field. The graph stage
creates that lineage; the indexer does not write it. The test now names that graph-only
exception and checks it has a graph producer and no search-index producer/projection.
All 38 focused search-wire/retrieval-stage tests pass. Initial failure artifacts remain;
`multilingual_suite_coverage_20260928.json` correctly remains incomplete. Its old collection
accounting excludes SDK nodes, so it is not a full-suite success report.

A new uninterrupted full suite, including the default configured SDK test paths, is queued
in `.bench_data/run-current-locomo-queued.py` (PID 27970 at launch). It waits for the owned
OCR screen (PID 23152), then runs Ruff plus pytest with live/model/Docker markers excluded.
Output: `.bench_data/multilingual-final-complete.log` and `.xml`. Only on success will it
start the three fresh database LoCoMo arms: Granite graph off, Granite graph on, Bekko a8m
graph on. Source is frozen during those runs. No benchmark DB has been repurposed.

The actual isolated Qdrant fusion screen completed on all 5,183 SciFact documents and
300 queries using cached embeddings. English Granite + BM25 scored 0.7409 nDCG@10 /
0.8912 recall@10; adding a separately ranked Bekko a8m dense vector scored **0.7557 /
0.8926**. Search-only p50/p99 changed from **16.10/54.77 ms** to **20.14/97.89 ms**.
Both inner prefetches and the outer query used production ACL filters. High-scoring
other-tenant and sibling-visibility sentinel records were excluded in both arms.
Artifact: `scifact_qdrant_ensemble_screen.json`; script:
`.bench_data/screen-qdrant-ensemble.py`. The dedicated temporary Qdrant container and
collection were removed after the screen; no shared collection was modified.

This is an exploratory component result using public test labels during model selection,
not a held-out gain, application deployment, answer accuracy, or endpoint load test.
Independent named-vector fusion is still not implemented in the application. In particular,
these numbers must not be attributed to the existing production default or a concatenated
embedding. Multilingual ensemble quality remains unmeasured. No paid calls were made.

Queue correction: PID 27970 was stopped while it was only waiting, before starting any
tests or LoCoMo arm. The current owned full-test/LoCoMo queue is PID **32270**. The first
real OCR screen used an older local base image with GUI OpenCV still installed; all 24
pages reported missing `libGL.so.1`. Its artifact is preserved as
`multilingual_ocr_tesseract_image_failure.json` and is a failed environment run, not OCR
quality evidence. Production `deploy/Dockerfile` already replaces GUI OpenCV with matching
headless OpenCV. The benchmark image now mirrors that step, successfully importing
`cv2 5.0.0`, under tag `memory-ocr-headless-screen:20260928`.

Owned follow-up queue PID **33979** waits for PID 32270, reruns OCR with the corrected image,
then runs `.bench_data/multilingual-fusion-screen.py`: 100 fixed evenly spaced XQuAD IDs
per language, all 12 languages, same-language and cross-language retrieval, equal RRF k=1
and depth 50. This evaluates the same fusion setting as SciFact/production rather than
mixing in the earlier multilingual k=60 hybrid results. No fitted weights or language
routing, no remote model requests. It caches per-model vectors for reproducibility.
Output: `multilingual_granite_bekko_fusion_screen.json`; not a promoted application feature.
