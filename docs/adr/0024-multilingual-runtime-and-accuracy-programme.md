# ADR 0024: The multilingual runtime and the accuracy programme (Phase 7: M2 + M4)

Date: 2026-09-28. Status: accepted. Plan of record: `docs/PLAN-MULTILINGUAL-PLATFORM-2026-09-28.md`
(D2, D6, D8, D9, section 4 rows M2 and M4). Spec: the gstack archive
`~/.gstack/projects/amitmohapatra-agent-memory-service/specs/20260928-161930-29573-phase-7-*.md`.
Results: `docs/PHASE7-RESULTS-2026-09-28.md`.

## Context

The CPU screens of 27-28 September selected the models (`docs/CPU-MULTILINGUAL-DECISION-20260928.md`)
but the runtime still embedded every query with an English encoder and verified claims with
an English NLI head. The measured numbers behind this record:

- XQuAD paragraph R@10 over 12 languages: English Granite 65.96%, Bekko a8m 98.83%.
- SciFact (5,183 documents / 300 queries): Granite + BM25 0.7409 nDCG@10 / 0.8912 R@10;
  Granite + Bekko + BM25 fused by RRF 0.7557 / 0.8926.
- mDeBERTa FP32 on the English golden grounding set: 36/40, the same as the English DeBERTa;
  its int8 graph lost eleven points and was rejected.
- Full LoCoMo: 72.92% strict / 81.17% on Mem0's ruler; 226 of 416 misses are fragmentation;
  every per-arm rank dump on disk reads `{"fusion": N}` because the store fuses server-side.

## Decisions

1. **Two dense spaces, one collection.** A collection carries named vectors `dense_en`
   (Granite small English r2) and `dense_ml` (Bekko a8m ONNX) beside `bm25`; every record
   carries both. The names are a closed vocabulary (`ports.search.VectorName`), and the
   collection name carries every space's fingerprint (`DenseSpaces.fingerprint()`), so a
   changed encoder mints a new collection instead of mixing spaces. `tools/reindex.py`
   moves an existing tenant; `--prune` retires the single-vector generation.
2. **The query's script decides the prefetch.** `domain/script.py` reads the dominant
   Unicode script from the letters' names (deterministic, no model, no new dependency).
   `dense_en` is searched for Latin-script queries only; `dense_ml` for every query; BM25
   always. A Cyrillic or Thai question therefore costs one encode and two prefetches. Every
   record is tagged with its script (indexed payload `script`).
3. **Both encoders run concurrently.** Each adapter owns a single-thread `SerialRunner`;
   `DenseSpaces.embed_query` gathers the spaces the script calls for, and the indexer gathers
   the spaces per batch. Two ONNX encoders at two intra-op threads each is four threads per
   worker; the thread budget in `DenseModel.threads` is arithmetic, not a measurement.
4. **The store fuses, weighted when told to.** `search_hybrid` sends one prefetch per arm and
   `Rrf(k, weights)` only when a weight differs from 1.0 - the historical wire query is
   unchanged otherwise. Weights are a frozen constant (`RetrievalSettings.hybrid_weights`),
   fitted offline from per-arm rank dumps (`benchmark/fit_rrf_weights.py`), never a setting.
5. **Entity routing as one more RRF list.** Memories carry their subject's name and extracted
   entities as an indexed `entities` payload; when `RetrievalSettings.entity_prefetch` is on,
   the query's entities anchor an extra prefetch on the primary space (`AnchoredPrefetch`).
   Off by default until its judged arm shows strict multi-hop does not lose by it.
6. **One NLI, one runtime.** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
   FP32 ONNX graph, revision `b5113eb3…`, through the ONNX session this repository owns.
   The English DeBERTa adapter (`TransformersNLI`) and its degrade-to-lexical branch are
   deleted: a frozen graph that cannot load is a startup error, not a quiet
   `representative: false`.
7. **Relative dates are resolved at ingest, and only where they can be resolved exactly.**
   `modules/memory/temporal.py` uses `dateparser` (BSD-3) with the observation's
   `occurred_at` as `RELATIVE_BASE` and only its relative-time parser, keeps only phrases the
   language's own cue words announce, and stores the pairs in
   `system_metadata["dated_mentions"]`; the renderer prints `three days ago (2023-05-05)`
   after the text. Nothing generated is stored: the phrase is the memory's words, the date is
   arithmetic on the timestamp.
   Named and counted offsets are covered (`yesterday`, `tomorrow`, `three days ago`,
   `last week`, `hace 2 días`, `вчера`, `昨天`) in all twelve languages. Weekday phrases
   (`last Tuesday`) are **not**: reaching them needs the `absolute-time` parser, which on this
   base resolves the weekday correctly but also reads the word "we" as a Wednesday and
   annotates the absolute dates this step exists to leave alone. A phrase resolved to the
   wrong day corrupts an answer silently, so the parser stays narrow - the measurement
   protocol's rule applied to a date: unmeasured, never wrong. Cost on this host: 0.84 ms
   when the cue pre-check rejects a text, 6-27 ms when it parses, on the ingest path only.
8. **Read-side aggregates and the reader protocol.** The judged-arm reader (`benchmark/locomo.py`)
   gets one bounded re-ask when it declines while the context names the question's entities
   (`--reask`); the rendered bundle is unchanged for callers. Write-path consolidation stays
   off (D1).
9. **The budget is read from the gateway that enforces it.** `benchmark/budget.py` reads the
   governance key's `current_usage` before and after every judged arm, records a ledger, and
   refuses any arm whose projection would exceed the phase cap ($6) or leave the key under
   its floor ($2). The key's id is configuration; its token never passes through the tool.
10. **One corpus per database, many arms.** `benchmark/corpus.py` keeps each LoCoMo
    conversation in its own tenant and a ledger of what was ingested (dataset hash, index
    fingerprint, ingestion settings, the observation-id -> turn-id map), so arms that change
    only the query side or the render share the ingested corpus; an arm that changes the
    write path (LLM-assisted extraction) gets its own database.
11. **A harness may only reset a store it owns, and the default is such a store.** A
    benchmark clears the whole schema between runs, so `benchmark/common.py:reset_store`
    refuses to TRUNCATE any database not named for a benchmark (`memory_bench*`,
    `memory_hi_*`, `p7_*`) - checked at the TRUNCATE, the one place every harness passes
    through, because the source harness carried that guard and the judged harness did not
    while the benchmark default URL named the shared `memory` database. `BENCH_QDRANT_URL`
    and `BENCH_QDRANT_GRPC_PORT` default to the isolated store on 16333/16334 for the same
    reason: the dev store on 6333 serves a running API, and a corpus ingested into it would
    pollute those reads and then be measured against the pollution.

## Consequences

- Every existing collection is superseded: the vector `dense` no longer exists and a
  deployment must run `make reindex-image REINDEX_ARGS="--drop"` then `--prune`. Rolling
  back is a revert; the previous collections keep their names.
- `dateparser` joins the core dependencies (with `regex`, `pytz`, `tzlocal`). The benchmark
  container installs it at start-up (`BENCH_PRELUDE`) because the image predates it.
- The hermetic suite stands in one hash space named `dense_ml`; the script-pruning path is
  covered by unit tests with spy encoders, and the real encoders by the container gates.
- The quality gates (`test_architecture.py`, `test_complexity_budget.py`) are not raised.
- What was measured, and what was not, is in `docs/PHASE7-RESULTS-2026-09-28.md`; no gate
  threshold in this record was moved to pass.
