# Codex commit and database handoff — 27 September 2026

This is the current handoff for independent measurement. Historical benchmark reports
are records of their runs, not evidence that the isolated commits below improve accuracy.
No new paid model calls or LoCoMo measurements were made while preparing these stacks.
The original working tree and the evaluator's `tier1-isolated` worktree were preserved.

## Commit boundaries

All code starts at `f555a68`. The retrieval stack has no new migrations and runs on
`0008_drop_entity_alias`:

| Commit | Scope | Dependency / interpretation |
|---|---|---|
| `13c04dd` | D: configurable inner fusion and deterministic returned-score ties | Inner default remains 1. Sorting ties is active behavior. |
| `0ee0b7c` | A: raw source lineage in index payloads, exact reads, context and stable source grouping | Rebuild an isolated index to populate old payloads. No fact extraction changes. |
| `d84a200` | B: graph companions survive a full primary-memory cap | Shared token cap still applies. More bundle items are not better fixed-depth ranking. |
| `66b8fa2` | C: bounded, authorized canonical memory hydration from graph relations | Uses A's `memory_candidate` helper; B allows companions through context packing. |
| `a121dc7` | Additional retrieval program | Provenance dedup, memory depth 100, optional actor/topic and source-turn controls, RAG isolation/filtering/diversity and ONNX spin control. Not an A–D-only arm. |

`codex/abcd-only-20260927` stops at C (`66b8fa2`), before the wider retrieval program.
`codex/retrieval-stack-20260927` stops at `a121dc7`.
Its worktree is `/Users/ricky/usage_data/ams-codex-retrieval`.

`codex/write-path-stack-20260927` adds the write-path program on top of the retrieval
stack: landing, narrative extraction, source-backed derived memories, continuous
reflection, lifecycle/visibility safeguards, related read guards and migrations 0009–0011.
Its worktree is `/Users/ricky/usage_data/ams-codex-write`.
This is a **dependent stack**, not an independently cherry-pickable write-only arm:
read-side lineage and derived-memory checks are required for safe use of its output.
Compare retrieval-only against retrieval-plus-write on independently prepared corpora.
To measure C alone, include A's shared projection dependency explicitly; do not call the
stacked D+A+B+C result an isolated C result.

The full write stack reproduces the frozen original runtime, migration and `.env.example`
files byte for byte. No claim is made that those combined defaults have earned promotion
from the new evaluator. The original tree remains dirty because it also contains historical
reports, raw outputs and the benchmark work owned by the evaluator.

## Intent of hybrid_rrf_k

`hybrid_rrf_k=1` is intentional, not a forgotten switch. It uses one-based ranks and keeps
Qdrant's historical `FusionQuery` request. Non-default values convert to Qdrant's zero-based
constant by adding one. The separate outer `rrf_k=60` controls strategy fusion.
The earlier inner-60 probe did not justify promotion; retain backward compatibility until
an independent, fixed-depth comparison says otherwise. D's tie ordering can still move
cutoff membership, so D as a whole is not guaranteed inert.

## Migration intent and data effects

The three migration identifiers and their ordering are intended for this write stack.
They are committed as executable schema changes, not an instruction to migrate a baseline.

- `0009_memory_dependencies`: derived slots, unique live-slot index, source revision
  dependencies. **Also retracts existing CURRENT BELIEF / ENTITY_SUMMARY and legacy
  reflection records**, increments their revisions and marks them for reindexing.
  Its downgrade removes schema objects; it does not undo those data changes.
- `0010_memory_recent_updates`: partial index on current, nondeleted memory update times.
- `0011_reflection_progress`: durable receipts keyed by tenant and source memory revision.

Read-only inspection during this handoff confirmed `memory_tests`, `memory_bench_conv`,
`memory_bench_docs`, `memory_bench_degen`, `memory_bench_golden`, and local `memory` are at
0011. `memory_bench_accuracy` remains at 0008; `memory_final_accuracy` is at 0010.
These are observed schema versions, not a claim that every database has an unchanged corpus.
No shared database was downgraded, stamped, reset or otherwise modified by this handoff.

The 709 errors are consistent with the session-autouse migration fixture failing before
individual tests execute. They are not 709 distinct product failures. A tree whose latest
revision is 0008 cannot resolve an already-recorded 0011 revision.

## Isolation traps the evaluator must address

1. **Pytest has its own URL.** Set `MEMORY_TEST_DATABASE_URL` to a fresh per-arm database;
   also set `MEMORY_TEST_APP_DB` and `MEMORY_TEST_QUEUE_DB` for tests using those fixtures.
   `BENCH_DB_CONV` and `MEMORY__DATABASE__URL` do not isolate the pytest database.
2. **The Makefile prerequisite ignores the override.** `make bench-locomo` depends on
   `bench-db`, whose loop hard-codes all four shared benchmark names. Merely overriding
   `BENCH_DB_CONV` still migrates those shared databases before starting the arm. Create
   and migrate the intended arm database explicitly from that arm's checkout, then invoke
   the benchmark directly, or use `make -o bench-db ...` only after explicit preparation.
   This handoff does not edit the evaluator's harness or Makefile.
3. **SQL isolation is not vector isolation.** Indexer collection names depend on model
   fingerprints, not SQL database names. Use independent Qdrant instances or collection
   namespaces, and independent tenant/cache state. Another arm's ingestion can overwrite
   records or change sparse IDF even when SQL databases differ. Hold the background corpus
   constant when comparing ranking. Copy/reindex from a verified original corpus rather
   than rerunning stochastic LLM ingestion for every retrieval-only arm.
4. **Code must match the checkout.** Set `PYTHONPATH` to the arm's `src` and root; an editable
   install or stale container image must not silently import the original dirty tree.
5. **Do not stamp around the error.** A revision marker is not a schema/data conversion.
   Keep shared databases intact. Fresh databases at the arm's own head are the clean baseline.

Example test invocation from an isolated checkout (names must be unique for the arm):

```bash
MEMORY_TEST_DATABASE_URL=postgresql+psycopg://memory:memory@localhost:5432/arm_d_tests \
MEMORY_TEST_APP_DB=arm_d_app \
MEMORY_TEST_QUEUE_DB=arm_d_queue \
PYTHONPATH=src:. \
/Users/ricky/usage_data/agent-memory-service/.venv/bin/python -m pytest <selection>
```

## Measurement ownership and caveats

The other AI owns the benchmark correction and independent arms. No changes to its
`ams-tier1` worktree were made. Current `benchmark/locomo.py` counts complete evidence over
all returned memories for `complete_in_candidates`; graph companions bypass primary item
caps. That is bundle coverage, **not complete@50** unless the instrument enforces a depth-50
cut. Report fixed-depth primary ranking and total evidence coverage separately, with
actual item/token counts. Do not credit C with a ranking gain merely because it added items.
The approximate 725 routed questions and four disputed headline numbers are the evaluator's
reported findings; this handoff verified the uncapped code path, not those counts.

No new answer-accuracy claim follows from these commits. The previously completed mini
reader comparison was 71.88% answerable native versus 71.56% with broad reflection,
using the same reader/ruler. It did not establish a reflection accuracy improvement.
Older partial DeepSeek output is not a controlled comparison to that full mini run.

## Artifact handling

Raw `benchmark/results` occupies about 2.1 GB and contains resumable responses, logs,
multiple copies and historical snapshots. It is retained in the original checkout, not
blindly added to a code commit. A separate evaluation archive records harness source,
notes and an artifact hash inventory. That archive is not a corrected measurement branch;
the evaluator still needs to fix the known depth issue before attributing gains.

## Validation of the split

- D: 29 fusion and adapter wire tests passed on fresh schema 0008.
- A: 18 lineage, rendering and memory integration tests passed on schema 0008.
- B: 23 context datapath tests passed on schema 0008.
- C: 22 graph budget and integration tests passed on schema 0008.
- Full retrieval stack: 251 passed, one skip (no local `.env` in the worktree).
- Full write stack: 160 passed, the same `.env` skip; fresh database upgraded to 0011.
- Ruff lint and formatting passed on both stacks. Architecture and complexity budgets
  were included. These are overlapping selections, not additive unique-test totals.
- The full write stack matches all 31 frozen runtime/schema/example-config files byte for
  byte. The split changed commit boundaries, not the final implementation.

The fresh databases used here are `codex_retrieval_split_tests` (0008) and
`codex_write_split_tests` (0011), with separately named app/queue databases configured.
Keep them separate from accuracy benchmark corpora. No LLM calls were made.

## Evaluation archive

`codex/evaluation-archive-20260927`, worktree
`/Users/ricky/usage_data/ams-codex-evaluation`, preserves the benchmark source separately
from both runtime stacks. Its parent write-path commit is `f988c26`.
The source snapshot, exact commit mapping, raw-artifact hash inventory (2,202 files),
and split test logs are in `docs/evidence/codex-stack-2026-09-27/`.
The raw artifacts remain under the original repository's `benchmark/results`.

The archived harness checks initially had 69 passes and one missing-data failure:
a unit test with synthetic conversations still hashed the ignored local LoCoMo file.
That unit test now creates and hashes its own temporary fixture; the benchmark implementation
and the evaluator's depth correction were not changed. Its focused recheck is recorded.
This removes a clean-checkout dependency without pretending the metric audit is finished.
The canonical external retrieval artifact is not regenerated by this handoff; rerun the
external gate against the final measured source when that evaluation is authorized.

There are no new accuracy or latency measurements here. The original `main` working tree
has not been reset, staged wholesale, merged, pushed or deployed. All branch refs are local
in the same Git repository, so the evaluator can use them immediately without a fetch.

The archived unit-test fixture recheck passed all five tests in its module. Committed
log copies trim trailing whitespace only; the original logs remain in
`/Users/ricky/usage_data/ams-codex-split-snapshot-20260927/`.
