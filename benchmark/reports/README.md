# Benchmark reports

Dated write-ups of measurement runs and reviews. They are **evidence**, not user documentation:
each one describes the code as it was on its date, and ADRs, `docs/MEASUREMENTS.md` or code
comments cite them for a specific number or decision. When a report and the current code
disagree, the code (and the ADR that changed it) is right. The raw result files are in
[`../results/`](../results).

The user documentation starts at [`docs/README.md`](../../docs/README.md).

| Report | Date | What it is | Cited by |
|---|---|---|---|
| [AUDIT-2026-09-23](AUDIT-2026-09-23.md) | 2026-09-23 | 50 confirmed findings from a code audit | ADR 0022 |
| [LATENCY-LAYERS-2026-09](LATENCY-LAYERS-2026-09.md) | 2026-09 | where the latency and throughput go, layer by layer | ADR 0022 |
| [ROADMAP-2026-09](ROADMAP-2026-09.md) | 2026-09 | the September roadmap; Phase 1 removed the separate model tier | ADR 0019, `challengers.txt` |
| [ENTITY-TOPIC-RETRIEVAL-2026-09-26](ENTITY-TOPIC-RETRIEVAL-2026-09-26.md) | 2026-09-26 | the actor/topic memory search experiment | `docs/MEASUREMENTS.md` |
| [CPU-MULTILINGUAL-DECISION-20260928](CPU-MULTILINGUAL-DECISION-20260928.md) | 2026-09-28 | the CPU multilingual encoder selection | ADR 0024, `config/constants.py` |
| [PLAN-MULTILINGUAL-PLATFORM-2026-09-28](PLAN-MULTILINGUAL-PLATFORM-2026-09-28.md) | 2026-09-28 | the plan of record for Phase 7 (M2 + M4) | ADR 0024, `runtime_retrieval.py` |
| [PHASE7-RESULTS-2026-09-28](PHASE7-RESULTS-2026-09-28.md) | 2026-09-28 | Phase 7: what was measured, and what was not | ADR 0024, `docs/MULTILINGUAL-RUNTIME.md`, `memory/connections.py` |
| [PR-CHECKPOINT-20260928](PR-CHECKPOINT-20260928.md) | 2026-09-28 | the validation state of the preserved checkpoint scripts | `experiments/checkpoint_20260928/` |
| [PHASE9-RESULTS-2026-09-29](PHASE9-RESULTS-2026-09-29.md) | 2026-09-29 | Phase 9 results (offline reranking and fusion) | `config/constants.py`, `cross_encoder.py` |
| [PHASE10-RESULTS-2026-10-01](PHASE10-RESULTS-2026-10-01.md) | 2026-10-01 | Phase 10: a turn indexed with the turn it answers | the evidence behind ADR 0027 (kept to keep the PHASE series whole) |
| [PHASE11-RESULTS-2026-10-01](PHASE11-RESULTS-2026-10-01.md) | 2026-10-01 | Phase 11: late interaction, two keys, a learned memory ranking | ADR 0025 |
| [DATA_PLACEMENT_REVIEW](DATA_PLACEMENT_REVIEW.md) | 2026-10-01 | which store holds what, what it costs, what drifts | a dated review, kept as evidence |
| [FINAL_REPORT](FINAL_REPORT.md) | 2026-09-15 | the M0-M13 per-gate account; says its own numbers are not representative | ADR 0017, `docs/guide/12-testing-and-gates.md` |

The design records that sit beside these, outside the user docs, are in
[`design/`](../../design/README.md): the tool-memory design history and the product decisions document.
