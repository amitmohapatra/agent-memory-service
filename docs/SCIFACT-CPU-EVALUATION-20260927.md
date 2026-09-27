# Full SciFact CPU encoder/fusion screen — September 27, 2026

Both models processed the same complete corpus: **5,183 documents, 300 queries**. There
were **zero generation or judge calls**. Each document is its title plus abstract; vectors
are cached against dataset, model-file, configuration and embedding-source hashes.

This isolates dense retrieval and BM25/RRF fusion. It does not run Docling, document
chunking, scope/ACL checks, graph expansion, parent/neighbor expansion, context assembly
or generated answers. It is not a replacement for the earlier full application RAG score.

| Encoder | Arm | nDCG@10 | Recall@10 |
|---|---|---:|---:|
| English Granite | dense | 0.7429 | 0.8659 |
| English Granite | sparse | 0.6583 | 0.7780 |
| English Granite | hybrid_k1 | 0.7409 | 0.8912 |
| English Granite | hybrid_k60 | 0.7201 | 0.8494 |
| Multilingual E5 int8 | dense | 0.6749 | 0.7966 |
| Multilingual E5 int8 | sparse | 0.6583 | 0.7780 |
| Multilingual E5 int8 | hybrid_k1 | 0.7055 | 0.8482 |
| Multilingual E5 int8 | hybrid_k60 | 0.6942 | 0.8204 |

The `sparse` arm is shared Unicode BM25. `hybrid_k1` matches the dense/sparse RRF rank
constant used by the current search adapter; the application also has other retrieval
arms and later processing that this component screen does not model. `hybrid_k60` is a
separate fixed comparison, not a production change. Each sparse/dense arm contributes
up to 50 candidates. No reranker is used.

## Model decision

The production constant remains English Granite in this patch. In this full screen, E5
regressed hybrid nDCG@10 from **0.7409 to 0.7055** and recall@10 from **89.12% to 84.82%**.
That does not justify replacing the shared encoder across document and conversational paths. E5 is an evaluated
challenger, not a silent replacement of existing vector collections. It improves aggregate
LoCoMo source retrieval and multilingual paragraph retrieval, but has category regressions
on LoCoMo. A replacement additionally needs a reproducible quantized-model bootstrap,
reindexing and full document/abstention calibration. The currently measured E5 graph is
locally quantized with QInt8, per-channel and reduced-range AVX2 settings; merely switching
its name in the frozen catalogue would not reproduce that graph on a clean deployment.
The multilingual NLI/answer path is not certified by these retrieval screens.

No fusion, embedding or reranker default was tuned against these 300 test queries. The
screen records the two fixed RRF constants and two preselected encoder profiles. Public
benchmark data may overlap a model's training data.

## Reproduction and artifacts

Use `python -m benchmark.document_dense --data <scifact.json> --spec <profile.json>
--output <result.json>` on the isolated integration checkout. The input result manifests
contain the exact model profile, graph/tokenizer/config hashes and dataset hash. The session's
profiles are `.bench_data/models/granite-baseline.json` and
`.bench_data/models/multilingual-e5.json`; adapt only their local model paths when reproducing
elsewhere. Cached vectors live in `.bench_data/document-vectors/` and are never reused across
an unmatched manifest.

Artifacts: `benchmark/results/scifact_dense_granite.json`, `scifact_dense_e5.json`, and
`scifact_dense_comparison_20260927.json`. Both encoders ran serially, one real encoder at a
time. Other host tests overlapped portions of the runs, so recorded encoding times do not
establish a controlled speed ratio or application p99.
