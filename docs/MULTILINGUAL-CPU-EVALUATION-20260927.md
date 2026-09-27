# CPU multilingual retrieval evaluation — 2026-09-27

These are XQuAD paragraph-retrieval measurements, **not** generated-answer accuracy, LoCoMo accuracy, or a whole-service latency SLO. Each arm uses the same pinned 1,190 questions per language (14,280 total), exact search over translated paragraphs, and no LLM calls. Dense encoders run on CPU.

| Language | English Granite R@10 | E5-small int8 R@10 | E5 query → English R@10 |
|---|---:|---:|---:|
| ar | 28.91% | 98.32% | 94.29% |
| de | 88.99% | 99.24% | 98.57% |
| el | 43.11% | 98.24% | 95.46% |
| en | 99.50% | 99.92% | 99.92% |
| es | 92.61% | 99.58% | 99.33% |
| hi | 24.45% | 99.41% | 97.98% |
| ro | 86.13% | 98.99% | 98.15% |
| ru | 77.98% | 99.50% | 97.82% |
| th | 26.47% | 99.33% | 95.71% |
| tr | 75.97% | 98.99% | 95.88% |
| vi | 66.72% | 99.33% | 95.63% |
| zh | 80.67% | 98.91% | 94.29% |

English Granite mean dense R@10: 65.96%. E5-small int8: 99.15%. Cross-language queries against English paragraphs: 47.23% versus 96.92%. These are arithmetic means over equally sized language sets.

E5 English R@10 is 99.92%, versus the English Granite baseline at 99.50%. MiniLM loses English recall (96.47%); it is not the selected replacement. E5 hybrid RRF (k=60, depth 50) averages 98.68%, below its dense-only result. Fusion must be evaluated on entity/identifier/document tasks before changing production weights.

No model is promoted from this component benchmark alone. E5’s score distribution needs abstention and duplicate-detection calibration. A fresh LoCoMo source-coverage arm and document RAG evaluation are required; old reader predictions cannot judge new contexts. The 97M multilingual Granite challenger completed all 12 languages: mean dense R@10 96.37%, hybrid 98.15%, and cross-language dense 92.56%. E5 wins the aggregate comparison; Granite is better on Chinese cross-language retrieval (96.72% versus 94.29%), so E5 is not uniformly best in every language.

## Reproducibility and limits

- Dataset: `google-deepmind/xquad`, revision `7d30520c717524000f0d9d2f9c10a069acd9d285`; per-file hashes are included in result artifacts.
- E5: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, QInt8 per-channel/reduced range, 384 dimensions, 512-token cap, `query: ` and `passage: ` prefixes.
- English Granite uses the existing local FP32 ONNX graph. Its historical downloader manifest is not independently sufficient to authenticate the graph; preserve exact local file hashes for comparisons.
- All result rows are stored in `benchmark/results/multilingual_dense_{granite,e5,minilm,granite_multilingual}.json`. The multilingual sparse comparison is in `multilingual_sparse_{baseline,unicode}.json`.
- Query-encoding timings in these exploratory runs encountered other test/process activity. They exclude authorization, network, SQL, search and context construction. Do not present them as controlled application p99.
- This public dataset may overlap training data. It has no unanswerable examples and does not validate multilingual temporal parsing, entity graphs, grounding, or cross-run agent behavior.

## Product fixes covered separately

Unicode tokenization preserves non-Latin scripts and uses character features for unsegmented text. Multilingual source ingestion no longer requires three English-style words. Verbatim statements cannot fuzzy-merge across small negation changes. Integration tests cover 12 languages through actual SQL ingestion, retrieval, authorization and forgetting with hash embeddings; those functional tests are distinct from the real-model quality runs.
