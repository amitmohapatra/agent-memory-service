# ADR 0008: Baseline retrieval — hybrid BM25 + dense with store-side isolation

**Status:** accepted · **Date:** 2026-09-15

## Decision
- **One index, two named vectors.** Every chunk is stored once in Qdrant with a `dense`
  (cosine) vector and a `bm25` sparse vector. The sparse side is client-encoded BM25 term
  saturation (k1 = 1.2, b = 0.75, deterministic tokenizer with stop words and light stemming,
  no model download) and Qdrant applies IDF server-side (`Modifier.IDF`). Queries carry
  weight-1 term ids. Hybrid retrieval is a single `query_points` call: two prefetches fused by
  native reciprocal-rank fusion; a client-side RRF exists for stores that cannot fuse.
- **Isolation happens inside the store, never in memory.** Each point carries `tenant_id` and
  `visibility_keys` (ADR 0005). Every search applies `tenant_id == t AND visibility_keys ∩
  authorized_keys ≠ ∅` as a Qdrant filter, so the engine never sees another principal's data
  and cannot forget to drop it. An empty key set is a real filter that matches nothing.
  `tests/security/test_retrieval_isolation.py` indexes all visibility variants of two tenants
  with *identical* text into one collection and runs 96 reader configurations through the
  whole pipeline against the independent oracle: returned ⊆ allowed and returned ⊇ allowed.
- **Collections are bound to the embedding space.** Collection names embed the embedding
  fingerprint (`knowledge_<fingerprint>`), and chunks record the `embedding|sparse`
  fingerprint they were indexed with, so a model change is a rebuild into a new space, never
  a mix. The search index is derived state: `Indexer.rebuild_document` re-indexes from
  PostgreSQL.
- **Pipeline order.** Authorized scope → exact identifiers (O(1) lookups by id prefix; M7
  registers `mem_`) → deterministic `QueryRouter` (regex rules; no model when
  `llm.enabled=false`) → hybrid per kind (`chunk` / `memory`) → dedup + prune to `fused_k` →
  bounded CPU rerank of the top `candidate_k` (20) → `post_stages` hooks (graph M8, expansion
  and evidence verification M9). Reranking is bounded so p95 stays predictable.
- **ContextBuilder is thin.** It selects ranked evidence within a token budget, keeps the most
  recent visible turns that fit the conversation budget, and caches the bundle by
  `tenant + scope fingerprint + revision fingerprint + config fingerprint + query + budget`.
  Any write bumps a revision, so a cached bundle can never outlive the data it summarizes.
- **Authorization grants invalidate scope caches.** `grant_membership(..., revisions=)`
  bumps the user's revision so a cached `AuthorizedScope` is dropped immediately rather than
  at the cache TTL (found by the retrieval tests: a workspace grant was invisible for 60 s).
- **Providers.** `HashEmbedding` (deterministic feature hashing, unit norm) is the sandbox
  stand-in and is labelled non-representative in every result. `SentenceTransformersEmbedding`
  (torch/onnx/openvino backends, `local_files_only` when `model_path` is set) and
  `FastEmbedEmbedding` load Granite / MiniLM from local directories. `LexicalReranker` is the
  dependency-free reranker; `CrossEncoderReranker` wraps a local cross-encoder. Both real
  adapters are contract-tested with tiny randomly initialised models built offline, so the
  loading, normalisation, batching and ordering paths are verified without downloads.
- **Gates.** `tests/eval/golden/acme_fy26.json` lists the evidence groups a correct answer must
  be grounded in (ACME cross-page chain + GLOBEX distractor in the same scope).
  `tests/eval/test_retrieval_gate.py` asserts critical Recall@20 = 1.00 and Evidence-Group
  Recall = 1.00 and writes `benchmark/results/retrieval_gate.json` with provider provenance
  for `release_gate`.

## Evidence
- 180 tests pass (unit: router/RRF/BM25/hash/lexical/window; integration: index job → Qdrant
  local hybrid, workspace/thread/tenant filtering, exact lookups, rerank bounds, bundle budget
  and revision-driven cache invalidation; security: retrieval isolation gate; contract: model
  adapters + OpenAPI; e2e: `/v1/recall`, `/v1/context` over HTTP and the SDK; eval: gates).
- `benchmark/results/retrieval_gate.json`: critical Recall@20 = 1.00, EGR = 1.00, routing
  accuracy 1.00 with the hash embedding (labelled non-representative).
- `benchmark/results/retrieval.json` (25 salted copies, 548 points, Qdrant local, hash
  embedding): recall p95 ≈ 36 ms, cold context p95 ≈ 38 ms, cached p95 ≈ 2.6 ms — all inside
  the configured budgets (300/400/75 ms) on a shared sandbox CPU. Under exact-duplicate
  crowding secondary evidence groups fall out of the top 20 (Recall@20 0.84): recorded as an
  M9/M10 target (diversity-aware ranking, per-document caps), not hidden.

## Consequences
- Qdrant is required for retrieval; the `memory` search provider is qdrant-client local mode
  (`:memory:` or a path), which shares the adapter code with the server mode.
- Payload indexes on `tenant_id` / `visibility_keys` / `document_id` are created only against
  a server (local mode has no indexes).
- Real-model quality numbers must be produced on a machine with the weights
  (`MEMORY__MODELS__EMBEDDING__MODEL_PATH`); until then every result file says
  `"representative": false` and the milestone is not claimed production-ready.
