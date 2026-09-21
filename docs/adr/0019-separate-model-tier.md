# ADR-0019: The model tier is deployed separately, on CPU

## Status

Accepted. Adapters and compose services exist; in-process remains the default.

## Context

Three models were loaded inside both the API and the worker: a 384-dim embedding encoder, a
cross-encoder reranker, and a ~700MB DeBERTa NLI head. Measured consequences:

* **~6 minutes of cold start.** The healthcheck allowed 100 seconds, so `docker compose up`
  reported an unhealthy stack that was about to be fine (fixed separately, ADR note in
  `docker-compose.yml`).
* **Two resident copies of the same weights.** The worker never calls the reranker or the
  NLI head — retrieval is their only consumer — but wiring loaded them anyway.
* **No independent scaling.** Indexing a 1,000-document corpus starved the API, because
  ingestion and query traffic competed for the same cores. This was observed, not predicted.
* **Sizing questions were unanswerable.** "What VM reaches 20 RPS" has no answer while one
  process does HTTP, ingestion and inference on shared cores.

The per-request model cost is not evenly spread. Read from configuration rather than
estimated: one retrieval request costs **1 query embedding** and **`reranker.candidate_k`
(20) cross-encoder pairs**. The reranker is ~87% of it; the NLI head is only used by the
grounding cascade, whose traffic is unrelated to retrieval volume.

## Decision

Serve each model over HTTP from its own container, using HuggingFace
`text-embeddings-inference` on CPU, behind the model ports that already existed.

* Each model section gains a `url`. Set it and the model is served over HTTP; leave it unset
  and `provider` chooses the in-process runtime. **The wire dialect is detected, not
  declared**: TEI answers `/info`, an OpenAI-compatible gateway answers `/v1/models`. A first
  draft of this added `remote`, `openai` and `rerank_api` provider values — four new enum
  entries for a fact the server already knows, in a configuration surface that was already
  criticised for having too many. One URL replaced all of them.
* `RemoteEmbedding`, `RemoteReranker` and `RemoteNLI` import no inference library. The
  service knows a URL and a model *name*.
* `fingerprint()` returns the **served** model identity — asked of the server at startup, not
  taken from configuration. A URL says where to send text, not which model answers: built
  from the declared name, a URL pointing at a different encoder would write foreign vectors
  into the collection belonging to the declared one, with no error anywhere. The startup
  handshake also embeds a probe string and refuses to start if the width does not match
  `models.embedding.dimension`, because that value is what creates the collection.
* `tei-embed`, `tei-rerank` and `tei-nli` are compose services under the `models` profile.
* These adapters report `locality="local"`, matching Qdrant and Dragonfly: a separate
  process inside the operator's deployment is not a third-party API.
  `provider_policy.allow_remote_models` exists to stop data leaving the deployment, and a
  self-hosted sidecar does not.

## Consequences

**Scaling is per-model.** `tei-rerank` is the capacity-defining tier — scale it against RPS
while the API scales on request volume. Ingestion load lands on `tei-embed` and no longer
competes with query latency.

**Dynamic batching becomes possible.** In-process, each request reranked its own batch of 16
with no cross-request batching. An inference server batches across concurrent callers, which
is where most of the CPU throughput gain is.

**In-process stays the default.** A single-container deployment must keep working with no
extra moving parts, and the stand-in providers (`hash`, `lexical`) must stay available for
tests. `remote` is opt-in.

**Sizing must be measured on target hardware.** `make bench-model-throughput` reports items
per second per thread. Transformer inference leans on AVX2/AVX-512/AMX; the development
machine for this ADR (Intel i5-5257U, 2015, no AVX2) measured 7.9 cross-encoder pairs/sec,
roughly an order of magnitude below a current server core. Numbers from a laptop are not a
basis for an instance type.

**A cross-encoder at 20 pairs per request is the real cost driver.** Before buying cores,
`reranker.candidate_k` is the dial: halving it halves the model cost of every request.
Measure the recall consequence with `make bench-external` — an external corpus — rather than
against fixtures written here.

## Alternatives considered

**Keep everything in-process.** Simplest to deploy, and it is why the default has not
changed. Rejected as the only option because it makes the tiers inseparable.

**A third-party embedding API.** `vertex` and `openai` were declared in the embedding enum
with no implementation: they passed validation and then killed startup with
`NotImplementedError`. Removed. Data residency is a deployment decision, not a default.

**GPU.** Out of scope: the requirement is CPU with full capability. Nothing here prevents it
— the model tier is a container, and swapping its image is a deployment change.
