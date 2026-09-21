# Where data lives, what it costs, and what drifts

A review of storage placement, duplication and synchronisation across the five stores.
Every quantity below is measured on this repository's own data (1,000 SciFact documents →
1,303 chunks), not estimated. Prices are list-price assumptions and are labelled as such.

## 1. The map

| Store | Holds | Measured |
|---|---|---|
| **PostgreSQL** | source of truth: documents, nodes, chunks, memories, observations, conversation, tool invocations, context edges | `chunks` 4.3 MB, `document_nodes` 3.6 MB, `context_edges` 3.2 MB per 1,000 docs |
| **Qdrant** | derived index: dense + sparse vectors, plus a payload copy of text and lineage | 1,536 B/chunk vector (384-dim f32) + ≤2,000 B payload |
| **Dragonfly** | context bundles, authorization scopes | TTL 300 s, revision-keyed |
| **Blob (GCS/filesystem)** | original uploads, archive segments | `archive_segments` 1.1 MB per 1,000 docs |
| **OpenFGA** | relationship tuples for authorization | 50 tuples for 10 threads |

## 2. Duplication — which copies earn their keep

The same document text exists in up to five places. Three are justified, one is transient
and correct, one is questionable.

**Justified — `chunks.text` + Qdrant payload.** Retrieval fuses 40 candidates and reranks 20.
The reranker needs the *text* of each. Without the payload copy every query becomes a Qdrant
round trip plus a Postgres `WHERE chunk_id IN (...)` before reranking can start — on the hot
path. The copy is truncated to 2,000 characters, so it is explicitly a rerank/display cache,
not a substitute for the source. **Keep.**

**Justified — `archive_segments`.** A compressed cold copy with its own checksum and
verification path. That is the point of an archive. **Keep.**

**Correct and transient — `file_staging.data`.** Raw bytes held durably until the blob
archive is verified, then purged. Verified working: **0 of 1,000** staged rows still hold
bytes after archiving. **Keep.**

**Questionable — `document_nodes.text` (3.6 MB) alongside `chunks.text` (4.3 MB).** Nodes are
the parsed structure; chunks are derived from them. After chunking, node text is read for
context-graph expansion (parent sections, footnotes, definitions) — but those lookups could
be satisfied by chunk ranges plus the node tree without storing the text twice. This is the
one duplication worth measuring: it is roughly 45% of the Postgres document footprint.
**Investigate.**

### Does duplication make it faster? Yes, measurably — but only the payload copy

The Qdrant payload copy removes 20 Postgres lookups from every retrieval. The others do not
buy latency; they buy durability (staging) or recovery (archive), which is a different and
also legitimate reason.

## 3. Synchronisation — three derived stores, three different guarantees

**Qdrant ← Postgres: sound.** Postgres is authoritative, `make reindex` rebuilds from it, and
collection names embed the embedding fingerprint so a model change creates a new vector space
rather than mixing two. Two real bugs in this path were found and fixed: index entries for
chunks a document no longer has (`_purge_superseded`), and a fingerprint containing `:` that
produced an illegal collection name.

**Dragonfly ← Postgres: sound.** Context bundles are keyed by scope + **revision fingerprint**
+ query hash + retrieval config, with a 300 s TTL. A revision bump invalidates; a stale read
is bounded by the TTL. This is the best-behaved of the three.

**OpenFGA ← Postgres: broken.** No revocation exists anywhere in the code base — grep for a
tuple delete on entity deletion returns nothing. Tuples are written and never removed. Today
there are 50 tuples for 10 threads.

This is not theoretical. It has already caused a production-shaped failure: a thread whose
Postgres rows were deleted kept its tuples, and re-using that thread id failed permanently
with `OpenFGA write failed: ValidationException` on every message write. That was fixed by
making tuple writes idempotent — but idempotent writes treat the symptom. **The orphaned
tuples are still there, they still grow without bound, and a deleted user's grants are still
live in the authorization store.** For a system whose isolation guarantees rest on OpenFGA,
that is the most serious finding in this review.

**A fourth, unmanaged: Qdrant collections accumulate.** 20 collections exist on this machine;
the service uses 2. Every embedding/sparse fingerprint change creates a new pair and nothing
ever drops the old ones. Each holds full vectors. There is a `drop_collection` call in
`reindex --drop`, but nothing reclaims superseded generations on its own.

## 4. Cost — where the money actually is

Measured per chunk: text **1,081 B**, dense vector **1,536 B**. *The vector is larger than the
text it describes.* Any plan that moves text around to save money is optimising the smaller
half.

**Moving chunk text to GCS would cost ~250× what it saves.** At 1M chunks the text is 1.1 GB:
$0.18/month on Postgres SSD, $0.02 on GCS — a saving of **$0.16/month**. The access pattern
is 20 RPS × 20 reranked chunks = 400 object reads/second ≈ 1B class-B operations/month ≈
**$415/month**, plus 20–100 ms added to every query. GCS is cheap per byte and expensive per
request; this is a hot path, so it is the wrong store. The blob store is already used
correctly for what suits it: large objects fetched rarely.

**The real levers, in order of size:**

1. **`reranker.candidate_k: 20`** — one request costs 1 embedding and 20 cross-encoder pairs;
   the reranker is ~87% of per-request model cost. Halving this halves the compute bill.
   Measure the recall consequence on an external corpus before choosing.
2. **`on_disk_payload`** — keeps vectors in RAM and payload on disk. At 10M chunks this is the
   difference between a small and a large Qdrant node. Already configurable, currently off
   outside local mode.
3. **Orphaned collections** — 20 exist where 2 are used. Full vector sets, paid for, unread.
4. **`context_edges`: 12,670 rows for 1,303 chunks — a ~10× fan-out**, and 3.2 MB per 1,000
   documents, comparable to the chunks themselves. Worth checking whether every edge kind
   earns its storage.

## 5. What I would change

**Must fix — authorization drift.** Add tuple revocation on delete for threads, documents,
memories and users, and a reconciliation job that removes tuples whose subject or object no
longer exists. This is a correctness and security issue, not a cost one: deleted principals
currently retain live grants.

**Should fix — reclaim superseded collections.** After a successful reindex to a new
fingerprint, drop the previous generation (or age it out), rather than leaving it to
`--drop`.

**Should measure — `document_nodes.text`.** Determine whether context-graph expansion can work
from the node tree plus chunk ranges. If it can, that is ~45% of the document footprint.

**Should measure — `context_edges` fan-out.** Ten edges per chunk is a lot; establish which
kinds are actually followed at retrieval time.

**Do not change — the Qdrant payload copy.** It is the one duplication that buys latency, it
is bounded at 2,000 characters, and it is rebuildable from the source of truth.

**Do not move hot text to blob storage.** The arithmetic is in §4.

## 6. What this review does not cover

No compute cost model exists. `benchmark/storage.json` models archive storage only; nothing
models Postgres, Qdrant, Dragonfly, OpenFGA or egress. The inputs exist across 21 benchmark
result files but have never been assembled into a total, and the throughput numbers that
would feed it were measured on a 2015 dual-core CPU without AVX2 — roughly an order of
magnitude off any current server. They must be re-measured on target hardware
(`make bench-model-throughput`) before any figure is quoted.
