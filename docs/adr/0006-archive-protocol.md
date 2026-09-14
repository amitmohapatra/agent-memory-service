# ADR 0006: Chat archive protocol (staged -> immutable verified segment -> purge)

**Status:** accepted · **Date:** 2026-09-14

## Decision
- Messages are acknowledged into PostgreSQL (`messages.content`, status `STAGED`). An archive
  job per thread (queueing-lock deduplicated, 60 s coalescing delay) compacts staged messages
  into JSONL+zstd segments named
  `tenant-shard=<sha256(tenant) % 64>/tenant=<t>/year=/month=/thread=/segment=<ulid>.jsonl.zst`.
- Upload is immutable (`if_generation_match=0`), then verified (generation + SHA-256 + size).
  Only then are the manifest marked `VERIFIED` and the messages marked `ARCHIVED` in one
  transaction. Large payloads (>= `purge_min_payload_bytes`, default 4 KiB) are nulled in
  the hot store only after `purge_grace_seconds` (default 24 h); reads of purged messages
  hydrate from the segment and re-check the content hash.
- A manifest row is written in `UPLOADING` state *before* the upload so a crash between
  upload and commit is repairable: the reconciler re-verifies the object and completes the
  commit, requeues ACKed-but-unarchived threads, retires unverifiable manifests (messages stay
  `STAGED`), and samples verified segments for checksum drift.
- Lifecycle: `blob.lifecycle_policy: autoclass` by default; explicit Nearline/Coldline/Archive
  tiers are generated from settings. The choice is data-driven (`make bench-storage`).

## Measurements (benchmark/results/storage.json, synthetic 20k-message chat, this sandbox)
zstd level 6: 5.6x compression at ~39 MB/s; level 3: 5.5x at ~54 MB/s; level 12: 6.1x at
~19 MB/s. Level 6 stays the default. The segment planner's expected ratio was raised from 3 to
5 based on this. Real corpora must be re-measured before freezing the 1–8 MB target.

## Consequences
- `acknowledged data loss = 0` under blob outage: the archive job fails and retries while the
  payload remains in PostgreSQL (`test_blob_outage_leaves_messages_staged`).
- Cost figures in the benchmark are list-price *assumptions*, not measurements.
