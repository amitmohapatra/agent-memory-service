# Qdrant: scaling the search index out

Qdrant holds a rebuildable projection of PostgreSQL (ADR 0001, 0008): two collections per
embedding fingerprint, `knowledge` (chunks and summaries) and `memories`. This page covers
running them on a cluster. The decisions are ADR 0031.

## Layout of a new collection

| setting | default | meaning |
|---|---|---|
| `MEMORY__SEARCH__SHARD_NUMBER` | 1 | shards a collection's points are spread over; at least the node count, so every node takes part |
| `MEMORY__SEARCH__REPLICATION_FACTOR` | 1 | copies of each shard; 2 survives the loss of one node |
| `MEMORY__SEARCH__WRITE_CONSISTENCY_FACTOR` | 1 | copies that must acknowledge a write before it returns; at most the replication factor (validated) |

They are passed to `create_collection` (`adapters/search/qdrant_store.py`). Local mode, the
test stand-in, ignores them.

## The tenant index

Every search filters on exactly one `tenant_id`. The payload index on it is created with
`is_tenant=True`, which tells Qdrant to group storage by tenant: a tenant's search then reads
that tenant's points instead of filtering every segment. A collection created before this
change has a plain keyword index. The service re-creates it with the flag the first time a
process ensures the collection. That is an in-place index rebuild, and search keeps working
while it runs. Points already stored are regrouped as the optimizer rewrites their segments,
or immediately by a rebuild (below).

## Memory payloads: RAM until they are big

The memories collection asks for its payloads in RAM. Every query reads them, and at
depth 100 a page-cache miss per returned hit costs more than the RAM. That holds only while
the collection is small. Past `constants.SEARCH.payload_in_ram_max_points` (500 000 points,
about 1 GB of RAM at 1-2 KB a payload) the payloads are moved to disk. The check runs when a
process first ensures the collection, so it takes effect at the next restart after the
collection crosses the line. `update_collection` changes the storage in place, without a
reindex.

## Applying a new layout to existing collections

Qdrant fixes `shard_number` when a collection is created. To move existing collections onto a
new layout, rebuild them from PostgreSQL:

```bash
# new collections with the configured layout, refilled from the canonical store
make reindex-image REINDEX_ARGS="--drop"
```

`--drop` deletes the current collections first, so search answers from an empty index until
the rebuild finishes. Run it in a maintenance window, or against a second deployment that
writes the same fingerprint's collections first and then switches. `reindex` reports every
document and memory that failed, and none is skipped silently. Replication factor and write
consistency can be raised on a live collection through Qdrant's own collection-update and
shard-replication APIs. This service never does that on its own.
