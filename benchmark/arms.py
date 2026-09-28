"""Where each retriever put the memories that carry a gold turn, for one question.

The store fuses its arms server-side, so every fused hit comes back labelled ``fusion`` and
the per-arm identity is gone at the store boundary: every LoCoMo dump on disk reads
``{"fusion": 1986}`` and cannot say whether a missed turn was never retrieved or was
retrieved by one arm and buried by the fusion. This asks the store the same question one
arm at a time - the query's own vectors, the same filter, the prefetch depth - and records,
per arm and for the fused bundle, the rank of every memory that carries a gold turn.

It runs inside the benchmark only: zero cost on the service path, one extra round trip per
arm per question in the harness. The dump is what ``fit_rrf_weights`` reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from memory_service.config.constants import derived_k
from memory_service.modules.rag.indexer import MEMORIES
from memory_service.ports.search import VectorName


@dataclass
class ArmDump:
    """Ranked memory ids per arm at prefetch depth, the fused bundle's memory ids, and the
    best rank of each gold turn in each list (``None`` when the list never carries it)."""

    depth: int
    arms: dict[str, list[str]] = field(default_factory=dict)
    fused: list[str] = field(default_factory=list)
    gold_ranks: dict[str, dict[str, int | None]] = field(default_factory=dict)
    #: memory id -> the gold turns it carries, for every memory any list returned
    carriers: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gold_ranks(
    ranked: list[str], gold: set[str], sources: Mapping[str, set[str]]
) -> dict[str, int | None]:
    """For each gold turn, the first (zero-based) rank of a memory carrying it, or None."""
    out: dict[str, int | None] = dict.fromkeys(sorted(gold))
    for rank, memory_id in enumerate(ranked):
        for turn in sources.get(memory_id, ()):
            if turn in out and out[turn] is None:
                out[turn] = rank
    return out


async def dump_arms(
    container: Any,
    ctx: Any,
    question: str,
    *,
    bundle: Any,
    gold: set[str],
    sources: Mapping[str, set[str]],
) -> ArmDump:
    """The per-arm view of one question, against the memories collection the engine read."""
    engine = container.services["retrieval"]
    indexer = engine.indexer
    cfg = engine.cfg
    depth = max(cfg.prefetch_k, derived_k(cfg.memory_recall_k))
    async with container.services["uow_factory"]() as uow:
        visibility = await container.services["authz"].visibility(ctx, revisions=uow.revisions)
    flt = visibility.search_filter(kind="memory")
    flt = flt.model_copy(update={"must": {**flt.must, "current": True}})
    vectors = await engine._encode(question)
    collection = indexer.collection(MEMORIES)
    dump = ArmDump(depth=depth)
    for space, vector in vectors.dense.items():
        hits = await container.search.search_dense(collection, space, vector, flt, limit=depth)
        dump.arms[space.value] = [h.record_id for h in hits]
    if vectors.sparse is not None and vectors.sparse.indices:
        hits = await container.search.search_sparse(collection, vectors.sparse, flt, limit=depth)
        dump.arms[VectorName.BM25.value] = [h.record_id for h in hits]
    dump.fused = [item.item_id for item in bundle.memories]
    dump.gold_ranks = {
        name: gold_ranks(ranked, gold, sources) for name, ranked in dump.arms.items()
    }
    dump.gold_ranks["fused"] = gold_ranks(dump.fused, gold, sources)
    seen = {memory_id for ranked in dump.arms.values() for memory_id in ranked} | set(dump.fused)
    dump.carriers = {
        memory_id: sorted(gold & set(sources.get(memory_id, ())))
        for memory_id in sorted(seen)
        if gold & set(sources.get(memory_id, ()))
    }
    return dump
