"""Would the entity prefetch narrow anything, on the corpus an arm reads?

    python -m benchmark.anchor_reach --dump benchmark/results/phase7/locomo_source_ensemble.json \
        --collection mem_memories_... --output benchmark/results/phase9/anchor_reach.json

D6 step 3 routes entity to memory: memories sharing an entity with the query enter the fusion
as one more RRF list (``ports.search.AnchoredPrefetch``, ``RetrievalTuning.entity_prefetch``).
Two things have to hold before that is worth an arm. The query has to name an entity, and the
anchor it produces has to match the way memories were indexed - and the second is a claim
about two pieces of code agreeing, which is cheaper to check than to run.

Both are counted here, against the store that would do the filtering:

* how many answerable questions ``query_entities`` would anchor, by category,
* how many points each of those anchors matches, and how many the same name matches in the
  scoped form the write path indexes a subject under (``memory_entities`` canonicalises
  ``m.subject``, which is ``user:john`` here, while the query side extracts ``john``).

An anchor that matches nothing adds a prefetch arm that returns nothing: it cannot move the
ranking, and it still costs a round trip inside the store. That is a finding about the wiring,
and it is not the same finding as "entity routing does not help".

No container and no encoder: this reads a finished dump and asks Qdrant to count. The
collection is named rather than derived, so the run says exactly which index it counted in.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.common import ISOLATED_QDRANT_PORT, provenance
from memory_service.modules.retrieval.engine import query_entities

#: how many distinct anchors to count points for, most-asked first
TOP_ANCHORS = 25


def questions_of(path: Path) -> list[tuple[str, str]]:
    """``(question, category)`` scanned line by line: the dump is 68 MB and the host is small."""
    out: list[tuple[str, str]] = []
    pending: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().rstrip(",")
            if stripped.startswith('"question": "'):
                pending = json.loads(stripped[len('"question": ') :])
            elif stripped.startswith('"category": "') and pending is not None:
                out.append((pending, json.loads(stripped[len('"category": ') :])))
                pending = None
    return out


def anchors_of(questions: list[tuple[str, str]]) -> tuple[dict[str, Any], Counter]:
    """Which questions would anchor, on what, and how often each anchor is asked for."""
    answerable = [(q, c) for q, c in questions if c != "adversarial"]
    firing = [(c, query_entities(q)) for q, c in answerable]
    hit = [(c, anchors) for c, anchors in firing if anchors]
    summary = {
        "answerable": len(answerable),
        "would_anchor": len(hit),
        "would_anchor_by_category": dict(Counter(c for c, _ in hit)),
        "answerable_by_category": dict(Counter(c for _, c in answerable)),
        "anchors_per_question": dict(Counter(len(a) for _, a in hit).most_common()),
    }
    return summary, Counter(a for _, anchors in hit for a in anchors)


def count(url: str, collection: str, tenant: str, anchor: str) -> int:
    """Exactly how many points in ``collection`` this anchor would narrow to.

    An empty ``tenant`` counts across the whole index. Both readings are worth having: a
    per-tenant count is what a query would see, and the index-wide count says whether the
    anchor's spelling exists anywhere at all - a name that is absent from its own tenant is
    an anchor for a different conversation, while a name absent from every tenant is a
    vocabulary mismatch between the two sides.
    """
    must: list[dict[str, Any]] = [{"key": "entities", "match": {"any": [anchor]}}]
    if tenant:
        must.insert(0, {"key": "tenant_id", "match": {"value": tenant}})
    body = json.dumps({"exact": True, "filter": {"must": must}}).encode()
    request = urllib.request.Request(  # noqa: S310 - the store is configuration
        f"{url.rstrip('/')}/collections/{collection}/points/count",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return int(json.load(response)["result"]["count"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", required=True, help="the memories collection to count in")
    parser.add_argument("--tenant", default="bench_conv_0")
    parser.add_argument("--qdrant-url", default=f"http://localhost:{ISOLATED_QDRANT_PORT}")
    args = parser.parse_args()

    summary, counter = anchors_of(questions_of(args.dump))
    top = [anchor for anchor, _ in counter.most_common(TOP_ANCHORS)]
    rows = [
        {
            "anchor": anchor,
            "questions": counter[anchor],
            "matched_as_asked": count(args.qdrant_url, args.collection, args.tenant, anchor),
            "matched_when_scoped": count(
                args.qdrant_url, args.collection, args.tenant, f"user:{anchor}"
            ),
        }
        for anchor in top
    ]
    result = {
        "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
        "collection": args.collection,
        "tenant": args.tenant,
        "dump": str(args.dump),
        **summary,
        "distinct_anchors": len(counter),
        "top_anchors": rows,
        "anchors_matching_nothing": sum(1 for row in rows if row["matched_as_asked"] == 0),
        "anchors_matching_only_when_scoped": sum(
            1 for row in rows if row["matched_as_asked"] == 0 < row["matched_when_scoped"]
        ),
        "limitations": [
            f"Counts are over {'tenant ' + args.tenant if args.tenant else 'every tenant'} in "
            "this collection, not per question.",
            "An anchor that matches points can still fail to help; this only says whether the "
            "prefetch can return anything at all.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2, default=str))


if __name__ == "__main__":
    main()
