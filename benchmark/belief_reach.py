"""Does the read path reach a dated per-(subject, predicate) aggregate on the LoCoMo arms?

    python -m benchmark.belief_reach --dump benchmark/results/phase7/locomo_source_ensemble.json \
        --output benchmark/results/phase9/belief_reach.json

D6 step 4 renders one dated line per (subject, predicate) slot instead of several scattered
memories, and its gate is strict multi-hop. Before measuring a delta, this establishes whether
there is anything to measure, from the corpus an arm actually read and the dump it produced.
Three questions, three numbers:

1. **Does the corpus hold any belief?** ``LandingReflection`` mints one per (scope, subject,
   multi-valued predicate) once ``belief_min_support`` memories share the slot - but only when
   ``consolidation_enabled`` is on, and it is off by default in the product and was off in both
   Phase 7 arms. The count of BELIEF and ENTITY_SUMMARY rows settles it.
2. **How many would there be?** The qualifying slots, their sizes, and the memories inside them,
   so "there are none" carries the size of what is missing.
3. **How big is the bucket it targets?** The questions whose gold-carrying memories sit in one
   multi-valued slot - the fragmentation shape an aggregate collapses - split by whether the
   arm already carries all their evidence at depth 10, because only the rest is addressable.

Read only: it runs SELECTs against the corpus and arithmetic over a finished dump.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from benchmark.common import dedicated_database, provenance
from benchmark.retrieval import _settings
from memory_service.config.constants import MEMORY_INTELLIGENCE
from memory_service.domain.predicates import is_multi_valued

#: the depth the fragmentation bucket is read at: what the arm already carries there is not
#: addressable by rendering the rest of it differently
DEPTH = "10"


async def corpus_facts(url: str) -> dict[str, Any]:
    """What the corpus holds: memory types, and the slots a belief would be minted for."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            types = dict(
                (
                    await conn.execute(
                        text("SELECT memory_type, count(*) FROM memories GROUP BY 1")
                    )
                ).all()  # type: ignore[arg-type]
            )
            slots = (
                await conn.execute(
                    text(
                        "SELECT tenant_id, subject, predicate, count(*) FROM memories "
                        "WHERE subject IS NOT NULL AND predicate IS NOT NULL "
                        "AND deleted_at IS NULL GROUP BY 1, 2, 3"
                    )
                )
            ).all()
            owners = {
                memory_id: (subject, predicate)
                for memory_id, subject, predicate in (
                    await conn.execute(
                        text(
                            "SELECT memory_id, subject, predicate FROM memories "
                            "WHERE subject IS NOT NULL AND predicate IS NOT NULL "
                            "AND deleted_at IS NULL"
                        )
                    )
                ).all()
            }
    finally:
        await engine.dispose()
    support = MEMORY_INTELLIGENCE.belief_min_support
    multi = [row for row in slots if is_multi_valued(str(row[2]))]
    qualifying = [row for row in multi if row[3] >= support]
    return {
        "memory_types": {str(k): v for k, v in sorted(types.items())},
        "derived_memories": sum(
            v for k, v in types.items() if str(k) in {"BELIEF", "ENTITY_SUMMARY"}
        ),
        "consolidation_enabled": MEMORY_INTELLIGENCE.consolidation_enabled,
        "belief_min_support": support,
        "slots": len(slots),
        "multi_valued_slots": len(multi),
        "slots_a_belief_would_be_minted_for": len(qualifying),
        "memories_inside_those_slots": sum(row[3] for row in qualifying),
        "widest_slot": max((row[3] for row in qualifying), default=0),
        "widest_slot_predicate": max(qualifying, key=lambda row: row[3])[2] if qualifying else None,
        "top_predicates": Counter(str(row[2]) for row in qualifying).most_common(8),
        "_slot_of": owners,
    }


def fragmentation(
    records: list[dict[str, Any]], slot_of: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    """Questions whose gold carriers share one multi-valued slot, and how many still miss.

    The slot is (subject, predicate), not the predicate alone: two people who each `said`
    something are two slots, and one aggregate line cannot gather them. Grouping on the
    predicate by itself counted those as fragmentation and inflated the bucket by 8%.
    """
    fragmented: list[dict[str, Any]] = []
    unknown = 0
    for row in records:
        if row.get("category") == "adversarial" or not row.get("gold") or not row.get("arms"):
            continue
        groups: dict[tuple[str, str], set[str]] = defaultdict(set)
        for memory_id in row["arms"]["carriers"]:
            slot = slot_of.get(memory_id)
            if slot is None:
                unknown += 1
            elif is_multi_valued(slot[1]):
                groups[slot].add(memory_id)
        if max((len(v) for v in groups.values()), default=0) >= 2:
            fragmented.append(row)
    missing = [row for row in fragmented if not row["coverage"][DEPTH]["complete"]]
    return {
        "questions": len(fragmented),
        "by_category": dict(Counter(row["category"] for row in fragmented)),
        "still_incomplete_at_10": len(missing),
        "still_incomplete_by_category": dict(Counter(row["category"] for row in missing)),
        "carriers_absent_from_the_corpus": unknown,
    }


async def run(args: argparse.Namespace) -> None:
    settings = _settings()
    database = dedicated_database(settings.database.url.get_secret_value())
    facts = await corpus_facts(settings.database.url.get_secret_value())
    slot_of = facts.pop("_slot_of")
    raw = args.dump.read_bytes()
    dump = json.loads(raw)
    result = {
        "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
        "database": database,
        "dump": str(args.dump),
        "dump_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus": facts,
        "fragmentation_bucket": fragmentation(dump["records"], slot_of),
        "limitations": [
            "A belief's evidence is memory-typed, so it can only raise the lineage metric, "
            "never the direct source-turn metric the arms are read on.",
            "The read-side aggregate render changes the rendered text, not which memories are "
            "in the bundle, so no source-ID arm can see its contribution.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
