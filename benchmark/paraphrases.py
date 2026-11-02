"""Duplicates, false merges and stale values on one user's paraphrases (ADR 0045).

    BENCH_NLI=frozen uv run python -m benchmark.paraphrases --arm nollm
    BENCH_LLM=on BENCH_NLI=frozen BENCH_LLM_MODEL=<model> uv run python -m benchmark.paraphrases \
        --arm llm            # the tenant's model policy, read rules-first (today)
    ... --arm model_first    # the same policy, Tenant.model_first_extraction on

``tests/eval/golden/paraphrase_sets.json`` is said turn by turn, round-robin over its groups,
two conversations alternating, a day apart, through the real write path: the observation
pipeline with its consolidation and cross-conversation replacement (the frozen NLI head with
``BENCH_NLI=frozen``). Then the store is read:

* duplicate rate - a ``same`` group is one fact said five ways; each current typed memory (not
  the turn as said) past the first one is a duplicate, over the repeats said;
* false-merge rate - a decision that joined a candidate to a memory of another group
  (reinforced, merged, superseded, contradicted or ignored as known), over the candidates;
* stale rate - an ``update`` group whose old value is still current: in a typed memory, or in a
  turn as said that nothing marks as an earlier value;
* the model's tokens per message and the write's seconds per message (the job, not the chat:
  ``memory.process_observation`` runs in the background).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

from benchmark.common import provenance, reset_store, submit_observation, write_result
from benchmark.env import bench_overrides, pin_model_policy
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision
from memory_service.domain.memory import replaced_by
from memory_service.domain.tenancy import Tenant
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.llm.cost import llm_accounting

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "tests" / "eval" / "golden" / "paraphrase_sets.json"
#: the write-path uses a tenant's model policy names (``data/_modes.json``'s llm mode has them)
WRITE_USES = ("contextual_extraction", "conflict_adjudication")
START = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
_JOINED = frozenset(
    {
        DedupDecision.REINFORCE,
        DedupDecision.MERGE,
        DedupDecision.SUPERSEDE,
        DedupDecision.UPDATE,
        DedupDecision.CONTRADICT,
        DedupDecision.IGNORE,
    }
)


def schedule(groups: list[dict[str, Any]]) -> list[tuple[str, int, str, str]]:
    """``(group id, turn index, user, turn)`` in the order they are said: round-robin over
    the groups."""
    out: list[tuple[str, int, str, str]] = []
    for i in range(max(len(g["turns"]) for g in groups)):
        out += [
            (g["id"], i, g.get("user", "u1"), g["turns"][i]) for g in groups if i < len(g["turns"])
        ]
    return out


async def run(arm: str, *, tenant_id: str) -> dict[str, Any]:
    groups = json.loads(SETS.read_text(encoding="utf-8"))["groups"]
    settings = _settings()
    container = await build_container(settings, __version__, overrides=bench_overrides())
    try:
        await reset_store(container, tenant_id)
        register_handlers(container)
        uow_factory = container.services["uow_factory"]
        pipeline = container.services["observation_pipeline"]
        await pin_model_policy(container, tenant_id, WRITE_USES if arm != "nollm" else ())
        async with uow_factory() as uow:
            switch = {"model_first_extraction": arm == "model_first"}
            if await uow.tenants.get(tenant_id) is None:
                await uow.tenants.add(Tenant(tenant_id=tenant_id, name="paraphrases", **switch))
            else:
                await uow.tenants.update(tenant_id, switch)
            await uow.commit()
        said_in: dict[str, tuple[str, int]] = {}  # observation id -> (group, turn index)
        decisions: list[tuple[str, DedupDecision, str | None]] = []
        seconds: list[float] = []
        tokens_in = tokens_out = 0
        for i, (group, turn, user, said) in enumerate(schedule(groups)):
            ctx = MemoryExecutionContext(
                tenant_id=tenant_id, user_id=user, thread_id=f"{user}-t{i % 2}", workspace_id="ws1"
            )
            async with uow_factory() as uow:
                observed = await submit_observation(
                    uow, ctx, content=said, occurred_at=START + timedelta(days=i)
                )
                await uow.commit()
            said_in[observed.observation_id] = (group, turn)
            started = time.perf_counter()
            with llm_accounting() as used:
                outcomes = await pipeline.run(
                    {"tenant_id": tenant_id, "observation_id": observed.observation_id}
                )
            seconds.append(time.perf_counter() - started)
            tokens_in += used.input
            tokens_out += used.output
            decisions += [(group, o.decision, o.target_memory_id) for o in outcomes]
        await container.tasks.drain()
        async with container.database.engine.connect() as conn:
            ids = (
                await conn.execute(
                    text("SELECT memory_id FROM memories WHERE tenant_id = :t"), {"t": tenant_id}
                )
            ).scalars()
            ids = list(ids)
        async with uow_factory() as uow:
            memories = await uow.memories.get_many(tenant_id, ids)
    finally:
        await container.close()
    turn_of = {
        m.memory_id: said_in.get(m.evidence[0].source_id, (None, -1))
        for m in memories
        if m.evidence
    }
    origin = {memory_id: group for memory_id, (group, _) in turn_of.items()}
    current = [m for m in memories if m.temporal.status.value == "CURRENT" and not m.deleted_at]
    typed = [m for m in current if m.system_metadata.get("category") != "verbatim_turn"]
    said = len(schedule(groups))
    report: dict[str, Any] = {"arm": arm, "messages": said}
    repeats = duplicates = 0
    per_group: dict[str, Any] = {}
    for g in groups:
        mine = [m for m in typed if origin.get(m.memory_id) == g["id"]]
        entry: dict[str, Any] = {"typed_current": [m.content for m in mine]}
        if g["kind"] == "same":
            repeats += len(g["turns"]) - 1
            duplicates += max(0, len(mine) - 1)
        if g["kind"] == "update":
            earlier = [
                m
                for m in current
                if turn_of.get(m.memory_id) is not None
                and turn_of[m.memory_id][0] == g["id"]
                and turn_of[m.memory_id][1] < g["changes_at"]
                and replaced_by(m.system_metadata) is None
            ]
            entry["stale_typed"] = [
                m.content for m in earlier if m.system_metadata.get("category") != "verbatim_turn"
            ]
            entry["stale_turns"] = [
                m.content for m in earlier if m.system_metadata.get("category") == "verbatim_turn"
            ]
            entry["new_current"] = any(turn_of[m.memory_id][1] >= g["changes_at"] for m in mine)
        per_group[g["id"]] = entry
    false = [
        (group, d.value, origin.get(target))
        for group, d, target in decisions
        if d in _JOINED and target is not None and origin.get(target) not in (None, group)
    ]
    updates = [g for g in groups if g["kind"] == "update"]
    report |= {
        "duplicate_rate": round(duplicates / repeats, 4) if repeats else 0.0,
        "duplicates": duplicates,
        "repeats": repeats,
        "false_merge_rate": round(len(false) / len(decisions), 4) if decisions else 0.0,
        "false_merges": false,
        "candidates": len(decisions),
        "stale_typed_rate": round(
            sum(bool(per_group[g["id"]]["stale_typed"]) for g in updates) / len(updates), 4
        ),
        "stale_any_rate": round(
            sum(
                bool(per_group[g["id"]]["stale_typed"] or per_group[g["id"]]["stale_turns"])
                for g in updates
            )
            / len(updates),
            4,
        ),
        "new_value_kept": sum(per_group[g["id"]]["new_current"] for g in updates),
        "typed_current": len(typed),
        "decisions": {d.value: sum(1 for _, x, _ in decisions if x is d) for d in DedupDecision},
        "tokens_per_message": {"input": tokens_in / said, "output": tokens_out / said},
        "write_seconds_per_message": {
            "mean": round(statistics.fmean(seconds), 3),
            "p50": round(statistics.median(seconds), 3),
            "max": round(max(seconds), 3),
        },
        "per_group": per_group,
        "provenance": provenance(),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=("nollm", "llm", "model_first"), required=True)
    p.add_argument("--tenant", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = asyncio.run(run(args.arm, tenant_id=args.tenant or f"paraphrase-{args.arm}"))
    path = write_result(args.out or f"paraphrases_{args.arm}.json", report)
    summary = {k: v for k, v in report.items() if k not in ("per_group", "provenance")}
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
