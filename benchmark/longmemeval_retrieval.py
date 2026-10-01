"""LongMemEval-S (cleaned), no LLM: does the shipped read path put the evidence in front of the
answerer? Turn- and session-level evidence recall at fixed memory depths, per question type.

Each question brings its own haystack - about fifty dated sessions between a user and an
assistant, some 500 turns - with the turns that hold the answer marked ``has_answer``. Every
turn is ingested the way the LoCoMo harness ingests one (an observation in the speaker's own
context, the session's date as its time), the question is asked of the production context
builder, and the memories it returns are traced back to the turns they were read out of:

* **turn recall@k**: the share of the question's ``has_answer`` turns among the sources of
  the first k memories;
* **session recall@k**: the same over the sessions those turns belong to (the measure the
  LongMemEval paper reports for retrieval).

The abstention questions (``*_abs``) have no evidence turn and are left out. A stratified
sample (``--questions``, seed 7) keeps a run CPU-feasible; every haystack is ingested from
an empty store, so a run is ``questions`` times one LoCoMo conversation of ingestion.

    PYTHONPATH=src:. python -m benchmark.longmemeval_retrieval \\
        --data benchmark/data/longmemeval/longmemeval_s_cleaned.json --questions 60 \\
        --output benchmark/results/phase10/longmemeval_retrieval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.common import (
    dedicated_database,
    isolated_qdrant,
    provenance,
    reset_store,
    submit_observation,
)
from benchmark.env import bench_overrides, bench_retrieval
from benchmark.harness import stats
from benchmark.native_source_retrieval import DEPTHS, coverage, observation_sources
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

TENANT = "bench_lme"
_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{2}):(\d{2})")


def sample(entries: list[dict[str, Any]], n: int | None, seed: int = 7) -> list[dict[str, Any]]:
    """The answerable questions, or ``n`` of them stratified by type (proportional, at least
    one each), in a fixed order."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if not str(entry["question_id"]).endswith("_abs"):
            by_type[entry["question_type"]].append(entry)
    if n is None:
        return [e for kind in sorted(by_type) for e in by_type[kind]]
    total = sum(len(v) for v in by_type.values())
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for _, members in sorted(by_type.items()):
        group = list(members)
        rng.shuffle(group)
        out += group[: max(1, round(n * len(group) / total))]
    return out


def session_time(value: str) -> datetime | None:
    m = _DATE.match(value or "")
    if not m:
        return None
    y, mo, d, h, mi = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


async def ingest(container: Any, ctx: MemoryExecutionContext, entry: dict[str, Any]) -> dict:
    """Every turn of the haystack as an observation. Returns observation id -> ``sid:k``."""
    uow_factory = container.services["uow_factory"]
    source_ids: dict[str, str] = {}
    for sid, date, session in zip(
        entry["haystack_session_ids"],
        entry["haystack_dates"],
        entry["haystack_sessions"],
        strict=True,
    ):
        when = session_time(date)
        for k, turn in enumerate(session):
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            speaker = ctx.model_copy(update={"user_id": str(turn.get("role") or "user")})
            async with uow_factory() as uow:
                observed = await submit_observation(
                    uow,
                    speaker,
                    content=content,
                    hints=ProcessingHints(visibility=Visibility.TENANT),
                    occurred_at=when,
                )
                source_ids[observed.observation_id] = f"{sid}:{k}"
                await uow.commit()
        await container.tasks.drain()
    await container.tasks.drain()
    return source_ids


def gold_of(entry: dict[str, Any]) -> list[str]:
    return [
        f"{sid}:{k}"
        for sid, session in zip(
            entry["haystack_session_ids"], entry["haystack_sessions"], strict=True
        )
        for k, turn in enumerate(session)
        if turn.get("has_answer")
    ]


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row[key]:
            groups[row["type"]].append(row)
            groups["all"].append(row)
    return {
        name: {
            "questions": len(group),
            "at": {
                str(k): {
                    metric: sum(r[key][str(k)][metric] for r in group) / len(group)
                    for metric in ("recall", "complete")
                }
                for k in DEPTHS
            },
        }
        for name, group in sorted(groups.items())
    }


async def run(args: argparse.Namespace) -> None:
    settings = _settings()
    dedicated_database(settings.database.url.get_secret_value())
    isolated_qdrant(settings.search.qdrant_url)
    settings = settings.model_copy(update={"bifrost_url": None, "bifrost_virtual_key": None})
    base = bench_overrides()
    overrides = replace(base, search=None, retrieval=bench_retrieval(base))
    container = await build_container(settings, __version__, overrides=overrides)
    register_handlers(container)
    entries = sample(json.loads(args.data.read_text()), args.questions)
    indexer = container.services["indexer"]
    result: dict[str, Any] = {
        "provenance": provenance(llm={"enabled": False, "provider": "disabled"}),
        "index_fingerprint": indexer.fingerprint,
        "retrieval": container.tuning.retrieval.model_dump(mode="json"),
        "questions": len(entries),
        "paid_llm_calls": 0,
        "limitations": [
            "Evidence recall of the context builder's memories, not answer accuracy.",
            "Abstention questions are excluded: they have no evidence turn.",
            "Sequential, one question at a time; in-process authorization and cache.",
        ],
    }
    rows: list[dict[str, Any]] = []
    builder = container.services["context_builder"]
    try:
        for n, entry in enumerate(entries):
            await reset_store(container, TENANT)
            ctx = MemoryExecutionContext(tenant_id=TENANT, user_id="asker", workspace_id="ws")
            started = time.perf_counter()
            source_ids = await ingest(container, ctx, entry)
            ingest_s = time.perf_counter() - started
            started = time.perf_counter()
            bundle = await builder.build(ctx, entry["question"])
            elapsed = (time.perf_counter() - started) * 1000
            sources = [
                sorted(observation_sources(item.evidence, source_ids)) for item in bundle.memories
            ]
            gold = gold_of(entry)
            sessions = [sorted({s.rsplit(":", 1)[0] for s in group}) for group in sources]
            rows.append(
                {
                    "question_id": entry["question_id"],
                    "type": entry["question_type"],
                    "turns": len(source_ids),
                    "gold": gold,
                    "turn": coverage(sources, gold),
                    "session": coverage(sessions, sorted({g.rsplit(":", 1)[0] for g in gold})),
                    "returned_memories": len(bundle.memories),
                    "ingest_seconds": round(ingest_s, 1),
                    "latency_ms": elapsed,
                }
            )
            result.update(
                {
                    "completed": len(rows),
                    "turn_recall": summarize(rows, "turn"),
                    "session_recall": summarize(rows, "session"),
                    "latency_ms": stats([r["latency_ms"] for r in rows]),
                    "records": rows,
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            done = result["turn_recall"].get("all", {}).get("at", {}).get("10", {})
            print(f"{n + 1}/{len(entries)} {entry['question_type']} turn@10 {done}", flush=True)
            await builder.drain()
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--questions", type=int, default=None, help="a stratified sample of this many (seed 7)"
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
