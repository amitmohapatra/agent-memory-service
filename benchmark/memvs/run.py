"""One system, one workload, one phase per process (stdlib + httpx + the system's library).

    python -m benchmark.memvs.run ingest --system mem0 --workload lme18.json --out DIR
    python -m benchmark.memvs.run query  --system mem0 --workload lme18.json --out DIR
    python -m benchmark.memvs.run answer --system mem0 --workload lme18.json --out DIR

``ingest`` writes every unit's sessions in order, waits until the system says all of it is
readable, and asks the first ``PROBE_QUERIES`` questions. ``query`` runs in a new process -
for this service, after the API and worker were restarted - so every read is a read after a
restart, and the probe questions are asked again: ``durability`` is the share whose hits are
the same. ``answer`` gives the answer-sample questions to the model (and the judge, where the
dataset has a gold answer); it resumes where an earlier run stopped.

Writes ``DIR/<system>.<dataset>.{ingest.json,rows.jsonl,query.json}``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark.memvs import llm
from benchmark.memvs.meter import TokenMeter
from benchmark.memvs.model import DatasetName, Phase, Query, QueryRow, SystemName, Unit, Workload
from benchmark.memvs.resources import Meter
from benchmark.memvs.systems.base import Embedder, Environment, MemorySystem

log = logging.getLogger("memvs.run")

#: the questions asked before and after the restart (the first ones, in workload order)
PROBE_QUERIES = 20
#: datasets graded by a judge model against a gold answer (the capability set is graded by
#: its own assertions in ``score``; isolation by its secrets)
JUDGED = frozenset(
    {DatasetName.LME18, DatasetName.LME_HELDOUT, DatasetName.LOCOMO, DatasetName.LOCOMO_C0}
)

#: agent-harness's gateway address (its ReAct and memory model calls)
ENV_HARNESS_GATEWAY = "BIFROST_URL"

#: system -> "module:Class", imported only in the process that runs it
REGISTRY: dict[SystemName, str] = {
    SystemName.TRELLIS: "benchmark.memvs.systems.trellis:Trellis",
    SystemName.TRELLIS_REACT: "benchmark.memvs.systems.trellis:TrellisReAct",
    SystemName.LANGGRAPH_STORE: "benchmark.memvs.systems.langgraph_store:LangGraphStore",
    SystemName.LANGMEM: "benchmark.memvs.systems.langmem_manager:LangMem",
    SystemName.DEEPAGENTS: "benchmark.memvs.systems.deepagents_files:DeepAgentsFiles",
    SystemName.OPENAI_SESSION: "benchmark.memvs.systems.openai_session:OpenAISession",
    SystemName.CREWAI: "benchmark.memvs.systems.crewai_memory:CrewAIMemory",
    SystemName.CREWAI_LLM: "benchmark.memvs.systems.crewai_memory:CrewAIMemoryLLM",
    SystemName.MEM0: "benchmark.memvs.systems.mem0_memory:Mem0",
    SystemName.MEM0_LLM: "benchmark.memvs.systems.mem0_memory:Mem0LLM",
    SystemName.GRAPHITI: "benchmark.memvs.systems.graphiti_graph:Graphiti",
}


def load_system(name: SystemName, env: Environment) -> MemorySystem:
    module, _, cls = REGISTRY[name].partition(":")
    return getattr(importlib.import_module(module), cls)(env)


def _paths(out: Path, system: SystemName, dataset: DatasetName) -> dict[str, Path]:
    stem = out / f"{system.value}.{dataset.value}"
    return {
        "ingest": stem.with_name(stem.name + ".ingest.json"),
        "rows": stem.with_name(stem.name + ".rows.jsonl"),
        "query": stem.with_name(stem.name + ".query.json"),
    }


def _hits_digest(hits: Sequence[Any]) -> list[str]:
    return [h.text for h in hits]


async def ingest(
    system: MemorySystem, workload: Workload, paths: dict[str, Path], tokens: TokenMeter | None
) -> None:
    write_ms: list[float] = []
    started = time.perf_counter()
    with Meter() as meter:
        await system.open()
        for n, unit in enumerate(workload.units):
            for session in unit.sessions:
                t0 = time.perf_counter()
                await system.write(unit, session)
                write_ms.append((time.perf_counter() - t0) * 1000)
                await system.after_write(unit)
            log.info("ingested unit %d/%d", n + 1, len(workload.units))
        written = time.perf_counter()
        await system.settle()
        settled = time.perf_counter()
        probe = {
            q.query_id: _hits_digest((await system.search(u, q)).hits) for u, q in _probe(workload)
        }
        await system.close()
    assert meter.usage is not None
    paths["ingest"].write_text(
        json.dumps(
            {
                "system": system.name.value,
                "dataset": workload.dataset.value,
                "units": len(workload.units),
                "sessions": sum(len(u.sessions) for u in workload.units),
                "turns": sum(len(s.turns) for u in workload.units for s in u.sessions),
                "write_ms": write_ms,
                "write_seconds": round(written - started, 2),
                "settle_seconds": round(settled - written, 2),
                "usage": asdict(meter.usage),
                "llm": tokens.snapshot() if tokens else None,
                "probe": probe,
            }
        )
    )


def _probe(workload: Workload) -> list[tuple[Unit, Query]]:
    pairs = [(u, q) for u in workload.units for q in u.queries]
    return pairs[:PROBE_QUERIES]


async def query(
    system: MemorySystem, workload: Workload, paths: dict[str, Path], tokens: TokenMeter | None
) -> None:
    before = json.loads(paths["ingest"].read_text())["probe"]
    rows: list[QueryRow] = []
    same = 0
    with Meter() as meter:
        await system.open()
        for unit in workload.units:
            for q in unit.queries:
                t0 = time.perf_counter()
                try:
                    got = await system.search(unit, q)
                    error = None
                except Exception as exc:  # noqa: BLE001 - a failed read is recorded, not fatal
                    log.warning("search failed: %s", type(exc).__name__)
                    got, error = None, f"{type(exc).__name__}: {exc}"[:300]
                latency = (time.perf_counter() - t0) * 1000
                if got is not None and got.latency_ms is not None:
                    latency = got.latency_ms  # the system timed the read its product makes
                if q.query_id in before and got is not None:
                    same += _hits_digest(got.hits) == before[q.query_id]
                rows.append(
                    QueryRow(
                        system=system.name.value,
                        dataset=workload.dataset.value,
                        unit_id=unit.unit_id,
                        query_id=q.query_id,
                        category=q.category,
                        language=q.language,
                        hits=[asdict(h) for h in got.hits] if got else [],
                        context=got.context if got else "",
                        context_tokens=len(got.context) // llm.CHARS_PER_TOKEN if got else 0,
                        latency_ms=latency,
                        insufficient=bool(got and got.insufficient),
                        error=error,
                    )
                )
        await system.close()
    assert meter.usage is not None
    with paths["rows"].open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    paths["query"].write_text(
        json.dumps(
            {
                "system": system.name.value,
                "dataset": workload.dataset.value,
                "queries": len(rows),
                "errors": sum(1 for r in rows if r.error),
                "usage": asdict(meter.usage),
                "llm": tokens.snapshot() if tokens else None,
                "durability": {"probed": len(before), "same_after_restart": same},
            }
        )
    )


async def answer(
    system: MemorySystem, workload: Workload, paths: dict[str, Path], tokens: TokenMeter | None
) -> None:
    gateway = llm.Gateway.from_env()
    queries = {(u.unit_id, q.query_id): (u, q) for u in workload.units for q in u.queries}
    rows = [json.loads(line) for line in paths["rows"].read_text().splitlines()]
    # only a system whose product builds the prompt itself needs its store to answer
    own = type(system).answer is not MemorySystem.answer
    if own:
        await system.open()
    try:
        for n, row in enumerate(rows):
            unit, q = queries[(row["unit_id"], row["query_id"])]
            if not q.answer_sample or row.get("answer") is not None:
                continue
            try:
                done = await system.answer(unit, q, row["context"], gateway)
            except Exception as exc:  # noqa: BLE001 - recorded on the row, the run goes on
                log.warning("answer failed: %s", type(exc).__name__)
                row["error"] = f"answer: {type(exc).__name__}: {exc}"[:300]
                continue
            row.update(
                answer=done.text,
                answer_prompt_tokens=done.prompt_tokens,
                answer_completion_tokens=done.completion_tokens,
                answer_latency_ms=done.latency_ms,
            )
            if workload.dataset in JUDGED and q.answer:
                row["judge"] = llm.judge(gateway, q.category, q.text, q.answer, done.text)
            # resumable: every answered row is on disk before the next model call
            paths["rows"].write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            )
            log.info("answered %d/%d", n + 1, len(rows))
    finally:
        if own:
            await system.close()
        if tokens is not None:
            stem = paths["query"].name.removesuffix(".query.json")
            paths["query"].with_name(stem + ".answer.json").write_text(
                json.dumps({"llm": tokens.snapshot()})
            )


PHASES = {Phase.INGEST: ingest, Phase.QUERY: query, Phase.ANSWER: answer}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", type=Phase)
    parser.add_argument("--system", type=SystemName, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True, help="the system's own directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    workload = Workload.load(args.workload)
    args.out.mkdir(parents=True, exist_ok=True)
    args.store.mkdir(parents=True, exist_ok=True)
    gateway, tokens = None, None
    try:
        upstream = llm.Gateway.from_env()
    except llm.LLMError:
        log.info("no gateway: a system that needs a model cannot run")
    else:
        # every model call of this process goes through the meter, the harness's included
        tokens = TokenMeter(upstream.base_url).start()
        gateway = llm.Gateway(tokens.url, upstream.key, upstream.model)
        os.environ[ENV_HARNESS_GATEWAY] = tokens.url
    env = Environment(
        store_dir=args.store,
        embedder=Embedder.from_env(),
        gateway=gateway,
        dataset=workload.dataset,
    )
    system = load_system(args.system, env)
    try:
        asyncio.run(
            PHASES[args.phase](
                system, workload, _paths(args.out, args.system, workload.dataset), tokens
            )
        )
    finally:
        if tokens is not None:
            tokens.stop()


if __name__ == "__main__":
    main()
