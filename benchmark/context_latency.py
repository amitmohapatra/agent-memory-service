# ruff: noqa: RUF001 - a literal multilingual off-topic question.
"""Latency of the read path with the model off: ``/v1/context`` without and with the tools
section, and ``/v1/recall`` - p50/p95 per endpoint and the stage split behind them.

The corpus is what the push serves in practice: one LoCoMo conversation recorded turn by
turn through ``/v1/messages`` (memories, entities, the graph), the golden documents
(``--copies`` salted copies each) and a tool catalog. Each measured request carries a unique
suffix so the bundle cache and the semantic cache miss; ``cached`` measures the same request
twice and keeps the second. ``diagnostics.timings_ms`` of every uncached context is kept,
so a slow series names the stage that made it slow.

    make bench-context-latency CONTEXT_LATENCY_ARGS="--label baseline"

In-process ASGI (no network hop), real encoders, the isolated Qdrant server and Dragonfly
when run through ``bench-run``; the providers are recorded next to the numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as redis_async

from benchmark.common import provenance, reset_store, write_result
from benchmark.env import BENCH, bench_overrides
from benchmark.evaluation.golden import GoldenSet
from benchmark.harness import H, build_corpus, new_scope, stats
from benchmark.retrieval import GOLDEN, _settings
from memory_service.__about__ import __version__
from memory_service.api.app import create_app
from memory_service.application.container import build_container
from memory_service.modules.jobs.registry import register_handlers

__all__ = ["main", "run"]

LOCOMO = Path(__file__).resolve().parent / "data" / "locomo10.json"
TOOLS = [
    {
        "name": f"{server}-{verb}",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {arg: {"type": "string"} for arg in args},
            "required": list(args),
        },
        "side_effects": effect,
        "source": "mcp",
        "server": server,
    }
    for server, verb, description, args, effect in [
        ("calendar", "create_event", "Create a calendar event", ("title", "date"), "write"),
        ("calendar", "list_events", "List events in a date range", ("start", "end"), "read"),
        ("mail", "send", "Send an email", ("to", "subject", "body"), "irreversible"),
        ("mail", "search", "Search the mailbox", ("query",), "read"),
        ("crm", "update_contact", "Update a contact's details", ("contact", "field"), "write"),
        ("crm", "find_contact", "Find a contact by name", ("name",), "read"),
        ("docs", "search", "Search company documents", ("query",), "read"),
        ("finance", "report", "Fetch a financial report line", ("metric", "period"), "read"),
    ]
]
SERIES = ("recall", "context", "context_tools", "cached")
#: Questions nothing in the corpus answers: what a floor is for.
OFF_TOPIC = (
    "What is the boiling point of mercury at sea level?",
    "How do I configure a VLAN on a Cisco switch?",
    "Who won the 1998 football world cup final?",
    "What is the capital of Mongolia?",
    "Explain the difference between TCP and UDP.",
    "How many moons does Neptune have?",
    "What is a good recipe for sourdough bread?",
    "Welche Programmiersprache ist am schnellsten?",
    "¿Cuál es la montaña más alta de África?",
    "量子コンピュータとは何ですか？",
)


def _turns(limit: int) -> tuple[list[tuple[str, str]], list[str], list[list[str]]]:
    """The first ``limit`` turns of LoCoMo conversation 0 (speaker, text), the questions
    whose evidence is among them, and each such question's evidence texts."""
    sample = json.loads(LOCOMO.read_text(encoding="utf-8"))[0]
    conversation = sample["conversation"]
    turns: list[tuple[str, str]] = []
    by_id: dict[str, str] = {}
    session = 1
    while f"session_{session}" in conversation and len(turns) < limit:
        for turn in conversation[f"session_{session}"]:
            turns.append((turn["speaker"], turn["text"]))
            by_id[turn["dia_id"]] = turn["text"]
            if len(turns) >= limit:
                break
        session += 1
    questions: list[str] = []
    evidence: list[list[str]] = []
    for qa in sample["qa"]:
        texts = [by_id[e] for e in qa.get("evidence", []) if e in by_id]
        if texts and len(texts) == len(qa.get("evidence", [])):
            questions.append(str(qa["question"]))
            evidence.append(texts)
    return turns, questions, evidence


#: Relevance floors the offline analysis replays over one floor-free pass.
FLOORS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)


def _floor_analysis(rows: list[tuple[list[tuple[float, bool]], int]]) -> dict[str, Any]:
    """Per floor: the share of evidence items and of other items that would still be packed,
    the share of questions that keep at least one evidence item, and the packed chars.

    ``rows``: per question, (similarity, is_evidence) of every packed memory, and the
    question's rendered chars. The replay only removes items, so it understates a floor:
    the budget a removed item frees is not refilled here."""
    out: dict[str, Any] = {}
    evidence = [s for items, _ in rows for s, hit in items if hit]
    other = [s for items, _ in rows for s, hit in items if not hit]
    answered = [items for items, _ in rows if any(hit for _, hit in items)]
    for floor in FLOORS:
        out[f"{floor:.2f}"] = {
            "evidence_kept": round(sum(s >= floor for s in evidence) / max(1, len(evidence)), 4),
            "other_kept": round(sum(s >= floor for s in other) / max(1, len(other)), 4),
            "questions_with_evidence": round(
                sum(any(s >= floor and hit for s, hit in items) for items in answered)
                / max(1, len(answered)),
                4,
            ),
            "items_per_question": round(
                sum(sum(s >= floor for s, _ in items) for items, _ in rows) / max(1, len(rows)), 1
            ),
        }
    return {
        "floors": out,
        "evidence_items": len(evidence),
        "other_items": len(other),
        "questions": len(rows),
        "questions_with_evidence_at_all": len(answered),
        "rendered_chars_mean": round(statistics.fmean(n for _, n in rows), 1) if rows else 0,
    }


async def _timed(client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    response = await client.post(path, headers=H, json=body)
    elapsed = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return elapsed, response.json()


def _stage_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    stages = sorted({name for row in rows for name in row})
    out: dict[str, dict[str, float]] = {}
    for name in stages:
        values = sorted(row.get(name, 0.0) for row in rows)
        out[name] = {
            "mean": round(statistics.fmean(values), 1),
            "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
        }
    return out


async def run(copies: int, turns: int, requests: int) -> dict[str, Any]:
    settings = _settings()
    container = await build_container(settings, __version__, overrides=bench_overrides())
    register_handlers(container)
    app = create_app(settings, container=container)
    golden = GoldenSet.load(GOLDEN)
    history, questions, evidence = _turns(turns)
    queries = [*questions, *(q.query for q in golden.questions)]
    try:
        await reset_store(container, "acme")
        if BENCH.cache == "dragonfly":
            # reset_store clears PostgreSQL and Qdrant; a run on the real cache must clear its
            # database too, or the upload dedup answers with a document the TRUNCATE removed
            # (measured: every document of a second run was a 404 behind a 202)
            client_cache = redis_async.from_url(settings.cache.url.get_secret_value())
            await client_cache.flushdb()
            await client_cache.aclose()
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=120
            ) as client,
        ):
            scope = new_scope()
            seeded = time.perf_counter()
            await build_corpus(client, H, scope, golden, copies)
            for i, (speaker, text) in enumerate(history):
                r = await client.post(
                    "/v1/messages",
                    headers=H,
                    json={"scope": scope, "role": "USER", "content": f"{speaker}: {text}"},
                )
                r.raise_for_status()
                if i % 20 == 19:
                    await container.tasks.drain()
            await container.tasks.drain()
            await container.tasks.drain()
            r = await client.put("/v1/tools/catalog", headers=H, json={"tools": TOOLS})
            r.raise_for_status()
            await container.tasks.drain()
            seed_seconds = round(time.perf_counter() - seeded, 1)
            available = [tool["name"] for tool in TOOLS]
            for query in queries[:5]:  # warm the encoders, the pools and the scope cache
                await _timed(client, "/v1/context", {"scope": scope, "query": query})

            latencies: dict[str, list[float]] = {name: [] for name in SERIES}
            stage_rows: dict[str, list[dict[str, float]]] = {"context": [], "context_tools": []}
            sizes: list[int] = []
            for i in range(requests):
                query = f"{queries[i % len(queries)]} (variant {i})"
                ms, _ = await _timed(client, "/v1/recall", {"scope": scope, "query": query})
                latencies["recall"].append(ms)
                ms, body = await _timed(
                    client, "/v1/context", {"scope": scope, "query": f"{query} [c]"}
                )
                latencies["context"].append(ms)
                stage_rows["context"].append(body["diagnostics"].get("timings_ms", {}))
                sizes.append(len(json.dumps(body)))
                ms, body = await _timed(
                    client,
                    "/v1/context",
                    {"scope": scope, "query": f"{query} [t]", "tools": {"available": available}},
                )
                latencies["context_tools"].append(ms)
                stage_rows["context_tools"].append(body["diagnostics"].get("timings_ms", {}))
                repeat = {"scope": scope, "query": queries[i % len(queries)]}
                await _timed(client, "/v1/context", repeat)
                ms, _ = await _timed(client, "/v1/context", repeat)
                latencies["cached"].append(ms)
            floor_rows: list[tuple[list[tuple[float, bool]], int]] = []
            for question, texts in zip(questions, evidence, strict=True):
                _, body = await _timed(
                    client, "/v1/context", {"scope": scope, "query": f"{question} [f]"}
                )
                items = [
                    (
                        float(item["relevance"]),
                        any(text[:60] in item["text"] for text in texts),
                    )
                    for item in body["memories"]
                    if item["score_kind"] == "fusion" and not item.get("expanded_from")
                ]
                floor_rows.append((items, len(body["rendered"])))
            off_topic: list[list[float]] = []
            off_bytes: list[float] = []
            for question in OFF_TOPIC:
                _, body = await _timed(client, "/v1/context", {"scope": scope, "query": question})
                off_topic.append([float(item["relevance"]) for item in body["memories"]])
                off_bytes.append(float(len(json.dumps(body))))
        return {
            "series": {name: stats(values) for name, values in latencies.items()},
            "stages_ms": {name: _stage_summary(rows) for name, rows in stage_rows.items()},
            "context_response_bytes": stats([float(n) for n in sizes]),
            "relevance_floor": _floor_analysis(floor_rows),
            "off_topic": {
                "memories_packed_mean": round(statistics.fmean(len(r) for r in off_topic), 1),
                "response_bytes_mean": round(statistics.fmean(off_bytes)),
                "kept_at_floor": {
                    f"{floor:.2f}": round(
                        statistics.fmean(sum(s >= floor for s in r) for r in off_topic), 1
                    )
                    for floor in FLOORS
                },
            },
            "corpus": {
                "documents": copies * len(golden.documents),
                "turns": len(history),
                "tools": len(TOOLS),
                "seed_seconds": seed_seconds,
            },
            "requests": requests,
            "transport": "in-process ASGI (no network hop)",
            "providers": {
                "embedding": container.embedding.fingerprint(),
                "search": BENCH.search,
                "cache": BENCH.cache,
                "authorization": BENCH.authorization,
                "llm": "off",
            },
        }
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    payload = asyncio.run(run(args.copies, args.turns, args.requests))
    payload["label"] = args.label
    payload["provenance"] = provenance()
    path = write_result(f"overhaul/context_latency_{args.label}.json", payload)
    print(f"wrote {path}")
    for name, row in payload["series"].items():
        print(f"{name:14} p50 {row['p50']:8.1f} ms   p95 {row['p95']:8.1f} ms")


if __name__ == "__main__":
    main()
