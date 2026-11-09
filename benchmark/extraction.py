"""What the pipeline remembers from a conversation, scored memory by memory.

The capability eval scores what a question gets back. This scores what was stored: each case
(``benchmark/capabilities/data/extraction_hard.json``) is a short conversation, written turn
by turn through the conversation API in process, then read back from the memory table:

* ``expect``: facts that must be held, each as the words one current memory read from the
  conversation (not the turn as said) must all contain;
* ``forbid``: false facts no current memory may hold (a turn as said included);
* ``stale``: an earlier value an update or a contradiction must have closed or marked;
* ``keep``: a memory a restatement or an unrelated statement must not have closed;
* ``ask``: a question whose context must carry every word given (multi-hop, temporal);
* ``negative``: chit-chat - nothing may be read from it.

Metrics: recall (expected facts held), false-fact rate (forbidden facts held, and negative
cases that stored a reading), duplicate rate (expected facts held by more than one memory),
wrong-update rate (stale values still current, kept values closed), ask recall.

    BENCH_LLM=on ... python -m benchmark.extraction --uses contextual_extraction --out b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

from benchmark.common import reset_store, write_result
from benchmark.env import JUDGE_USES, bench_overrides, pin_model_policy
from benchmark.retrieval import _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.llm.cost import llm_accounting

DATA = Path(__file__).parent / "capabilities" / "data" / "extraction_hard.json"
TENANT = "extract-eval"
#: every case is said relative to this day, so a run is a function of the data alone
ANCHOR = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ASK_BUDGET = 2000


def _holds(content: str, words: list[str]) -> bool:
    """Every word starts a word of ``content`` (a stem matches its forms: "renew" in
    "renewal"; "art" is not in "started")."""
    folded = content.casefold()
    return all(re.search(r"(?<!\w)" + re.escape(w.casefold()), folded) for w in words)


async def _write(container: Any, case: dict[str, Any]) -> MemoryExecutionContext:
    conversation = container.services["conversation"]
    ctx = MemoryExecutionContext(tenant_id=TENANT, user_id=case["id"], workspace_id="eval")
    thread = new_id("thread")
    turn_id = new_id("turn")
    for n, turn in enumerate(case["turns"]):
        role = MessageRole.ASSISTANT if turn["role"] == "assistant" else MessageRole.USER
        if role is MessageRole.USER and n:
            turn_id = new_id("turn")
        at = ANCHOR + timedelta(days=turn.get("day", 0), minutes=n)
        async with container.services["uow_factory"]() as uow:
            await conversation.append_message(
                uow,
                ctx.model_copy(update={"thread_id": thread, "turn_id": turn_id}),
                role=role,
                content=turn["text"],
                occurred_at=at,
                source_system="extraction-eval",
                source_message_id=f"{case['id']}:{n}",
            )
            await uow.commit()
        await container.tasks.drain()
    return ctx


async def _memories(container: Any, user_id: str) -> list[dict[str, Any]]:
    async with container.database.engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT content, memory_type, temporal_status, confidence, "
                "system_metadata->>'category' AS category, system_metadata ? 'replaced_by' "
                "AS marked FROM memories WHERE tenant_id = :t AND owner_principal = 'user:' || :u "
                "AND deleted_at IS NULL"
            ),
            {"t": TENANT, "u": user_id},
        )
        return [dict(r._mapping) for r in rows]


def score_case(case: dict[str, Any], mems: list[dict[str, Any]], asked: list[bool]) -> dict:
    current = [m for m in mems if m["temporal_status"] == "CURRENT" and not m["marked"]]
    read = [m for m in current if m["category"] != "verbatim_turn"]
    held = [sum(_holds(m["content"], words) for m in read) for words in case.get("expect", [])]
    return {
        "id": case["id"],
        "kind": case["kind"],
        "expected": len(held),
        "held": sum(1 for h in held if h),
        "duplicated": sum(1 for h in held if h > 1),
        "forbidden": len(case.get("forbid", [])),
        "false": sum(
            1 for words in case.get("forbid", []) if any(_holds(m["content"], words) for m in read)
        ),
        "negative_stored": bool(case.get("negative")) and bool(read),
        "stale": len(case.get("stale", [])),
        "stale_current": sum(
            1 for words in case.get("stale", []) if any(_holds(m["content"], words) for m in read)
        ),
        "keep": len(case.get("keep", [])),
        "keep_lost": sum(
            1
            for words in case.get("keep", [])
            if not any(_holds(m["content"], words) for m in current)
        ),
        "asks": len(asked),
        "asks_hit": sum(asked),
        "readings": [m["content"] for m in read],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def ratio(num: str, den: str, *, where: Any = None) -> float | None:
        pick = [r for r in rows if where is None or where(r)]
        d = sum(r[den] for r in pick)
        return round(sum(r[num] for r in pick) / d, 4) if d else None

    negatives = [r for r in rows if r["kind"] == "chitchat"]
    false_den = sum(r["forbidden"] for r in rows) + len(negatives)
    false_num = sum(r["false"] for r in rows) + sum(r["negative_stored"] for r in negatives)
    kinds = sorted({r["kind"] for r in rows})
    return {
        "cases": len(rows),
        "recall": ratio("held", "expected"),
        "false_fact_rate": round(false_num / false_den, 4) if false_den else None,
        "duplicate_rate": ratio("duplicated", "held"),
        "wrong_update_rate": round(
            (sum(r["stale_current"] for r in rows) + sum(r["keep_lost"] for r in rows))
            / max(1, sum(r["stale"] + r["keep"] for r in rows)),
            4,
        ),
        "ask_recall": ratio("asks_hit", "asks"),
        "by_kind": {
            k: {
                "recall": ratio("held", "expected", where=lambda r, k=k: r["kind"] == k),
                "ask_recall": ratio("asks_hit", "asks", where=lambda r, k=k: r["kind"] == k),
            }
            for k in kinds
        },
    }


async def run(uses: list[str], only: list[str] | None) -> dict[str, Any]:
    cases = json.loads(DATA.read_text())["cases"]
    if only:
        cases = [c for c in cases if c["id"] in only or c["kind"] in only]
    container = await build_container(_settings(), __version__, overrides=bench_overrides())
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    messages = 0
    try:
        register_handlers(container)
        await reset_store(container, TENANT)
        await pin_model_policy(container, TENANT, [*JUDGE_USES, *uses])
        builder = container.services["context_builder"]
        with llm_accounting() as tokens:
            for case in cases:
                ctx = await _write(container, case)
                messages += len(case["turns"])
                asked = []
                for ask in case.get("ask", []):
                    bundle = await builder.build(ctx, ask["q"], token_budget=ASK_BUDGET)
                    asked.append(_holds(bundle.render() or "", ask["expect"]))
                rows.append(score_case(case, await _memories(container, case["id"]), asked))
                print(f"[extraction] {case['id']}", file=sys.stderr, flush=True)
    finally:
        await container.close()
    seconds = time.perf_counter() - started
    return {
        "uses": uses,
        "summary": summarise(rows),
        "messages": messages,
        "messages_per_second": round(messages / seconds, 3),
        "llm_tokens": {"input": tokens.input, "output": tokens.output},
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uses", default="", help="model uses beside the judge, comma-separated")
    parser.add_argument("--only", default="", help="case ids or kinds, comma-separated")
    parser.add_argument("--out", default="extraction.json")
    args = parser.parse_args()
    out = asyncio.run(
        run([u for u in args.uses.split(",") if u], [o for o in args.only.split(",") if o])
    )
    write_result(args.out, out)
    sys.stdout.write(json.dumps(out["summary"], indent=2) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
