"""Run the accumulated-memory stress set against a memory service, no model anywhere.

    python -m benchmark.capabilities.stress.run --run-date 2026-10-18 --out stress.json

Writes every script's earlier runs and its final run for one user per (script, language), waits
on the service's own readiness signals (``runner.settle``), then asks each final-run question
in the final run's own thread (``same_session``: the same chat, no window sent - the framework
keeps its own) and in a new thread (``new_session``), and scores the context
(``judge.py``). It needs a store of its own: run it under ``isolated_stack.sh`` (the
``run_isolated.sh`` wrapper does), which creates and drops a private database, Qdrant, Redis db.

Deterministic: users and threads are named from the script and language (``runner.identity``),
the values from ``--seed``, the dates from ``--run-date``; two runs of one service tree with the
same arguments write the same thing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from benchmark.capabilities import runner
from benchmark.capabilities.dataset import Case, Selection
from benchmark.capabilities.scoring import items_from_bundle, preview
from benchmark.capabilities.stress import build
from benchmark.capabilities.stress.judge import judge
from benchmark.capabilities.stress.report import summarize, to_markdown

PLACEMENTS = ("same_session", "new_session")


def _service_rev() -> str | None:
    root = os.environ.get("CAPEVAL_ROOT")
    if not root:
        return None
    try:
        rev = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "diff", "--quiet"], check=False).returncode != 0
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


async def _ask(
    service: runner.Service, w: runner.Written, b: build.Built, case: Case, request_timeout: float
) -> list[dict[str, Any]]:
    e = build.expect(b, case.query.id, case.language)
    out = []
    for placement in PLACEMENTS:
        if placement == "new_session":
            thread = runner.thread_name(
                w.user_id, f"ask-{hashlib.sha1(case.key.encode()).hexdigest()[:10]}"
            )
            window = True
        else:
            thread = w.threads[-1]  # the final run's conversation
            window = False
        ctx = runner._ctx(service.client, w, thread)
        rec: dict[str, Any] = {
            "key": case.key,
            "script": b.id,
            "domain": b.script["domain"],
            "query": case.query.id,
            "category": e.category,
            "split": case.query.split,
            "kind": case.phrasing.kind,
            "language": case.language,
            "placement": placement,
            "question": case.text,
            "expect": {"contains": list(e.contains), "stale": len(e.stale), "anchor": e.anchor},
        }
        t = runner._now()
        try:
            bundle = await ctx.context(
                case.text,
                token_budget=runner.TOKEN_BUDGET,
                window=window,
                format="full",
                timeout=request_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - a failed request is a failed answer, recorded
            rec.update(
                passed=False,
                reason="error",
                latency_ms=0.0,
                evidence_status=None,
                preview=[f"{type(exc).__name__}: {exc}"[:200]],
            )
            out.append(rec)
            continue
        latency = (runner._now() - t) * 1000
        items = items_from_bundle(bundle.model_dump(mode="json"))
        j = judge(e, items, bundle.evidence_status)
        rec.update(j.as_dict())
        rec.update(
            latency_ms=round(latency, 2),
            evidence_status=bundle.evidence_status,
            preview=[] if j.passed else preview(items, n=4, width=120),
        )
        out.append(rec)
    return out


async def run(a: argparse.Namespace) -> dict[str, Any]:
    built = build.build(a.prior_runs, a.seed)
    if a.scripts:
        built = [b for b in built if b.id in a.scripts]
    langs = tuple(a.languages)
    settings = runner.Settings(
        url=a.url,
        mode="nollm",
        tenant=a.tenant,
        admin_key=a.admin_key,
        api_key=None,
        worker_metrics_url=a.worker_metrics_url,
        selection=Selection(languages=langs),
        write_concurrency=a.write_concurrency,
        timeout=a.timeout,
        run_date=date.fromisoformat(a.run_date) if a.run_date else None,
    )
    today = runner.run_day(settings)
    factory = runner.default_factory(settings)
    token, policy = await runner._tenant(settings, factory)
    client = factory(token)
    service = runner.Service(client, settings.worker_metrics_url)
    work = [(b, build.scenario(b), lang) for b in built for lang in langs]
    records: list[dict[str, Any]] = []
    try:
        version = await client.version()
        sem = asyncio.Semaphore(a.write_concurrency)

        async def _write(b: build.Built, s: Any, lang: str) -> tuple[build.Built, runner.Written]:
            async with sem:
                user = runner.identity(s.id, lang)
                try:
                    return b, await runner.write_scenario(service, s, lang, user, today, settings)
                except Exception as exc:  # noqa: BLE001 - recorded; its questions still asked
                    print(f"write failed {s.id}/{lang}: {exc!r}", file=sys.stderr)
                    w = runner.Written(s, lang, user, error=f"{type(exc).__name__}: {exc}"[:300])
                    w.threads.append(runner.thread_name(user, "unwritten"))
                    return b, w

        t0 = runner._now()
        written = list(await asyncio.gather(*(_write(*x) for x in work)))
        write_seconds = runner._now() - t0
        turns = sum(w.turns for _, w in written)
        print(f"wrote {turns} turns in {write_seconds:.0f}s", file=sys.stderr)
        signals = await runner.settle(
            service,
            [w for _, w in written],
            time.time(),
            a.probe_seconds,
            a.enrich_seconds,
            a.settle_seconds,
        )
        print(f"settled: {signals}", file=sys.stderr)
        qsem = asyncio.Semaphore(a.query_concurrency)

        async def _one(b: build.Built, w: runner.Written, case: Case) -> None:
            async with qsem:
                records.extend(await _ask(service, w, b, case, a.timeout))

        t1 = runner._now()
        await asyncio.gather(
            *(
                _one(b, w, Case(w.scenario, q, p, w.language))
                for b, w in written
                for q in w.scenario.queries
                if not a.split or q.split == a.split
                for p in q.phrasings
            )
        )
        ask_seconds = runner._now() - t1
        llm_calls = await service.llm_calls()
    finally:
        await client.aclose()
        await runner._revoke(settings, factory, policy)
    records.sort(key=lambda r: (r["key"], r["placement"]))
    meta = {
        "set": "stress_accumulated",
        "mode": "nollm",
        "service_version": version,
        "service_rev": _service_rev(),
        "service_root": os.environ.get("CAPEVAL_ROOT"),
        "run_date": today.date().isoformat(),
        "seed": a.seed,
        "prior_runs": a.prior_runs,
        "languages": list(langs),
        "scripts": [b.id for b in built],
        "split": a.split or "all",
        "placements": list(PLACEMENTS),
        "token_budget": runner.TOKEN_BUDGET,
        "identities": runner.IDENTITIES,
        "policy": {k: v for k, v in policy.items() if k not in ("key_id", "admin_key")},
        "ingestion": {
            "turns": turns,
            "jobs": sum(w.jobs for _, w in written),
            "failed_jobs": sum(w.failed_jobs for _, w in written),
            "failed_writes": [
                f"{w.scenario.id}/{w.language}: {w.error}" for _, w in written if w.error
            ],
            "write_seconds": round(write_seconds, 1),
            "settle": signals,
        },
        "ask_seconds": round(ask_seconds, 1),
        "llm_calls": llm_calls,
    }
    return {"meta": meta, "summary": summarize(records), "records": records}


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=os.environ.get("MEMORY_URL", "http://localhost:8091"))
    p.add_argument("--tenant", default="capeval-stress")
    p.add_argument("--admin-key", default=os.environ.get("CAPEVAL_ADMIN_KEY"))
    p.add_argument("--worker-metrics-url", default=os.environ.get("MEMORY_WORKER_METRICS_URL"))
    p.add_argument("--languages", default="en,de,es,ar,hi")
    p.add_argument("--scripts", default=None, help="script ids (default all)")
    p.add_argument("--split", choices=("dev", "heldout"), default=None)
    p.add_argument("--prior-runs", type=int, default=build.DEFAULT_PRIOR_RUNS)
    p.add_argument("--seed", default=build.DEFAULT_SEED)
    p.add_argument("--run-date", default=os.environ.get("CAPEVAL_RUN_DATE"))
    p.add_argument("--write-concurrency", type=int, default=4)
    p.add_argument("--query-concurrency", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--probe-seconds", type=float, default=runner.PROBE_SECONDS)
    p.add_argument("--enrich-seconds", type=float, default=runner.ENRICH_SECONDS)
    p.add_argument("--settle-seconds", type=float, default=runner.SETTLE_SECONDS)
    p.add_argument("--out", required=True, help="result .json (a .md summary is written beside it)")
    a = p.parse_args(argv)
    a.languages = [x.strip() for x in a.languages.split(",") if x.strip()]
    a.scripts = [x.strip() for x in a.scripts.split(",")] if a.scripts else None
    return a


def main(argv: Sequence[str] | None = None) -> int:
    a = parse(argv)
    payload = asyncio.run(run(a))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(payload), encoding="utf-8")
    o = payload["summary"]["overall"]
    print(
        f"wrote {out}: pass {o['pass_rate']} stale_leak {o['stale_leak_rate']} "
        f"false_refusal {o['false_refusal_rate']} correct_refusal {o['correct_refusal_rate']} (n={o['n']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
