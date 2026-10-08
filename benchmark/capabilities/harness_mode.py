"""Mode (c): an agent wrapped by agent-harness answers, with the real model, and the final
answer is scored for the same assertions as the memory service's context.

    ../agent-harness/.venv/bin/python -m benchmark.capabilities.harness_mode --sample harness

Runs in agent-harness's environment (it imports ``trellis.harness``). The scenarios are
written through the memory SDK into the eval tenant exactly as ``runner`` writes them (the
transcript a past session would have left); then each question is one harness run of a plain
function agent: the harness pushes the memory context (``agent.context``) and the function asks
the gateway's model to answer from it. Runs record nothing back (``without={"records"}``), so
one question's answer never becomes the next question's memory, and grounding sampling is off.

The model on a CPU is slow, so this mode runs a named, deterministic sample
(``data/_modes.json``); every run is one model call and is timed end to end.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.capabilities.cleanup import purge, should_purge
from benchmark.capabilities.dataset import (
    Case,
    cases,
    languages_of,
    load_dataset,
    load_json,
    select,
)
from benchmark.capabilities.runner import (
    LoadSampler,
    Service,
    Settings,
    _revoke,
    _tenant,
    default_factory,
    owned_by,
    parse,
    settle,
    write_scenario,
)
from benchmark.capabilities.scoring import items_from_answer, preview, score

#: the agent's instruction; the memory context arrives as the harness pushes it
SYSTEM = (
    "You are an operations assistant for the user. Answer the user's request using only the "
    "memory context below. If the context does not contain what the request needs, say that "
    "you do not have that information on file - never guess a code, number, name or date. "
    "Follow any standing instructions or preferences the context shows.\n\n# Memory context\n"
)
MAX_TOKENS = 400
RUN_TIMEOUT = 600
#: how long to wait for the model to answer a one-token request before asking anything
MODEL_READY_SECONDS = 900


async def wait_for_model(
    gateway: Any, model: str, deadline_s: float = MODEL_READY_SECONDS
) -> float:
    """The model's own readiness signal: a one-token completion that succeeds. A model that is
    still loading (or a gateway whose breaker is open) answers with an error; retry with
    backoff until it answers or the deadline passes. Returns the seconds waited."""
    t0 = time.monotonic()
    delay = 2.0
    last: BaseException | None = None
    while time.monotonic() - t0 < deadline_s:
        try:
            await gateway.complete(
                [{"role": "user", "content": "ok"}], model=model, max_tokens=1, temperature=0
            )
            return time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001 - not ready yet; the deadline bounds it
            last = exc
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise RuntimeError(f"the model {model} did not answer within {deadline_s:.0f} s: {last!r}")


def _refusals(lang: str) -> list[str]:
    markers = load_json("_languages.json")["refusal"]
    return list(markers.get(lang, markers["en"])) + (markers["en"] if lang != "en" else [])


async def run(settings: Settings, model: str) -> dict[str, Any]:
    # agent-harness's package: this mode runs in its environment, not this repository's
    from trellis import Harness, Runtime  # pyright: ignore[reportAttributeAccessIssue]
    from trellis import Settings as HarnessSettings  # pyright: ignore[reportAttributeAccessIssue]
    from trellis.memory import MemoryClient

    dataset = load_dataset()
    scenarios = select(dataset, settings.selection)
    factory = default_factory(settings)
    token, policy = await _tenant(settings, factory)
    today = datetime.now(UTC)
    client = MemoryClient(settings.url, api_key=token, timeout=settings.timeout)
    service = Service(client, settings.worker_metrics_url)
    work = [
        (s, lang, f"ce{settings.run_id}-{i:03d}-{lang}")
        for i, s in enumerate(scenarios)
        for lang in languages_of(s, settings.selection)
    ]
    sem = asyncio.Semaphore(settings.write_concurrency)

    async def _write(s: Any, lang: str, user: str) -> Any:
        async with sem:
            return await write_scenario(service, s, lang, user, today, settings)

    queues_at_start = await service.queue_depths()
    print(f"worker queues at start: {queues_at_start}", file=sys.stderr)
    t0 = time.perf_counter()
    written = await asyncio.gather(*(_write(*x) for x in work))
    last_write = time.time()
    write_seconds = time.perf_counter() - t0
    signals = await settle(service, written, last_write, settings.probe_seconds)
    print(f"wrote {sum(w.turns for w in written)} turns; settled {signals}", file=sys.stderr)

    hs = HarnessSettings.from_env().model_copy(
        update={"memory_url": settings.url, "api_key": token, "grounding_sample": 0.0}
    )
    harness_memory = MemoryClient(settings.url, api_key=token, timeout=settings.timeout)
    records: list[dict[str, Any]] = []
    async with Harness(config=hs, memory=harness_memory) as h:
        assert h.gateway is not None, "the harness mode needs BIFROST_URL (the gateway)"
        gateway = h.gateway
        model_wait = await wait_for_model(gateway, model)
        print(f"model {model} answered after {model_wait:.1f} s", file=sys.stderr)

        async def answer(question: str, agent: Runtime) -> str:
            messages = [
                {"role": "system", "content": SYSTEM + (agent.context or "(nothing on file)")},
                {"role": "user", "content": question},
            ]
            reply = await gateway.complete(
                messages, model=model, max_tokens=MAX_TOKENS, temperature=0
            )
            return str(
                ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            )

        agents: dict[str | None, Any] = {}

        def agent_for(agent_id: str | None) -> Any:
            if agent_id not in agents:
                agents[agent_id] = h.wrap(
                    answer, id=agent_id or "capeval-agent", timeout=RUN_TIMEOUT
                )
            return agents[agent_id]

        # the answer latency includes the model, which on a laptop shares the CPU: the load
        # is recorded so a run beside another heavy job can be told apart
        async with LoadSampler() as load:
            for w in written:
                for case in cases(w.scenario, w.language):
                    records.append(await _ask(agent_for(w.scenario.agent), w, case, settings))
                    r = records[-1]
                    print(
                        f"{r['key']}: {'PASS' if r['passed'] else 'FAIL ' + str(r['reason'])} "
                        f"({r['latency_ms'] / 1000:.1f}s)",
                        file=sys.stderr,
                    )
    cleanup: dict[str, Any] = {"purged": False}
    if should_purge(settings):
        done = await purge(client, owned_by(written).items())
        cleanup = {"purged": True, **done.as_dict()}
        print(f"cleanup: {cleanup}", file=sys.stderr)
    await client.aclose()
    await _revoke(settings, factory, policy)

    from benchmark.capabilities.report import build, save_raw

    meta = {
        "mode": "harness",
        "tenant": settings.tenant,
        "policy": {k: v for k, v in policy.items() if k not in ("key_id", "admin_key")},
        "answer_model": model,
        "memory_url": settings.url,
        "run_id": settings.run_id,
        "run_date": today.date().isoformat(),
        "selection": settings.selection.describe(),
        "sample": settings.sample,
        "scale": settings.scale,
        "placements": ["new_session"],
        "surfaces": ["answer"],
        "model_ready_seconds": round(model_wait, 1),
        "host_load_during_asks": load.summary(),
        "cleanup": cleanup,
        "written": owned_by(written),
        "answer_errors": sum(1 for r in records if r["reason"] == "error"),
        "ingestion": {
            "queues_at_start": queues_at_start,
            "turns": sum(w.turns for w in written),
            "jobs": sum(w.jobs for w in written),
            "failed_jobs": sum(w.failed_jobs for w in written),
            "write_seconds": round(write_seconds, 1),
            "settle": signals,
        },
    }
    if settings.out is not None:
        if meta["answer_errors"]:
            # a run the model failed is not a measurement: keep its records for inspection,
            # beside (never as) the result the baseline reads
            failed = settings.out.with_name(settings.out.stem + ".failed.json")
            save_raw(records, meta, failed)
        else:
            save_raw(records, meta, settings.out)
    return build(dataset, records, meta)


async def _ask(agent: Any, w: Any, case: Case, settings: Settings) -> dict[str, Any]:
    expect = w.expects.get(case.query.id, case.query.expect)
    thread = f"ce-{settings.run_id}-{w.user_id}-h-{uuid.uuid4().hex[:8]}"
    t = time.perf_counter()
    error = None
    try:
        result = await agent.run(case.text, user=w.user_id, thread=thread, without={"records"})
        text = str(result.answer or "")
        if result.error:
            error = str(result.error)[:200]
    except Exception as exc:  # noqa: BLE001 - a failed run is a failed answer, recorded
        text, error = "", f"{type(exc).__name__}: {exc}"[:200]
    latency = (time.perf_counter() - t) * 1000
    items = items_from_answer(text)
    v = score(
        expect, items, case.language, surface="answer", refusal_markers=_refusals(case.language)
    ).as_dict()
    if error is not None:
        # the model call failed: an error, not the agent declining to answer
        v.update(passed=False, reason="error")
    s = case.scenario
    return {
        "key": case.key,
        "capability": s.capability,
        "scenario": s.id,
        "origin": s.origin,
        "domain": s.domain,
        "query": case.query.id,
        "split": case.query.split,
        "kind": case.phrasing.kind,
        "language": case.language,
        "question": case.text,
        "placement": "new_session",
        "surface": "answer",
        "latency_ms": round(latency, 2),
        "evidence_status": None,
        **v,
        "answer": text[:1500],
        "error": error,
        "preview": [] if v["passed"] else preview(items, n=1, width=300),
        "corpus_tokens": w.corpus_tokens or None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from benchmark.capabilities.report import write

    argv = list(sys.argv[1:] if argv is None else argv)
    model = (
        os.environ.get("CAPEVAL_ANSWER_MODEL")
        or os.environ.get("TRELLIS_LIVE_MODEL")
        or "granite/granite-3.3-2b"
    )
    settings, args = parse(["--mode", "nollm", *argv])
    settings.mode = "harness"
    settings.tenant = args.tenant or "capeval-harness"
    settings.out = settings.out or Path("benchmark/results/capabilities/harness.json")
    payload = asyncio.run(run(settings, model))
    out = settings.out or Path("benchmark/results/capabilities/harness.json")
    errors = payload["meta"].get("answer_errors") or 0
    if errors:
        out = out.with_name(out.stem + ".failed.json")
    json_path, md_path = write(payload, out)
    print(f"wrote {json_path} and {md_path}")
    print(
        f"harness: macro accuracy {payload['summary']['macro_accuracy']}, answer p50/p95 "
        f"{payload['summary']['latency_ms'].get('answer_p50')}/{payload['summary']['latency_ms'].get('answer_p95')} ms"
    )
    if errors:
        print(f"{errors} model calls failed: not a baseline (kept as {json_path})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
