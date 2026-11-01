"""Mode (c): an agent wrapped by agent-harness answers, with the real model, and the final
answer is scored for the same assertions as the memory service's context.

    ../agent-harness/.venv/bin/python -m benchmark.capabilities.harness_mode --sample harness

Runs in agent-harness's environment (it imports ``trellis.harness``). The scenarios are
written through the memory SDK into the eval tenant exactly as ``runner`` writes them (the
transcript a past session would have left); then each question is one harness run of a plain
function agent: the harness pushes the memory context (``agent.context``) and the function asks
the gateway's model to answer from it. Each question has a scope of its own — its own user
(:func:`question_user`), the scenario written again for it — so what its run records (the
transcript, the outcome) lands in that user's memory and never in another question's: the
memory service reads a user's memories for that user only, and an agent's own notes are
bound to the user it runs for. Grounding sampling is off (``without={"grounding"}`` on the
wrap).

The model on a CPU is slow, so this mode runs a named, deterministic sample
(``data/_modes.json``); every run is one model call and is timed end to end.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from collections.abc import Sequence
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
    identity,
    owned_by,
    parse,
    run_day,
    settle,
    thread_name,
    write_scenario,
)
from benchmark.capabilities.scoring import items_from_answer, preview, score

#: the agent's instruction; the memory context arrives as the harness pushes it. What the
#: user, their work and their records are comes from the context only; general know-how a
#: request needs (a recipe, installing a VPN client, what to do on a free afternoon in London)
#: is the model's own, shaped by what the context says of the user - the implicit and
#: procedural questions ask for exactly that, and "only the memory context" made the model
#: refuse them with the trait in front of it
SYSTEM = (
    "You are an operations assistant for the user. Answer the user's request using the "
    "memory context below for everything about the user, their work and their records. If "
    "the context does not hold a code, number, name or date about them that the request "
    "needs, say that you do not have that information on file - never guess one. For "
    "general know-how the request needs, use your own knowledge, shaped by what the context "
    "says about the user. Follow any standing instructions or preferences the context "
    "shows.\n\n# Memory context\n"
)
#: room for a whole report or table: at 400, 20 of 107 answers were cut mid-way (a service
#: report lost the signature line its standing rule ends it with)
MAX_TOKENS = 1024
RUN_TIMEOUT = 600
#: how long to wait for the model to answer a one-token request before asking anything
MODEL_READY_SECONDS = 900
#: how this mode names its users: one per question (``runner`` names one per scenario-language)
IDENTITIES = "question-language/v1"


def question_user(case: Case) -> str:
    """The user one question is asked as, and its scenario written as: its own, so what an
    earlier question's run recorded is never in its memory (stable across runs, opaque)."""
    return identity(case.key, case.language)


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
    today = run_day(settings)
    client = MemoryClient(settings.url, api_key=token, timeout=settings.timeout)
    service = Service(client, settings.worker_metrics_url)
    # one scope per question: the scenario written for each question's own user
    work = [
        (s, lang, question_user(case), case)
        for s in scenarios
        for lang in languages_of(s, settings.selection)
        for case in cases(s, lang)
    ]
    if len({user for _, _, user, _ in work}) != len(work):
        raise SystemExit("two questions share a user: their runs would read each other's records")
    sem = asyncio.Semaphore(settings.write_concurrency)

    async def _write(s: Any, lang: str, user: str, _case: Case) -> Any:
        async with sem:
            w = await write_scenario(service, s, lang, user, today, settings)
            written.append(w)
            return w

    written: list[Any] = []
    records: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {"purged": False}
    try:
        queues_at_start = await service.queue_depths()
        print(f"worker queues at start: {queues_at_start}", file=sys.stderr)
        t0 = time.perf_counter()
        # in scenario order (the asking order), not completion order
        written[:] = await asyncio.gather(*(_write(*x) for x in work))
        last_write = time.time()
        write_seconds = time.perf_counter() - t0
        signals = await settle(service, written, last_write, settings.probe_seconds)
        print(f"wrote {sum(w.turns for w in written)} turns; settled {signals}", file=sys.stderr)

        hs = HarnessSettings.from_env().model_copy(
            update={"memory_url": settings.url, "api_key": token}
        )
        harness_memory = MemoryClient(settings.url, api_key=token, timeout=settings.timeout)
        async with Harness(config=hs, memory=harness_memory) as h:
            assert h.gateway is not None, "the harness mode needs LLM_BASE_URL (the gateway)"
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
                    # grounding sampling off: the harness's switch for it (no setting holds it)
                    agents[agent_id] = h.wrap(
                        answer,
                        id=agent_id or "capeval-agent",
                        run_timeout=RUN_TIMEOUT,
                        without={"grounding"},
                    )
                return agents[agent_id]

            # the answer latency includes the model, which on a laptop shares the CPU: the load
            # is recorded so a run beside another heavy job can be told apart
            async with LoadSampler() as load:
                for (*_, case), w in zip(work, written, strict=True):
                    records.append(await _ask(agent_for(w.scenario.agent), w, case, settings))
                    r = records[-1]
                    print(
                        f"{r['key']}: {'PASS' if r['passed'] else 'FAIL ' + str(r['reason'])} "
                        f"({r['latency_ms'] / 1000:.1f}s)",
                        file=sys.stderr,
                    )
    finally:
        # a failed run purges what it wrote and gives its key back, as a finished one does
        if should_purge(settings) and written:
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
        "identities": IDENTITIES,
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
    thread = thread_name(w.user_id, f"h-{hashlib.sha1(case.key.encode()).hexdigest()[:10]}")
    # the run records into this thread (its transcript): the purge deletes it with the rest
    w.threads.append(thread)
    t = time.perf_counter()
    error = None
    try:
        result = await agent.run(case.text, user=w.user_id, thread=thread)
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
        v.update(passed=False, reason="error", abstained=None)
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
    model = os.environ.get("CAPEVAL_ANSWER_MODEL") or os.environ.get("TRELLIS_LIVE_MODEL") or ""
    if not model:
        raise SystemExit("harness mode needs the answer model in CAPEVAL_ANSWER_MODEL")
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
