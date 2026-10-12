"""Drive the real memory service through the SDK: write each scenario, wait until the service
says it has processed and indexed it, ask every phrasing, score what comes back.

    python -m benchmark.capabilities.runner --mode nollm [--languages en,de] [--split dev]

Modes differ only in the eval tenant's model policy (``data/_modes.json``): ``nollm`` runs no
model at all, ``llm`` the policy's uses through the gateway. ``harness`` is
``benchmark.capabilities.harness_mode`` (an agent wrapped by agent-harness answers).

Determinism: what a run writes is a function of the dataset, the selection and the run date
alone. Every user and thread is named from its scenario and language (``identity``), never
from anything drawn per run: the service indexes a memory's subject with its text, so a
random user id moved every cosine by a few hundredths and two runs of the same code disagreed
on cells near the relevance floor. The run date (``--run-date``, default the UTC date the run
starts) anchors every session and is recorded in the result.

Isolation: since the names repeat from run to run, a run needs a store of its own - the
isolated stack's (``isolated_stack.sh``: a fresh database, Qdrant, Redis db per run), or at
least a tenant never used before. The eval tenant (``capeval-<mode>`` unless ``--tenant``) is
created by the run and a run refuses one that already exists (``--reuse`` aside); a service
key is issued for the run and revoked after it.

Readiness, never a blind sleep: (1) every write's processing job reaches a terminal state
(``GET /v1/jobs/{id}``, archive staging excluded - it is scheduled a minute out by design);
sessions of a scenario are written in time order, each after the previous one was processed,
as a conversation days apart would be; (2) the outbox is drained (``memory_outbox_backlog``)
and, when the worker's metrics are reachable, two consecutive samples taken after the last
write show no job waiting on the queues that index; (3) the read path itself: the last turn
of every scenario is searchable in its own thread.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import Any

from benchmark.capabilities.cleanup import purge, should_purge
from benchmark.capabilities.dataset import (
    ALL_LANGUAGES,
    Case,
    Dataset,
    Scenario,
    Selection,
    cases,
    languages_of,
    load_dataset,
    load_json,
    select,
    text_for,
)
from benchmark.capabilities.generate import (
    GeneratedTurn,
    aggregate,
    aggregate_expect,
    build_corpus,
    expand_repeat,
    scale_tokens,
)
from benchmark.capabilities.scoring import (
    items_from_bundle,
    items_from_search,
    preview,
    scorable,
    score,
)

#: the agent's push, as agent-harness asks for it (docs/USAGE.md section 14)
TOKEN_BUDGET = 2000
SEARCH_KINDS = ("memory", "episode", "message")
SEARCH_LIMIT = 10
#: one history.add carries at most this many turns
BATCH = 25
#: queues whose waiting jobs mean "not indexed yet"; archive staging is scheduled a minute
#: out on purpose and reconcile carries the periodic jobs
SETTLE_QUEUES = ("chat-fast", "memory-extract", "embedding", "graph", "summary")
#: the queues a memory must clear before it can be found
INDEX_QUEUES = ("chat-fast", "memory-extract", "embedding")
#: how long to wait for graph enrichment and summaries once everything is indexed
ENRICH_SECONDS = 1800.0
#: all read-path probes together
PROBES_TOTAL_SECONDS = 900.0
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "DEAD"}
#: how long one scenario's read-path probe may wait once the queues are idle
PROBE_SECONDS = 60.0
#: how long the whole readiness wait may take before the run proceeds and says so
SETTLE_SECONDS = 3600.0


def _now() -> float:
    return time.perf_counter()


@dataclass
class Written:
    """What writing one scenario in one language left behind."""

    scenario: Scenario
    language: str
    user_id: str
    threads: list[str] = field(default_factory=list)
    last_probe: str | None = None
    probe_thread: str | None = None
    turns: int = 0
    jobs: int = 0
    failed_jobs: int = 0
    write_seconds: float = 0.0
    #: why writing it failed, when it did (its questions are still asked and scored)
    error: str | None = None
    #: set for generated corpora: the expectations of the aggregate queries
    expects: dict[str, dict[str, Any]] = field(default_factory=dict)
    corpus_tokens: int = 0


@dataclass
class Settings:
    url: str
    mode: str
    tenant: str
    admin_key: str | None
    api_key: str | None
    worker_metrics_url: str | None
    selection: Selection
    scale: str = "standard"
    write_concurrency: int = 4
    query_concurrency: int = 1
    placements: tuple[str, ...] = ("new_session", "same_session")
    surfaces: tuple[str, ...] = ("context", "search")
    timeout: float = 120.0
    #: how long one scenario's read-path probe may wait once the queues are idle
    probe_seconds: float = PROBE_SECONDS
    #: the named sample (data/_modes.json) the selection came from, if any
    sample: str | None = None
    #: ask again what an earlier run (its ``.raw.json.gz``) wrote, without writing it again
    reuse: Path | None = None
    #: leave the run's writes in the tenant (for a later ``--reuse``); by default the run
    #: forgets them when it ends (``cleanup.purge``)
    keep: bool = False
    #: how long to wait for graph enrichment and summaries once everything is indexed
    enrich_seconds: float = ENRICH_SECONDS
    out: Path | None = None
    #: the date every session is anchored to (``Session.day`` is relative to it); None is the
    #: UTC date the run starts. Recorded as ``meta.run_date``.
    run_date: date | None = None
    #: names this run's service key and log lines only - never a user, thread or anything the
    #: service stores and indexes (``identity``)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


#: the naming scheme ``identity`` implements, recorded with every run
IDENTITIES = "scenario-language/v1"


def identity(scenario_id: str, language: str) -> str:
    """The user a scenario is written as in one language: the same in every run, opaque like a
    production user id (the service indexes the subject with each memory's text, so a name
    that carried the scenario's words would lend its questions a match they did not earn)."""
    digest = hashlib.sha256(f"{scenario_id}/{language}".encode()).hexdigest()
    return f"ce{digest[:12]}"


def thread_name(user_id: str, name: str) -> str:
    """A thread of ``user_id``'s, named from the scenario's own session (or question) name."""
    return f"{user_id}-{name}"


def run_day(settings: Settings) -> datetime:
    """Midnight UTC of the run date: the anchor of every session and generated record."""
    day = settings.run_date or datetime.now(UTC).date()
    return datetime.combine(day, clock_time(0), tzinfo=UTC)


# --------------------------------------------------------------------------- the service


class Service:
    """The SDK client for the eval tenant, plus the two readiness signals that are not SDK
    verbs: a job's task name (to leave archive staging out) and the metrics endpoints."""

    def __init__(self, client: Any, worker_metrics_url: str | None) -> None:
        self.client = client
        self.worker_metrics_url = worker_metrics_url

    async def job(self, ctx: Any, job_id: str) -> dict[str, Any]:
        return await self.client.transport.request("GET", f"/v1/jobs/{job_id}", scope=ctx.scope)

    async def wait_jobs(self, ctx: Any, job_ids: Sequence[str], *, within: float = 900) -> int:
        """Wait for every processing job; returns how many ended other than SUCCEEDED."""
        pending = list(job_ids)
        failed = 0
        deadline = _now() + within
        delay = 0.1
        while pending:
            still = []
            for chunk in range(0, len(pending), 16):
                infos = await asyncio.gather(
                    *(self.job(ctx, j) for j in pending[chunk : chunk + 16])
                )
                for j, info in zip(pending[chunk : chunk + 16], infos, strict=True):
                    if str(info.get("task_name", "")).startswith("archive."):
                        continue
                    status = str(info.get("status"))
                    if status in TERMINAL:
                        failed += status != "SUCCEEDED"
                    else:
                        still.append(j)
            pending = still
            if pending:
                if _now() > deadline:
                    raise TimeoutError(f"{len(pending)} jobs still pending: {pending[:5]}")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 2.0)
        return failed

    async def outbox_backlog(self) -> float | None:
        return _metric(await self.client.metrics(), "memory_outbox_backlog")

    async def worker_sample(self) -> dict[str, float] | None:
        if not self.worker_metrics_url:
            return None
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as http:
                text = (await http.get(self.worker_metrics_url)).text
        except Exception:  # noqa: BLE001 - the signal is optional
            return None
        out = {"sampled_at": _metric(text, "memory_worker_last_sample_timestamp_seconds") or 0.0}
        for q in SETTLE_QUEUES:
            out[q] = _metric(text, "memory_queue_depth", queue=q) or 0.0
        return out

    async def queue_depths(self) -> dict[str, float] | None:
        """Every queue's depth on the worker, whoever's jobs they are: a backlog the eval
        inherits (another tenant's, or an earlier run's) is recorded with the run."""
        if not self.worker_metrics_url:
            return None
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as http:
                text = (await http.get(self.worker_metrics_url)).text
        except Exception:  # noqa: BLE001 - the signal is optional
            return None
        out: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith('memory_queue_depth{queue="'):
                queue = line.split('"', 2)[1]
                with contextlib.suppress(ValueError):
                    out[queue] = float(line.rsplit(" ", 1)[-1])
        return dict(sorted(out.items()))


def _metric(text: str, name: str, **labels: str) -> float | None:
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest[:1] not in ("{", " "):
            continue
        if labels and not all(f'{k}="{v}"' in rest for k, v in labels.items()):
            continue
        try:
            return float(rest.rsplit(" ", 1)[-1])
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- host load

#: Above this load per core while the questions were asked, the box was shared with other
#: work and the latency percentiles are an upper bound, not a measurement (the convention of
#: benchmark.common.provenance's host_load).
LOAD_PER_CORE_MAX = 1.0


class LoadSampler:
    """The machine's 1-minute load average, sampled every few seconds while the latency is
    measured. A latency taken while another heavy job ran is flagged, never silently kept:
    run every baseline under the machine-wide lock (scratchpad heavy.sh / ``HEAVY=``)."""

    def __init__(self, every: float = 5.0) -> None:
        self.every = every
        self.samples: list[float] = []
        self._task: asyncio.Task[None] | None = None

    async def _loop(self) -> None:
        while True:
            try:
                self.samples.append(os.getloadavg()[0])
            except (OSError, AttributeError):
                return
            await asyncio.sleep(self.every)

    async def __aenter__(self) -> LoadSampler:
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def summary(self) -> dict[str, Any]:
        cores = os.cpu_count() or 1
        peak = max(self.samples) if self.samples else None
        mean = sum(self.samples) / len(self.samples) if self.samples else None
        trustworthy = peak is not None and peak / cores <= LOAD_PER_CORE_MAX
        return {
            "samples": len(self.samples),
            "cpu_count": cores,
            "loadavg_1m_mean": round(mean, 2) if mean is not None else None,
            "loadavg_1m_max": round(peak, 2) if peak is not None else None,
            "max_load_per_core": round(peak / cores, 2) if peak is not None else None,
            "threshold_load_per_core": LOAD_PER_CORE_MAX,
            "latency_trustworthy": trustworthy,
        }


# --------------------------------------------------------------------------- writing


def _ctx(client: Any, w: Written, thread: str) -> Any:
    ctx = client.bind(user_id=w.user_id, thread_id=thread)
    return ctx.agent(w.scenario.agent) if w.scenario.agent else ctx


async def _add(service: Service, ctx: Any, turns: Sequence[GeneratedTurn], w: Written) -> None:
    from trellis.memory import Message

    job_ids: list[str] = []
    for i in range(0, len(turns), BATCH):
        batch = turns[i : i + BATCH]
        acks = await ctx.history.add(
            [Message(role=t.role, content=t.text, occurred_at=t.occurred_at) for t in batch]  # type: ignore[arg-type]
        )
        job_ids += [j for a in acks for j in a.job_ids]
    w.turns += len(turns)
    w.jobs += len(job_ids)
    w.failed_jobs += await service.wait_jobs(ctx, job_ids)


async def _tool_runs(client: Any, w: Written, session: Any, thread: str) -> None:
    base = client.bind(user_id=w.user_id, thread_id=thread)
    for run in session.tool_runs:
        task = text_for(run.task, w.language)
        for n in range(run.count):
            agent = base.agent(run.agent, agent_run_id=f"run-{w.user_id}-{session.id}-{n}")
            for step, call in enumerate(run.calls):
                await agent.record_tool(
                    call["tool"],
                    dict(call.get("args", {})),
                    output=call.get("output"),
                    task=task,
                    step=step,
                )
            await agent.feedback("run", str(agent.scope.agent_run_id), run.verdict, source="system")


def _probe_text(turns: Sequence[GeneratedTurn]) -> str | None:
    """The turn the read-path probe looks for: the last statement the user (or an event)
    made. A question is not a probe - a question may leave no memory behind."""
    said = [
        t.text
        for t in turns
        if t.role in ("USER", "EVENT") and not t.text.rstrip().endswith(("?", "\uff1f", "\u061f"))
    ]
    if not said:
        return None
    # a multi-line record (a report) is stored line by line: probe for its last line
    lines = [ln for ln in said[-1].splitlines() if ln.strip()]
    return lines[-1] if lines else said[-1]


async def write_scenario(
    service: Service, s: Scenario, lang: str, user_id: str, today: datetime, settings: Settings
) -> Written:
    client = service.client
    w = Written(s, lang, user_id)
    t0 = _now()
    if s.corpus:
        spec = load_json(f"{s.corpus['spec']}.json")
        corpus = build_corpus(spec, lang, today.date(), settings.scale)
        w.corpus_tokens = scale_tokens(corpus)
        for q in s.queries:
            if "aggregate" in q.expect:
                truth = aggregate(corpus, q.expect["aggregate"], today.date())
                w.expects[q.id] = aggregate_expect(truth, q.expect["aggregate"])
        for name, turns in corpus.sessions.items():
            thread = thread_name(user_id, name)
            w.threads.append(thread)
            if not settings.reuse:
                await _add(service, _ctx(client, w, thread), turns, w)
            if probe := _probe_text(turns):
                w.last_probe, w.probe_thread = probe, thread
    for session in sorted(s.sessions, key=lambda x: x.day):
        thread = thread_name(user_id, session.id)
        w.threads.append(thread)
        start = session.start(today)
        turns = [
            GeneratedTurn(t.role, text_for(t.text, lang), start + timedelta(minutes=t.minute))
            for t in session.turns
        ]
        if session.repeat:
            turns += expand_repeat(session.repeat, lang, start, seed=f"{s.id}/{session.id}")
        turns.sort(key=lambda t: t.occurred_at)
        if turns:
            if not settings.reuse:
                await _add(service, _ctx(client, w, thread), turns, w)
            if probe := _probe_text(turns):
                w.last_probe, w.probe_thread = probe, thread
        if session.tool_runs and not settings.reuse:
            await _tool_runs(client, w, session, thread)
    w.write_seconds = _now() - t0
    return w


# --------------------------------------------------------------------------- readiness


async def _calm(
    service: Service, queues: Sequence[str], since: float, until: float
) -> tuple[bool, int]:
    """Two consecutive worker samples, taken after ``since``, with nothing waiting on
    ``queues``; False when ``until`` passes first. The worker samples every 15 s and a job
    being run is not "waiting", so one empty sample is not proof - two in a row are."""
    calm = samples = 0
    seen_at = 0.0
    while _now() < until:
        sample = await service.worker_sample()
        if sample is None:
            return False, samples
        if sample["sampled_at"] > max(since, seen_at):
            seen_at = sample["sampled_at"]
            samples += 1
            calm = calm + 1 if sum(sample[q] for q in queues) == 0 else 0
            if calm >= 2:
                return True, samples
        await asyncio.sleep(2)
    return False, samples


async def settle(
    service: Service,
    written: Sequence[Written],
    since: float,
    probe_seconds: float = PROBE_SECONDS,
    enrich_seconds: float = ENRICH_SECONDS,
) -> dict[str, Any]:
    """The service's own signals that what was written is processed and indexed.

    Required before any question: the outbox drained and nothing waiting to be extracted or
    embedded (``INDEX_QUEUES``). Then, bounded by ``ENRICH_SECONDS``: graph enrichment and
    thread summaries, which run below indexing and on a large write trail it by many minutes
    (``queued_at_ask`` records what was still waiting when the questions started). Last, the
    read path: every scenario's last statement is findable in its own thread."""
    t0 = _now()
    signals: dict[str, Any] = {"outbox": None, "probes_found": 0}
    deadline = t0 + SETTLE_SECONDS
    while _now() < deadline:
        backlog = await service.outbox_backlog()
        signals["outbox"] = backlog
        if not backlog:
            break
        await asyncio.sleep(1)
    if service.worker_metrics_url:
        indexed, n = await _calm(service, INDEX_QUEUES, since, deadline)
        enriched, m = await _calm(
            service, SETTLE_QUEUES, since, min(deadline, _now() + enrich_seconds)
        )
        signals.update(indexed=indexed, enriched=enriched, worker_samples=n + m)
        if (sample := await service.worker_sample()) is not None:
            signals["queued_at_ask"] = {q: sample[q] for q in SETTLE_QUEUES}
    else:
        signals["worker_metrics"] = "not configured"
    signals["index_seconds"] = round(_now() - t0, 2)
    # the read path: each scenario's last statement is retrievable in its own thread (each
    # probe has its own budget and is always tried at least once; all of them together at
    # most PROBES_TOTAL_SECONDS, so a store that cannot find them cannot stall the run)
    probes_until = _now() + PROBES_TOTAL_SECONDS
    for w in written:
        if not w.last_probe or not w.probe_thread:
            continue
        ctx = _ctx(service.client, w, w.probe_thread)
        needle = " ".join(w.last_probe.split())[:40]
        found = False
        delay = 0.2
        until = min(_now() + probe_seconds, probes_until)
        while True:
            items = await ctx.search(w.last_probe[:300], kinds=["memory"], limit=10)
            if any(needle in " ".join(i.text.split()) for i in items):
                found = True
                break
            if _now() >= until:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 3.0)
        signals["probes_found"] += found
        if not found:
            signals.setdefault("probes_missing", []).append(f"{w.scenario.id}/{w.language}")
    signals["probes"] = sum(1 for w in written if w.last_probe)
    signals["seconds"] = round(_now() - t0, 2)
    return signals


# --------------------------------------------------------------------------- asking


@dataclass
class Asked:
    case: Case
    placement: str
    surface: str
    latency_ms: float
    verdict: dict[str, Any]
    evidence_status: str | None
    preview: list[str]


def _expect(w: Written, case: Case) -> Mapping[str, Any]:
    return w.expects.get(case.query.id, case.query.expect)


async def ask(service: Service, w: Written, case: Case, settings: Settings) -> list[Asked]:
    out: list[Asked] = []
    expect = _expect(w, case)
    for placement in settings.placements:
        if placement == "new_session":
            thread = thread_name(
                w.user_id, f"ask-{hashlib.sha1(case.key.encode()).hexdigest()[:10]}"
            )
            window = True
        else:
            thread = w.threads[-1]
            window = False  # the framework keeps its own history; memory must carry it
        ctx = _ctx(service.client, w, thread)
        if "context" in settings.surfaces:
            t = _now()
            bundle = await ctx.context(
                case.text,
                token_budget=TOKEN_BUDGET,
                tools=list(case.query.tools) or None,
                window=window,
                format="full",
                timeout=settings.timeout,
            )
            latency = (_now() - t) * 1000
            data = bundle.model_dump(mode="json")
            items = items_from_bundle(data)
            v = score(
                expect,
                items,
                case.language,
                surface="context",
                evidence_status=bundle.evidence_status,
            )
            out.append(
                Asked(
                    case,
                    placement,
                    "context",
                    latency,
                    v.as_dict(),
                    bundle.evidence_status,
                    [] if v.passed else preview(items),
                )
            )
        if (
            "search" in settings.surfaces
            and placement == "new_session"
            and scorable(expect, "search")
        ):
            t = _now()
            found = await ctx.search(
                case.text, kinds=list(SEARCH_KINDS), limit=SEARCH_LIMIT, timeout=settings.timeout
            )
            latency = (_now() - t) * 1000
            items = items_from_search([i.model_dump(mode="json") for i in found])
            v = score(expect, items, case.language, surface="search")
            out.append(
                Asked(
                    case,
                    placement,
                    "search",
                    latency,
                    v.as_dict(),
                    None,
                    [] if v.passed else preview(items),
                )
            )
    return out


def record(a: Asked, w: Written) -> dict[str, Any]:
    s = a.case.scenario
    return {
        "key": a.case.key,
        "capability": s.capability,
        "scenario": s.id,
        "origin": s.origin,
        "domain": s.domain,
        "query": a.case.query.id,
        "split": a.case.query.split,
        "kind": a.case.phrasing.kind,
        "language": a.case.language,
        "question": a.case.text,
        "placement": a.placement,
        "surface": a.surface,
        "latency_ms": round(a.latency_ms, 2),
        "evidence_status": a.evidence_status,
        **a.verdict,
        "preview": a.preview,
        "corpus_tokens": w.corpus_tokens or None,
    }


def _reused_meta(raw: Path) -> dict[str, Any]:
    import gzip

    body = gzip.decompress(raw.read_bytes()) if raw.suffix == ".gz" else raw.read_bytes()
    return json.loads(body.decode("utf-8"))["meta"]


def failed_record(case: Case, w: Written, exc: BaseException) -> dict[str, Any]:
    """A request that raised: scored as a failure with reason ``error``, never dropped."""
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
        "surface": "context",
        "latency_ms": 0.0,
        "evidence_status": None,
        "passed": False,
        "reason": "error",
        "groups_met": 0,
        "groups_required": 0,
        "missing": [],
        "leaks": [],
        "abstained": None,
        "items": 0,
        "preview": [f"{type(exc).__name__}: {exc}"[:200]],
        "corpus_tokens": w.corpus_tokens or None,
    }


# --------------------------------------------------------------------------- the run


ClientFactory = Callable[[str], Any]


def default_factory(settings: Settings) -> ClientFactory:
    """A real HTTP client of ``settings.url`` for a key."""
    from trellis.memory import MemoryClient

    return lambda key: MemoryClient(settings.url, api_key=key, timeout=settings.timeout)


async def _tenant(settings: Settings, factory: ClientFactory) -> tuple[str, Any]:
    """The eval tenant with the mode's policy, and a service key for this run."""
    from trellis.memory.errors import NotFoundError

    modes = load_json("_modes.json")["modes"]
    policy = modes[settings.mode]["policy"]
    if settings.api_key and not settings.admin_key:
        # the tenant cannot be checked from a service key: ``_fresh_users`` checks the users
        return settings.api_key, {"policy": "unchanged (service key given)"}
    assert settings.admin_key, "an admin credential (--admin-key / CAPEVAL_ADMIN_KEY) is needed"
    admin_key = settings.admin_key
    async with factory(settings.admin_key) as platform:
        try:
            await platform.admin.get_tenant(settings.tenant)
        except NotFoundError:
            created = await platform.admin.create_tenant(
                f"Capability eval ({settings.mode})", tenant_id=settings.tenant
            )
            # a platform key onboards a tenant and does not act for it: the new tenant's own
            # admin key administers it (a development key may do both)
            admin_key = created.admin_key.token or admin_key
        else:
            if not settings.reuse:
                raise SystemExit(STORE_IN_USE.format(what=f"tenant {settings.tenant!r} exists"))
    async with factory(admin_key) as admin:
        tenant = admin.administer(settings.tenant)
        model = os.environ.get(policy.get("model_env", ""), "")
        if policy.get("model_env") and not model:
            raise SystemExit(
                f"{settings.mode} mode needs the gateway model in {policy['model_env']}"
            )
        models = dict.fromkeys(policy["uses"], model) if model else {}
        stored = await tenant.set_model_policy(
            policy["uses"], read_assist=policy["read_assist"], models=models
        )
        key = await tenant.keys.issue("service", f"capeval-{settings.run_id}")
        assert key.token
        return key.token, {
            "uses": list(stored.uses),
            "read_assist": stored.read_assist,
            "models": dict(stored.models),
            "key_id": key.key_id,
            "admin_key": admin_key,
        }


STORE_IN_USE = (
    "{what}: the eval names its users by scenario and language, the same in every run, so a "
    "run must not share a store with an earlier one (its memories, graph and profiles would "
    "be read as this run's). Run it on a store of its own (benchmark/capabilities/"
    "isolated_stack.sh, make eval-capabilities-isolated), or name a tenant never used "
    "(--tenant); --reuse re-asks a kept run's writes on purpose."
)


async def _fresh_users(client: Any, users: Sequence[str]) -> None:
    """With only a service key the tenant cannot be checked: refuse when any of the run's
    users already holds a memory (a soft-deleted one is not listed: the admin path, which
    needs the tenant itself to be new, is the strict one)."""
    for user in users:
        async for _ in client.bind(user_id=user).advanced.memories.iter(include_superseded=True):
            raise SystemExit(STORE_IN_USE.format(what=f"user {user} already has memories"))


async def _revoke(settings: Settings, factory: ClientFactory, policy: Mapping[str, Any]) -> None:
    key_id, admin_key = policy.get("key_id"), policy.get("admin_key")
    if not key_id or not admin_key:
        return
    async with factory(admin_key) as admin:
        await admin.administer(settings.tenant).keys.revoke(key_id)


async def run(
    settings: Settings, dataset: Dataset | None = None, client_factory: ClientFactory | None = None
) -> dict[str, Any]:
    """Write, settle, ask and score; the report payload (``report.build``). ``client_factory``
    makes the SDK client for a key - an in-process app's transport in tests."""
    factory = client_factory or default_factory(settings)
    dataset = dataset or load_dataset()
    scenarios = select(dataset, settings.selection)
    if settings.reuse:
        # the same users and threads (named from the scenarios) and the same run date as the
        # run being asked again
        reused = _reused_meta(settings.reuse)
        if reused["selection"] != settings.selection.describe() or reused["mode"] != settings.mode:
            raise SystemExit("--reuse needs the same mode and selection as the run it reuses")
        if reused.get("identities") != IDENTITIES:
            raise SystemExit(
                "--reuse needs a run whose users were named the way this runner names them"
            )
        settings.run_id = reused["run_id"]
        settings.run_date = date.fromisoformat(reused["run_date"])
    today = run_day(settings)
    work = [
        (s, lang, identity(s.id, lang))
        for s in scenarios
        for lang in languages_of(s, settings.selection)
    ]
    if len({user for *_, user in work}) != len(work):
        raise SystemExit("two scenarios of the selection share an id")
    token, policy = await _tenant(settings, factory)
    client = factory(token)
    service = Service(client, settings.worker_metrics_url)
    written: list[Written] = []
    cleanup: dict[str, Any] = {"purged": False}
    try:
        version = await client.version()
        if not settings.reuse and settings.api_key and not settings.admin_key:
            await _fresh_users(client, [user for *_, user in work])
        sem = asyncio.Semaphore(settings.write_concurrency)

        async def _write(s: Scenario, lang: str, user: str) -> Written:
            async with sem:
                try:
                    return await write_scenario(service, s, lang, user, today, settings)
                except Exception as exc:  # noqa: BLE001 - one scenario's failure is recorded, not fatal
                    print(f"write failed {s.id}/{lang}: {exc!r}", file=sys.stderr)
                    w = Written(s, lang, user, error=f"{type(exc).__name__}: {exc}"[:300])
                    w.threads.append(thread_name(user, "unwritten"))
                    return w

        queues_at_start = await service.queue_depths()
        t_write = _now()
        written = list(await asyncio.gather(*(_write(*x) for x in work)))
        last_write = time.time()
        write_seconds = _now() - t_write
        print(
            f"wrote {sum(w.turns for w in written)} turns in {write_seconds:.0f}s", file=sys.stderr
        )
        signals = await settle(
            service, written, last_write, settings.probe_seconds, settings.enrich_seconds
        )
        print(f"settled: {signals}", file=sys.stderr)

        records: list[dict[str, Any]] = []
        qsem = asyncio.Semaphore(settings.query_concurrency)

        async def _ask(w: Written, case: Case) -> None:
            async with qsem:
                try:
                    for a in await ask(service, w, case, settings):
                        records.append(record(a, w))
                except Exception as exc:  # noqa: BLE001 - a failed request is a failed answer
                    records.append(failed_record(case, w, exc))

        t_ask = _now()
        async with LoadSampler() as load:
            await asyncio.gather(
                *(_ask(w, c) for w in written for c in cases(w.scenario, w.language))
            )
        ask_seconds = _now() - t_ask
    finally:
        # a run leaves the service as it found it, a failed one too
        if should_purge(settings) and written:
            done = await purge(client, owned_by(written).items())
            cleanup = {"purged": True, **done.as_dict()}
            print(f"cleanup: {cleanup}", file=sys.stderr)
        await client.aclose()
        await _revoke(settings, factory, policy)

    from benchmark.capabilities.report import build, save_raw

    meta = {
        "mode": settings.mode,
        "tenant": settings.tenant,
        "policy": {k: v for k, v in policy.items() if k not in ("key_id", "admin_key")},
        "service_version": version,
        "memory_url": settings.url,
        "run_id": settings.run_id,
        "run_date": today.date().isoformat(),
        #: how users and threads are named (``identity``): a pure function of the scenario and
        #: the language, so two runs on one run date write the same thing
        "identities": IDENTITIES,
        "selection": settings.selection.describe(),
        "sample": settings.sample,
        "scale": settings.scale,
        "placements": list(settings.placements),
        "surfaces": list(settings.surfaces),
        "token_budget": TOKEN_BUDGET,
        "ingestion": {
            "queues_at_start": queues_at_start,
            "turns": sum(w.turns for w in written),
            "jobs": sum(w.jobs for w in written),
            "failed_jobs": sum(w.failed_jobs for w in written),
            "failed_writes": [
                f"{w.scenario.id}/{w.language}: {w.error}" for w in written if w.error
            ],
            "write_seconds": round(write_seconds, 1),
            "settle": signals,
            "per_scenario_write_seconds": _pcts([w.write_seconds for w in written]),
            "corpus_tokens": {
                f"{w.scenario.id}/{w.language}": w.corpus_tokens for w in written if w.corpus_tokens
            },
        },
        "ask_seconds": round(ask_seconds, 1),
        "reused_writes_of": str(settings.reuse) if settings.reuse else None,
        "cleanup": cleanup,
        #: the users and threads written, so a kept run can be purged later
        #: (``python -m benchmark.capabilities.cleanup <raw.json.gz>``)
        "written": owned_by(written),
        "host_load_during_asks": load.summary(),
    }
    # the measurement itself, before anything that could fail summarising it
    if settings.out is not None:
        save_raw(records, meta, settings.out)
    return build(dataset, records, meta)


def owned_by(written: Sequence[Written]) -> dict[str, list[str]]:
    """User -> the threads the run wrote for it."""
    out: dict[str, dict[str, None]] = {}
    for w in written:
        out.setdefault(w.user_id, {}).update(dict.fromkeys(w.threads))
    return {user: list(threads) for user, threads in out.items()}


def _pcts(values: Sequence[float]) -> dict[str, float | None]:
    from benchmark.capabilities.report import pct

    return {
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "max": round(max(values), 2) if values else None,
    }


# --------------------------------------------------------------------------- CLI


def _csv(value: str | None) -> tuple[str, ...] | None:
    return tuple(v.strip() for v in value.split(",") if v.strip()) if value else None


def parse(argv: Sequence[str] | None = None) -> tuple[Settings, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=("nollm", "llm"), default="nollm")
    p.add_argument("--url", default=os.environ.get("MEMORY_URL", "http://localhost:8091"))
    p.add_argument("--tenant", default=None, help="default capeval-<mode>")
    p.add_argument(
        "--admin-key",
        default=os.environ.get("CAPEVAL_ADMIN_KEY") or os.environ.get("TRELLIS_ADMIN_KEY"),
    )
    p.add_argument("--api-key", default=os.environ.get("CAPEVAL_API_KEY"))
    p.add_argument("--worker-metrics-url", default=os.environ.get("MEMORY_WORKER_METRICS_URL"))
    p.add_argument("--languages", default=",".join(ALL_LANGUAGES))
    p.add_argument("--split", choices=("dev", "heldout"), default=None)
    p.add_argument("--capabilities", default=None)
    p.add_argument("--domains", default=None)
    p.add_argument("--kinds", default=None, help="phrasing kinds, e.g. direct,paraphrase")
    p.add_argument("--scenarios-per-capability", type=int, default=None)
    p.add_argument("--scenarios-per-domain", type=int, default=None)
    p.add_argument("--scenarios", default=None, help="scenario ids")
    p.add_argument("--sample", default=None, help="a named sample from data/_modes.json")
    p.add_argument("--scale", choices=("standard", "large"), default="standard")
    p.add_argument("--write-concurrency", type=int, default=4)
    p.add_argument("--query-concurrency", type=int, default=1)
    p.add_argument("--placements", default="new_session,same_session")
    p.add_argument("--surfaces", default="context,search")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--reuse", default=None, help="ask again what this run's .raw.json.gz wrote")
    p.add_argument(
        "--keep", action="store_true", help="leave the writes in the tenant (for a later --reuse)"
    )
    p.add_argument("--enrich-seconds", type=float, default=ENRICH_SECONDS)
    p.add_argument(
        "--run-date",
        default=os.environ.get("CAPEVAL_RUN_DATE"),
        help="YYYY-MM-DD the sessions are dated from (default: today, UTC); recorded in the result",
    )
    p.add_argument("--out", default=None, help="default benchmark/results/capabilities/<mode>.json")
    a = p.parse_args(argv)
    sample: dict[str, Any] = {}
    if a.sample:
        sample = load_json("_modes.json")["samples"][a.sample]
    sel = Selection(
        languages=_csv((sample.get("languages") and ",".join(sample["languages"])) or a.languages)
        or ALL_LANGUAGES,
        split=sample.get("split", a.split),
        capabilities=tuple(sample["capabilities"])
        if sample.get("capabilities")
        else _csv(a.capabilities),
        domains=_csv(a.domains),
        kinds=tuple(sample["kinds"]) if sample.get("kinds") else _csv(a.kinds),
        scenarios_per_capability=sample.get("scenarios_per_capability", a.scenarios_per_capability),
        scenarios_per_domain=sample.get("scenarios_per_domain", a.scenarios_per_domain),
        scenario_ids=tuple(sample["scenarios"]) if sample.get("scenarios") else _csv(a.scenarios),
    )
    settings = Settings(
        url=a.url,
        mode=a.mode,
        tenant=a.tenant or f"capeval-{a.mode}",
        admin_key=a.admin_key,
        api_key=a.api_key,
        worker_metrics_url=a.worker_metrics_url,
        selection=sel,
        scale=a.scale,
        write_concurrency=a.write_concurrency,
        query_concurrency=a.query_concurrency,
        placements=_csv(a.placements) or ("new_session",),
        surfaces=_csv(a.surfaces) or ("context",),
        timeout=a.timeout,
        sample=a.sample,
        reuse=Path(a.reuse) if a.reuse else None,
        keep=a.keep,
        enrich_seconds=float(sample.get("enrich_seconds", a.enrich_seconds)),
        out=Path(a.out) if a.out else None,
        run_date=date.fromisoformat(a.run_date) if a.run_date else None,
    )
    return settings, a


def default_out(mode: str, sample: str | None) -> Path:
    """``<mode>.json``, or ``<mode>_<set>.json`` for a named gate set (``nollm_core.json``)."""
    name = mode if sample in (None, mode) else f"{mode}_{sample}"
    return Path(f"benchmark/results/capabilities/{name}.json")


def main(argv: Sequence[str] | None = None) -> int:
    from benchmark.capabilities.report import write

    settings, _ = parse(argv)
    settings.out = settings.out or default_out(settings.mode, settings.sample)
    payload = asyncio.run(run(settings))
    json_path, md_path = write(payload, settings.out)
    print(f"wrote {json_path} and {md_path}")
    s = payload["summary"]
    print(
        f"{settings.mode}: macro accuracy {s['macro_accuracy']} (context, new session), retail track "
        f"{s['retail_track']}, context p50/p95 {s['latency_ms']['context_p50']}/"
        f"{s['latency_ms']['context_p95']} ms"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
