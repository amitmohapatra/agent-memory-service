"""What each registered background job does when it runs (``modules.jobs.registry``).

``test_job_registration`` proves every enqueued task has a handler; this runs the handlers
against recording stand-ins for the services and the unit of work, so each job's effect -
which service it calls with what, which rows it bumps, which follow-up jobs it queues, and
what it skips when a service is not wired - is asserted rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from memory_service.config.constants import TASKS
from memory_service.domain.enums import MemoryType
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.jobs import registry
from memory_service.modules.jobs.names import TASK_MEMORY_INDEX
from memory_service.modules.llm.cost import llm_accounting, record_llm_tokens
from memory_service.modules.profile.service import TASK_PROFILE_REFRESH
from memory_service.ports.tasks import JobSpec, Queue

pytestmark = pytest.mark.unit

NOW = datetime(2026, 6, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- stand-ins


class _Queue:
    """Records what ``register_handlers`` registers."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.options: dict[str, tuple[Queue, dict[str, Any]]] = {}
        self.periodic: dict[str, tuple[Queue, str]] = {}

    def register(self, name: str, queue: Queue, handler: Any, **kwargs: Any) -> None:
        self.handlers[name] = handler
        self.options[name] = (queue, kwargs)

    def register_periodic(self, name: str, queue: Queue, handler: Any, *, cron: str) -> None:
        self.handlers[name] = handler
        self.periodic[name] = (queue, cron)


class _Service:
    """Any service: every awaited method is recorded and answers from ``returns``."""

    def __init__(self, **returns: Any) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.returns = returns

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return self.returns.get(name)

        return method


class _Uow:
    def __init__(self, memories: list[CanonicalMemory] | None = None) -> None:
        self.stored = {m.memory_id: m for m in memories or []}
        self.expired: list[tuple[str, str]] = []
        self.calls: list[tuple[str, Any]] = []
        self.jobs: list[JobSpec] = []
        self.bumped: list[tuple[str, RevisionKind, str]] = []
        self.commits = 0
        self.observations = SimpleNamespace(mark_processed=self._mark_processed)
        self.memories = SimpleNamespace(get_many=self._get_many, expire_due=self._expire_due)
        self.revisions = SimpleNamespace(bump=self._bump)
        self.idempotency = SimpleNamespace(purge_expired=self._purge_expired)
        self.outbox = SimpleNamespace(purge_dispatched=self._purge_dispatched)
        self.purged = 0

    async def __aenter__(self) -> _Uow:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def _mark_processed(self, tenant_id: str, observation_id: str, *, status: str) -> None:
        self.calls.append(("mark_processed", (tenant_id, observation_id, status)))

    async def _get_many(self, tenant_id: str, ids: list[str]) -> list[CanonicalMemory]:
        self.calls.append(("get_many", (tenant_id, list(ids))))
        return [self.stored[i] for i in dict.fromkeys(ids) if i in self.stored]

    async def _expire_due(self, *, now: datetime) -> list[tuple[str, str]]:
        self.calls.append(("expire_due", now))
        return list(self.expired)

    async def _bump(self, tenant_id: str, kind: RevisionKind, object_id: str = "") -> int:
        self.bumped.append((tenant_id, kind, object_id))
        return 1

    async def _purge_expired(self, *, now: datetime) -> int:
        self.calls.append(("purge_expired", now))
        return self.purged

    async def _purge_dispatched(self, *, older_than_seconds: int) -> int:
        self.calls.append(("purge_dispatched", older_than_seconds))
        return self.purged

    async def enqueue(self, spec: JobSpec) -> None:
        self.jobs.append(spec)

    async def commit(self) -> None:
        self.commits += 1


def _memory(memory_id: str, memory_type: MemoryType, user_id: str | None = "u1") -> CanonicalMemory:
    scope: dict[str, Any] = {"level": "USER" if user_id else "TENANT", "tenant_id": "acme"}
    if user_id:
        scope["user_id"] = user_id
    return CanonicalMemory.model_validate(
        {
            "memory_id": memory_id,
            "tenant_id": "acme",
            "content": f"content of {memory_id}",
            "memory_type": memory_type,
            "lifetime": "LONG_TERM",
            "visibility": "USER" if user_id else "TENANT",
            "scope": scope,
            "owner_principal": f"user:{user_id or 'svc'}",
            "normalized_hash": memory_id,
            "temporal": {"observed_at": NOW},
            "evidence": [{"source_type": "message", "source_id": "msg_1", "observed_at": NOW}],
        }
    )


def _wire(
    services: dict[str, Any] | None = None,
    *,
    uow: _Uow | None = None,
    llm: bool = False,
    tasks: Any = None,
) -> tuple[_Queue, _Uow, SimpleNamespace]:
    queue = tasks if tasks is not None else _Queue()
    unit = uow or _Uow()
    container = SimpleNamespace(
        tasks=queue,
        services={"uow_factory": lambda: unit, **(services or {})},
        settings=SimpleNamespace(llm=SimpleNamespace(enabled=llm)),
    )
    registry.register_handlers(container)  # type: ignore[arg-type]
    return queue, unit, container


@pytest.fixture
def logs() -> Iterator[list[Any]]:
    """Structured log events, on structlog's default configuration (restored after)."""
    saved = structlog.get_config()
    structlog.reset_defaults()
    try:
        with capture_logs() as events:
            yield events
    finally:
        structlog.configure(**saved)


# --------------------------------------------------------------------------- registration


def test_nothing_is_registered_without_a_task_queue() -> None:
    container = SimpleNamespace(tasks=None, services={}, settings=None)
    assert registry.register_handlers(container) is None  # type: ignore[arg-type,func-returns-value]


def test_every_job_is_registered_on_its_queue_with_its_retry_budget() -> None:
    queue, _, _ = _wire()
    assert queue.options[registry.TASK_PROCESS_OBSERVATION] == (Queue.CHAT_FAST, {"retries": 5})
    assert queue.options["document.parse"] == (Queue.DOCUMENT_PARSE, {"retries": 3})
    assert queue.options[TASK_MEMORY_INDEX] == (Queue.EMBEDDING, {"retries": 5})
    assert queue.options[registry.TASK_MEMORY_FORGET] == (Queue.RECONCILE, {"retries": 0})
    assert queue.options[registry.TASK_ARCHIVE_STAGE] == (Queue.ARCHIVE, {"retries": 10})
    assert queue.options[TASK_PROFILE_REFRESH] == (Queue.SUMMARY, {"retries": 3})


def test_periodic_jobs_are_scheduled_by_cron() -> None:
    queue, _, _ = _wire()
    assert queue.periodic["periodic.outbox_sweep"] == (Queue.RECONCILE, "* * * * *")
    assert queue.periodic["periodic.retention"] == (Queue.RECONCILE, "37 3 * * *")
    assert queue.periodic["periodic.memory_forget"] == (Queue.RECONCILE, "11 4 * * *")
    every = max(1, TASKS.periodic_reconcile_seconds // 60)
    assert queue.periodic["periodic.reconcile"] == (Queue.RECONCILE, f"*/{min(every, 59)} * * * *")


def test_model_driven_passes_are_registered_only_when_a_model_is_reachable() -> None:
    without, _, _ = _wire(llm=False)
    assert registry.TASK_MEMORY_REFLECT not in without.handlers
    assert "periodic.memory_connect" not in without.periodic
    with_model, _, _ = _wire(llm=True)
    assert with_model.options[registry.TASK_MEMORY_REFLECT] == (Queue.RECONCILE, {"retries": 0})
    assert with_model.periodic["periodic.memory_reflect"] == (Queue.RECONCILE, "53 */6 * * *")
    assert with_model.periodic["periodic.memory_connect"] == (Queue.RECONCILE, "19 */6 * * *")


def test_extra_registrars_are_handed_the_container() -> None:
    seen: list[Any] = []
    _, _, container = _wire({"extra_task_registrars": [seen.append]})
    assert seen == [container]


# --------------------------------------------------------------------------- accounting


async def test_a_job_s_model_tokens_are_counted_in_its_own_scope_and_logged(logs) -> None:
    async def uses_the_model(payload: dict[str, Any]) -> str:
        record_llm_tokens(30, 12)
        return "done"

    queue = _Queue()
    registry._AccountedQueue(queue).register("t.model", Queue.RECONCILE, uses_the_model)
    with llm_accounting() as outer:
        assert await queue.handlers["t.model"]({"tenant_id": "acme"}) == "done"
    assert outer.total == 0, "a job's tokens never leak into the request that ran it inline"
    [event] = [e for e in logs if e["event"] == "job.llm_tokens"]
    assert event["task"] == "t.model" and event["tenant_id"] == "acme"
    assert (event["input_tokens"], event["output_tokens"]) == (30, 12)


async def test_a_job_that_used_no_model_logs_no_tokens_and_errors_propagate(logs) -> None:
    async def fails(payload: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    queue = _Queue()
    registry._AccountedQueue(queue).register_periodic("t.fail", Queue.RECONCILE, fails, cron="*")
    with pytest.raises(RuntimeError, match="boom"):
        await queue.handlers["t.fail"]({})
    assert queue.periodic["t.fail"] == (Queue.RECONCILE, "*")
    assert not [e for e in logs if e["event"] == "job.llm_tokens"]


# --------------------------------------------------------------------------- ingestion jobs


async def test_an_observation_runs_through_the_pipeline_when_one_is_wired() -> None:
    pipeline = _Service()
    queue, uow, _ = _wire({"observation_pipeline": pipeline})
    payload = {"tenant_id": "acme", "observation_id": "obs_1"}
    await queue.handlers[registry.TASK_PROCESS_OBSERVATION](payload)
    assert pipeline.calls == [("run", (payload,), {})]
    assert uow.calls == [] and uow.commits == 0


async def test_without_a_pipeline_an_observation_is_only_marked_recorded() -> None:
    queue, uow, _ = _wire()
    await queue.handlers[registry.TASK_PROCESS_OBSERVATION](
        {"tenant_id": "acme", "observation_id": "obs_1"}
    )
    assert uow.calls == [("mark_processed", ("acme", "obs_1", "RECORDED"))]
    assert uow.commits == 1


async def test_archive_staging_archives_the_thread_when_an_archiver_is_wired() -> None:
    archiver = _Service()
    queue, _, _ = _wire({"archive_service": archiver})
    await queue.handlers[registry.TASK_ARCHIVE_STAGE]({"tenant_id": "acme", "thread_id": "thr_1"})
    assert archiver.calls == [("archive_thread", ("acme", "thr_1"), {})]
    bare, _, _ = _wire()
    assert await bare.handlers[registry.TASK_ARCHIVE_STAGE]({"tenant_id": "a"}) is None


async def test_a_document_is_parsed_then_indexed_and_enriched() -> None:
    ingestion, indexer, graph = _Service(), _Service(), _Service()
    queue, _, _ = _wire({"ingestion": ingestion, "indexer": indexer, "graph": graph})
    payload = {"tenant_id": "acme", "document_id": "doc_1"}
    await queue.handlers["document.parse"](payload)
    await queue.handlers["document.index"](payload)
    assert ingestion.calls == [("parse_document", ("acme", "doc_1"), {})]
    assert indexer.calls == [("index_document", ("acme", "doc_1"), {})]
    assert graph.calls == [("enrich_document", ("acme", "doc_1"), {})]


async def test_document_jobs_keep_the_chain_intact_when_nothing_is_wired() -> None:
    queue, _, _ = _wire()
    payload = {"tenant_id": "acme", "document_id": "doc_1"}
    assert await queue.handlers["document.parse"](payload) is None
    assert await queue.handlers["document.index"](payload) is None


async def test_feedback_is_projected_when_the_service_is_wired() -> None:
    feedback = _Service()
    queue, _, _ = _wire({"feedback": feedback})
    task = next(n for n in queue.handlers if n.startswith("feedback"))
    await queue.handlers[task]({"tenant_id": "acme", "feedback_id": "fb_1"})
    assert feedback.calls == [("project", ("acme", "fb_1"), {})]
    bare, _, _ = _wire()
    assert await bare.handlers[task]({"tenant_id": "acme", "feedback_id": "fb_1"}) is None


# --------------------------------------------------------------------------- memory index


async def test_indexing_memories_makes_them_findable_then_moves_their_revisions() -> None:
    fact = _memory("mem_1", MemoryType.SEMANTIC, user_id="u1")
    pref = _memory("mem_2", MemoryType.PREFERENCE, user_id="u2")
    indexer, graph = _Service(), _Service()
    queue, uow, _ = _wire({"indexer": indexer, "graph": graph}, uow=_Uow([fact, pref]))

    await queue.handlers[TASK_MEMORY_INDEX]({"tenant_id": "acme", "memory_ids": ["mem_1", "mem_2"]})

    assert indexer.calls == [("index_memories", ("acme", ["mem_1", "mem_2"]), {})]
    assert graph.calls == [("enrich_memories", ("acme", ["mem_1", "mem_2"]), {})]
    assert ("acme", RevisionKind.USER, "u1") in uow.bumped
    assert ("acme", RevisionKind.USER, "u2") in uow.bumped
    assert ("acme", RevisionKind.TENANT, "") not in uow.bumped, "nothing was deleted"
    # only what a user said about themselves refreshes their profile block
    assert [(j.task_name, j.payload) for j in uow.jobs] == [
        (TASK_PROFILE_REFRESH, {"tenant_id": "acme", "user_id": "u2"})
    ]
    assert uow.commits == 1


async def test_a_deleted_memory_in_the_batch_moves_the_tenant_revision() -> None:
    queue, uow, _ = _wire(uow=_Uow([_memory("mem_1", MemoryType.USER)]))
    await queue.handlers[TASK_MEMORY_INDEX](
        {"tenant_id": "acme", "memory_ids": ["mem_1", "mem_gone", "mem_1"]}
    )
    assert ("acme", RevisionKind.TENANT, "") in uow.bumped
    assert [j.payload["user_id"] for j in uow.jobs] == ["u1"]


async def test_a_memory_with_no_user_refreshes_no_profile() -> None:
    queue, uow, _ = _wire(uow=_Uow([_memory("mem_1", MemoryType.USER, user_id=None)]))
    await queue.handlers[TASK_MEMORY_INDEX]({"tenant_id": "acme", "memory_ids": ["mem_1"]})
    assert uow.jobs == [] and uow.commits == 1


async def test_an_empty_index_batch_touches_nothing() -> None:
    queue, uow, _ = _wire()
    await queue.handlers[TASK_MEMORY_INDEX]({"tenant_id": "acme", "memory_ids": []})
    assert uow.calls == [] and uow.bumped == [] and uow.commits == 0


# --------------------------------------------------------------------------- expiry / forgetting


async def test_expiry_queues_one_projection_cleanup_per_tenant(logs) -> None:
    uow = _Uow([_memory("mem_1", MemoryType.SEMANTIC)])
    uow.expired = [("acme", "mem_1"), ("globex", "mem_9"), ("acme", "mem_2")]
    queue, _, _ = _wire(uow=uow)
    await queue.handlers[registry.TASK_MEMORY_EXPIRE]({})
    assert [(j.task_name, j.queue, j.payload) for j in uow.jobs] == [
        (
            TASK_MEMORY_INDEX,
            Queue.EMBEDDING,
            {"tenant_id": "acme", "memory_ids": ["mem_1", "mem_2"]},
        ),
        (TASK_MEMORY_INDEX, Queue.EMBEDDING, {"tenant_id": "globex", "memory_ids": ["mem_9"]}),
    ]
    assert ("get_many", ("acme", ["mem_1", "mem_2"])) in uow.calls
    assert ("acme", RevisionKind.USER, "u1") in uow.bumped
    assert uow.commits == 1
    assert [e["count"] for e in logs if e["event"] == "memory.expired"] == [3]


async def test_expiry_with_nothing_due_queues_nothing(logs) -> None:
    queue, uow, _ = _wire()
    await queue.handlers["periodic.memory_expire"]({})
    assert uow.jobs == [] and uow.commits == 1
    assert not [e for e in logs if e["event"] == "memory.expired"]


async def test_forgetting_runs_a_sweep_when_wired() -> None:
    forgetting = _Service()
    queue, _, _ = _wire({"forgetting": forgetting})
    await queue.handlers[registry.TASK_MEMORY_FORGET]({})
    assert forgetting.calls == [("sweep", (), {})]
    bare, _, _ = _wire()
    assert await bare.handlers["periodic.memory_forget"]({}) is None


async def test_retention_runs_its_sweep() -> None:
    retention = _Service()
    queue, _, _ = _wire({"retention": retention})
    await queue.handlers["periodic.retention"]({})
    assert retention.calls == [("sweep", (), {})]


@pytest.mark.parametrize("purged", [0, 12])
async def test_the_read_audit_is_purged_past_its_retention(logs, purged: int) -> None:
    audit = _Service(purge=purged)
    queue, _, _ = _wire({"read_audit": audit})
    await queue.handlers["periodic.read_audit_purge"]({})
    assert audit.calls == [("purge", (), {"older_than_days": TASKS.read_audit_retention_days})]
    counts = [e["count"] for e in logs if e["event"] == "read_audit.purged"]
    assert counts == ([purged] if purged else [])


async def test_reflection_and_connections_run_over_everyone_when_wired() -> None:
    reflection, connections = _Service(), _Service()
    queue, _, _ = _wire({"reflection": reflection, "connections": connections}, llm=True)
    await queue.handlers[registry.TASK_MEMORY_REFLECT]({})
    await queue.handlers["periodic.memory_connect"]({})
    assert reflection.calls == [("reflect_all", (), {})]
    assert connections.calls == [("connect_all", (), {})]
    bare, _, _ = _wire(llm=True)
    assert await bare.handlers["periodic.memory_reflect"]({}) is None
    assert await bare.handlers["periodic.memory_connect"]({}) is None


# --------------------------------------------------------------------------- summaries / profile / tools


async def test_a_summary_refresh_folds_the_thread_then_indexes_its_episode() -> None:
    summaries, indexer = _Service(), _Service()
    queue, _, _ = _wire({"thread_summaries": summaries, "indexer": indexer})
    task = next(n for n in queue.options if n.startswith("summary"))
    await queue.handlers[task]({"tenant_id": "acme", "thread_id": "thr_1", "principal_id": "u:1"})
    await queue.handlers[task]({"tenant_id": "acme", "thread_id": "thr_2"})
    assert summaries.calls == [
        ("refresh", ("acme", "thr_1"), {"principal_id": "u:1"}),
        ("refresh", ("acme", "thr_2"), {"principal_id": None}),
    ]
    assert indexer.calls == [
        ("index_episode", ("acme", "thr_1"), {}),
        ("index_episode", ("acme", "thr_2"), {}),
    ]


async def test_an_episode_is_reindexed_on_its_own() -> None:
    indexer = _Service()
    queue, _, _ = _wire({"indexer": indexer})
    task = next(n for n in queue.options if n.startswith("episode"))
    await queue.handlers[task]({"tenant_id": "acme", "thread_id": "thr_1"})
    assert indexer.calls == [("index_episode", ("acme", "thr_1"), {})]


async def test_profile_jobs_refresh_answer_and_schedule() -> None:
    profile = _Service()
    queue, _, _ = _wire({"profile": profile})
    await queue.handlers[TASK_PROFILE_REFRESH]({"tenant_id": "acme", "user_id": "u1"})
    query = next(
        n
        for n, (q, _) in queue.options.items()
        if n.startswith("profile.") and n != TASK_PROFILE_REFRESH
    )
    await queue.handlers[query]({"tenant_id": "acme", "scope_key": "user:u1", "block": "user"})
    await queue.handlers["periodic.profile_queries"]({})
    assert profile.calls == [
        ("refresh_user", ("acme", "u1"), {}),
        ("answer_query", ("acme", "user:u1", "user"), {}),
        ("schedule_due", (), {}),
    ]


async def test_tool_jobs_index_learn_and_fold_prefetch() -> None:
    tool_index, learning, agent_tools = _Service(), _Service(), _Service()
    queue, _, _ = _wire(
        {"tool_index": tool_index, "tool_learning": learning, "agent_tools": agent_tools}
    )
    index_task = next(
        n for n, (q, _) in queue.options.items() if q is Queue.EMBEDDING and n.startswith("tools")
    )
    learn_task = next(
        n for n, (q, _) in queue.options.items() if q is Queue.RECONCILE and n.startswith("tools")
    )
    await queue.handlers[index_task]({"tenant_id": "acme", "tool_ids": ["tool_1"]})
    await queue.handlers[learn_task]({"tenant_id": "acme"})
    await queue.handlers["periodic.tools_learn"]({})
    await queue.handlers["periodic.prefetch_learn"]({})
    assert tool_index.calls == [("index", ("acme", ["tool_1"]), {})]
    assert learning.calls == [("learn", ("acme",), {}), ("learn", (None,), {})]
    assert agent_tools.calls == [("learn_prefetch", (), {})]


# --------------------------------------------------------------------------- system jobs


async def test_the_outbox_sweep_dispatches_rows_older_than_asked(logs) -> None:
    relay = _Service(sweep=4)
    queue, _, _ = _wire({"outbox_relay": relay})
    await queue.handlers[registry.TASK_OUTBOX_SWEEP]({"older_than_seconds": "90"})
    await queue.handlers["periodic.outbox_sweep"]({})
    assert relay.calls == [
        ("sweep", (), {"older_than_seconds": 90}),
        ("sweep", (), {"older_than_seconds": 30}),
    ]
    assert [e["dispatched"] for e in logs if e["event"] == "outbox.swept"] == [4, 4]


async def test_an_outbox_sweep_that_found_nothing_logs_nothing(logs) -> None:
    queue, _, _ = _wire({"outbox_relay": _Service(sweep=0)})
    await queue.handlers[registry.TASK_OUTBOX_SWEEP]({})
    assert not [e for e in logs if e["event"] == "outbox.swept"]
    bare, _, _ = _wire()
    assert await bare.handlers[registry.TASK_OUTBOX_SWEEP]({}) is None


@pytest.mark.parametrize("purged", [0, 3])
async def test_expired_idempotency_keys_are_purged(logs, purged: int) -> None:
    uow = _Uow()
    uow.purged = purged
    queue, _, _ = _wire(uow=uow)
    before = datetime.now(UTC)
    await queue.handlers[registry.TASK_IDEMPOTENCY_PURGE]({})
    [(name, at)] = uow.calls
    assert name == "purge_expired" and before <= at <= datetime.now(UTC)
    assert uow.commits == 1
    counts = [e["count"] for e in logs if e["event"] == "idempotency.purged"]
    assert counts == ([purged] if purged else [])


@pytest.mark.parametrize("purged", [0, 7])
async def test_dispatched_outbox_rows_are_purged_past_their_retention(logs, purged: int) -> None:
    uow = _Uow()
    uow.purged = purged
    queue, _, _ = _wire(uow=uow)
    await queue.handlers["periodic.outbox_purge"]({})
    assert uow.calls == [("purge_dispatched", TASKS.outbox_retention_seconds)]
    assert uow.commits == 1
    counts = [e["count"] for e in logs if e["event"] == "outbox.purged"]
    assert counts == ([purged] if purged else [])


async def test_reconcile_sweeps_the_outbox_reconciles_archives_and_recovers_stalled_jobs(
    logs,
) -> None:
    relay = _Service(sweep=0)
    archiver = _Service(reconcile={"restaged": 2, "verified": 0})
    extra_runs: list[str] = []

    async def extra() -> None:
        extra_runs.append("ran")

    class _RecoveringQueue(_Queue):
        def __init__(self) -> None:
            super().__init__()
            self.recovered: list[float] = []

        async def recover_stalled(self, *, seconds_since_heartbeat: float) -> int:
            self.recovered.append(seconds_since_heartbeat)
            return 0

    tasks = _RecoveringQueue()
    queue, _, _ = _wire(
        {"outbox_relay": relay, "archive_service": archiver, "extra_reconcilers": [extra]},
        tasks=tasks,
    )
    await queue.handlers[registry.TASK_RECONCILE]({})
    assert relay.calls == [("sweep", (), {"older_than_seconds": 30})]
    assert archiver.calls == [("reconcile", (), {})]
    assert tasks.recovered == [TASKS.stalled_after_seconds]
    assert extra_runs == ["ran"]
    [event] = [e for e in logs if e["event"] == "reconcile.report"]
    assert event["restaged"] == 2


async def test_a_quiet_reconcile_logs_no_report(logs) -> None:
    queue, _, _ = _wire({"archive_service": _Service(reconcile={"restaged": 0})})
    await queue.handlers["periodic.reconcile"]({})
    assert not [e for e in logs if e["event"] == "reconcile.report"]


async def test_reconcile_with_nothing_wired_does_nothing() -> None:
    queue, _, _ = _wire()
    assert await queue.handlers[registry.TASK_RECONCILE]({}) is None


async def test_staged_archive_payloads_are_purged_when_an_archiver_is_wired() -> None:
    archiver = _Service()
    queue, _, _ = _wire({"archive_service": archiver})
    await queue.handlers[registry.TASK_ARCHIVE_PURGE]({})
    await queue.handlers["periodic.archive_purge"]({})
    assert archiver.calls == [("purge_staged_payloads", (), {})] * 2
    bare, _, _ = _wire()
    assert await bare.handlers[registry.TASK_ARCHIVE_PURGE]({}) is None
