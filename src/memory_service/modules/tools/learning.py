"""The ``tools.learn`` job: fold recorded calls into stored procedures and graph edges.

It reads the calls not learned yet (a partial index), and for every (tenant, audience, task
pattern) they touch it re-mines that pattern from its newest calls with the prefix-tree miner
(``modules.tools.procedures``) and updates the one stored procedure of that pattern - a delta,
never a rewrite of the others. A procedure is ``active`` (offered) only when enough runs
support it and enough of them succeeded.

An agent's own calls (its PRIVATE records, for whichever user it ran) share one audience,
``agent_audience``: the agent learns from all of its users, and the procedure says how many
users produced it, so it reaches the agent's other users only once two of them did
(``StoredProcedure``). Calls shared wider keep their own audience key.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from memory_service.domain.feedback import FeedbackSource
from memory_service.domain.revisions import RevisionKind, audience_revision_keys
from memory_service.domain.tools import (
    RunOutcome,
    StoredProcedure,
    ToolInvocation,
    agent_audience,
    stable_hash,
)
from memory_service.modules.tools.edges import tool_edges
from memory_service.modules.tools.procedures import Procedure, decayed, mine_procedure
from memory_service.modules.tools.trajectories import Trajectory, build_trajectory
from memory_service.observability.logging import get_logger
from memory_service.ports.intelligence import GraphStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

#: Calls one run of the job learns from.
LEARN_BATCH: Final = 500
#: The newest calls of one pattern a procedure is re-mined from.
PATTERN_CALLS_MAX: Final = 480
#: Runs that must follow a procedure before it is offered, and the share that must succeed.
PROCEDURE_MIN_SUPPORT: Final = 2
PROCEDURE_MIN_SUCCESS_RATE: Final = 0.6


def succeeded(outcome: RunOutcome | None, calls: Sequence[ToolInvocation]) -> bool | None:
    """A run is a success or a failure when labelled so, and a failure when nobody labelled
    it and one of its calls failed. ``None`` - no evidence either way - for a run nobody
    labelled whose calls all worked (however old: a fresh run is no failure), one whose label
    is its own unverified report of success (``RunOutcome.teaches``), and one the system
    reported failed though every call of it worked: a time limit, a model error or a stall is
    the run's failure, not its procedure's."""
    clean = all(c.succeeded for c in calls)
    if outcome is None:
        return False if not calls or not clean else None
    if not outcome.teaches:
        return None
    if outcome.source == FeedbackSource.SYSTEM and not outcome.success and calls and clean:
        return None
    return outcome.success


def _steps(procedure: Procedure) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps, bindings = [], []
    for step in procedure.steps:
        payload = step.to_payload()
        for binding in payload.pop("bindings"):
            bindings.append({"step": step.ordinal, **binding})
        steps.append(payload)
    return steps, bindings


def _status(mined: Procedure, steps_hash: str, existing: StoredProcedure | None) -> str:
    if existing is not None and existing.status == "rejected" and existing.steps_hash == steps_hash:
        return "rejected"
    admitted = (
        mined.support >= PROCEDURE_MIN_SUPPORT
        and mined.success_rate >= PROCEDURE_MIN_SUCCESS_RATE
        and not decayed(mined)
    )
    if admitted:
        return "active"
    return "retired" if existing is not None and existing.status != "candidate" else "candidate"


def audience_of(call: ToolInvocation) -> tuple[str, str | None]:
    """Where a call is learned: an agent's own (PRIVATE) record under the agent's audience,
    whichever user it ran for, with that agent's id; any other under its first visibility
    key."""
    key = call.visibility_keys[0]
    if call.agent_id and key == _agent_private_key(call):
        return agent_audience(call.tenant_id, call.agent_id), call.agent_id
    return key, None


def _agent_private_key(call: ToolInvocation) -> str:
    principal = (
        f"agent:{call.user_id}/{call.agent_id}" if call.user_id else f"agent:{call.agent_id}"
    )
    return f"principal:{call.tenant_id}/{principal}"


def contributors(calls: Sequence[ToolInvocation]) -> tuple[int, str | None]:
    """How many distinct users the calls ran for (no user counts as one), and that user when
    there is exactly one."""
    users = {c.user_id for c in calls}
    sole = next(iter(users)) if len(users) == 1 else None
    return len(users), sole


def merge(
    existing: StoredProcedure | None,
    mined: Procedure | None,
    *,
    tenant_id: str,
    audience: str,
    owner: ToolInvocation,
    agent_id: str | None = None,
    users: int = 0,
    sole_user: str | None = None,
) -> StoredProcedure | None:
    """The stored procedure after a re-mine: only its own row changes."""
    if mined is None:
        if existing is None or existing.status in ("retired", "rejected"):
            return None
        return existing.model_copy(update={"status": "retired", "updated_at": datetime.now(UTC)})
    steps, bindings = _steps(mined)
    steps_hash = stable_hash({"tools": mined.tools, "bindings": bindings})
    base = existing or StoredProcedure(
        tenant_id=tenant_id, scope_key=audience, pattern=mined.task_pattern, agent_id=agent_id
    )
    return base.model_copy(
        update={
            "steps": steps,
            "bindings": bindings,
            "success_rate": round(mined.success_rate, 4),
            "support": mined.support,
            "status": _status(mined, steps_hash, existing),
            "steps_hash": steps_hash,
            "workspace_id": owner.workspace_id,
            "users": users,
            "sole_user": sole_user,
            "updated_at": datetime.now(UTC),
        }
    )


def _unchanged(merged: StoredProcedure, existing: StoredProcedure | None) -> bool:
    return (
        existing is not None
        and merged.model_copy(update={"updated_at": existing.updated_at}) == existing
    )


class ToolLearning:
    def __init__(self, uow_factory: UnitOfWorkFactory, graph: GraphStore | None) -> None:
        self.uow_factory = uow_factory
        self.graph = graph

    async def learn(self, tenant_id: str | None = None) -> int:
        """One batch: re-mine every pattern the unlearned calls touch, write their graph
        edges, mark them learned. Returns how many calls were learned."""
        async with self.uow_factory() as uow:
            calls = await uow.tools.unlearned(tenant_id=tenant_id, limit=LEARN_BATCH)
        if not calls:
            return 0
        groups = sorted(
            {
                (c.tenant_id, *audience_of(c), c.task_pattern)
                for c in calls
                if c.task_pattern and c.visibility_keys
            },
            key=lambda g: (g[0], g[1], g[2] or "", g[3]),
        )
        for tenant, audience, agent_id, pattern in groups:
            await self._mine(tenant, audience, agent_id, str(pattern))
        await self._edges(calls)
        by_tenant: dict[str, list[str]] = defaultdict(list)
        for call in calls:
            by_tenant[call.tenant_id].append(call.invocation_id)
        async with self.uow_factory() as uow:
            for tenant, ids in by_tenant.items():
                await uow.tools.mark_learned(tenant, ids)
            await uow.commit()
        log.info("tools.learned", calls=len(calls), patterns=len(groups))
        return len(calls)

    async def _mine(
        self, tenant_id: str, audience: str, agent_id: str | None, pattern: str
    ) -> None:
        async with self.uow_factory() as uow:
            if agent_id is None:
                calls = await uow.tools.for_pattern(
                    tenant_id, audience, pattern, limit=PATTERN_CALLS_MAX
                )
            else:
                calls = [
                    c
                    for c in await uow.tools.for_agent_pattern(
                        tenant_id, agent_id, pattern, limit=PATTERN_CALLS_MAX
                    )
                    if audience_of(c)[0] == audience
                ]
            trajectories = await self._trajectories(uow, tenant_id, calls)
            existing = await uow.procedures.by_pattern(tenant_id, audience, pattern)
            owner = next((c for c in calls if c.succeeded), calls[0]) if calls else None
            if owner is None:
                return
            mined = mine_procedure(pattern, trajectories)
            users, sole_user = contributors(calls)
            merged = merge(
                existing,
                mined,
                tenant_id=tenant_id,
                audience=audience,
                owner=owner,
                agent_id=agent_id,
                users=users,
                sole_user=sole_user,
            )
            if merged is None or _unchanged(merged, existing):
                return
            await uow.procedures.save(merged)
            await uow.revisions.bump(tenant_id, RevisionKind.TENANT, "")
            await uow.commit()

    @staticmethod
    async def _trajectories(
        uow: Any, tenant_id: str, calls: Sequence[ToolInvocation]
    ) -> list[Trajectory]:
        by_run: dict[str, list[ToolInvocation]] = defaultdict(list)
        for call in calls:
            if call.run_id:
                by_run[call.run_id].append(call)
        outcomes = await uow.tools.outcomes(tenant_id, list(by_run))
        labelled = {run: succeeded(outcomes.get(run), runs) for run, runs in by_run.items()}
        return [
            build_trajectory(run, by_run[run], succeeded=success)
            for run, success in labelled.items()
            if success is not None
        ]

    async def _edges(self, calls: Sequence[ToolInvocation]) -> None:
        if self.graph is None:
            return
        by_tenant: dict[str, list[ToolInvocation]] = defaultdict(list)
        for call in calls:
            by_tenant[call.tenant_id].append(call)
        for tenant_id, tenant_calls in by_tenant.items():
            async with self.uow_factory() as uow:
                entries = {
                    e.tool_id: e
                    for e in await uow.tools.catalog_by_ids(
                        tenant_id, sorted({c.tool_id for c in tenant_calls})
                    )
                }
            audiences: set[tuple[RevisionKind, str]] = set()
            for call in tenant_calls:
                entry = entries.get(call.tool_id)
                entities, relations = tool_edges(call, entry) if entry else ([], [])
                if relations:
                    await self.graph.upsert_entities(entities)
                    await self.graph.upsert_relations(relations)
                    # the edges' readers, not the whole tenant (ADR 0031)
                    for relation in relations:
                        audiences |= audience_revision_keys(
                            tenant_id, relation.visibility_keys
                        ) or {(RevisionKind.GRAPH, "")}
            if audiences:
                async with self.uow_factory() as uow:
                    for kind, identifier in sorted(audiences):
                        await uow.revisions.bump(tenant_id, kind, identifier)
                    await uow.commit()
