"""The ``tools.learn`` job: fold recorded calls into stored procedures and graph edges.

It reads the calls not learned yet (a partial index), and for every (tenant, audience, task
pattern) they touch it re-mines that pattern from its newest calls with the prefix-tree miner
(``modules.tools.procedures``) and updates the one stored procedure of that pattern - a delta,
never a rewrite of the others. A procedure is ``active`` (offered) only when enough runs
support it and enough of them succeeded.

With the tenant's model (use ``procedure_abstraction``), an active procedure whose steps
changed is distilled into a title and a strategy from what succeeded AND what failed
(ReasoningBank / AWM style). Without one, the title is the pattern and the strategy the
miner's own rendering, so behaviour is uniform.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from memory_service.domain.revisions import RevisionKind, audience_revision_keys
from memory_service.domain.tools import RunOutcome, StoredProcedure, ToolInvocation, stable_hash
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.tools.edges import tool_edges
from memory_service.modules.tools.procedures import Procedure, decayed, mine_procedure
from memory_service.modules.tools.trajectories import Trajectory, build_trajectory
from memory_service.observability.logging import get_logger
from memory_service.ports.credentials import ModelIdentity
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
#: An unlabelled run counts as a success once this old, when none of its calls failed.
WEAK_POSITIVE_AFTER: Final = timedelta(hours=24)
#: Example runs of each kind the distillation prompt sees.
DISTIL_EXAMPLES: Final = 3
TITLE_MAX_CHARS: Final = 120
STRATEGY_MAX_CHARS: Final = 1200

DISTIL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "strategy": {"type": "string"},
        "avoid": {"type": "string"},
    },
    "required": ["title", "strategy", "avoid"],
    "additionalProperties": False,
}
_DISTIL_SYSTEM = (
    "You distil a reusable procedure from an agent's recorded tool calls. You are given a "
    "task pattern (values replaced by typed placeholders), the steps that succeeded with "
    "where each argument came from, and what happened in runs that failed. Write a short "
    "title, the strategy (when to apply the steps and how their arguments are filled) and "
    "what to avoid (from the failures; empty when none failed). Use only what is given. "
    "Write in the language of the task pattern. Return JSON only: "
    '{"title": ..., "strategy": ..., "avoid": ...}.'
)


def succeeded(outcome: RunOutcome | None, calls: Sequence[ToolInvocation], now: datetime) -> bool:
    """A run is a success when labelled so, or - as a weak positive - when it is older than
    the window and none of its calls failed."""
    if outcome is not None:
        return outcome.success
    if not calls or any(not c.succeeded for c in calls):
        return False
    newest = max(c.occurred_at for c in calls)
    newest = newest if newest.tzinfo else newest.replace(tzinfo=UTC)
    return now - newest >= WEAK_POSITIVE_AFTER


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


def merge(
    existing: StoredProcedure | None,
    mined: Procedure | None,
    *,
    tenant_id: str,
    audience: str,
    owner: ToolInvocation,
) -> StoredProcedure | None:
    """The stored procedure after a re-mine: only its own row changes, and its distilled
    title and strategy are kept for as long as its steps are."""
    if mined is None:
        if existing is None or existing.status in ("retired", "rejected"):
            return None
        return existing.model_copy(update={"status": "retired", "updated_at": datetime.now(UTC)})
    steps, bindings = _steps(mined)
    steps_hash = stable_hash({"tools": mined.tools, "bindings": bindings})
    base = existing or StoredProcedure(
        tenant_id=tenant_id, scope_key=audience, pattern=mined.task_pattern
    )
    return base.model_copy(
        update={
            "steps": steps,
            "bindings": bindings,
            "success_rate": round(mined.success_rate, 4),
            "support": mined.support,
            "status": _status(mined, steps_hash, existing),
            "steps_hash": steps_hash,
            "owner_principal": owner.principal_id,
            "workspace_id": owner.workspace_id,
            "strategy": base.strategy if base.steps_hash == steps_hash else mined.render(),
            "title": base.title or mined.task_pattern[:TITLE_MAX_CHARS],
            "updated_at": datetime.now(UTC),
        }
    )


def _render_run(trajectory: Trajectory) -> str:
    calls = ", ".join(
        f"{s.tool_name}({','.join(sorted(s.args_redacted))})"
        + ("" if s.succeeded else f" -> {s.status} {s.error_class or ''}".rstrip())
        for s in trajectory.steps
    )
    return f"- {calls}"


def distil_prompt(procedure: StoredProcedure, trajectories: Sequence[Trajectory]) -> str:
    wins = [t for t in trajectories if t.succeeded][:DISTIL_EXAMPLES]
    losses = [t for t in trajectories if not t.succeeded][:DISTIL_EXAMPLES]
    return "\n".join(
        [
            f"Task pattern: {procedure.pattern}",
            f"Steps (support {procedure.support}, success rate {procedure.success_rate:.0%}):",
            procedure.strategy,
            "Runs that succeeded:",
            *(_render_run(t) for t in wins),
            "Runs that failed:",
            *([_render_run(t) for t in losses] or ["- none"]),
        ]
    )


def accept_distilled(result: dict[str, Any] | None) -> tuple[str, str] | None:
    """The model's title and strategy (with what to avoid), or None when unusable."""
    if not result:
        return None
    title = " ".join(str(result.get("title", "")).split())[:TITLE_MAX_CHARS]
    strategy = str(result.get("strategy", "")).strip()
    avoid = str(result.get("avoid", "")).strip()
    if not title or not strategy:
        return None
    text = f"{strategy}\nAvoid: {avoid}" if avoid else strategy
    return title, text[:STRATEGY_MAX_CHARS]


def _unchanged(merged: StoredProcedure, existing: StoredProcedure | None) -> bool:
    return (
        existing is not None
        and merged.model_copy(update={"updated_at": existing.updated_at}) == existing
    )


class ToolLearning:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, assist: LLMAssist, graph: GraphStore | None
    ) -> None:
        self.uow_factory = uow_factory
        self.assist = assist
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
                (c.tenant_id, c.visibility_keys[0], c.task_pattern)
                for c in calls
                if c.task_pattern and c.visibility_keys
            }
        )
        for tenant, audience, pattern in groups:
            await self._mine(tenant, audience, str(pattern))
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

    async def _mine(self, tenant_id: str, audience: str, pattern: str) -> None:
        async with self.uow_factory() as uow:
            calls = await uow.tools.for_pattern(
                tenant_id, audience, pattern, limit=PATTERN_CALLS_MAX
            )
            trajectories = await self._trajectories(uow, tenant_id, calls)
            existing = await uow.procedures.by_pattern(tenant_id, audience, pattern)
            owner = next((c for c in calls if c.succeeded), calls[0]) if calls else None
            if owner is None:
                return
            mined = mine_procedure(pattern, trajectories)
            merged = merge(existing, mined, tenant_id=tenant_id, audience=audience, owner=owner)
            if merged is None or _unchanged(merged, existing):
                return
            await uow.procedures.save(merged)
            await uow.revisions.bump(tenant_id, RevisionKind.TENANT, "")
            await uow.commit()
        if merged.status == "active" and merged.distilled != merged.steps_hash:
            await self._distil(merged, trajectories)

    @staticmethod
    async def _trajectories(
        uow: Any, tenant_id: str, calls: Sequence[ToolInvocation]
    ) -> list[Trajectory]:
        by_run: dict[str, list[ToolInvocation]] = defaultdict(list)
        for call in calls:
            if call.run_id:
                by_run[call.run_id].append(call)
        outcomes = await uow.tools.outcomes(tenant_id, list(by_run))
        now = datetime.now(UTC)
        return [
            build_trajectory(run, runs, succeeded=succeeded(outcomes.get(run), runs, now))
            for run, runs in by_run.items()
        ]

    async def _distil(self, procedure: StoredProcedure, trajectories: Sequence[Trajectory]) -> None:
        """Title and strategy from the tenant's model; the deterministic ones stay otherwise."""
        if not procedure.owner_principal:
            return
        identity = ModelIdentity(procedure.tenant_id, procedure.owner_principal)
        async with self.assist.bound(identity):
            if not self.assist.wants("procedure_abstraction"):
                return
            result = await self.assist.structured(
                "procedure_abstraction",
                system=_DISTIL_SYSTEM,
                user=distil_prompt(procedure, trajectories),
                schema=DISTIL_SCHEMA,
                max_tokens=600,
            )
        accepted = accept_distilled(result)
        if accepted is None:
            return
        title, strategy = accepted
        async with self.uow_factory() as uow:
            current = await uow.procedures.get(procedure.tenant_id, procedure.procedure_id)
            if current is None or current.steps_hash != procedure.steps_hash:
                return  # re-mined meanwhile: the next run distils the new steps
            await uow.procedures.save(
                current.model_copy(
                    update={"title": title, "strategy": strategy, "distilled": current.steps_hash}
                )
            )
            await uow.revisions.bump(procedure.tenant_id, RevisionKind.TENANT, "")
            await uow.commit()

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
