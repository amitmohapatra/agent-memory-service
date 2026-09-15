"""Map a LangGraph ``RunnableConfig`` to Memory Service lineage.

LangGraph identifies a conversation by ``configurable.thread_id`` and every executing
task by ``configurable.checkpoint_ns`` — ``"node:task"`` segments joined by ``|``, one
per graph level (``"supervisor:A|research:B|worker:C"`` is node ``worker`` of the
``research`` subgraph invoked by the ``supervisor`` subgraph of the root graph). Task
ids are derived deterministically from the checkpoint, the step and the node, so they
are stable across retries of the same superstep — which makes them the right agent run
ids and the right idempotency material.

Mapping rules (the generic part lives in :mod:`universal_memory.integrations.core`):

- ``thread_id`` -> ``Scope.thread_id`` (optionally prefixed, ids must match the service's
  ``ID_PATTERN``: letters, digits, ``._:-``).
- every segment *except the last* is a subgraph invocation and becomes an agent run:
  ``agent_id`` = the node name, ``agent_run_id`` = its task id, parent = the enclosing
  segment's task id. The last segment is the executing node; it is an agent run only
  when the wrapper asks for one (``agent=...``).
- root-graph nodes therefore act *as the user* (their memories are USER/THREAD scoped),
  subgraphs act as agents (RUN-scoped working memory, ADR 0013).
- ``session_id`` defaults to one session per thread and ``turn_id`` to the pending human
  message's id (or the superstep) — apps with real sessions/turns pass them explicitly.
- ``configurable.memory`` (a dict) can set or override any ``Scope`` field explicitly —
  ``tenant_id``, ``user_id``, ``workspace_id``, ``group_ids``, ``agent_group_id``,
  ``work_id``, ``session_id``, ``turn_id`` — for apps that carry identity in the config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from universal_memory.integrations.core import AgentRun, safe_id
from universal_memory.integrations.core import scope_fields as _core_scope_fields

__all__ = ["Lineage", "Segment", "lineage_from_config", "safe_id", "scope_fields"]


@dataclass(frozen=True)
class Segment:
    node: str
    task_id: str

    @classmethod
    def parse(cls, raw: str) -> Segment:
        node, _, task = raw.partition(":")
        return cls(node=node or raw, task_id=task or raw)


@dataclass(frozen=True)
class Lineage:
    """What the config says about *where* we are in the graph."""

    thread_id: str | None
    segments: tuple[Segment, ...] = ()
    checkpoint_id: str | None = None
    run_id: str | None = None
    step: int | None = None
    node: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    @property
    def subgraphs(self) -> tuple[Segment, ...]:
        """Enclosing subgraph invocations (every segment but the executing node's)."""
        return self.segments[:-1]

    @property
    def task(self) -> Segment | None:
        return self.segments[-1] if self.segments else None

    @property
    def path(self) -> str:
        return "|".join(f"{s.node}:{s.task_id}" for s in self.segments)


def lineage_from_config(config: Mapping[str, Any] | None) -> Lineage:
    conf = dict((config or {}).get("configurable") or {})
    meta = dict((config or {}).get("metadata") or {})
    ns = str(conf.get("checkpoint_ns") or "")
    segments = tuple(Segment.parse(s) for s in ns.split("|") if s)
    node = meta.get("langgraph_node") or (segments[-1].node if segments else None)
    step = meta.get("langgraph_step")
    return Lineage(
        thread_id=conf.get("thread_id"),
        segments=segments,
        checkpoint_id=conf.get("checkpoint_id"),
        run_id=str((config or {}).get("run_id") or "") or None,
        step=int(step) if step is not None else None,
        node=str(node) if node else None,
        overrides=dict(conf.get("memory") or {}),
    )


def scope_fields(
    lineage: Lineage,
    *,
    defaults: Mapping[str, Any],
    thread_prefix: str = "",
    agent: str | None = None,
    turn_hint: str | None = None,
) -> dict[str, Any]:
    """Build the ``Scope`` keyword arguments for this position in the graph.

    ``defaults`` are the adapter-level identity (tenant, user, workspace...); the config's
    ``memory`` dict overrides them; the checkpoint namespace supplies the agent lineage;
    ``agent`` turns the executing node itself into an agent run.
    """
    runs = [AgentRun(s.node, s.task_id) for s in lineage.subgraphs]
    if agent and lineage.task is not None:
        runs.append(AgentRun(agent, lineage.task.task_id))
    elif agent:
        runs.append(AgentRun(agent, lineage.run_id or "run"))
    return _core_scope_fields(
        defaults=defaults,
        overrides=lineage.overrides,
        thread_id=lineage.thread_id,
        thread_prefix=thread_prefix,
        turn_hint=turn_hint,
        step=lineage.step,
        runs=runs,
        run_prefix="lg-",
        correlation_id=lineage.run_id,
    )
