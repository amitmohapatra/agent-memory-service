"""Map a LangGraph ``RunnableConfig`` to Memory Service lineage.

LangGraph identifies a conversation by ``configurable.thread_id`` and every executing
task by ``configurable.checkpoint_ns`` — ``"node:task"`` segments joined by ``|``, one
per graph level (``"supervisor:A|research:B|worker:C"`` is node ``worker`` of the
``research`` subgraph invoked by the ``supervisor`` subgraph of the root graph). Task
ids are derived deterministically from the checkpoint, the step and the node, so they
are stable across retries of the same superstep — which makes them the right agent run
ids and the right idempotency material.

Mapping rules:

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

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9._:\-]")
_SCOPE_FIELDS = (
    "tenant_id",
    "workspace_id",
    "user_id",
    "group_ids",
    "thread_id",
    "session_id",
    "turn_id",
    "work_id",
    "task_id",
    "agent_id",
    "agent_group_id",
    "agent_run_id",
    "parent_agent_run_id",
    "trace_id",
    "correlation_id",
)


def safe_id(value: str, *, max_len: int = 200) -> str:
    """Coerce an arbitrary LangGraph id into the service's id alphabet."""
    cleaned = _SAFE.sub("-", str(value)).strip("-.")[:max_len]
    return cleaned or "x"


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
    out: dict[str, Any] = {k: v for k, v in defaults.items() if k in _SCOPE_FIELDS}
    out.update({k: v for k, v in lineage.overrides.items() if k in _SCOPE_FIELDS})
    if lineage.thread_id and not out.get("thread_id"):
        out["thread_id"] = safe_id(f"{thread_prefix}{lineage.thread_id}")
    thread = out.get("thread_id")
    if thread and not out.get("session_id"):
        # one session per LangGraph thread unless the app supplies its own
        out["session_id"] = safe_id(f"{thread}-session")
    if thread and not out.get("turn_id"):
        # a turn = one human input: its message id when the messages convention is used,
        # otherwise the superstep (apps with their own turn ids pass configurable.memory)
        out["turn_id"] = safe_id(
            f"turn-{turn_hint}" if turn_hint else f"{thread}-step{lineage.step or 0}"
        )
    runs = list(lineage.subgraphs)
    if agent and lineage.task is not None:
        runs.append(Segment(node=agent, task_id=lineage.task.task_id))
    elif agent:
        runs.append(Segment(node=agent, task_id=lineage.run_id or "run"))
    if runs and not lineage.overrides.get("agent_run_id"):
        last = runs[-1]
        out["agent_id"] = lineage.overrides.get("agent_id") or safe_id(last.node)
        out["agent_run_id"] = safe_id(f"lg-{last.task_id}")
        out["parent_agent_run_id"] = safe_id(f"lg-{runs[-2].task_id}") if len(runs) > 1 else None
    if lineage.run_id and not out.get("correlation_id"):
        out["correlation_id"] = safe_id(lineage.run_id)
    return {k: v for k, v in out.items() if v is not None}
