"""Trajectory and chain mining (TOOL_MEMORY.md §30.9 steps 2 and 4).

A run's invocation records, in step order, are a trajectory. Two kinds of edge are derived
from it, both deterministically and both with evidence:

``feeds``        a value produced by step *i* reappears in the arguments of step *j > i*.
                 The edge keeps both field paths, which is the data-flow binding a plan
                 later uses to fill an argument from an earlier output.
``followed_by``  step *j* simply came after step *i*. Weaker, but it is what lets a chain
                 be proposed before any data flow has been observed.

Nothing here invents an edge: every edge points at the invocations that produced it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from memory_service.domain.tools import ToolInvocation

MAX_DEPTH = 6
MAX_FIELDS = 200
_MIN_VALUE_LEN = 3


def flatten(value: Any, *, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Scalar leaves of a nested structure as ``path -> value``. Bounded in depth and width so
    a pathological tool output cannot blow up the miner."""
    out: dict[str, Any] = {}
    if depth > MAX_DEPTH or len(out) > MAX_FIELDS:
        return out
    if isinstance(value, dict):
        for key, sub in value.items():
            if len(out) >= MAX_FIELDS:
                break
            out.update(
                flatten(sub, prefix=f"{prefix}.{key}" if prefix else str(key), depth=depth + 1)
            )
    elif isinstance(value, list):
        for i, sub in enumerate(value[:20]):
            out.update(flatten(sub, prefix=f"{prefix}[{i}]", depth=depth + 1))
    elif value is not None and not isinstance(value, bool):
        out[prefix] = value
    return out


def normalise(value: Any) -> str | None:
    """Comparison form for value matching: strings casefolded and stripped, numbers rendered
    without trailing zeros. Values too short to be discriminating are ignored, otherwise every
    ``1`` or ``true`` in a payload would look like data flow."""
    if isinstance(value, (int, float)):
        text = f"{value:.10g}"
    elif isinstance(value, str):
        text = value.strip().casefold()
    else:
        return None
    return text if len(text) >= _MIN_VALUE_LEN else None


@dataclass(frozen=True)
class ChainEdge:
    """One observed link between two steps of a run."""

    source_tool: str
    target_tool: str
    kind: str  # "feeds" | "followed_by"
    source_field: str | None = None
    target_field: str | None = None
    source_invocation_id: str = ""
    target_invocation_id: str = ""

    @property
    def signature(self) -> tuple[str, str, str, str | None, str | None]:
        return (self.source_tool, self.target_tool, self.kind, self.source_field, self.target_field)


@dataclass
class Trajectory:
    run_id: str
    task: str
    task_pattern: str | None
    steps: list[ToolInvocation] = field(default_factory=list)
    succeeded: bool = False

    @property
    def tool_sequence(self) -> list[str]:
        return [s.tool_name for s in self.steps]


def build_trajectory(
    run_id: str, invocations: Sequence[ToolInvocation], *, succeeded: bool = False
) -> Trajectory:
    steps = sorted(invocations, key=lambda i: (i.step, i.occurred_at))
    task = next((s.task for s in steps if s.task), "")
    pattern = next((s.task_pattern for s in steps if s.task_pattern), None)
    return Trajectory(
        run_id=run_id, task=task, task_pattern=pattern, steps=steps, succeeded=succeeded
    )


def mine_edges(trajectory: Trajectory) -> list[ChainEdge]:
    """Every edge the trajectory supports. Only successful source steps can feed a later step:
    a failed call's output is not data another step legitimately consumed."""
    edges: list[ChainEdge] = []
    steps = trajectory.steps
    for index, target in enumerate(steps):
        target_args = flatten(target.args_redacted)
        target_values = {
            path: norm for path, value in target_args.items() if (norm := normalise(value))
        }
        for source in steps[:index]:
            if not source.succeeded:
                continue
            source_fields = source.output_fields or {}
            for src_path, src_value in source_fields.items():
                src_norm = normalise(src_value)
                if src_norm is None:
                    continue
                for tgt_path, tgt_norm in target_values.items():
                    if src_norm == tgt_norm:
                        edges.append(
                            ChainEdge(
                                source_tool=source.tool_name,
                                target_tool=target.tool_name,
                                kind="feeds",
                                source_field=src_path,
                                target_field=tgt_path,
                                source_invocation_id=source.invocation_id,
                                target_invocation_id=target.invocation_id,
                            )
                        )
        if index:
            previous = steps[index - 1]
            edges.append(
                ChainEdge(
                    source_tool=previous.tool_name,
                    target_tool=target.tool_name,
                    kind="followed_by",
                    source_invocation_id=previous.invocation_id,
                    target_invocation_id=target.invocation_id,
                )
            )
    return _dedupe(edges)


def _dedupe(edges: Iterable[ChainEdge]) -> list[ChainEdge]:
    seen: dict[tuple[Any, ...], ChainEdge] = {}
    for edge in edges:
        seen.setdefault(edge.signature, edge)
    return list(seen.values())


@dataclass
class EdgeSupport:
    """How an edge has behaved across every run of one task pattern."""

    source_tool: str
    target_tool: str
    kind: str
    source_field: str | None = None
    target_field: str | None = None
    support: int = 0
    successes: int = 0
    failures: int = 0
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    failure_modes: dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.support if self.support else 0.0

    @property
    def median_latency_ms(self) -> float | None:
        return _median(self.latencies)

    @property
    def median_cost(self) -> float | None:
        return _median(self.costs)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def accumulate(trajectories: Sequence[Trajectory]) -> dict[tuple[Any, ...], EdgeSupport]:
    """Fold every trajectory of one task pattern into per-edge statistics. An edge seen in a
    failed run accumulates its error class, and the correction — the tool that succeeded in
    its place later in the same run — when there is one."""
    table: dict[tuple[Any, ...], EdgeSupport] = {}
    for trajectory in trajectories:
        for edge in mine_edges(trajectory):
            entry = table.setdefault(
                edge.signature,
                EdgeSupport(
                    source_tool=edge.source_tool,
                    target_tool=edge.target_tool,
                    kind=edge.kind,
                    source_field=edge.source_field,
                    target_field=edge.target_field,
                ),
            )
            entry.support += 1
            target = next(
                (s for s in trajectory.steps if s.invocation_id == edge.target_invocation_id), None
            )
            if target is None:
                continue
            if target.succeeded and trajectory.succeeded:
                entry.successes += 1
            else:
                entry.failures += 1
                if target.error_class:
                    entry.failure_modes[target.error_class] = _correction(trajectory, target)
            if target.latency_ms is not None:
                entry.latencies.append(target.latency_ms)
            if target.cost is not None:
                entry.costs.append(target.cost)
    return table


def _correction(trajectory: Trajectory, failed: ToolInvocation) -> str:
    """What the run did after ``failed`` and got away with: the next successful call. That is
    the adjustment a procedure records, rather than a second procedure for the failure."""
    later = [s for s in trajectory.steps if s.step > failed.step and s.succeeded]
    if not later:
        return ""
    fix = later[0]
    if fix.tool_name == failed.tool_name:
        return f"retry {fix.tool_name} with adjusted arguments"
    return f"use {fix.tool_name}"
