"""Procedures: what worked, in what order, and what to do next (TOOL_MEMORY.md §30.3, §30.6).

Successful trajectories for one task pattern are folded into a prefix tree. The
highest-support path through it is the procedure: an ordered list of steps, each with the
tool, an argument template whose bindings point at earlier outputs, the preconditions those
bindings imply, and the failure branches observed after that step.

Mem^p update rules are followed literally:

*validation* only a run labelled successful contributes a procedure;
*adjustment*  a step that failed is corrected in place (the correction recorded on the edge),
              never duplicated into a second competing procedure;
*decay*       a procedure that stops being used, or starts failing, loses score and is
              archived rather than deleted.

Ranking is deterministic and explainable. Nothing here calls a model; the optional Bifrost
reflection only renames and narrates a procedure that was already mined, and is rejected
unless every step exists in the registry and every binding resolves against a real run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from memory_service.domain.tools import ToolDescriptor, ToolInvocation
from memory_service.modules.tools.trajectories import EdgeSupport, Trajectory, accumulate, flatten

# ranking weights (documented in docs/adr/0018-tool-memory.md)
W_PROCEDURE = 0.5
W_OUTCOME = 0.3
W_RECENCY = 0.2
FAILURE_PENALTY = 0.4
RECENCY_HALF_LIFE_DAYS = 14.0


@dataclass
class Binding:
    """Where one argument of a step comes from."""

    argument: str
    source_step: int | None = None
    source_field: str | None = None
    literal: Any | None = None

    @property
    def resolvable_from_trajectory(self) -> bool:
        return self.source_step is not None and self.source_field is not None

    def render(self) -> str:
        if self.resolvable_from_trajectory:
            return f"{self.argument}=step{self.source_step}.{self.source_field}"
        return f"{self.argument}={self.literal!r}"


@dataclass
class ProcedureStep:
    ordinal: int
    tool: str
    bindings: list[Binding] = field(default_factory=list)
    expected_output: list[str] = field(default_factory=list)
    support: int = 0
    success_rate: float = 0.0
    median_latency_ms: float | None = None
    median_cost: float | None = None
    failure_modes: dict[str, str] = field(default_factory=dict)

    @property
    def preconditions(self) -> list[str]:
        """What must already be known before this step can run: every binding that reads an
        earlier step's output is a precondition on that step having produced it."""
        return [
            f"step{b.source_step}.{b.source_field}"
            for b in self.bindings
            if b.resolvable_from_trajectory
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "tool": self.tool,
            "bindings": [
                {
                    "argument": b.argument,
                    "source_step": b.source_step,
                    "source_field": b.source_field,
                    "literal": b.literal,
                }
                for b in self.bindings
            ],
            "preconditions": self.preconditions,
            "expected_output": self.expected_output,
            "support": self.support,
            "success_rate": round(self.success_rate, 4),
            "median_latency_ms": self.median_latency_ms,
            "median_cost": self.median_cost,
            "failure_modes": self.failure_modes,
        }


@dataclass
class Procedure:
    task_pattern: str
    steps: list[ProcedureStep] = field(default_factory=list)
    support: int = 0
    success_rate: float = 0.0
    run_ids: list[str] = field(default_factory=list)
    invocation_ids: list[str] = field(default_factory=list)
    last_used_at: datetime | None = None
    when_not_to_use: str | None = None
    script: str | None = None

    @property
    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    def render(self) -> str:
        lines = [f"Procedure for: {self.task_pattern}"]
        for step in self.steps:
            args = ", ".join(b.render() for b in step.bindings) or "-"
            lines.append(
                f"{step.ordinal + 1}. {step.tool}({args})"
                f"  [support {step.support}, success {step.success_rate:.0%}]"
            )
            for error, fix in step.failure_modes.items():
                if fix:
                    lines.append(f"     on {error}: {fix}")
        if self.when_not_to_use:
            lines.append(f"Do not use when: {self.when_not_to_use}")
        return "\n".join(lines)

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_pattern": self.task_pattern,
            "steps": [s.to_payload() for s in self.steps],
            "support": self.support,
            "success_rate": round(self.success_rate, 4),
            "run_ids": self.run_ids[:20],
            "invocation_ids": self.invocation_ids[:50],
            "when_not_to_use": self.when_not_to_use,
            "script": self.script,
        }

    def render_script(self) -> str:
        """Starlark-shaped rendering for Bifrost code mode: each step becomes a call whose
        arguments reference earlier results by name."""
        lines = []
        for step in self.steps:
            args = []
            for b in step.bindings:
                if b.resolvable_from_trajectory:
                    args.append(f"{b.argument}=step{b.source_step}[{b.source_field!r}]")
                else:
                    args.append(f"{b.argument}={b.literal!r}")
            server, _, tool = step.tool.partition("-")
            call = f"{server}.{tool or step.tool}({', '.join(args)})"
            lines.append(f"step{step.ordinal} = {call}")
        if lines:
            lines.append(f"result = step{self.steps[-1].ordinal}")
        return "\n".join(lines)


def mine_procedure(pattern: str, trajectories: list[Trajectory]) -> Procedure | None:
    """Build the procedure for one task pattern from its successful trajectories.

    Validation rule: only successful runs are considered. Failed runs still contribute their
    failure modes to the steps they share, which is how a correction is learned without a
    competing procedure being created.
    """
    successful = [t for t in trajectories if t.succeeded and t.steps]
    if not successful:
        return None
    edges = accumulate(trajectories)
    best_sequence = _best_sequence(successful)
    if not best_sequence:
        return None
    by_tool: dict[str, list[ToolInvocation]] = defaultdict(list)
    for trajectory in successful:
        for step in trajectory.steps:
            by_tool[step.tool_name].append(step)

    steps: list[ProcedureStep] = []
    for ordinal, tool in enumerate(best_sequence):
        samples = by_tool.get(tool, [])
        step = ProcedureStep(
            ordinal=ordinal,
            tool=tool,
            bindings=_bindings(tool, ordinal, best_sequence, successful),
            expected_output=_expected_output(samples),
        )
        _apply_edge_stats(step, edges, previous=best_sequence[ordinal - 1] if ordinal else None)
        if not step.support:
            step.support = len(samples)
            step.success_rate = (
                sum(1 for s in samples if s.succeeded) / len(samples) if samples else 0.0
            )
        steps.append(step)

    matching = [t for t in successful if t.tool_sequence[: len(best_sequence)] == best_sequence]
    total = [t for t in trajectories if t.tool_sequence[: len(best_sequence)] == best_sequence]
    last_used = max(
        (s.occurred_at for t in total for s in t.steps),
        default=None,
    )
    return Procedure(
        task_pattern=pattern,
        steps=steps,
        support=len(matching),
        success_rate=len(matching) / len(total) if total else 0.0,
        run_ids=[t.run_id for t in matching],
        invocation_ids=[s.invocation_id for t in matching for s in t.steps],
        last_used_at=last_used,
    )


def _best_sequence(trajectories: list[Trajectory]) -> list[str]:
    """Highest-support path through the prefix tree of successful tool sequences. Ties break on
    the longer sequence, then alphabetically, so the result is stable across runs."""
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for trajectory in trajectories:
        sequence = tuple(trajectory.tool_sequence)
        for length in range(1, len(sequence) + 1):
            counts[sequence[:length]] += 1
    if not counts:
        return []
    best = max(counts.items(), key=lambda kv: (kv[1], len(kv[0]), [-ord(c) for c in kv[0][0]]))
    return list(best[0])


def _apply_edge_stats(
    step: ProcedureStep, edges: dict[tuple[Any, ...], EdgeSupport], *, previous: str | None
) -> None:
    if previous is None:
        return
    relevant = [
        e for e in edges.values() if e.source_tool == previous and e.target_tool == step.tool
    ]
    if not relevant:
        return
    step.support = max(e.support for e in relevant)
    step.success_rate = max(e.success_rate for e in relevant)
    latencies = [e.median_latency_ms for e in relevant if e.median_latency_ms is not None]
    costs = [e.median_cost for e in relevant if e.median_cost is not None]
    step.median_latency_ms = min(latencies) if latencies else None
    step.median_cost = min(costs) if costs else None
    for edge in relevant:
        step.failure_modes.update(edge.failure_modes)


def _bindings(
    tool: str, ordinal: int, sequence: list[str], trajectories: list[Trajectory]
) -> list[Binding]:
    """Arguments of ``tool`` at this position, bound to an earlier step's output whenever the
    value was observed to come from there, otherwise kept as the literal that recurred."""
    observed: dict[str, list[Any]] = defaultdict(list)
    sources: dict[str, tuple[int, str]] = {}
    for trajectory in trajectories:
        if trajectory.tool_sequence[: len(sequence)] != sequence:
            continue
        if ordinal >= len(trajectory.steps):
            continue
        step = trajectory.steps[ordinal]
        if step.tool_name != tool:
            continue
        for path, value in flatten(step.args_redacted).items():
            observed[path].append(value)
            if path in sources:
                continue
            for earlier_index in range(ordinal):
                earlier = trajectory.steps[earlier_index]
                for out_path, out_value in (earlier.output_fields or {}).items():
                    if _same(out_value, value):
                        sources[path] = (earlier_index, out_path)
                        break
                if path in sources:
                    break
    bindings: list[Binding] = []
    for path, values in sorted(observed.items()):
        if path in sources:
            source_step, source_field = sources[path]
            bindings.append(
                Binding(argument=path, source_step=source_step, source_field=source_field)
            )
        elif len(set(map(str, values))) == 1:
            bindings.append(Binding(argument=path, literal=values[0]))
        else:
            bindings.append(Binding(argument=path, literal=None))
    return bindings


def _same(left: Any, right: Any) -> bool:
    from memory_service.modules.tools.trajectories import normalise

    a, b = normalise(left), normalise(right)
    return a is not None and a == b


def _expected_output(samples: list[ToolInvocation]) -> list[str]:
    """Output field paths present in every successful sample: the shape a caller may rely on."""
    successful = [s for s in samples if s.succeeded and s.output_fields]
    if not successful:
        return []
    common: set[str] | None = None
    for sample in successful:
        paths = set(sample.output_fields or {})
        common = paths if common is None else (common & paths)
    return sorted(common or set())


# --------------------------------------------------------------------------- ranking


def recency_weight(when: datetime | None, *, now: datetime | None = None) -> float:
    """1.0 for something used right now, halving every ``RECENCY_HALF_LIFE_DAYS``."""
    if when is None:
        return 0.0
    now = now or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    age_days = max(0.0, (now - when).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def score_tool(
    *,
    in_procedure: bool,
    procedure_position: int | None,
    success_rate: float,
    invocations: int,
    last_used_at: datetime | None,
    failed_recently: bool,
    now: datetime | None = None,
) -> float:
    """procedure match x outcome statistics x recency, minus a penalty for a known failure.

    Confidence in the outcome term grows with the number of observations, so one lucky call
    does not outrank a tool with a long record.
    """
    procedure_term = 0.0
    if in_procedure:
        procedure_term = 1.0 if procedure_position == 0 else 0.8
    confidence = min(1.0, invocations / 5.0)
    outcome_term = success_rate * confidence
    score = (
        W_PROCEDURE * procedure_term
        + W_OUTCOME * outcome_term
        + W_RECENCY * recency_weight(last_used_at, now=now)
    )
    if failed_recently:
        score -= FAILURE_PENALTY
    return max(0.0, round(score, 6))


def decayed(procedure: Procedure, *, idle_days: float = 60.0, now: datetime | None = None) -> bool:
    """Decay rule: a procedure nobody has used for ``idle_days``, or whose success rate has
    fallen below half, is archived rather than offered."""
    now = now or datetime.now(UTC)
    if procedure.success_rate < 0.5:
        return True
    if procedure.last_used_at is None:
        return False
    last = procedure.last_used_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return now - last > timedelta(days=idle_days)


def validate_against_registry(
    procedure: Procedure, registry: dict[str, ToolDescriptor]
) -> list[str]:
    """Problems that disqualify a procedure: an unknown tool, or a binding that points at a
    step which does not exist or does not produce the field. Returns an empty list when valid."""
    problems: list[str] = []
    for step in procedure.steps:
        if step.tool not in registry:
            problems.append(f"step {step.ordinal}: unknown tool {step.tool!r}")
        for binding in step.bindings:
            if binding.source_step is None:
                continue
            if binding.source_step >= step.ordinal:
                problems.append(
                    f"step {step.ordinal}: binding {binding.argument!r} reads a later step"
                )
    return problems
