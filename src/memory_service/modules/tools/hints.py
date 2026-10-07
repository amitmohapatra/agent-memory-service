"""Tool hints: which tool fits a task, the learned plan, the next step and its arguments.

- candidates: tool search over the catalog (hybrid, per tenant and workspace), narrowed to
  the tools the caller can call (a callable tool the search missed still competes on its
  record), re-scored by how well each has worked and how recently;
- plan / next: the best stored procedure for the task pattern that uses only callable tools,
  and its first step this run has not done yet;
- prefill: each argument of every candidate (keyed ``tool.arg``), resolved in order from the
  procedure's bindings (a literal, or an earlier step's output in this run), the knowledge
  graph (entities of the argument's type named in the task, and the ids tools returned for
  them), the pinned profile and the memories in hand, and finally the values the task itself
  names - a value the task names right after an argument's own words ("cost centre CC-7")
  fills that argument and no other - each cast to the type the tool's schema declares;
- missing: required arguments nothing resolved, with the question to ask.
- confidence: each candidate's score as a number in 0..1 (``1 - e^-score``, so the order
  is the score's).

Every read is indexed and bounded; the only model is the query encoder of the tool search.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.graph import IDENTIFIED_BY
from memory_service.domain.tools import (
    MissingArgument,
    Prefill,
    SkillView,
    StoredProcedure,
    ToolCandidate,
    ToolDescriptor,
    ToolHints,
    ToolInvocation,
    ToolStats,
)
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.modules.tools.index import ToolIndex
from memory_service.modules.tools.patterns import similarity, task_pattern, task_slots
from memory_service.modules.tools.skills import skill_view
from memory_service.ports.intelligence import GraphStore
from memory_service.ports.uow import UnitOfWorkFactory

#: The most candidates a caller may ask for.
HINTS_K_MAX: Final = 20
#: Pattern similarity at which a stored procedure is taken to be about the task.
PROCEDURE_MATCH: Final = 0.6
#: Active procedures read per request before ranking (the newest learned).
VISIBLE_PROCEDURES_MAX: Final = 100
#: How much a tool's success rate moves its score around the neutral 0.5.
SUCCESS_WEIGHT: Final = 0.5
#: Added to the score of a tool the learned plan uses.
PLAN_BONUS: Final = 0.5
#: Added to the plan's next step: it leads whatever the search found (the step to take is
#: known, a closer text match is not a reason to take another).
NEXT_STEP_BONUS: Final = 1.0
#: Added when the tool was used within ``RECENT``.
RECENCY_BONUS: Final = 0.1
RECENT: Final = timedelta(days=30)
#: Entities of the task looked up in the graph for one argument.
TASK_ENTITIES_MAX: Final = 6
#: Words before a value the task names that may say which argument it is for.
SLOT_CONTEXT_WORDS: Final = 3
#: Argument-name words too generic to say which value is meant ("supplier_id": supplier).
_GENERIC_ARG_WORDS: Final = frozenset(
    {"id", "ids", "no", "num", "number", "code", "ref", "value", "name", "the", "of", "to", "for"}
)
#: Words in an argument's name that ask for an identifier, and the value kinds that are one.
_IDENTIFIER_WORDS: Final = frozenset({"id", "ids", "code", "ref", "number", "no"})
_IDENTIFIER_KINDS: Final = frozenset({"id", "num"})
#: Word prefixes that compare as the same word ("centre"/"center", "supplier"/"suppliers").
STEM_CHARS: Final = 5
_WORD = re.compile(r"[A-Za-z][a-z]*|[a-z]+|\d+")

#: Argument-name cues for the task's own values, most specific first.
_SLOT_CUES: Final = (
    ("email", ("email", "mail")),
    ("url", ("url", "link", "href")),
    ("date", ("date", "day", "deadline", "when", "due")),
    ("money", ("amount", "price", "total", "cost", "budget", "value")),
    ("id", ("id", "number", "code", "ref", "reference", "sku")),
    ("num", ("qty", "quantity", "count", "number", "amount")),
)


def rank_procedures(
    task: str, procedures: Sequence[StoredProcedure], k: int
) -> list[StoredProcedure]:
    """The procedures about this task (pattern similarity), best first."""
    pattern = task_pattern(task)
    scored = [(similarity(pattern, p.pattern), p) for p in procedures]
    kept = [(s, p) for s, p in scored if s >= PROCEDURE_MATCH]
    kept.sort(key=lambda sp: (-sp[0], -sp[1].success_rate, -sp[1].support, sp[1].procedure_id))
    return [p for _, p in kept[:k]]


def _norm(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def next_tool(plan: StoredProcedure | None, done: Sequence[ToolInvocation]) -> str | None:
    """The plan's first step this run has not completed (in order), or None."""
    if plan is None:
        return None
    tools, at = plan.tools, 0
    for call in done:
        if call.succeeded and at < len(tools) and call.tool_name == tools[at]:
            at += 1
    return tools[at] if at < len(tools) else None


def _candidates(
    found: Sequence[tuple[str, float]],
    plan: StoredProcedure | None,
    step: str | None,
    stats: dict[str, ToolStats],
    available: Sequence[str] | None,
    k: int,
) -> list[ToolCandidate]:
    allowed = set(available) if available is not None else None
    relevance = dict(found)
    # a callable tool search did not surface (never catalogued, or nothing like the task)
    # still competes on its record, after every tool the search did find
    for tool in [*(plan.tools if plan else ()), *(available or ())]:
        relevance.setdefault(tool, 0.0)
    now = datetime.now(UTC)
    out = [
        _candidate(name, rel, stats.get(name), plan, now, next_step=name == step)
        for name, rel in relevance.items()
        if allowed is None or name in allowed
    ]
    out.sort(key=lambda c: (-c.score, c.name))
    return out[:k]


def _candidate(
    name: str,
    relevance: float,
    stats: ToolStats | None,
    plan: StoredProcedure | None,
    now: datetime,
    *,
    next_step: bool,
) -> ToolCandidate:
    rate = stats.success_rate if stats else None
    score = relevance * (1 + SUCCESS_WEIGHT * ((0.5 if rate is None else rate) - 0.5))
    why = [f"matches the task ({relevance:.2f})"] if relevance else []
    if plan is not None and name in plan.tools:
        score += PLAN_BONUS
        why.append(f"step {plan.tools.index(name) + 1} of the learned plan")
    if next_step:
        score += NEXT_STEP_BONUS
        why.append("the plan's next step")
    if stats is not None and stats.calls:
        why.append(f"succeeded {stats.successes}/{stats.calls}")
    last = stats.last_used_at if stats else None
    if last is not None and now - (last if last.tzinfo else last.replace(tzinfo=UTC)) <= RECENT:
        score += RECENCY_BONUS
        why.append("used recently")
    return ToolCandidate(
        name=name,
        score=round(score, 4),
        confidence=round(1 - math.exp(-score), 2),
        success_rate=rate,
        why="; ".join(why),
    )


def _plan_hint(plan: StoredProcedure | None) -> SkillView | None:
    return skill_view(plan) if plan is not None else None


def _arguments(entry: ToolDescriptor, plan: StoredProcedure | None) -> list[str]:
    """The tool's arguments: required first, then its schema's, then any the learned plan
    binds for it (a tool recorded but never catalogued has only those)."""
    properties = list(((entry.input_schema or {}).get("properties") or {}).keys())
    bound: list[str] = []
    if plan is not None and entry.name in plan.tools:
        ordinal = plan.tools.index(entry.name)
        bound = [str(b["argument"]) for b in plan.bindings if b.get("step") == ordinal]
    return list(dict.fromkeys([*entry.required, *properties, *bound]))


@dataclass
class _Resolution:
    """What one argument search has to work with."""

    ctx: MemoryExecutionContext
    task: str
    tool: ToolDescriptor
    plan: StoredProcedure | None
    done: Sequence[ToolInvocation]
    scope_keys: Sequence[str]
    memories: Sequence[Any] = ()
    profile: Sequence[Any] = ()
    #: the values the task names: (kind, value, the argument the words before it name)
    slots: list[tuple[str, str, str | None]] = field(default_factory=list)
    #: every candidate's arguments: a value named for another tool's argument is not this one's
    labels: Sequence[str] = ()


class ToolHintsService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, index: ToolIndex, graph: GraphStore | None
    ) -> None:
        self.uow_factory = uow_factory
        self.index = index
        self.graph = graph

    async def procedures(
        self, ctx: MemoryExecutionContext, task: str, scope_keys: Sequence[str], *, k: int
    ) -> list[StoredProcedure]:
        """Active procedures the caller may read, about this task, best first: its audience's,
        and its agent's learned skills (``ProcedureRepository.visible``)."""
        async with self.uow_factory() as uow:
            visible = await uow.procedures.visible(
                ctx.tenant_id,
                scope_keys,
                agent_id=ctx.agent_id,
                user_id=ctx.user_id,
                limit=VISIBLE_PROCEDURES_MAX,
            )
        return rank_procedures(task, visible, k)

    async def hints(
        self,
        ctx: MemoryExecutionContext,
        task: str,
        *,
        available: Sequence[str] | None,
        k: int,
        scope_keys: Sequence[str],
        memories: Sequence[Any] = (),
        profile: Sequence[Any] = (),
        vectors: QueryVectors | None = None,
    ) -> ToolHints:
        # the stored procedures and the catalog search are independent reads
        procedures, found = await asyncio.gather(
            self.procedures(ctx, task, scope_keys, k=3),
            self.index.search(ctx.tenant_id, ctx.workspace_id, task, vectors=vectors),
        )
        plan = next(
            (p for p in procedures if available is None or set(p.tools) <= set(available)), None
        )
        # every tool that may become a candidate: its catalog entry is what its arguments
        # are read from
        names = sorted(
            {n for n, _ in found} | set(plan.tools if plan else ()) | set(available or ())
        )
        async with self.uow_factory() as uow:
            entries = {
                e.name: e
                for e in await uow.tools.catalog(
                    ctx.tenant_id, workspace_id=ctx.workspace_id, names=names, limit=len(names) + 1
                )
            }
            stats = await uow.tools.stats(ctx.tenant_id, names)
            done = (
                await uow.tools.invocations_for_run(
                    ctx.tenant_id, ctx.agent_run_id, scope_keys=scope_keys
                )
                if ctx.agent_run_id
                else []
            )
        step = next_tool(plan, done)
        candidates = _candidates(found, plan, step, stats, available, min(k, HINTS_K_MAX))
        tool = step or (candidates[0].name if candidates else None)
        # every candidate's arguments, not only the next step's: the caller may take another
        prefill, missing = await self._every_argument(
            [
                _Resolution(ctx, task, entries[c.name], plan, done, scope_keys, memories, profile)
                for c in candidates
                if c.name in entries
            ]
        )
        return ToolHints(
            candidates=candidates,
            plan=_plan_hint(plan),
            next=tool,
            prefill=prefill,
            missing=missing,
        )

    async def _every_argument(
        self, jobs: list[_Resolution]
    ) -> tuple[dict[str, Prefill], list[MissingArgument]]:
        """The arguments of every tool, each read knowing all of their names: a value the
        task names for one tool's argument is not another tool's to take."""
        labels = sorted({arg for job in jobs for arg in _arguments(job.tool, job.plan)})
        found = await asyncio.gather(*(self.arguments(replace(job, labels=labels)) for job in jobs))
        prefill: dict[str, Prefill] = {}
        missing: list[MissingArgument] = []
        for values, absent in found:
            prefill.update(values)
            missing.extend(absent)
        return prefill, missing

    async def arguments(self, job: _Resolution) -> tuple[dict[str, Prefill], list[MissingArgument]]:
        arguments = _arguments(job.tool, job.plan)
        job.slots = _labelled_slots(job.task, sorted({*arguments, *job.labels}))
        prefill: dict[str, Prefill] = {}
        missing: list[MissingArgument] = []
        for arg in arguments:
            found = await self._resolve(job, arg)
            if found is not None:
                prefill[f"{job.tool.name}.{arg}"] = _as_declared(found, job.tool, arg)
            elif arg in job.tool.required:
                missing.append(_missing(job.tool, arg))
        return prefill, missing

    async def _resolve(self, job: _Resolution, arg: str) -> Prefill | None:
        return (
            _from_procedure(job, arg)
            or await self._from_graph(job, arg)
            or _from_profile(job, arg)
            or _from_memories(job, arg)
            or _from_task(job, arg)
            or _from_memory_text(job, arg)
        )

    async def _from_graph(self, job: _Resolution, arg: str) -> Prefill | None:
        entity_type = job.tool.argument_entity_types.get(arg)
        if self.graph is None or not entity_type:
            return None
        names = [canonical_entity(n) for n in extract_entities(job.task)[:TASK_ENTITIES_MAX]]
        if not names:
            return None
        entities = await self.graph.find_entities(
            job.ctx.tenant_id, names, scope_keys=job.scope_keys
        )
        typed = [e for e in entities if e.entity_type.casefold() == entity_type.casefold()]
        if not typed:
            return None
        entity = sorted(typed, key=lambda e: (-e.mention_count, e.entity_id))[0]
        if _norm(arg).endswith("id"):
            return await self._identifier(job, arg, entity)
        return Prefill(
            tool=job.tool.name, value=entity.name, source="graph", evidence_id=entity.entity_id
        )

    async def _identifier(self, job: _Resolution, arg: str, entity: Any) -> Prefill | None:
        assert self.graph is not None
        relations = await self.graph.entity_relations(
            job.ctx.tenant_id, entity.entity_id, scope_keys=job.scope_keys, current=True, limit=20
        )
        ids = [
            r
            for r in relations
            if r.predicate == IDENTIFIED_BY and r.subject_id == entity.entity_id
        ]
        if not ids:
            return None
        objects = await self.graph.get_entities(
            job.ctx.tenant_id, [ids[0].object_id], scope_keys=job.scope_keys
        )
        if not objects:
            return None
        return Prefill(
            tool=job.tool.name,
            value=objects[0].name,
            source="graph",
            evidence_id=ids[0].relation_id,
        )


def _from_procedure(job: _Resolution, arg: str) -> Prefill | None:
    """A binding of the plan's step for this tool: an earlier step's output in this run, or
    the literal every successful run used."""
    if job.plan is None or job.tool.name not in job.plan.tools:
        return None
    ordinal = job.plan.tools.index(job.tool.name)
    binding = next(
        (b for b in job.plan.bindings if b.get("step") == ordinal and b.get("argument") == arg),
        None,
    )
    if binding is None:
        return None
    if binding.get("source_step") is not None:
        return _from_earlier_step(job, arg, binding)
    if binding.get("literal") is not None:
        return Prefill(tool=job.tool.name, value=binding["literal"], source="procedure")
    return None


def _from_earlier_step(job: _Resolution, arg: str, binding: dict[str, Any]) -> Prefill | None:
    assert job.plan is not None
    source_tool = job.plan.tools[int(binding["source_step"])]
    field_path = str(binding.get("source_field") or "")
    for call in reversed(job.done):
        if call.tool_name == source_tool and call.succeeded and field_path in call.output_fields:
            return Prefill(
                tool=job.tool.name,
                value=call.output_fields[field_path],
                source="procedure",
                evidence_id=call.invocation_id,
            )
    return None


def _from_profile(job: _Resolution, arg: str) -> Prefill | None:
    """A ``name: value`` line of a pinned profile block whose name is the argument's."""
    wanted = _norm(arg)
    for block in job.profile:
        for line in str(getattr(block, "text", "")).splitlines():
            name, sep, value = line.lstrip("-* ").partition(":")
            if sep and value.strip() and _norm(name) == wanted:
                return Prefill(
                    tool=job.tool.name,
                    value=value.strip(),
                    source="profile",
                    evidence_id=str(getattr(block, "block", "")),
                )
    return None


def _from_memories(job: _Resolution, arg: str) -> Prefill | None:
    """A memory in hand whose predicate is the argument's name."""
    wanted = _norm(arg)
    for item in job.memories:
        attributes = getattr(item, "attributes", {}) or {}
        if _norm(str(attributes.get("predicate") or "")) == wanted and attributes.get("object"):
            return Prefill(
                tool=job.tool.name,
                value=attributes["object"],
                source="memory",
                evidence_id=str(getattr(item, "item_id", "")),
            )
    return None


def _from_memory_text(job: _Resolution, arg: str) -> Prefill | None:
    """A value a memory in hand names for this argument ("their supplier id is SUP-40" for
    ``supplier_id``): only a labelled one, and only after the task, which always wins."""
    for item in job.memories:
        for _, value, label in _labelled_slots(str(getattr(item, "text", "")), [arg]):
            if label == arg:
                return Prefill(
                    tool=job.tool.name,
                    value=value,
                    source="memory",
                    evidence_id=str(getattr(item, "item_id", "")),
                )
    return None


def _slot_kinds(job: _Resolution, arg: str) -> list[str]:
    if job.tool.argument_entity_types.get(arg):
        return ["entity"]
    name = arg.casefold()
    return [kind for kind, cues in _SLOT_CUES if any(cue in name for cue in cues)]


def _from_task(job: _Resolution, arg: str) -> Prefill | None:
    """A value the task names for this argument: first one the words before it name it for
    ("cost centre CC-7" for ``cost_centre``), then one of the kind the argument's name asks
    for that the task does not name for another argument. Each value fills one argument."""
    for index, (_, value, label) in enumerate(job.slots):
        if label == arg:
            job.slots.pop(index)
            return Prefill(tool=job.tool.name, value=value, source="task")
    for wanted in _slot_kinds(job, arg):
        for index, (kind, value, label) in enumerate(job.slots):
            if kind == wanted and label is None:
                job.slots.pop(index)
                return Prefill(tool=job.tool.name, value=value, source="task")
    return None


def _asks_identifier(arg: str) -> bool:
    """``supplier_id`` wants an identifier, not the supplier's name the task gives."""
    return any(w in _IDENTIFIER_WORDS for w in (m.casefold() for m in _WORD.findall(arg)))


def _labelled_slots(task: str, arguments: Sequence[str]) -> list[tuple[str, str, str | None]]:
    """The values the task names, each with the argument the words just before it name, if
    any: "for cost centre CC-7" names ``cost_centre``. Only the words since the previous
    value count, so a label never reaches back past one. An argument that asks for an
    identifier is no label for a name ("from supplier Acme" is not ``supplier_id``): the
    name stays free for the argument that takes one."""
    out: list[tuple[str, str, str | None]] = []
    previous_end = 0
    for kind, value in task_slots(task):
        at = task.find(value, previous_end)
        at = at if at >= 0 else task.find(value)
        named = {
            arg: _name_words(arg)
            for arg in arguments
            if kind in _IDENTIFIER_KINDS or not _asks_identifier(arg)
        }
        between = [w.casefold() for w in _WORD.findall(task[previous_end:at])]
        # an identifier that spells its argument ("SKU-22" for ``sku``) names it first
        own = _WORD.findall(value.casefold()) if kind == "id" else []
        label = _named(named, own) or _named(named, between[-SLOT_CONTEXT_WORDS:])
        out.append((kind, value, label))
        previous_end = max(previous_end, at + len(value))
    return out


def _named(named: dict[str, list[str]], near: Sequence[str]) -> str | None:
    """The first argument one of whose name words is among ``near``."""
    return next(
        (arg for arg, words in named.items() if any(_same(w, n) for w in words for n in near)),
        None,
    )


def _name_words(arg: str) -> list[str]:
    """The words of an argument's name that say what it is (``cost_centre``: cost, centre)."""
    return [w for w in (m.casefold() for m in _WORD.findall(arg)) if w not in _GENERIC_ARG_WORDS]


def _same(left: str, right: str) -> bool:
    if left == right:
        return True
    return min(len(left), len(right)) >= STEM_CHARS and left[:STEM_CHARS] == right[:STEM_CHARS]


def _as_declared(found: Prefill, tool: ToolDescriptor, arg: str) -> Prefill:
    """The value as the tool's schema types the argument: "700" or "EUR 1,200" for a number
    argument is a number. A value that is not one is left as it was found."""
    properties = (tool.input_schema or {}).get("properties") or {}
    declared = (properties.get(arg) or {}).get("type")
    if declared not in ("number", "integer") or not isinstance(found.value, str):
        return found
    digits = re.sub(r"[^\d.\-]", "", found.value.replace(",", ""))
    try:
        number = float(digits)
    except ValueError:
        return found
    value: int | float = int(number) if declared == "integer" or number.is_integer() else number
    return found.model_copy(update={"value": value})


def _missing(tool: ToolDescriptor, arg: str) -> MissingArgument:
    entity_type = tool.argument_entity_types.get(arg)
    words = arg.replace("_", " ").strip()
    return MissingArgument(
        tool=tool.name,
        arg=arg,
        entity_type=entity_type,
        question=f"{tool.name} needs {words!r}: what should it be?",
    )
