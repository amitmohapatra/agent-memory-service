"""Tool hints: which tool fits a task, the learned plan, the next step and its arguments.

- candidates: tool search over the catalog (hybrid, per tenant and workspace), narrowed to
  the tools the caller can call (a callable tool the search missed still competes on its
  record), re-scored by how well each has worked and how recently; none when nothing matched
  the task and no plan is learned (an index not built yet is no reason to name any tool),
  and no general view (an overview, a summary, a dashboard) ranked below a specific tool;
- plan / next: the best stored procedure about the task (by its pattern's words or meaning:
  ``matching``) that uses only callable tools, and its first step this run has not done yet;
  never a tool that changes something when the task is a question;
- prefill: each argument of every candidate (keyed ``tool.arg``) - but none of one that
  changes something when the task is a question, nor asked for - resolved in order from the
  procedure's bindings (a literal, or an earlier step's output in this run), the knowledge
  graph (entities of the argument's type named in the task, and the ids tools returned for
  them), the pinned profile and the memories in hand (one about what the task is about),
  and finally the values the task itself
  names - a value the task names right after an argument's own words ("cost centre CC-7")
  fills that argument and no other, as does one of the shape of the argument's own example
  ("such as S12": S40) - each cast to the type the tool's schema declares, and none that is
  not of it, whatever its source ("ST-101" is no ``min_units``);
- missing: required arguments nothing resolved, with the question to ask - but not a
  free-text one the catalog describes (a team's name, a question): the model words it from
  the request.
- forbidden: the candidates a current standing rule of the user forbids for this call
  ("Never create a stock rule at store S40", the ``standing_rules`` section the context
  lists): never the next step, nothing prefilled or asked for, and the rule beside them. A
  caller with no context (``POST /v1/tools/hints``, the ``tool_search`` agent tool) passes
  its visibility, and the rules that apply to the task are read here, beside the search
  (``standing.standing_blocks``, on the task's own vectors): a stored profile never lists
  them.
- confidence: each candidate's score as a number in 0..1 (``1 - e^-score``), less the
  search's part when the tool is not close to the task (``ToolMatch.matches``): the fusion
  ranks a first tool for any task, a greeting or the weather on Mars included, and that
  rank is no evidence the tool answers it. The order stays the score's.

Every read is indexed and bounded; the only model is the query encoder of the tool search
(the task is encoded once, for the search, the procedures' meaning and the standing rules
alike: the tool index and the memories share one indexer, so one vector space), and the same
encoder once per process for each procedure pattern a match reads by meaning.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ProfileBlockView
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
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.context.standing import forbids, listed_rules, standing_blocks
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.retrieval.engine import QueryVectors, RetrievalEngine
from memory_service.modules.tools.index import ToolIndex, ToolMatch
from memory_service.modules.tools.matching import ProcedureMatcher, TaskVectors
from memory_service.modules.tools.patterns import task_slots
from memory_service.modules.tools.skills import skill_view
from memory_service.ports.intelligence import GraphStore
from memory_service.ports.uow import UnitOfWorkFactory

#: The most candidates a caller may ask for.
HINTS_K_MAX: Final = 20
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
#: The value kinds that say what a text is about (a store, a supplier), not how much or when.
_SUBJECT_KINDS: Final = frozenset({"id", "entity", "email", "url"})
#: Word prefixes that compare as the same word ("centre"/"center", "supplier"/"suppliers").
STEM_CHARS: Final = 5
_WORD = re.compile(r"[A-Za-z][a-z]*|[a-z]+|\d+")
#: Words of a tool's name that say it gives the general picture, not one answer.
_BROAD_WORDS: Final = frozenset({"overview", "summary", "dashboard"})
#: How a question ends, in the scripts the service reads.
_QUESTION_MARKS: Final = ("?", "\uff1f", "\u061f")
#: An example an argument's description gives ("The store's id, such as S12.").
_EXAMPLE = re.compile(r"\b(?:such as|e\.g\.|for example|for instance)\s+([^\s,;]+)", re.I)
_TOKEN = re.compile(r"[\w-]+")

#: Argument-name cues for the task's own values, most specific first.
_SLOT_CUES: Final = (
    ("email", ("email", "mail")),
    ("url", ("url", "link", "href")),
    ("date", ("date", "day", "deadline", "when", "due")),
    ("money", ("amount", "price", "total", "cost", "budget", "value")),
    ("id", ("id", "number", "code", "ref", "reference", "sku")),
    ("num", ("qty", "quantity", "count", "number", "amount")),
)


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
    found: Sequence[ToolMatch],
    plan: StoredProcedure | None,
    step: str | None,
    stats: dict[str, ToolStats],
    available: Sequence[str] | None,
    k: int,
) -> list[ToolCandidate]:
    allowed = set(available) if available is not None else None
    relevance = {m.name: m.relevance for m in found}
    matched = {m.name for m in found if m.matches}
    # a callable tool search did not surface (never catalogued, or nothing like the task)
    # still competes on its record, after every tool the search did find
    for tool in [*(plan.tools if plan else ()), *(available or ())]:
        relevance.setdefault(tool, 0.0)
    names = [n for n in relevance if allowed is None or n in allowed]
    if plan is None and not any(relevance[n] for n in names):
        return []  # nothing matched the task (no index yet): naming any tool would be a guess
    now = datetime.now(UTC)
    out = [
        _candidate(
            name,
            relevance[name],
            stats.get(name),
            plan,
            now,
            next_step=name == step,
            matches=name in matched,
        )
        for name in names
    ]
    out.sort(key=lambda c: (-c.score, c.name))
    return _specific(out, plan)[:k]


def _broad(name: str) -> bool:
    """A tool whose name says it gives the general picture (an overview, a summary, a
    dashboard) rather than one answer."""
    return any(w in _BROAD_WORDS for w in _WORD.findall(name.casefold()))


def _specific(ranked: list[ToolCandidate], plan: StoredProcedure | None) -> list[ToolCandidate]:
    """``ranked`` less the general views ranked below a specific tool: the task asked for
    something in particular. One ranked first is what the task asked for, and a step of the
    learned plan always stays."""
    out: list[ToolCandidate] = []
    for candidate in ranked:
        broad = _broad(candidate.name) and not (plan is not None and candidate.name in plan.tools)
        if broad and any(not _broad(c.name) for c in out):
            continue
        out.append(candidate)
    return out


def _changes(entry: ToolDescriptor | None) -> bool:
    """Whether calling the tool changes something (its catalog risk; unknown: it does)."""
    return entry is None or entry.risk != "read"


def _may_lead(tool: str | None, task: str, entries: dict[str, ToolDescriptor]) -> str | None:
    """``tool`` as the next step, unless it changes something and the task is a question: a
    question is answered before anything is changed."""
    if tool is None or asks_change(task) or not _changes(entries.get(tool)):
        return tool
    return None


def _rescored(
    candidates: Sequence[ToolCandidate],
    found: Sequence[ToolMatch],
    plan: StoredProcedure | None,
    stats: dict[str, ToolStats],
    available: Sequence[str] | None,
) -> list[ToolCandidate]:
    """The same candidates scored with no next step (it is forbidden): none leads on the
    plan's bonus."""
    kept = {c.name for c in candidates}
    return [
        c for c in _candidates(found, plan, None, stats, available, HINTS_K_MAX) if c.name in kept
    ]


def _guarded(hints: ToolHints, forbidden: dict[str, str], unasked: set[str]) -> ToolHints:
    """``hints`` less what they would hand over of a forbidden tool: it is never the next
    step, and no argument of it is found or asked for; nor of a tool ``unasked`` for
    (``_unasked``)."""
    held = unasked | set(forbidden)
    return hints.model_copy(
        update={
            "next": None if hints.next in forbidden else hints.next,
            "prefill": {k: p for k, p in hints.prefill.items() if p.tool not in held},
            "missing": [m for m in hints.missing if m.tool not in held],
            "forbidden": forbidden,
        }
    )


def _unasked(
    candidates: Sequence[ToolCandidate], task: str, entries: dict[str, ToolDescriptor]
) -> set[str]:
    """The candidates that change something when the task is a question (``_may_lead``): it
    is answered before anything is changed, so no argument of such a call is found or asked
    for - a value an earlier request named ("min 1 max 3") is no argument of a call nobody
    asked for. They stay candidates."""
    return {c.name for c in candidates if _may_lead(c.name, task, entries) is None}


def _forbidden(
    tools: Iterable[str],
    task: str,
    profile: Sequence[Any],
    prefill: dict[str, Prefill] | None = None,
) -> dict[str, str]:
    """The tools a standing rule of the user forbids for this call, each with the rule: a
    rule the context's ``standing_rules`` section lists (current, and applying to the
    request) that ``standing.forbids`` the tool on the task's values and those found for its
    arguments (``prefill``) ("Never create a stock rule at store S40": ``create_stock_rule``
    at S40)."""
    rules = listed_rules(profile)
    if not rules:
        return {}
    out: dict[str, str] = {}
    for tool in dict.fromkeys(tools):
        values = [task, *(p.value for p in (prefill or {}).values() if p.tool == tool)]
        rule = next((r for r in rules if forbids(r, tool, values)), None)
        if rule is not None:
            out[tool] = rule
    return out


def ruled(
    skills: Sequence[SkillView], task: str, profile: Sequence[Any], hints: ToolHints | None
) -> list[SkillView]:
    """``skills`` with the steps a standing rule of the user forbids for this task
    (``SkillView.forbidden``), with or without tool hints: the rule (``_forbidden``) on the
    task's values, and the hints' own verdict (``ToolHints.forbidden``, which also read the
    values found for each candidate's arguments) for a step they hinted."""
    known = hints.forbidden if hints is not None else {}
    return [
        skill.model_copy(
            update={
                "forbidden": {
                    **_forbidden(skill.steps, task, profile),
                    **{step: known[step] for step in skill.steps if step in known},
                }
            }
        )
        for skill in skills
    ]


def asks_change(task: str) -> bool:
    """Whether the task asks for something to be done rather than asks a question."""
    return not any(mark in task for mark in _QUESTION_MARKS)


def _candidate(
    name: str,
    relevance: float,
    stats: ToolStats | None,
    plan: StoredProcedure | None,
    now: datetime,
    *,
    next_step: bool,
    matches: bool,
) -> ToolCandidate:
    """``matches``: the tool's text is close to the task (``ToolMatch``); when it is not, the
    search's part orders the tool but adds nothing to its confidence."""
    rate = stats.success_rate if stats else None
    fit = relevance * (1 + SUCCESS_WEIGHT * ((0.5 if rate is None else rate) - 0.5))
    score = fit
    why = [f"matches the task ({relevance:.2f})"] if relevance and matches else []
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
        confidence=round(1 - math.exp(-(score if matches else score - fit)), 2),
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
    #: those that take only a number: a value that is none is never named for one
    numbers: frozenset[str] = frozenset()


class ToolHintsService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        index: ToolIndex,
        graph: GraphStore | None,
        matcher: ProcedureMatcher | None = None,
        *,
        engine: RetrievalEngine | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.index = index
        self.graph = graph
        #: which procedures a task is about; without one, by the pattern's words alone
        self.matcher = matcher or ProcedureMatcher(None)
        #: the memories' search the user's standing rules are read with (``_standing``)
        self.engine = engine

    async def procedures(
        self,
        ctx: MemoryExecutionContext,
        task: str,
        scope_keys: Sequence[str],
        *,
        k: int,
        vectors: TaskVectors = None,
    ) -> list[StoredProcedure]:
        """Active procedures the caller may read, about this task, best first: its audience's,
        and its agent's learned skills (``ProcedureRepository.visible``), matched by
        ``ProcedureMatcher``. ``vectors``: the task's, when the request has (or is computing)
        them."""
        async with self.uow_factory() as uow:
            visible = await uow.procedures.visible(
                ctx.tenant_id,
                scope_keys,
                agent_id=ctx.agent_id,
                user_id=ctx.user_id,
                limit=VISIBLE_PROCEDURES_MAX,
            )
        return await self.matcher.rank(task, visible, k, vectors=vectors)

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
        visibility: VisibilitySpecification | None = None,
    ) -> ToolHints:
        """``visibility``: the caller's, when ``profile`` is the stored blocks alone (no
        context was built): the standing rules that apply to the task are read in it
        (``_standing``). The context passes none: its profile already lists them."""
        if vectors is None:
            # encoded once: the catalog search, a procedure the words miss and the standing
            # rules all read them
            vectors = await self.index.encode(task)
        # the stored procedures, the catalog search and the standing rules are independent
        procedures, found, standing = await asyncio.gather(
            self.procedures(ctx, task, scope_keys, k=3, vectors=vectors),
            self.index.search(ctx.tenant_id, ctx.workspace_id, task, vectors=vectors),
            self._standing(task, visibility, vectors),
        )
        profile = [*standing, *profile]
        plan = next(
            (p for p in procedures if available is None or set(p.tools) <= set(available)), None
        )
        # every tool that may become a candidate: its catalog entry is what its arguments
        # are read from
        names = sorted(
            {m.name for m in found} | set(plan.tools if plan else ()) | set(available or ())
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
        step = _may_lead(next_tool(plan, done), task, entries)
        candidates = _candidates(found, plan, step, stats, available, min(k, HINTS_K_MAX))
        # every candidate's arguments, not only the next step's: the caller may take another
        prefill, missing = await self._every_argument(
            [
                _Resolution(ctx, task, entries[c.name], plan, done, scope_keys, memories, profile)
                for c in candidates
                if c.name in entries
            ]
        )
        forbidden = _forbidden([c.name for c in candidates], task, profile, prefill)
        if step in forbidden:
            # the plan's next step is a call the user's standing rule forbids: it leads nothing
            step = None
            candidates = _rescored(candidates, found, plan, stats, available)
        tool = step or next((c.name for c in candidates if _may_lead(c.name, task, entries)), None)
        hints = ToolHints(
            candidates=candidates,
            plan=_plan_hint(plan),
            next=tool,
            prefill=prefill,
            missing=missing,
        )
        return _guarded(hints, forbidden, _unasked(candidates, task, entries))

    async def _standing(
        self, task: str, visibility: VisibilitySpecification | None, vectors: QueryVectors | None
    ) -> list[ProfileBlockView]:
        """The standing rules and preferences that apply to ``task`` in ``visibility``, as the
        context lists them (``standing.standing_blocks``): what ``_forbidden`` reads. None
        without a visibility (the caller's profile already lists them) or a memories'
        search."""
        if visibility is None or self.engine is None:
            return []
        # no memory indexed yet is no rule, not a missing collection
        await self.engine.ensure_collections()
        return await standing_blocks(self.engine, task, visibility, vectors)

    async def _every_argument(
        self, jobs: list[_Resolution]
    ) -> tuple[dict[str, Prefill], list[MissingArgument]]:
        """The arguments of every tool, each read knowing all of their names: a value the
        task names for one tool's argument is not another tool's to take."""
        labels = sorted({arg for job in jobs for arg in _arguments(job.tool, job.plan)})
        numbers = frozenset(n for job in jobs for n in _numeric(job.tool, job.plan))
        found = await asyncio.gather(
            *(self.arguments(replace(job, labels=labels, numbers=numbers)) for job in jobs)
        )
        prefill: dict[str, Prefill] = {}
        missing: list[MissingArgument] = []
        for values, absent in found:
            prefill.update(values)
            missing.extend(absent)
        return prefill, missing

    async def arguments(self, job: _Resolution) -> tuple[dict[str, Prefill], list[MissingArgument]]:
        arguments = _arguments(job.tool, job.plan)
        job.slots = _labelled_slots(
            job.task,
            sorted({*arguments, *job.labels}),
            numbers=job.numbers | _numeric(job.tool, job.plan),
        )
        prefill: dict[str, Prefill] = {}
        missing: list[MissingArgument] = []
        for arg in arguments:
            found = await self._resolve(job, arg)
            if found is not None:
                prefill[f"{job.tool.name}.{arg}"] = found
            elif arg in job.tool.required and not _worded(job, arg):
                missing.append(_missing(job.tool, arg))
        return prefill, missing

    async def _resolve(self, job: _Resolution, arg: str) -> Prefill | None:
        """The first value a source finds for ``arg`` that is of the type the tool declares
        for it (``_as_declared``): one that is not is no value of the argument, wherever it
        was found, and the next source is read."""
        fit = partial(_as_declared, tool=job.tool, arg=arg)
        return (
            fit(_from_procedure(job, arg))
            or fit(await self._from_graph(job, arg))
            or fit(_from_profile(job, arg))
            or fit(_from_memories(job, arg))
            or fit(_from_task(job, arg))
            or fit(_from_memory_text(job, arg))
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
    the literal every successful run used - never for a task that names another value of
    the literal's kind ("style ZZ999999" against a learned 'ST-101'): the task's own wins."""
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
    literal = binding.get("literal")
    if literal is not None and not _names_another(job.task, literal):
        return Prefill(tool=job.tool.name, value=literal, source="procedure")
    return None


def _names_another(task: str, literal: Any) -> bool:
    """Whether ``task`` names a value of the kind ``literal`` is, other than it."""
    kinds = {kind for kind, _ in task_slots(str(literal))}
    named = {value.casefold() for kind, value in task_slots(task) if kind in kinds}
    return bool(named) and str(literal).casefold() not in named


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
    ``supplier_id``): only a labelled one, only after the task, which always wins, and only
    from a memory about what the task is about (``_about_task``)."""
    for item in job.memories:
        text = str(getattr(item, "text", ""))
        for _, value, label in _labelled_slots(text, [arg]):
            if label == arg and _about_task(text, job.task, value):
                return Prefill(
                    tool=job.tool.name,
                    value=value,
                    source="memory",
                    evidence_id=str(getattr(item, "item_id", "")),
                )
    return None


def _about_task(text: str, task: str, value: str) -> bool:
    """Whether a memory that names ``value`` is about what the task is about: it names
    something the task names (an identifier or a name: "Acme Steel"), or nothing but the
    value. An earlier request ("Create a stock rule for ST-101 at S40, min 1 max 3") names
    its own store and style: its "min 1" is no value of a call about style ZZ999999."""
    named = {v.casefold() for kind, v in task_slots(task) if kind in _SUBJECT_KINDS}
    said = {v.casefold() for kind, v in task_slots(text) if kind in _SUBJECT_KINDS}
    return bool(said & named) or said <= {value.casefold()}


def _slot_kinds(job: _Resolution, arg: str) -> list[str]:
    if job.tool.argument_entity_types.get(arg):
        return ["entity"]
    name = arg.casefold()
    return [kind for kind, cues in _SLOT_CUES if any(cue in name for cue in cues)]


def _from_task(job: _Resolution, arg: str) -> Prefill | None:
    """A value the task names for this argument, that the task does not name for another:
    first one the words before it name it for ("cost centre CC-7" for ``cost_centre``), then
    one of the shape of the argument's own example ("such as S12": S40, and the S7 of "Store
    S7"), then one of the kind the argument's name asks for - but never an identifier only
    because the argument's name asks for one (an asset tag is no user id). Each value fills
    one argument."""
    for index, (_, value, label) in enumerate(job.slots):
        if label == arg:
            job.slots.pop(index)
            return Prefill(tool=job.tool.name, value=value, source="task")
    shapes = _example_shapes(job.tool, arg)
    for index, (_, value, label) in enumerate(job.slots):
        shaped = next((w for w in _TOKEN.findall(value) if _shape(w) in shapes), None)
        if shaped is not None and label is None:
            job.slots.pop(index)
            return Prefill(tool=job.tool.name, value=shaped, source="task")
    for wanted in _slot_kinds(job, arg):
        if wanted == "id":
            continue
        for index, (kind, value, label) in enumerate(job.slots):
            if kind == wanted and label is None:
                job.slots.pop(index)
                return Prefill(tool=job.tool.name, value=value, source="task")
    return None


def _declared(tool: ToolDescriptor, arg: str) -> dict[str, Any]:
    """The argument's schema, as the tool declares it (empty when it does not)."""
    return ((tool.input_schema or {}).get("properties") or {}).get(arg) or {}


def _numeric(tool: ToolDescriptor, plan: StoredProcedure | None) -> frozenset[str]:
    """The tool's arguments that take only a number."""
    return frozenset(
        a for a in _arguments(tool, plan) if _declared(tool, a).get("type") in ("number", "integer")
    )


def _shape(value: str) -> str:
    """A value with each run of digits as one 9: S12 and S7 are both s9, LT-2290 is lt-9."""
    return re.sub(r"\d+", "9", value.strip().casefold())


def _example_shapes(tool: ToolDescriptor, arg: str) -> frozenset[str]:
    """The shapes of the identifiers the argument's schema gives as examples (its
    ``examples``, or "such as S12" in its description)."""
    spec = _declared(tool, arg)
    examples = [
        *(spec.get("examples") or []),
        *_EXAMPLE.findall(str(spec.get("description") or "")),
    ]
    return frozenset(
        _shape(str(e).rstrip(".)")) for e in examples if any(ch.isdigit() for ch in str(e))
    )


def _worded(job: _Resolution, arg: str) -> bool:
    """Whether ``arg`` is free text the model words from the request: no identifier, number or
    entity (those are found, or asked for), and the catalog describes it ("The team's name,
    such as network or desktop."). Such an argument is never "missing": the task says it in
    words ("the network team") the hints do not extract. An undescribed one is asked for."""
    if _slot_kinds(job, arg) or _example_shapes(job.tool, arg):
        return False
    if arg in _numeric(job.tool, job.plan):
        return False
    return bool(_declared(job.tool, arg).get("description"))


def _number(value: str) -> float | None:
    """The number ``value`` is, with a currency and thousands separators around it ("EUR
    1,200.50"); None for anything else - an identifier's digits ("ST-101") are no number."""
    bare = re.sub(r"\b[A-Za-z]{3}\b|[$€£,\s]", "", value)
    return float(bare) if re.fullmatch(r"-?\d+(?:\.\d+)?", bare) else None


def _asks_identifier(arg: str) -> bool:
    """``supplier_id`` wants an identifier, not the supplier's name the task gives."""
    return any(w in _IDENTIFIER_WORDS for w in (m.casefold() for m in _WORD.findall(arg)))


def _labelled_slots(
    task: str, arguments: Sequence[str], *, numbers: frozenset[str] = frozenset()
) -> list[tuple[str, str, str | None]]:
    """The values the task names, each with the argument the words just before it name, if
    any: "for cost centre CC-7" names ``cost_centre``. Only the words since the previous
    value count, so a label never reaches back past one. An argument that asks for an
    identifier is no label for a name ("from supplier Acme" is not ``supplier_id``): the
    name stays free for the argument that takes one; and one of ``numbers`` (it takes only a
    number) is no label for what is none ("units of ST-101" is not ``max_units``)."""
    out: list[tuple[str, str, str | None]] = []
    previous_end = 0
    for kind, value in task_slots(task):
        at = task.find(value, previous_end)
        at = at if at >= 0 else task.find(value)
        named = {
            arg: _name_words(arg)
            for arg in arguments
            if (kind in _IDENTIFIER_KINDS or not _asks_identifier(arg))
            and (arg not in numbers or _number(value) is not None)
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
    """The words of an argument's name that say what it is (``cost_centre``: cost, centre);
    a name of generic words only is its own word (``ref``: ref)."""
    words = [m.casefold() for m in _WORD.findall(arg)]
    return [w for w in words if w not in _GENERIC_ARG_WORDS] or words


def _same(left: str, right: str) -> bool:
    if left == right:
        return True
    return min(len(left), len(right)) >= STEM_CHARS and left[:STEM_CHARS] == right[:STEM_CHARS]


def _as_declared(found: Prefill | None, tool: ToolDescriptor, arg: str) -> Prefill | None:
    """The value as the tool's schema types the argument: "700" or "EUR 1,200" for a number
    argument is a number. None for a value that is not one ("ST-101" is no ``min_units``) or
    no whole one for an integer argument; any value of an argument declared otherwise."""
    declared = _declared(tool, arg).get("type")
    if found is None or declared not in ("number", "integer"):
        return found
    value = found.value
    if isinstance(value, str):
        number = _number(value)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
    else:
        number = None
    if number is None or (declared == "integer" and not number.is_integer()):
        return None
    if not isinstance(value, str):
        return found
    return found.model_copy(update={"value": int(number) if number.is_integer() else number})


def _missing(tool: ToolDescriptor, arg: str) -> MissingArgument:
    entity_type = tool.argument_entity_types.get(arg)
    words = arg.replace("_", " ").strip()
    return MissingArgument(
        tool=tool.name,
        arg=arg,
        entity_type=entity_type,
        question=f"{tool.name} needs {words!r}: what should it be?",
    )
