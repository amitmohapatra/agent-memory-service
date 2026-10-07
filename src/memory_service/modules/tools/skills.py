"""Learned skills: what an agent's successful runs proved, offered back to that agent.

The learning job (``modules.tools.learning``) keeps one procedure per task pattern and
audience: the tool steps that worked, their track record and the fixes for the errors they
met. An agent's own calls are learned across all of its users (``learning.audience_of``). An
``active`` procedure *is* the agent's learned skill for that kind of task: the context offers
the ones that match the task, in full (``modules.context.sections``), and ``tool_search``
returns the best as its plan. Nothing is published anywhere and nobody approves it: a skill
that stops working is retired by the learning job, and an administrator can dismiss one
(``rejected``), which holds until its steps change.

A learned skill whose runs opened one of the agent's own skills (the harness's
``load_skill``, recorded like any tool call) is shown as what those runs added to that skill,
never as a copy of it: the written skill stays the person's.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.errors import NotFound
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import SkillView, StoredProcedure
from memory_service.ports.uow import UnitOfWorkFactory

#: The learned skills one listing reads (best supported first).
SKILLS_MAX: Final = 100
#: The longest name a learned skill gets.
NAME_MAX: Final = 64
#: The harness tools that open a person's skill: steps of the skill machinery, not the task.
SKILL_TOOLS: Final = frozenset({"load_skill", "read_skill_file"})
LOAD_SKILL: Final = "load_skill"
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_STATE: Final[dict[str, Literal["active", "retired", "dismissed"]]] = {
    "active": "active",
    "retired": "retired",
    "rejected": "dismissed",
}


def _words(text: str) -> list[str]:
    """Its words in any script: letters and digits, with the combining marks that belong to
    them (Devanagari vowel signs, Arabic harakat), split at everything else."""
    words, current = [], []
    for ch in text:
        if ch.isalnum() or (current and unicodedata.category(ch).startswith("M")):
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def skill_name(procedure: StoredProcedure) -> str:
    """A name from the title (or the pattern without its placeholders): its words, in any
    script, lowercased and joined by hyphens, at most 64 characters, cut at a word."""
    source = _PLACEHOLDER.sub(" ", (procedure.title or procedure.pattern).casefold())
    name = ""
    for word in _words(source):
        longer = f"{name}-{word}" if name else word
        if len(longer) > NAME_MAX:
            break
        name = longer
    return name or "procedure"


def with_skill(procedure: StoredProcedure) -> str | None:
    """The person's skill this one adds to: the name every run passed to ``load_skill``."""
    ordinals = {
        int(step.get("ordinal", i))
        for i, step in enumerate(procedure.steps)
        if step.get("tool") == LOAD_SKILL
    }
    for binding in procedure.bindings:
        literal = binding.get("literal")
        if (
            binding.get("step") in ordinals
            and binding.get("argument") == "name"
            and isinstance(literal, str)
            and literal
        ):
            return literal
    return None


def task_steps(procedure: StoredProcedure) -> list[str]:
    """The tools the skill calls, in order, without the skill machinery."""
    return [
        str(step.get("tool"))
        for step in procedure.steps
        if step.get("tool") and step.get("tool") not in SKILL_TOOLS
    ]


def fixes(procedure: StoredProcedure) -> list[str]:
    """What worked when a step failed: ``on <error>: <fix>``, in step order."""
    out: list[str] = []
    for step in procedure.steps:
        for error, fix in (step.get("failure_modes") or {}).items():
            if fix:
                out.append(f"{step.get('tool')} on {error}: {fix}")
    return out


def skill_view(procedure: StoredProcedure) -> SkillView:
    """The learned skill as the agent is offered it."""
    return SkillView(
        id=procedure.procedure_id,
        name=skill_name(procedure),
        steps=task_steps(procedure),
        with_skill=with_skill(procedure),
        fixes=fixes(procedure),
        success_rate=procedure.success_rate,
        support=procedure.support,
    )


class LearnedSkill(BaseModel):
    """What an agent learned for one kind of task, as an administrator sees it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="The skill's id (prc_...): what dismiss takes.")
    agent_id: str | None = Field(
        description="The agent that learned it; null for a skill learned from records shared "
        "with a group or workspace."
    )
    name: str = Field(description="Its name: the task, in a few words.")
    status: Literal["active", "retired", "dismissed"] = Field(
        description="active: offered to the agent; retired: stopped working (offered again if "
        "it works again); dismissed: an administrator dismissed it (until its steps change)."
    )
    pattern: str = Field(description="The kind of task it was learned for (values as {slots}).")
    steps: list[str] = Field(description="The tools it calls, in order.")
    with_skill: str | None = Field(
        default=None,
        description="The agent's own (written) skill these runs opened: this one is what they "
        "added to it.",
    )
    fixes: list[str] = Field(
        default_factory=list, description="What worked when a step failed: tool on error: fix."
    )
    success_rate: float = Field(ge=0.0, le=1.0, description="The share of its runs that worked.")
    runs: int = Field(description="The runs it was learned from.")
    users: int = Field(
        description="How many users' runs it was learned from; another user of the agent is "
        "offered it from two on."
    )
    updated_at: datetime = Field(description="When it was last re-learned.")


def learned_skill(procedure: StoredProcedure) -> LearnedSkill:
    return LearnedSkill(
        id=procedure.procedure_id,
        agent_id=procedure.agent_id,
        name=skill_name(procedure),
        status=_STATE.get(procedure.status, "retired"),
        pattern=procedure.pattern,
        steps=task_steps(procedure),
        with_skill=with_skill(procedure),
        fixes=fixes(procedure),
        success_rate=procedure.success_rate,
        runs=procedure.support,
        users=procedure.users,
        updated_at=procedure.updated_at,
    )


class LearnedSkills:
    """List and dismiss the tenant's learned skills."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self.uow_factory = uow_factory

    async def list(self, tenant_id: str, *, agent_id: str | None = None) -> list[LearnedSkill]:
        async with self.uow_factory() as uow:
            found = await uow.procedures.learned(tenant_id, agent_id=agent_id, limit=SKILLS_MAX)
        return [learned_skill(p) for p in found]

    async def dismiss(self, tenant_id: str, skill_id: str) -> LearnedSkill:
        """Not offered again until its steps change (a re-learned skill with new steps is a
        new offer). Dismissing a dismissed skill is a no-op."""
        async with self.uow_factory() as uow:
            found = await uow.procedures.get(tenant_id, skill_id)
            if found is None or found.status == "candidate":
                raise NotFound("no such learned skill")
            if found.status != "rejected":
                await uow.procedures.reject(tenant_id, skill_id)
                # the contexts that offered it are read again (ADR 0031)
                await uow.revisions.bump(tenant_id, RevisionKind.TENANT, "")
                await uow.commit()
                found = await uow.procedures.get(tenant_id, skill_id)
        if found is None:  # deleted meanwhile (tenant purge)
            raise NotFound("no such learned skill")
        return learned_skill(found)
