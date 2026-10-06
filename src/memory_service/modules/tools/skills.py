"""Learned skills: an active procedure, as a draft Agent Skill a person approves.

The learning job already keeps what worked for each task pattern (``modules.tools.learning``:
the steps, their track record, and - with a model - a distilled title and strategy). A
procedure that is ``active`` is offered to the tenant's administrator as a skill draft: a
``SKILL.md`` built from it, nothing invented. The administrator publishes it - to the team's
skills store (``SKILLS_DIR``, else the Bifrost gateway's skills repository), where agents
already load skills from - or dismisses it. Either decision is kept with the steps it was
about; when the steps change, the draft comes back (as ``changed`` when it was published).

No new job and no new model call: the draft is read from the procedure on request.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.errors import Conflict, NotFound, ProviderNotConfigured
from memory_service.domain.tools import SkillDecision, StoredProcedure
from memory_service.ports.skills import (
    NAME_MAX,
    PROCEDURE_KEY,
    SOURCE,
    SOURCE_KEY,
    TENANT_KEY,
    SkillContent,
    SkillStore,
    valid_name,
)
from memory_service.ports.uow import UnitOfWorkFactory

#: The active procedures one listing reads (best supported first).
DRAFTS_MAX: Final = 100
DESCRIPTION_MAX: Final = 1024
_WORDS = re.compile(r"[a-z0-9]+")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def skill_name(procedure: StoredProcedure) -> str:
    """A name from the title (or the pattern without its placeholders): lowercase words joined
    by hyphens, at most 64 characters, cut at a word."""
    source = _PLACEHOLDER.sub(" ", (procedure.title or procedure.pattern).casefold())
    name = ""
    for word in _WORDS.findall(source):
        longer = f"{name}-{word}" if name else word
        if len(longer) > NAME_MAX:
            break
        name = longer
    return name or "procedure"


def _description(procedure: StoredProcedure) -> str:
    what = procedure.title or procedure.pattern
    text = (
        f"{what.rstrip('.')}. Use for tasks like: {procedure.pattern}. "
        f"Worked in {procedure.success_rate:.0%} of {procedure.support} runs."
    )
    return text[:DESCRIPTION_MAX]


def _steps(procedure: StoredProcedure) -> list[str]:
    lines = []
    for i, step in enumerate(procedure.steps, start=1):
        line = f"{i}. `{step.get('tool')}`"
        if expected := step.get("expected_output"):
            line += f" - returns {', '.join(map(str, expected))}"
        lines.append(line)
        for error, fix in (step.get("failure_modes") or {}).items():
            if fix:
                lines.append(f"   - on {error}: {fix}")
    return lines


def skill_body(procedure: StoredProcedure) -> str:
    """The ``SKILL.md`` body: the title, the strategy (distilled, or the mined steps as
    text), and the steps in order."""
    title = procedure.title or procedure.pattern
    parts = [f"# {title}", "", f"Tasks like: `{procedure.pattern}`", ""]
    if procedure.strategy:
        parts += [procedure.strategy.strip(), ""]
    parts += ["## Steps", "", *_steps(procedure)]
    return "\n".join(parts)


class SkillDraft(BaseModel):
    """A skill as it would be published from one procedure."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="The procedure's id: what publish and dismiss take.")
    state: Literal["new", "changed"] = Field(
        description="new: never published; changed: published, and its steps changed since"
    )
    name: str = Field(description="The skill name it would be published under.")
    description: str = Field(
        description="The SKILL.md description: what it does, when to use it, its track record."
    )
    body: str = Field(description="The SKILL.md body.")
    pattern: str = Field(description="The task pattern it was learned for.")
    support: int = Field(description="Runs that followed it.")
    success_rate: float = Field(description="The share of them that succeeded (0-1).")
    published: SkillDecision | None = Field(
        default=None, description="The last publication, for a changed draft."
    )


def draft(procedure: StoredProcedure) -> SkillDraft | None:
    """The draft an active procedure offers, or None: not active, or decided for these
    steps already."""
    decided = procedure.skill
    if procedure.status != "active":
        return None
    if decided is not None and decided.steps_hash == procedure.steps_hash:
        return None
    # a dismissal after a publication keeps naming it: the skill is still out there
    published = decided if decided is not None and decided.name else None
    return SkillDraft(
        id=procedure.procedure_id,
        state="changed" if published else "new",
        name=published.name if published and published.name else skill_name(procedure),
        description=_description(procedure),
        body=skill_body(procedure),
        pattern=procedure.pattern,
        support=procedure.support,
        success_rate=procedure.success_rate,
        published=published,
    )


class SkillDrafts:
    """List, publish and dismiss the tenant's skill drafts."""

    def __init__(self, uow_factory: UnitOfWorkFactory, store: SkillStore | None) -> None:
        self.uow_factory = uow_factory
        self.store = store

    async def list(self, tenant_id: str) -> list[SkillDraft]:
        async with self.uow_factory() as uow:
            procedures = await uow.procedures.active(tenant_id, limit=DRAFTS_MAX)
        return [d for p in procedures if (d := draft(p)) is not None]

    async def _draft(self, tenant_id: str, procedure_id: str) -> tuple[StoredProcedure, SkillDraft]:
        async with self.uow_factory() as uow:
            procedure = await uow.procedures.get(tenant_id, procedure_id)
        if procedure is None:
            raise NotFound("no such procedure")
        found = draft(procedure)
        if found is None:
            raise Conflict("no draft: the procedure is not active, or decided for its steps")
        return procedure, found

    async def publish(
        self,
        tenant_id: str,
        procedure_id: str,
        *,
        by: str | None,
        name: str | None = None,
        description: str | None = None,
    ) -> SkillDecision:
        """Publish the draft (under ``name`` / with ``description`` when given) and record
        it. ``ProviderNotConfigured`` when the deployment has no skills store."""
        if self.store is None:
            raise ProviderNotConfigured("no skills store: set SKILLS_DIR or BIFROST_URL")
        procedure, found = await self._draft(tenant_id, procedure_id)
        chosen = name or found.name
        if not valid_name(chosen):
            raise Conflict(f"{chosen!r} is not a skill name (lowercase letters, digits, hyphens)")
        content = SkillContent(
            name=chosen,
            description=(description or found.description)[:DESCRIPTION_MAX],
            body=found.body,
            metadata={
                SOURCE_KEY: SOURCE,
                TENANT_KEY: tenant_id,
                PROCEDURE_KEY: procedure.procedure_id,
                "support": str(procedure.support),
                "success_rate": f"{procedure.success_rate:.2f}",
            },
        )
        version = await self.store.publish(content, tenant_id=tenant_id)
        decision = SkillDecision(
            state="published",
            steps_hash=procedure.steps_hash,
            name=chosen,
            version=version,
            destination=self.store.destination,
            decided_by=by,
        )
        await self._record(tenant_id, procedure_id, decision)
        return decision

    async def dismiss(self, tenant_id: str, procedure_id: str, *, by: str | None) -> SkillDecision:
        """Not a skill: the draft is not offered again until the procedure's steps change
        (a published skill stays where it is)."""
        procedure, found = await self._draft(tenant_id, procedure_id)
        decision = SkillDecision(
            state="dismissed",
            steps_hash=procedure.steps_hash,
            name=found.published.name if found.published else None,
            version=found.published.version if found.published else None,
            destination=found.published.destination if found.published else None,
            decided_by=by,
        )
        await self._record(tenant_id, procedure_id, decision)
        return decision

    async def _record(self, tenant_id: str, procedure_id: str, decision: SkillDecision) -> None:
        async with self.uow_factory() as uow:
            await uow.procedures.decide(tenant_id, procedure_id, decision)
            await uow.commit()
