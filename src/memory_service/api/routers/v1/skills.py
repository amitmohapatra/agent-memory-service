"""/v1/skills: what the tenant's agents learned (docs/api/skills.md).

An agent's successful runs become its learned skills on their own (``modules.tools.skills``):
the context offers the ones that match the task, nobody publishes or approves them. These two
routes are the administrator's view: what each agent learned, and dismissing one that should
not be offered.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field

from memory_service.api.deps import AdministeredTenantDep, ContainerDep, request_context
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.params import SkillIdPath
from memory_service.modules.tools.skills import SKILLS_MAX, LearnedSkill

router = APIRouter()


class LearnedSkillsResponse(BaseModel):
    skills: list[LearnedSkill] = Field(
        description=f"The learned skills, best supported first (at most {SKILLS_MAX})."
    )


AgentFilter = Annotated[
    str | None,
    Query(
        min_length=1,
        max_length=200,
        description="Only the skills this agent learned (e.g. support); every agent's when "
        "absent. (Not agent_id: that names the agent a call acts as.)",
    ),
]


@router.get(
    "/skills",
    response_model=LearnedSkillsResponse,
    tags=["skills"],
    summary="What the tenant's agents learned from their successful runs",
    description="The tenant's administrator credential. Each skill is what one agent's runs "
    "of one kind of task proved: its steps, what fixed a failing step, its track record, and "
    "its state (active: offered to the agent; retired: stopped working; dismissed). A skill "
    "is offered to the agent's other users once two users produced it.",
    responses=error_responses(401, 403, 422, 503),
)
async def learned_skills(
    container: ContainerDep, tenant_id: AdministeredTenantDep, agent: AgentFilter = None
) -> LearnedSkillsResponse:
    return LearnedSkillsResponse(
        skills=await container.services["learned_skills"].list(tenant_id, agent_id=agent)
    )


@router.post(
    "/skills/{skill_id}/dismiss",
    response_model=LearnedSkill,
    tags=["skills"],
    summary="Stop offering a learned skill until its steps change",
    description="The tenant's administrator credential. The agent is no longer offered it; "
    "if its runs later settle on different steps, that is a new skill. Dismissing a "
    "dismissed skill returns it unchanged. 404 when there is no such skill.",
    responses=error_responses(401, 403, 404, 422, 503),
)
async def dismiss_learned_skill(
    request: Request,
    skill_id: SkillIdPath,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> Response:
    async def handler(uow):  # type: ignore[no-untyped-def]
        skill = await container.services["learned_skills"].dismiss(tenant_id, skill_id)
        return 200, skill.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        request_context(request, tenant_id),
        key=request.state.idempotency_key,
        payload={"action": "dismiss", "skill": skill_id},
        handler=handler,
    )
