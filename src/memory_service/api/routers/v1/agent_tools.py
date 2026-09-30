"""Public /v1/agent-tools routes: the memory tools an agent calls itself (pull mode), listed
with their JSON input schemas and called in the caller's scope."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.modules.agent_tools.service import AgentTools

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)


class AgentToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(description="JSON schema of the tool's arguments")


class AgentToolsResponse(BaseModel):
    tools: list[AgentToolSpec]


class CallRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"scope": {"agent_run_id": "run_01J8ZK"}, "args": {"query": "supplier"}}]
        },
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    args: dict[str, Any] = Field(default_factory=dict, description="the tool's arguments")


class CallResponse(BaseModel):
    result: Any = Field(description="what the tool returned (items carry an id)")


@router.get(
    "/agent-tools",
    response_model=AgentToolsResponse,
    tags=["agent_tools"],
    summary="The memory tools an agent may call, with their input schemas",
    responses=_ERRORS,
)
async def list_agent_tools(_: ServicePrincipalDep) -> AgentToolsResponse:
    return AgentToolsResponse(tools=[AgentToolSpec(**spec) for spec in AgentTools.specs()])


@router.post(
    "/agent-tools/{name}",
    response_model=CallResponse,
    tags=["agent_tools"],
    summary="Call one memory tool in this scope",
    responses=_ERRORS,
)
async def call_agent_tool(
    name: str, request: Request, body: CallRequest, container: ContainerDep, _: ServicePrincipalDep
) -> CallResponse:
    ctx = build_context(request, container, body.scope)
    return CallResponse(result=await container.services["agent_tools"].call(ctx, name, body.args))
