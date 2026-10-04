"""Public /v1/agent-tools routes: the memory tools an agent calls itself (pull mode), listed
with their JSON input schemas and called in the caller's scope."""

from __future__ import annotations

import functools
from typing import Any, Final

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.caching import conditional, etag_of
from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
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
    toolbox: list[str] | None = Field(
        default=None,
        max_length=500,
        description="the caller's own tools, which tool_search chooses among (every catalog "
        "tool when omitted); not part of the arguments the model sees",
    )


class CallResponse(BaseModel):
    result: Any = Field(description="what the tool returned (items carry an id)")


#: The set changes only with a release: a client may keep it five minutes without asking,
#: and revalidates with If-None-Match after that.
AGENT_TOOLS_CACHE_CONTROL: Final = "private, max-age=300"


@functools.cache
def _listing() -> tuple[bytes, str]:
    """The listing's bytes and tag, built once per process: the set is fixed."""
    body = AgentToolsResponse(tools=[AgentToolSpec(**spec) for spec in AgentTools.specs()])
    raw = body.model_dump_json().encode()
    return raw, etag_of(raw)


@router.get(
    "/agent-tools",
    response_model=AgentToolsResponse,
    tags=["agent_tools"],
    summary="The memory tools an agent may call, with their input schemas",
    description="The set is fixed per release: the response carries `ETag` and "
    "`Cache-Control: private, max-age=300`, and a request whose `If-None-Match` names the tag "
    "is a `304` without a body.",
    responses={**_ERRORS, 304: {"description": "Not modified: If-None-Match names the ETag"}},
)
async def list_agent_tools(request: Request, _: ServicePrincipalDep) -> Response:
    body, etag = _listing()
    return conditional(request, body, etag=etag, cache_control=AGENT_TOOLS_CACHE_CONTROL)


@router.post(
    "/agent-tools/{name}",
    response_model=CallResponse,
    tags=["agent_tools"],
    summary="Call one memory tool in this scope",
    responses=_ERRORS,
)
async def call_agent_tool(
    name: str, request: Request, body: CallRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response | CallResponse:
    """With ``Idempotency-Key``, a retried call (a ``remember`` whose answer was lost) gets
    the first result instead of a second write. The tool writes in units of work of its own;
    the key's record is committed once the tool has answered."""
    ctx = build_context(request, container, body.scope)

    async def call() -> CallResponse:
        result = await container.services["agent_tools"].call(
            ctx, name, body.args, toolbox=body.toolbox
        )
        return CallResponse(result=result)

    if request.state.idempotency_key is None:
        return await call()

    async def handler(uow):  # type: ignore[no-untyped-def]
        return 200, (await call()).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload={"name": name, **body.model_dump(mode="json")},
        handler=handler,
    )
