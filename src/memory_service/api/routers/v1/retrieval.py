"""Public /v1 routes: recall (ranked evidence) and context (ContextBundle)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.api.schemas.context import ContextResponse, PromptContextResponse
from memory_service.application.container import Container
from memory_service.domain.audit import ReadKind
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
from memory_service.modules.context.sections import ToolsRequest
from memory_service.modules.retrieval.engine import PointInTime
from memory_service.modules.retrieval.search import DEFAULT_KINDS, SearchItem, SearchKind


def _audit(
    request: Request,
    container: Container,
    ctx: MemoryExecutionContext,
    kind: ReadKind,
    query: str,
    record_ids: Iterable[str],
) -> None:
    """Record a read for the audit trail. A side channel, never a dependency: a container
    without the service (unit tests build partial ones) reads exactly as before. The entry
    names the authenticated credential as well as the principal it acted for, because the
    principal is asserted by the caller and the credential is not."""
    audit = container.services.get("read_audit")
    if audit is not None:
        principal = getattr(request.state, "service_principal", None)
        audit.record(
            ctx,
            kind,
            query,
            record_ids,
            scope_fingerprint=ctx.scope_fingerprint(),
            credential=principal.service_id if principal is not None else "",
        )


router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)

_QUERY_TYPE_DESCRIPTION = (
    "How the deterministic router classified the query: EXACT_IDENTIFIER (an id or code was "
    "looked up), CONVERSATION_HISTORY, USER_MEMORY, DECISION, DOCUMENT_LOCAL (one passage "
    "answers it), DOCUMENT_MULTI_HOP (several passages must be combined), ENTITY_RELATION "
    "(graph traversal), TEMPORAL, GLOBAL_SUMMARY or GENERAL_SEMANTIC (the default)."
)

_SCOPE: dict[str, Any] = {
    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
}


_RECALL_EXAMPLE: dict[str, Any] = {
    "scope": _SCOPE,
    "query": "Why did Adjusted EBITDA increase despite lower revenue?",
    "limit": 20,
}
_ITEM_EXAMPLE: dict[str, Any] = {
    "id": "chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "kind": "chunk",
    "text": "Adjusted EBITDA increased to EUR 98 million…",
    "observed_on": None,
    "document_id": "doc_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "page": 11,
}
_CONTEXT_EXAMPLE: dict[str, Any] = {
    "scope": _SCOPE,
    "query": "Why did Adjusted EBITDA increase despite lower revenue?",
    "token_budget": 6000,
}


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_RECALL_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody, examples=[_SCOPE])
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["Why did Adjusted EBITDA increase despite lower revenue?"],
    )
    limit: int = Field(default=20, ge=1, le=100, examples=[20])
    kinds: list[SearchKind] = Field(
        default_factory=lambda: list(DEFAULT_KINDS),
        min_length=1,
        max_length=5,
        description="What to search: memory (what was learned or stated), chunk (document "
        "passages), summary (document summaries), episode (earlier conversations of this "
        "user, one per thread), message (this thread's history).",
        examples=[["chunk", "memory"]],
    )
    time_from: datetime | None = Field(
        default=None, description="only what was observed since (filters before ranking)"
    )
    time_to: datetime | None = Field(default=None, description="only what was observed until")
    as_of: datetime | None = Field(
        default=None,
        description="memories as they were true at this moment, including ones later "
        "replaced (valid time)",
    )
    known_at: datetime | None = Field(
        default=None,
        description="memories as they were known at this moment: learned by then and not yet "
        "replaced (knowledge time, for audit)",
    )
    document_ids: list[str] | None = Field(
        default=None,
        max_length=100,
        description="Restrict knowledge retrieval to these documents",
        examples=[None],
    )
    debug: bool = Field(default=False, description="add each item's ranking detail")


class RecallResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"items": [_ITEM_EXAMPLE]}]})

    items: list[SearchItem]
    query_type: QueryType | None = Field(
        default=None, description=_QUERY_TYPE_DESCRIPTION + " Only with debug."
    )
    diagnostics: dict[str, Any] | None = Field(default=None, description="only with debug")


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_CONTEXT_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody, examples=[_SCOPE])
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["Why did Adjusted EBITDA increase despite lower revenue?"],
    )
    token_budget: int | None = Field(default=None, ge=200, le=16_000, examples=[6000])
    document_ids: list[str] | None = Field(default=None, max_length=100, examples=[None])
    tools: ToolsRequest | None = Field(
        default=None,
        description="the agent's callable tools (available: null means any catalog tool): "
        "adds the procedures learned for the task and the tool hints",
    )
    window: bool = Field(
        default=True,
        description="carry the thread's recent messages; false when the framework keeps its "
        "own history (a LangGraph checkpointer, an OpenAI session). The thread's summary "
        "comes either way",
    )
    format: Literal["prompt", "full"] = Field(
        default="prompt",
        description="prompt: what to put in front of the model (rendered, bundle_id, "
        "token_estimate, evidence_status, and the tools that fit when tools were given); full: "
        "the same content as structured data, without the rendering",
    )
    debug: bool = Field(default=False, description="add the build diagnostics")


@router.post(
    "/recall",
    response_model=RecallResponse,
    response_model_exclude_none=True,
    tags=["retrieval"],
    summary="Ranked, scope-filtered items for a query",
    responses=_ERRORS,
)
async def recall(
    request: Request, body: RecallRequest, container: ContainerDep, _: ServicePrincipalDep
) -> RecallResponse:
    ctx = build_context(request, container, body.scope)
    found = await container.services["search"].search(
        ctx,
        body.query,
        kinds=body.kinds,
        limit=body.limit,
        observed=(body.time_from, body.time_to) if body.time_from or body.time_to else None,
        at=PointInTime(as_of=body.as_of, known_at=body.known_at),
        document_ids=body.document_ids,
        debug=body.debug,
    )
    _audit(request, container, ctx, "recall", body.query, (i.id for i in found.items))
    if not body.debug:
        return RecallResponse(items=found.items)
    return RecallResponse(
        items=found.items, query_type=found.query_type, diagnostics=found.diagnostics
    )


@router.post(
    "/context",
    # The body is the bytes the builder produced (a cache hit is the stored bytes), so the
    # route returns a Response and FastAPI validates nothing; the documented 200 is either
    # model, and tests/unit/test_context_datapath.py checks the bytes against them.
    response_model=None,
    tags=["retrieval"],
    summary="The context for the current turn (format=prompt: rendered; full: the bundle)",
    responses={
        200: {
            "model": PromptContextResponse | ContextResponse,
            "description": "format=prompt: PromptContextResponse; format=full: ContextResponse",
        },
        **_ERRORS,
    },
)
async def context(
    request: Request, body: ContextRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response:
    ctx = build_context(request, container, body.scope)
    async with container.services["llm_assist"].reading(ctx):
        payload = await container.services["context_builder"].build_api(
            ctx,
            body.query,
            token_budget=body.token_budget,
            document_ids=body.document_ids,
            tools=body.tools,
            window=body.window,
            output=body.format,
            debug=body.debug,
        )
    # The bundle is opaque bytes here on purpose (see above), so the audit records who asked
    # what under which scope; the records served are in the bundle itself.
    _audit(request, container, ctx, "context", body.query, ())
    return Response(content=payload, media_type="application/json")
