"""Public /v1 routes: recall (ranked evidence) and context (ContextBundle)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.api.schemas.context import (
    REPRESENTATION_DESCRIPTION,
    ContextItemBody,
    ConversationWindowBody,
    EvidenceReportBody,
)
from memory_service.domain.enums import QueryType, Representation
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.context.builder import bundle_to_api, candidate_to_item
from memory_service.modules.grounding.cascade import attach

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)

RecallKind = Literal["chunk", "memory", "summary"]

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
    kinds: list[RecallKind] = Field(
        default_factory=lambda: ["chunk", "memory"],
        min_length=1,
        max_length=3,
        description=(
            "Which record kinds to search and return: chunk (document passages), memory "
            "(canonical memories) and summary (rolled-up summaries; also added by the router "
            "when the query asks for an overview). Graph facts are not a recall kind: ask "
            "/v1/context or /v1/graph/query for them."
        ),
        examples=[["chunk", "memory"]],
    )
    document_ids: list[str] | None = Field(
        default=None,
        max_length=100,
        description="Restrict knowledge retrieval to these documents",
        examples=[None],
    )


class RecallItem(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "item_id": "chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "representation": "CHUNK",
                    "text": "Adjusted EBITDA increased to EUR 98 million…",
                    "score": 0.83,
                    "retrievers": ["fusion"],
                    "citation": "chunk_id:chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "document_id": "doc_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "page": 11,
                    "section_path": "ACME FY26 Annual Report > 3. Financial Results",
                    "evidence": [],
                }
            ]
        }
    )

    item_id: str
    representation: Representation = Field(..., description=REPRESENTATION_DESCRIPTION)
    text: str
    score: float
    retrievers: list[str]
    citation: str
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = None
    expanded_from: str | None = None
    expansion_edge: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class RecallResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Why did Adjusted EBITDA increase?",
                    "query_type": "DOCUMENT_MULTI_HOP",
                    "results": [],
                    "diagnostics": {"fused_candidates": 12, "duplicates_collapsed": 2},
                }
            ]
        }
    )

    query: str
    query_type: QueryType = Field(..., description=_QUERY_TYPE_DESCRIPTION)
    results: list[RecallItem]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    evidence: EvidenceReportBody | None = Field(
        default=None,
        description="EvidenceReport (COMPLETE | INCOMPLETE | INSUFFICIENT) when the "
        "verification stage ran; absent otherwise.",
    )


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
    answer: str | None = Field(
        default=None,
        max_length=8_000,
        description="When given, the grounding cascade verifies this answer against the "
        "bundle and the report is attached as evidence.grounding",
        examples=[None],
    )


class ContextResponse(BaseModel):
    """ContextBundle: bounded, ranked, provenance-carrying context. ``rendered`` is prompt-ready."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "Why did Adjusted EBITDA increase?",
                    "query_type": "DOCUMENT_MULTI_HOP",
                    "conversation": {
                        "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                        "message_ids": ["msg_1"],
                        "rendered": "USER: …",
                        "token_estimate": 120,
                        "summary": None,
                    },
                    "memories": [],
                    "knowledge": [],
                    "graph_facts": [],
                    "summaries": [],
                    "evidence": {
                        "status": "COMPLETE",
                        "required_groups": [],
                        "satisfied_groups": [],
                        "missing_groups": [],
                        "escalations": [],
                        "notes": [],
                        "unused": [],
                        "grounding": None,
                        "llm_tokens": 0,
                    },
                    "bundle_id": "6f1c…",
                    "token_budget": 6000,
                    "token_estimate": 1840,
                    "cache_hit": False,
                    "revision_fingerprint": "…",
                    "built_at": "2026-09-14T10:00:00Z",
                    "diagnostics": {},
                    "rendered": "## Recent conversation\nUSER: …",
                }
            ]
        },
    )

    query: str
    query_type: QueryType = Field(..., description=_QUERY_TYPE_DESCRIPTION)
    bundle_id: str = Field(
        default="", description="tenant-bound handle for /v1/verify while the bundle is cached"
    )
    conversation: ConversationWindowBody
    memories: list[ContextItemBody]
    knowledge: list[ContextItemBody]
    graph_facts: list[ContextItemBody]
    summaries: list[ContextItemBody]
    evidence: EvidenceReportBody
    token_budget: int
    token_estimate: int
    cache_hit: bool
    revision_fingerprint: str = Field(
        default="", description="revisions this bundle was built from"
    )
    built_at: datetime
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    rendered: str


@router.post(
    "/recall",
    response_model=RecallResponse,
    tags=["retrieval"],
    summary="Scope-filtered ranked recall",
    responses=_ERRORS,
)
async def recall(
    request: Request, body: RecallRequest, container: ContainerDep, _: ServicePrincipalDep
) -> RecallResponse:
    ctx = build_context(request, container, body.scope)
    engine = container.services["retrieval"]
    result = await engine.retrieve(
        ctx,
        body.query,
        limit=body.limit,
        kinds=tuple(k for k in body.kinds if k in ("chunk", "memory")),
        document_ids=body.document_ids,
    )
    wanted = set(body.kinds)
    items = [candidate_to_item(c) for c in result.candidates if c.kind in wanted][: body.limit]
    evidence = result.diagnostics.get("evidence")
    return RecallResponse(
        query=body.query,
        query_type=result.routed.query_type,
        results=[RecallItem.model_validate(i.model_dump(mode="json")) for i in items],
        diagnostics={
            k: v for k, v in result.diagnostics.items() if k not in ("evidence", "evidence_targets")
        },
        evidence=EvidenceReportBody.model_validate(evidence) if evidence is not None else None,
    )


@router.post(
    "/context",
    # The unverified answer is sent as the bytes the builder already produced, so the route
    # returns a Response and FastAPI validates nothing. ContextResponse stays the documented
    # 200 below, and tests/unit/test_context_datapath.py asserts those bytes still validate
    # against it - the contract is checked where it can be checked once, not per request.
    response_model=None,
    tags=["retrieval"],
    summary="Build a ContextBundle for the current turn",
    responses={200: {"model": ContextResponse, "description": "Successful Response"}, **_ERRORS},
)
async def context(
    request: Request, body: ContextRequest, container: ContainerDep, _: ServicePrincipalDep
) -> Response | ContextResponse:
    ctx = build_context(request, container, body.scope)
    builder = container.services["context_builder"]
    if not body.answer:
        # One serialisation for the whole request: a cache hit is the stored bytes, a miss is
        # one dump. Parsing the cached bundle only to dump it, validate it and dump it again
        # was most of what a 30-80 KB hit cost.
        payload = await builder.build_api(
            ctx, body.query, token_budget=body.token_budget, document_ids=body.document_ids
        )
        return Response(content=payload, media_type="application/json")
    # Grounding needs the bundle itself, so this arm keeps the model round trip.
    cascade = container.services.get("grounding")
    if cascade is None:
        raise ProviderNotConfigured("the NLI classifier is disabled in this process")
    bundle = await builder.build(
        ctx, body.query, token_budget=body.token_budget, document_ids=body.document_ids
    )
    bundle = attach(bundle, await cascade.verify_bundle(bundle, body.answer))
    return ContextResponse.model_validate(bundle_to_api(bundle))
