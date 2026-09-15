"""Public /v1 routes: recall (ranked evidence) and context (ContextBundle)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses
from memory_service.modules.context.builder import bundle_to_api, candidate_to_item

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)

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
    kinds: list[str] = Field(
        default_factory=lambda: ["chunk", "memory"],
        description="chunk | memory | fact (graph facts are included only when requested)",
        examples=[["chunk", "memory"]],
    )
    document_ids: list[str] | None = Field(
        default=None, description="Restrict knowledge retrieval to these documents", examples=[None]
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
    representation: str
    text: str
    score: float
    retrievers: list[str]
    citation: str
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = None
    expanded_from: str | None = None
    expansion_edge: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class RecallResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Why did Adjusted EBITDA increase?",
                    "query_type": "DOCUMENT_MULTI_HOP",
                    "results": [],
                    "diagnostics": {"fused_candidates": 12, "reranked": True},
                }
            ]
        }
    )

    query: str
    query_type: str
    results: list[RecallItem]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] | None = Field(
        default=None, description="EvidenceReport: COMPLETE | INCOMPLETE | INSUFFICIENT"
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
    token_budget: int | None = Field(default=None, ge=200, le=200_000, examples=[6000])
    document_ids: list[str] | None = Field(default=None, examples=[None])


class ContextResponse(BaseModel):
    """ContextBundle: bounded, ranked, provenance-carrying context. ``rendered`` is prompt-ready."""

    model_config = ConfigDict(
        extra="allow",
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
                    },
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
    query_type: str
    conversation: dict[str, Any]
    memories: list[dict[str, Any]]
    knowledge: list[dict[str, Any]]
    graph_facts: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    evidence: dict[str, Any]
    token_budget: int
    token_estimate: int
    cache_hit: bool
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
    return RecallResponse(
        query=body.query,
        query_type=result.routed.query_type.value,
        results=[
            RecallItem(**{**i.model_dump(mode="json"), "representation": i.representation.value})
            for i in items
        ],
        diagnostics={
            k: v for k, v in result.diagnostics.items() if k not in ("evidence", "evidence_targets")
        },
        evidence=result.diagnostics.get("evidence"),
    )


@router.post(
    "/context",
    response_model=ContextResponse,
    tags=["retrieval"],
    summary="Build a ContextBundle for the current turn",
    responses=_ERRORS,
)
async def context(
    request: Request, body: ContextRequest, container: ContainerDep, _: ServicePrincipalDep
) -> ContextResponse:
    ctx = build_context(request, container, body.scope)
    builder = container.services["context_builder"]
    bundle = await builder.build(
        ctx, body.query, token_budget=body.token_budget, document_ids=body.document_ids
    )
    return ContextResponse.model_validate(bundle_to_api(bundle))
