"""Operational endpoints: liveness, readiness, metrics, version."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from memory_service.api.errors import error_responses
from memory_service.application.container import Container
from memory_service.observability.metrics import render_metrics

router = APIRouter(tags=["operations"])


class LiveResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class DependencyStatus(BaseModel):
    ok: bool
    mandatory: bool
    error: str | None = None


class ReadyResponse(BaseModel):
    status: str = Field(..., description="ready | degraded | not_ready", examples=["ready"])
    dependencies: dict[str, DependencyStatus] = Field(
        default_factory=dict,
        examples=[
            {
                "postgres": {"ok": True, "mandatory": True},
                "cache": {"ok": False, "mandatory": False},
            }
        ],
    )


class VersionResponse(BaseModel):
    service: str = Field(..., examples=["memory-service"])
    version: str = Field(..., examples=["0.1.0"])
    api_version: str = Field(..., examples=["v1"])
    environment: str = Field(..., examples=["dev"])
    providers: dict[str, Any] = Field(
        default_factory=dict,
        description="Enabled provider per port (no secrets).",
        examples=[
            {
                "cache": "dragonfly",
                "search": "qdrant",
                "authorization": "openfga",
                "llm": "disabled",
            }
        ],
    )


def _container(request: Request) -> Container:
    return request.app.state.container


@router.get("/health/live", response_model=LiveResponse, summary="Liveness probe")
async def live() -> LiveResponse:
    return LiveResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Verifies mandatory backing stores. Optional providers never fail readiness when disabled."
    ),
    responses={**error_responses(503)},
)
async def ready(request: Request, response: Response) -> ReadyResponse:
    results = await _container(request).readiness()
    mandatory_down = [n for n, r in results.items() if r["mandatory"] and not r["ok"]]
    optional_down = [n for n, r in results.items() if not r["mandatory"] and not r["ok"]]
    if mandatory_down:
        response.status_code = 503
        status = "not_ready"
    elif optional_down:
        status = "degraded"
    else:
        status = "ready"
    return ReadyResponse(
        status=status, dependencies={k: DependencyStatus(**v) for k, v in results.items()}
    )


@router.get(
    "/metrics", summary="Prometheus metrics", response_class=Response, include_in_schema=True
)
async def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@router.get("/version", response_model=VersionResponse, summary="Build and provider information")
async def version(request: Request) -> VersionResponse:
    c = _container(request)
    s = c.settings
    return VersionResponse(
        service=s.service.name,
        version=c.version,
        api_version=s.service.api_version,
        environment=s.service.environment,
        providers={
            "cache": s.cache.provider,
            "search": s.search.provider,
            "blob": s.blob.provider,
            "tasks": s.tasks.provider,
            "authorization": s.authorization.provider,
            "policy": s.policy.provider,
            "embedding": f"{s.models.embedding.provider}:{s.models.embedding.model}",
            "reranker": f"{s.models.reranker.provider}:{s.models.reranker.model}",
            "llm": s.models.llm.provider if s.models.llm.enabled else "disabled",
            "memory_intelligence": s.memory_intelligence.provider,
            "graph_enrichment": s.graph_enrichment.provider,
            "document_parser": s.documents.parser,
        },
    )
