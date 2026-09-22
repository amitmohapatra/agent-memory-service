"""Operational endpoints: liveness, readiness, metrics, version."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from memory_service.api.errors import error_responses
from memory_service.application.container import Container
from memory_service.config import constants
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
    degraded: list[str] = Field(
        default_factory=list,
        description=(
            "Capabilities running below what was configured — a parser that fell back, a "
            "stand-in classifier. Empty means the service is what it was asked to be."
        ),
    )
    providers: dict[str, Any] = Field(
        default_factory=dict,
        description="The provider actually running per port (no secrets).",
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
        service=constants.SERVICE_NAME,
        version=c.version,
        api_version=constants.API_VERSION,
        environment=s.service.environment,
        providers={
            # what is *running*: the stand-ins a test or benchmark asked for through
            # build_container(overrides=...) are not settings, so they can only be read here
            "cache": _active(c.cache, "redis-protocol"),
            "search": "qdrant-local"
            if c.overrides.search or c.overrides.search_local_path
            else "qdrant",
            "blob": _active(c.blob, s.blob.provider),
            "tasks": _active(c.tasks, "procrastinate"),
            "authorization": _active(c.authorization, "openfga"),
            "embedding": _active(c.embedding, constants.FROZEN_MODELS.dense.id),
            "reranker": _active(c.reranker, "none"),
            "llm": "bifrost" if s.models.llm.enabled else "disabled",
            "memory_intelligence": "native",
            "graph_enrichment": _active(c.graph_enrichment, "native"),
            # what is *running*, not what was asked for: the parser is "docling" even in an
            # image built without it, where the builtin is doing the work. An endpoint that
            # reports the request rather than the reality is worse than silent.
            "document_parser": _active(c.document_parser, c.tuning.documents.parser),
        },
        degraded=_degraded(c),
    )


def _active(provider: Any, configured: str) -> str:
    """What is actually running, not what was configured.

    ``None`` means the component was never built, and saying so matters: the reranker is
    off by default and used to be reported here by its configured provider name, so
    /version named a cross-encoder that had never been loaded and would never be called.
    """
    if provider is None:
        return "disabled"
    info = getattr(provider, "info", None)
    return getattr(info, "name", None) or configured


def _degraded(c: Any) -> list[str]:
    """Where the running service is not what it was asked to be.

    Each of these is a capability that falls back rather than failing, so nothing else in the
    system reports it: the document parser downgrades to the builtin when docling is absent,
    and the NLI head downgrades to a lexical stand-in. Both keep serving — with quietly worse
    output — which is exactly the kind of thing that should be visible without reading logs.
    """
    notes: list[str] = []
    wanted_parser = c.overrides.document_parser or c.tuning.documents.parser
    active_parser = _active(c.document_parser, wanted_parser)
    if wanted_parser != active_parser:
        notes.append(f"document_parser: configured {wanted_parser!r}, running {active_parser!r}")
    if c.nli is not None and getattr(c.nli, "representative", True) is False:
        notes.append("nli: running a non-representative stand-in; grounding verdicts are weak")
    if c.tuning.retrieval.rerank and c.reranker is None:
        notes.append("rerank: requested, but no reranker was built; results are unreranked")
    return notes
