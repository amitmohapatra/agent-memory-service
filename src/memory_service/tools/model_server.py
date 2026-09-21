"""Serve one model, over HTTP, from this image.

The model tier is a separate deployment, not a library the service imports. Each model runs
as its own process behind its own URL, so it can be scaled, restarted, pinned to different
hardware and replaced without touching the service — and so a native crash in one model
cannot take down the API or the worker with it.

Making that split does not require a separate framework. This image already carries torch,
the weights and FastAPI, so serving them is one more entrypoint and the extra containers cost
no extra bytes — Docker shares the layers. (The first attempt reached for a 14.3 GB
third-party inference server to serve a 130 MB model. Reaching for the framework was the
mistake; the capability was already here.)

One process serves one role, chosen by ``MEMORY_MODEL_ROLE``:

    embed    POST /embed     {"inputs": [...]}        -> [[float]]      RemoteEmbedding
    rerank   POST /rerank    {"query":..,"texts":[..]} -> [{index,score}] RemoteReranker
    nli      POST /predict   {"inputs": [[p,h],..]}   -> [[{label,score}]] RemoteNLI
    sparse   POST /sparse    {"inputs": [...]}        -> [{indices,values}] RemoteSparse

Every role also answers ``GET /info`` in the shape TEI uses, so the client's dialect
detection identifies the server and can verify it is serving the model it was promised, and
``GET /health``, which builds the model — so "still loading" is distinguishable from "failed
to start" rather than both looking like a refused connection.

    MEMORY_MODEL_ROLE=rerank memory-model
    MEMORY__MODELS__RERANKER__URL=http://memory-rerank:80
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from memory_service.config.settings import get_settings

#: Roles this server can take. One per process: a container that loads every model has the
#: memory profile of the monolith the split exists to avoid.
ROLES = ("embed", "rerank", "nli", "sparse")


class TextsRequest(BaseModel):
    inputs: list[str] | str

    def texts(self) -> list[str]:
        return [self.inputs] if isinstance(self.inputs, str) else self.inputs


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    # TEI spells this field `texts`; some clients send `documents`. Accept both rather than
    # making the caller care which server implementation is behind the URL.
    documents: list[str] | None = None

    def candidates(self) -> list[str]:
        return self.documents if self.documents is not None else self.texts


class PredictRequest(BaseModel):
    #: pairs of (premise, hypothesis)
    inputs: list[list[str]]


def _role() -> str:
    role = os.environ.get("MEMORY_MODEL_ROLE", "embed").strip().lower()
    if role not in ROLES:
        raise SystemExit(f"MEMORY_MODEL_ROLE must be one of {', '.join(ROLES)}; got {role!r}")
    return role


def _build_model(role: str, settings: Any) -> Any:
    """Construct the one model this process serves. Imported lazily, per role, so a container
    never pays for weights belonging to a role it is not running."""
    if role == "embed":
        from memory_service.adapters.models.embeddings import SentenceTransformersEmbedding

        return SentenceTransformersEmbedding(settings.models.embedding)
    if role == "rerank":
        from memory_service.adapters.models.rerankers import CrossEncoderReranker

        return CrossEncoderReranker(settings.models.reranker)
    if role == "nli":
        from memory_service.adapters.models.nli import TransformersNLI

        return TransformersNLI(settings.models.nli)
    from memory_service.adapters.models.advanced import FastEmbedSparseEncoder

    return FastEmbedSparseEncoder(
        settings.models.sparse_model, model_path=settings.models.sparse_model_path
    )


def _identity(role: str, settings: Any) -> tuple[str, int | None]:
    cfg = {
        "embed": settings.models.embedding,
        "rerank": settings.models.reranker,
        "nli": settings.models.nli,
    }.get(role)
    if cfg is None:  # sparse has no settings section of its own
        return (settings.models.sparse_model_path or settings.models.sparse_model, None)
    return (
        getattr(cfg, "model_path", None) or getattr(cfg, "model", "") or role,
        getattr(cfg, "max_tokens", None) or getattr(cfg, "max_length", None),
    )


def _mount_embed(app: FastAPI, model: Callable[[], Any]) -> None:
    @app.post("/embed")
    async def embed(request: TextsRequest) -> list[list[float]]:
        return await model().embed_documents(request.texts())


def _mount_rerank(app: FastAPI, model: Callable[[], Any]) -> None:
    @app.post("/rerank")
    async def rerank(request: RerankRequest) -> list[dict[str, Any]]:
        candidates = request.candidates()
        # `top_k` is required by the port; the server ranks everything it was given and lets
        # the caller cut, because a remote reranker that silently truncates makes the
        # client's own top-k meaningless.
        results = await model().rerank(request.query, candidates, top_k=len(candidates))
        # TEI answers with index/score pairs, ordered best first
        return [
            {"index": r.index, "score": r.score}
            for r in sorted(results, key=lambda r: r.score, reverse=True)
        ]


def _mount_nli(app: FastAPI, model: Callable[[], Any]) -> None:
    @app.post("/predict")
    async def predict(request: PredictRequest) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        for pair in request.inputs:
            scores = await model().entail([pair[0]], pair[1])
            if not scores:
                out.append([{"label": "neutral", "score": 1.0}])
                continue
            # An MNLI head produces all three probabilities and the client maps the labels
            # back. Sending only the winner would discard the contradiction signal, which is
            # the half the grounding cascade uses to detect conflict.
            score = scores[0]
            out.append(
                [
                    {"label": "entailment", "score": score.entailment},
                    {"label": "neutral", "score": score.neutral},
                    {"label": "contradiction", "score": score.contradiction},
                ]
            )
        return out


def _mount_sparse(app: FastAPI, model: Callable[[], Any]) -> None:
    @app.post("/sparse")
    async def sparse(request: TextsRequest) -> list[dict[str, Any]]:
        vectors = model().encode_documents(request.texts())
        return [{"indices": list(v.indices), "values": list(v.values)} for v in vectors]


#: role -> the one endpoint it exposes. A process that mounted every role would have the
#: memory profile of the monolith this split exists to avoid.
_MOUNT = {
    "embed": _mount_embed,
    "rerank": _mount_rerank,
    "nli": _mount_nli,
    "sparse": _mount_sparse,
}


def create_app(role: str | None = None) -> FastAPI:
    settings = get_settings()
    role = role or _role()
    app = FastAPI(title=f"memory-model-{role}", version="1")
    state: dict[str, Any] = {}

    def model() -> Any:
        """Built on first use so the port binds immediately and /health means something."""
        if "model" not in state:
            state["model"] = _build_model(role, settings)
        return state["model"]

    @app.get("/info")
    async def info() -> dict[str, Any]:
        model_id, max_input = _identity(role, settings)
        return {
            "model_id": model_id,
            "model_dtype": "float32",
            "max_input_length": max_input,
            # not part of the TEI shape, but the client logs it and a mismatched role is
            # otherwise only visible as a 404 on the endpoint it expected
            "role": role,
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        model()
        return {"status": "ok", "role": role}

    _MOUNT[role](app, model)
    return app


def run() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    role = _role()
    uvicorn.run(
        create_app(role),
        host=os.environ.get("MEMORY_MODEL_HOST", "0.0.0.0"),  # noqa: S104 - in-cluster service
        port=int(os.environ.get("MEMORY_MODEL_PORT", "80")),
        log_level="warning",
    )
