"""The service's multilingual encoder behind an OpenAI-compatible ``/v1/embeddings``.

Every library in the head-to-head that takes an embedder is pointed here, so every system
ranks with the same vectors this service's ``dense_ml`` space holds (``FROZEN_MODELS.dense_ml``,
loaded the way the service loads it: ``load_dense``). Run from the repository root (the model
directory is ``./models``):

    PYTHONPATH=src:sdk/python/src:. python -m benchmark.memvs.embed_server --port 18311
"""

from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from memory_service.adapters.models.embeddings import load_dense
from memory_service.config.constants import FROZEN_MODELS

HOST = "127.0.0.1"


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = None


def build_app() -> FastAPI:
    encoder = load_dense(FROZEN_MODELS.dense_ml)
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"model": FROZEN_MODELS.dense_ml.id, "dimension": encoder.dimension}

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        texts = [request.input] if isinstance(request.input, str) else list(request.input)
        vectors = await encoder.embed_documents(texts)
        return {
            "object": "list",
            "model": FROZEN_MODELS.dense_ml.id,
            "data": [
                {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(build_app(), host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
