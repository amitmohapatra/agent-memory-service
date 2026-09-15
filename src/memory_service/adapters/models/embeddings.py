"""Embedding adapters.

* ``HashEmbedding``: deterministic feature-hashed bag-of-words + fixed random projection.
  Zero dependencies, zero downloads. Used for tests and benchmarks as a labelled
  *non-representative* baseline; never a production default.
* ``SentenceTransformersEmbedding``: Granite / any HF model, CPU-first, with PyTorch, ONNX or
  OpenVINO backends (``backend`` maps to sentence-transformers' native backends).
* ``FastEmbedEmbedding``: Qdrant's ONNX runtime for the models it packages.

All adapters expose ``fingerprint()`` (model + version + backend + dimension) which is baked
into collection names and cache keys so a model swap can never mix vector spaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import re
from collections.abc import Sequence
from typing import Any

from memory_service.config.settings import EmbeddingSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedding:
    """Deterministic lexical embedding: unigram + bigram feature hashing, L2-normalised."""

    info = ProviderInfo(
        name="hash-embedding",
        version="1",
        license="Apache-2.0",
        origin="internal",
        locality="local",
    )

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = _TOKEN.findall(text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for f in feats:
            h = int(hashlib.blake2b(f.encode(), digest_size=8).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def fingerprint(self) -> str:
        return f"hash-v1-d{self.dimension}"


class SentenceTransformersEmbedding:
    info: ProviderInfo

    def __init__(self, settings: EmbeddingSettings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DependencyUnavailable(
                "sentence-transformers is required (install [models])"
            ) from exc
        backend = {"sentence_transformers": "torch", "onnx": "onnx", "openvino": "openvino"}.get(
            settings.provider, "torch"
        )
        source = settings.model_path or settings.model
        kwargs: dict[str, Any] = {"device": settings.device, "backend": backend}
        if settings.model_path:
            kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(source, **kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"embedding model {source!r} could not be loaded ({type(exc).__name__}); "
                "set MEMORY__MODELS__EMBEDDING__MODEL_PATH to a local directory"
            ) from exc
        if settings.threads:
            import torch

            torch.set_num_threads(settings.threads)
        self.settings = settings
        self.dimension = int(self._model.get_embedding_dimension() or settings.dimension)
        self.max_tokens = int(
            getattr(self._model, "max_seq_length", settings.max_tokens) or settings.max_tokens
        )
        self.info = ProviderInfo(
            name=settings.model,
            version=_st_version(),
            license="Apache-2.0" if "granite" in settings.model.lower() else "see model card",
            origin="huggingface/" + settings.model,
            locality="local",
        )
        self._backend = backend

    def _encode(self, texts: Sequence[str], *, prompt_name: str | None = None) -> list[list[float]]:
        out = self._model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in out]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [text]))[0]

    def fingerprint(self) -> str:
        name = (self.settings.model_path or self.settings.model).rstrip("/").split("/")[-1]
        return f"st-{name}-{self._backend}-d{self.dimension}"


class FastEmbedEmbedding:
    def __init__(self, settings: EmbeddingSettings) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise DependencyUnavailable("fastembed is required (install [models])") from exc
        kwargs: dict[str, Any] = {"model_name": settings.model}
        if settings.model_path:
            kwargs["cache_dir"] = settings.model_path
            kwargs["local_files_only"] = True
        if settings.threads:
            kwargs["threads"] = settings.threads
        try:
            self._model = TextEmbedding(**kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"fastembed model {settings.model!r} could not be loaded ({type(exc).__name__})"
            ) from exc
        self.settings = settings
        self.dimension = settings.dimension
        self.info = ProviderInfo(
            name=settings.model, license="Apache-2.0", origin="qdrant/fastembed", locality="local"
        )

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(x) for x in vec]
            for vec in self._model.embed(list(texts), batch_size=self.settings.batch_size)
        ]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts) if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [text]))[0]

    def fingerprint(self) -> str:
        return f"fastembed-{self.settings.model.split('/')[-1]}-d{self.dimension}"


def _st_version() -> str:
    try:
        from importlib.metadata import version

        return version("sentence-transformers")
    except Exception:
        return "unknown"
