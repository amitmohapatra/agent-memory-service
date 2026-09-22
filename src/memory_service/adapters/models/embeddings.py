"""Embedding adapters.

* ``HashEmbedding``: deterministic feature-hashed bag-of-words + fixed random projection.
  Zero dependencies, zero downloads. Used for tests and benchmarks as a labelled
  *non-representative* baseline; never a production default.
* ``SentenceTransformersEmbedding``: the frozen dense encoder (``constants.FROZEN_MODELS``),
  CPU-first, with PyTorch, ONNX or OpenVINO backends (``backend`` maps to
  sentence-transformers' native backends).

Both expose ``fingerprint()`` (model + backend + graph file + dimension) which is baked into
collection names and cache keys so a model swap can never mix vector spaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import re
from collections.abc import Sequence
from typing import Any

from memory_service.adapters.models._precision import cpu_dtype_kwargs
from memory_service.config.constants import DenseModel
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

    def __init__(self, spec: DenseModel, *, threads: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DependencyUnavailable(
                "sentence-transformers is required (install [models])"
            ) from exc
        source = spec.source
        local = source != spec.id
        kwargs: dict[str, Any] = {"device": spec.device, "backend": spec.backend}
        if spec.backend == "torch" and (dtype := cpu_dtype_kwargs(spec.device)):
            kwargs["model_kwargs"] = dtype  # see _precision: half precision on CPU is worse
        if spec.backend != "torch" and spec.graph_file:
            kwargs["model_kwargs"] = {"file_name": spec.graph_file}
        if local:
            kwargs["local_files_only"] = True
        if spec.revision and not local:
            kwargs["revision"] = spec.revision
        try:
            self._model = SentenceTransformer(source, **kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"embedding model {source!r} could not be loaded ({type(exc).__name__}); "
                "run `make models` or bake the weights under /models"
            ) from exc
        if threads:
            import torch

            torch.set_num_threads(threads)
        self.spec = spec
        self.threads = threads
        self.dimension = int(self._model.get_embedding_dimension() or spec.dimension)
        # The checkpoint's own limit governs truncation, as it always has (granite-small
        # declares 8192). ``spec.max_seq_length`` is the value the ONNX export will be pinned
        # to once it is measured (Phase 2); pinning it here now would change what today's
        # collections were embedded with.
        self.max_tokens = int(getattr(self._model, "max_seq_length", None) or spec.max_seq_length)
        self.info = ProviderInfo(
            name=spec.id,
            version=_st_version(),
            license="Apache-2.0" if "granite" in spec.id.lower() else "see model card",
            origin="huggingface/" + spec.id,
            locality="local",
        )

    def _encode(self, texts: Sequence[str], *, prompt_name: str | None = None) -> list[list[float]]:
        out = self._model.encode(
            list(texts),
            batch_size=self.spec.batch_size,
            normalize_embeddings=self.spec.normalize,
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
        return dense_fingerprint(self.spec, self.dimension)


def dense_fingerprint(spec: DenseModel, dimension: int | None = None) -> str:
    """The name of the vector space: model, backend, the ONNX graph when one is named, and
    the width. Two graphs of one model (int8 and fp32) must never share a collection, so the
    graph file is part of it; without one the name is exactly what it was before graphs
    existed, so frozen collections keep their names."""
    name = (spec.model_path or spec.id).rstrip("/").split("/")[-1]
    graph = (
        f"-{spec.graph_file.rsplit('/', 1)[-1].removesuffix('.onnx')}" if spec.graph_file else ""
    )
    return f"st-{name}-{spec.backend}{graph}-d{dimension or spec.dimension}"


def _st_version() -> str:
    try:
        from importlib.metadata import version

        return version("sentence-transformers")
    except Exception:
        return "unknown"
