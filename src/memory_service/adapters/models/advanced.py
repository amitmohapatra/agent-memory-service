"""Benchmark-gated model adapters (M10): learned sparse (SPLADE / miniCOIL / BM42 via
fastembed), late interaction (ColBERT via fastembed) and late chunking (long-context
sentence-transformers). Each declares its license/origin, loads only from local files when a
``model_path`` is given, and fails with ``DependencyUnavailable`` (never a silent fallback)
when the weights are absent — the benchmark harness records that as *skipped*.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from memory_service.config.settings import EmbeddingSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo
from memory_service.ports.search import SparseVector

_SPARSE_LICENSES = {
    "prithivida/Splade_PP_en_v1": "Apache-2.0",
    "Qdrant/minicoil-v1": "Apache-2.0",
    "Qdrant/bm42-all-minilm-l6-v2-attentions": "Apache-2.0",
}
_LATE_LICENSES = {
    "colbert-ir/colbertv2.0": "MIT",
    "answerdotai/answerai-colbert-small-v1": "Apache-2.0",
    "jinaai/jina-colbert-v2": "CC-BY-NC-4.0",
}


class FastEmbedSparseEncoder:
    """SPLADE / miniCOIL / BM42 as a :class:`SparseEncoder`. Server-side IDF is off for
    these (the model already weights terms), so collections created for them use
    ``sparse_idf=False``."""

    info: ProviderInfo

    def __init__(self, model: str, *, model_path: str | None = None, threads: int | None = None):
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as exc:
            raise DependencyUnavailable("fastembed is required (install [models])") from exc
        kwargs: dict[str, Any] = {"model_name": model, "threads": threads}
        if model_path:
            kwargs["specific_model_path"] = model_path
            kwargs["local_files_only"] = True
        try:
            self._model = SparseTextEmbedding(**kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"sparse model {model!r} could not be loaded ({type(exc).__name__}); "
                "download it into models/ and set the model_path"
            ) from exc
        self.model = model
        self.info = ProviderInfo(
            name=model,
            license=_SPARSE_LICENSES.get(model, "see model card"),
            origin="huggingface/" + model,
            locality="local",
        )

    @staticmethod
    def _to_vector(emb: Any) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in emb.indices], values=[float(v) for v in emb.values]
        )

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._to_vector(e) for e in self._model.embed(list(texts))]

    def encode_query(self, text: str) -> SparseVector:
        return self._to_vector(next(iter(self._model.query_embed(text))))

    def fingerprint(self) -> str:
        return "sparse-" + self.model.rsplit("/", 1)[-1].lower()

    @property
    def server_side_idf(self) -> bool:
        return False


class FastEmbedLateInteraction:
    """ColBERT-style multivector encoder (one vector per token)."""

    info: ProviderInfo

    def __init__(self, model: str, *, model_path: str | None = None, threads: int | None = None):
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError as exc:
            raise DependencyUnavailable("fastembed is required (install [models])") from exc
        kwargs: dict[str, Any] = {"model_name": model, "threads": threads}
        if model_path:
            kwargs["specific_model_path"] = model_path
            kwargs["local_files_only"] = True
        try:
            self._model = LateInteractionTextEmbedding(**kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"late-interaction model {model!r} could not be loaded ({type(exc).__name__})"
            ) from exc
        self.model = model
        self.dimension = int(
            next(
                (
                    m["dim"]
                    for m in LateInteractionTextEmbedding.list_supported_models()
                    if m["model"] == model
                ),
                128,
            )
        )
        self.info = ProviderInfo(
            name=model,
            license=_LATE_LICENSES.get(model, "see model card"),
            origin="huggingface/" + model,
            locality="local",
        )

    def _docs(self, texts: Sequence[str]) -> list[list[list[float]]]:
        return [[[float(x) for x in row] for row in emb] for emb in self._model.embed(list(texts))]

    async def embed_documents_multi(self, texts: Sequence[str]) -> list[list[list[float]]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._docs, texts)

    async def embed_query_multi(self, text: str) -> list[list[float]]:
        def _q() -> list[list[float]]:
            emb = next(iter(self._model.query_embed(text)))
            return [[float(x) for x in row] for row in emb]

        return await asyncio.to_thread(_q)

    def fingerprint(self) -> str:
        return "colbert-" + self.model.rsplit("/", 1)[-1].lower()


class LateChunkingEmbedding:
    """Late chunking on top of a sentence-transformers model: embed the *whole* contextual
    document once with token embeddings, then mean-pool the token span of each chunk. Chunk
    vectors therefore carry document context (anaphora, definitions) that isolated chunk
    embeddings lose. Requires a long-context model; texts beyond ``max_tokens`` fall back to
    per-chunk embedding for the overflow."""

    info: ProviderInfo

    def __init__(self, settings: EmbeddingSettings) -> None:
        from memory_service.adapters.models.embeddings import SentenceTransformersEmbedding

        self._base = SentenceTransformersEmbedding(settings)
        self.settings = settings
        self.dimension = self._base.dimension
        self.max_tokens = self._base.max_tokens
        self.info = self._base.info.model_copy(
            update={"name": self._base.info.name + "+late-chunking"}
        )

    async def embed_query(self, text: str) -> list[float]:
        return await self._base.embed_query(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._base.embed_documents(texts)

    def _pool_spans(self, document: str, spans: Sequence[tuple[int, int]]) -> list[list[float]]:
        import numpy as np

        model = self._base._model  # adapter-internal access
        tokenizer = model.tokenizer
        enc = tokenizer(
            document,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        out = model.encode(document, output_value="token_embeddings", convert_to_numpy=True)
        tokens = np.asarray(out)[: len(offsets)]
        vectors: list[list[float]] = []
        for start, end in spans:
            idx = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]
            if not idx:
                vectors.append([])
                continue
            v = tokens[idx].mean(axis=0)
            if self.settings.normalize:
                norm = float(np.linalg.norm(v)) or 1.0
                v = v / norm
            vectors.append([float(x) for x in v])
        return vectors

    async def embed_spans(
        self, document: str, spans: Sequence[tuple[int, int]], fallback_texts: Sequence[str]
    ) -> list[list[float]]:
        pooled = await asyncio.to_thread(self._pool_spans, document, spans)
        missing = [i for i, v in enumerate(pooled) if not v]
        if missing:  # spans beyond the model window -> classic per-chunk embedding
            extra = await self._base.embed_documents([fallback_texts[i] for i in missing])
            for i, vec in zip(missing, extra, strict=True):
                pooled[i] = vec
        return pooled

    def fingerprint(self) -> str:
        return self._base.fingerprint() + "-late"
