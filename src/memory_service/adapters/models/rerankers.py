"""Reranker adapters: cross-encoder (MiniLM / Granite reranker) and a lexical fallback."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from memory_service.adapters.models.sparse import tokenize
from memory_service.config.settings import RerankerSettings
from memory_service.adapters.models._precision import cpu_dtype_kwargs
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo, RerankResult


class LexicalReranker:
    """BM25-style overlap between query and document terms. Deterministic; tests/fallback."""

    info = ProviderInfo(
        name="lexical-reranker",
        version="1",
        license="Apache-2.0",
        origin="internal",
        locality="local",
    )

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int
    ) -> list[RerankResult]:
        q = Counter(tokenize(query))
        if not q:
            return [RerankResult(index=i, score=0.0) for i in range(min(top_k, len(documents)))]
        docs = [Counter(tokenize(d)) for d in documents]
        n = len(docs) or 1
        df = Counter()
        for d in docs:
            df.update(d.keys())
        avg = sum(sum(d.values()) for d in docs) / n if docs else 1.0
        scores = []
        for i, d in enumerate(docs):
            dl = sum(d.values()) or 1
            s = 0.0
            for term, qtf in q.items():
                tf = d.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += (
                    idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * dl / max(avg, 1.0))) * min(qtf, 2)
                )
            scores.append((s, i))
        scores.sort(key=lambda t: (-t[0], t[1]))
        return [RerankResult(index=i, score=float(s)) for s, i in scores[:top_k]]

    def fingerprint(self) -> str:
        return "lexical-v1"


class CrossEncoderReranker:
    info: ProviderInfo

    def __init__(self, settings: RerankerSettings) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise DependencyUnavailable(
                "sentence-transformers is required (install [models])"
            ) from exc
        source = settings.model_path or settings.model
        kwargs: dict[str, Any] = {"device": "cpu"}
        if settings.provider == "onnx":
            kwargs["backend"] = "onnx"
        else:
            kwargs["model_kwargs"] = cpu_dtype_kwargs()
        if settings.model_path:
            kwargs["local_files_only"] = True
        try:
            self._model = CrossEncoder(source, **kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"reranker {source!r} could not be loaded ({type(exc).__name__}); "
                "set MEMORY__MODELS__RERANKER__MODEL_PATH"
            ) from exc
        self.settings = settings
        self.info = ProviderInfo(
            name=settings.model,
            license="Apache-2.0",
            origin="huggingface/" + settings.model,
            locality="local",
        )

    def _score(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [(query, d) for d in documents]
        out = self._model.predict(
            pairs, batch_size=self.settings.batch_size, show_progress_bar=False
        )
        return [float(x) for x in out]

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int
    ) -> list[RerankResult]:
        if not documents:
            return []
        scores = await asyncio.to_thread(self._score, query, documents)
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        return [RerankResult(index=i, score=scores[i]) for i in order[:top_k]]

    def fingerprint(self) -> str:
        return f"ce-{(self.settings.model_path or self.settings.model).rstrip('/').split('/')[-1]}"
