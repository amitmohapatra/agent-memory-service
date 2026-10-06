"""The cross-encoder challenger the offline reranking benchmarks score.

The service ships no reranker: SciFact-1000 nDCG@10 fell from 84.51 to 79.33 with one (paired
sign test p = 0.012) at 21x the latency, and LoCoMo source ranking was worse than the fused
order (``benchmark/reports/PHASE9-RESULTS-2026-09-29.md``). The model spec and the adapter live here, beside
the benchmarks that measure them, and nowhere in ``src/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.adapters.models._precision import cpu_dtype_kwargs
from memory_service.adapters.models._runner import SerialRunner
from memory_service.config.constants import local_model_path
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.ids import stable_key
from memory_service.ports.models import ProviderInfo


class CrossEncoderModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    local_dir: str = "ms-marco-MiniLM-L6-v2"
    model_path: str | None = None
    revision: str | None = None
    backend: Literal["torch", "onnx"] = "torch"
    batch_size: int = 16
    max_length: int = Field(default=512, ge=32, le=2048)

    @property
    def source(self) -> str:
        return self.model_path or local_model_path(self.local_dir) or self.id


@dataclass(frozen=True)
class Ranked:
    index: int
    score: float


class CrossEncoderReranker:
    info: ProviderInfo

    def __init__(self, spec: CrossEncoderModel) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise DependencyUnavailable(
                "sentence-transformers is required (install [models])"
            ) from exc
        source = spec.source
        kwargs: dict[str, Any] = {
            "device": "cpu",
            "max_length": spec.max_length,
            "revision": spec.revision,
        }
        if spec.backend == "onnx":
            kwargs["backend"] = "onnx"
        else:
            kwargs["model_kwargs"] = cpu_dtype_kwargs()
        if source != spec.id:
            kwargs["local_files_only"] = True
        try:
            self._model = CrossEncoder(source, **kwargs)
        except Exception as exc:
            raise DependencyUnavailable(
                f"reranker {source!r} could not be loaded ({type(exc).__name__}); "
                "download it under models/ first"
            ) from exc
        import torch

        self._sigmoid = torch.nn.Sigmoid()
        self._runner = SerialRunner("reranker")
        self.spec = spec
        profile = stable_key(spec.model_dump_json())[:12]
        self._fingerprint = f"ce-{spec.id.rsplit('/', 1)[-1]}-{profile}"
        self.info = ProviderInfo(
            name=spec.id,
            license="Apache-2.0",
            origin="huggingface/" + spec.id,
            locality="local",
        )

    def _score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance in 0..1, not a raw logit.

        Sigmoid produces bounded scores; calibration on the target corpus is a separate
        measurement. Ours came back with ``activation_fn=Identity()``
        because the model directory carries no ``modules.json``, so a freshly constructed
        CrossEncoder gets no activation and we were publishing logits in the -11..+11 range
        as if they were scores. Asking for the sigmoid explicitly removes the dependence on
        what happens to be in the weights directory.
        """
        pairs = [(query, d) for d in documents]
        out = self._model.predict(
            pairs,
            batch_size=self.spec.batch_size,
            show_progress_bar=False,
            activation_fn=self._sigmoid,
        )
        return [float(x) for x in out]

    async def rerank(self, query: str, documents: Sequence[str], *, top_k: int) -> list[Ranked]:
        if not documents:
            return []
        scores = await self._runner.run(self._score, query, documents)
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        return [Ranked(index=i, score=scores[i]) for i in order[:top_k]]

    def fingerprint(self) -> str:
        return self._fingerprint

    def close(self) -> None:
        self._runner.close()
