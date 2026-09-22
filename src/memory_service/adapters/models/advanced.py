"""Benchmark-gated learned-sparse encoder (SPLADE / miniCOIL / BM42 via fastembed). It
declares its license/origin, loads only from local files when a ``model_path`` is given, and
fails with ``DependencyUnavailable`` (never a silent fallback) when the weights are absent —
the benchmark harness records that as *skipped*.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo
from memory_service.ports.search import SparseVector

_SPARSE_LICENSES = {
    "prithivida/Splade_PP_en_v1": "Apache-2.0",
    "Qdrant/bm42-all-minilm-l6-v2-attentions": "Apache-2.0",
}


def _flat_model_dir(model_path: str) -> str:
    """Give fastembed the one flat directory it expects.

    It loads ``<dir>/model.onnx`` and the tokenizer from that same ``<dir>``, but upstream
    repositories disagree on the layout: some ship ``model.onnx`` at the top level while
    ``Splade_PP_en_v1`` ships it under ``onnx/``. Rather than ask
    whoever downloads the weights to rearrange them — or copy hundreds of megabytes — link
    both layouts into one directory under the cache. The weights are never modified, and a
    read-only model directory stays read-only.
    """
    root = Path(model_path)
    if (root / "model.onnx").exists() or not (root / "onnx" / "model.onnx").exists():
        return str(root)

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    flat = cache / "memory-service" / "flat-models" / root.name
    flat.mkdir(parents=True, exist_ok=True)
    for source in [*root.iterdir(), *(root / "onnx").iterdir()]:
        if source.is_dir():
            continue
        link = flat / source.name
        if not link.exists():
            link.symlink_to(source.resolve())
    return str(flat)


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
            kwargs["specific_model_path"] = _flat_model_dir(model_path)
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
