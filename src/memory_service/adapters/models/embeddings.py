"""Embedding adapters.

* ``HashEmbedding``: deterministic feature-hashed bag-of-words + fixed random projection.
  Zero dependencies, zero downloads. Used for tests and benchmarks as a labelled
  *non-representative* baseline; never a production default.
* ``SentenceTransformersEmbedding``: the frozen dense encoder (``constants.FROZEN_MODELS``),
  CPU-first, with PyTorch, ONNX or OpenVINO backends (``backend`` maps to
  sentence-transformers' native backends).
* ``OnnxEmbedding``: the same encoder on an ``onnxruntime`` session this repository builds
  itself, because sentence-transformers' ONNX backend is unreachable here (see the class).

All three expose ``fingerprint()`` (model + runtime + graph file + dimension), which is
baked into collection names and cache keys so a model swap can never mix vector spaces.
``load_dense(spec)`` is how the two real ones are reached: the runtime is chosen once, here,
not once per caller.

The two real encoders are entered through a ``SerialRunner``: one thread, one caller at a
time, with the intra-op thread count pinned to ``DenseModel.threads``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# numpy arrives with qdrant-client, a core dependency, and again with onnxruntime.
import numpy as np

from memory_service.adapters.models._precision import cpu_dtype_kwargs
from memory_service.adapters.models._runner import SerialRunner
from memory_service.config.constants import DenseModel
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo

_TOKEN = re.compile(r"[a-z0-9]+")

#: the graph an ONNX spec means when it names none
DEFAULT_GRAPH_FILE = "onnx/model.onnx"
#: sentence-transformers writes the pooling module's configuration here
POOLING_CONFIG = "1_Pooling/config.json"


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
        self.threads = threads or spec.threads
        import torch

        # Process-wide, and deliberately so: the NLI head sets the same number.
        torch.set_num_threads(self.threads)
        self.spec = spec
        self._runner = SerialRunner("encoder")
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
        return await self._runner.run(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self._runner.run(self._encode, [text]))[0]

    def fingerprint(self) -> str:
        return dense_fingerprint(self.spec, self.dimension)

    def close(self) -> None:
        self._runner.close()


class _OnnxEncoder:
    """The tensor path, and nothing else: tokenise, feed the session exactly the inputs it
    declares, pool, normalise. It opens no files and imports no runtime, so the arithmetic
    that decides what a vector *is* can be tested against a fake session."""

    def __init__(
        self,
        tokenizer: Any,
        session: Any,
        *,
        pooling: str,
        dimension: int,
        pad_id: int,
        normalize: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.session = session
        # ModernBERT declares input_ids and attention_mask and no token_type_ids; a
        # BERT-shaped graph declares all three. Feed what the graph asks for, not what the
        # family is assumed to want.
        self.inputs = tuple(declared.name for declared in session.get_inputs())
        unknown = sorted(set(self.inputs) - {"input_ids", "attention_mask", "token_type_ids"})
        if unknown:
            raise DependencyUnavailable(f"the graph declares inputs with no source: {unknown}")
        self.pooling = pooling
        self.dimension = dimension
        self.pad_id = pad_id
        self.normalize = normalize

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        ids, mask = self._batch(texts)
        feed: dict[str, Any] = {}
        for name in self.inputs:
            if name == "input_ids":
                feed[name] = ids
            elif name == "attention_mask":
                feed[name] = mask
            else:
                feed[name] = np.zeros_like(ids)
        hidden = np.asarray(self.session.run(None, feed)[0])
        return [[float(x) for x in row] for row in self._reduce(hidden, mask)]

    def _batch(self, texts: Sequence[str]) -> tuple[Any, Any]:
        """Truncated at the tokenizer, padded to the longest member of *this* batch — a
        query is 12 tokens and padding it to 512 would be forty times the work."""
        encoded = self.tokenizer.encode_batch(list(texts))
        width = max(1, *(len(e.ids) for e in encoded))
        ids = [list(e.ids) + [self.pad_id] * (width - len(e.ids)) for e in encoded]
        mask = [list(e.attention_mask) + [0] * (width - len(e.ids)) for e in encoded]
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.int64)

    def _reduce(self, hidden: Any, mask: Any) -> Any:
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            weights = mask[:, :, None].astype(hidden.dtype)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        if not self.normalize:
            return pooled
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)


class OnnxEmbedding:
    """The dense encoder on an ``onnxruntime`` session this repository builds itself.

    sentence-transformers' ``backend="onnx"`` is not reachable from this image: it loads the
    graph through ``optimum.onnxruntime``, and ``optimum-onnx`` pins ``optimum~=2.1``, which
    resolves against sentence-transformers 6 only by downgrading it. What that backend does
    for an encoder is four steps long, and owning them buys the two things the hot path
    needs and the library does not expose: ``intra_op_num_threads`` pinned before the
    session exists, and a graph file that is part of the vector space's name.

    The weights are never downloaded here — the graph is a file under the model directory,
    written by ``tools/download_models.py --export-onnx``.

    ``encoder`` is the loaded graph. The default builds it from ``spec``; passing one is how
    the tensor path is exercised without weights, which is the only way these tests can run
    on a machine torch has no wheel for.
    """

    info: ProviderInfo

    def __init__(
        self,
        spec: DenseModel,
        *,
        threads: int | None = None,
        encoder: _OnnxEncoder | None = None,
    ) -> None:
        self.threads = threads or spec.threads
        self._encoder = encoder if encoder is not None else _load_graph(spec, self.threads)
        self._runner = SerialRunner("encoder")
        self.spec = spec
        self.dimension = self._encoder.dimension
        self.max_tokens = spec.max_seq_length
        self.info = ProviderInfo(
            name=spec.id,
            version=_ort_version(),
            license="Apache-2.0" if "granite" in spec.id.lower() else "see model card",
            origin="huggingface/" + spec.id,
            locality="local",
        )

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        size = max(1, self.spec.batch_size)
        out: list[list[float]] = []
        for start in range(0, len(texts), size):
            out.extend(self._encoder.encode(texts[start : start + size]))
        return out

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._runner.run(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self._runner.run(self._encode, [text]))[0]

    def fingerprint(self) -> str:
        return onnx_fingerprint(self.spec, self.dimension)

    def close(self) -> None:
        self._runner.close()


def load_dense(
    spec: DenseModel, *, threads: int | None = None
) -> OnnxEmbedding | SentenceTransformersEmbedding:
    """The dense encoder ``spec.runtime`` names.

    One place makes this choice. Two runners that answer the same calls are exactly the
    shape of bug where a harness times one and stamps the artifact with the other's
    fingerprint, so wiring and the benchmarks come through here rather than each deciding.
    """
    if spec.runtime == "onnx":
        return OnnxEmbedding(spec, threads=threads)
    return SentenceTransformersEmbedding(spec, threads=threads)


def _load_graph(spec: DenseModel, threads: int) -> _OnnxEncoder:
    """Open the tokenizer and the session, with the thread counts set before the session
    exists — ``SessionOptions`` is read at construction and ignored afterwards."""
    directory = Path(spec.source)
    graph = directory / (spec.graph_file or DEFAULT_GRAPH_FILE)
    tokenizer_file = directory / "tokenizer.json"
    # Before the imports: a missing graph is the far more likely of the two, and saying
    # "install [models]" to someone whose image has the runtime but not the file sends them
    # to the wrong place.
    for path in (directory, graph, tokenizer_file):
        if not path.exists():
            raise DependencyUnavailable(
                f"the ONNX encoder needs {path}; run `python -m "
                f"memory_service.tools.download_models --export-onnx {directory}` "
                "inside the runtime image"
            )
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise DependencyUnavailable(
            "onnxruntime and tokenizers are required for the ONNX encoder (install [models])"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.no_padding()  # padded per batch, not to the model's limit
    tokenizer.enable_truncation(max_length=spec.max_seq_length)

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise DependencyUnavailable(
            f"the ONNX graph {graph} could not be loaded ({type(exc).__name__})"
        ) from exc

    pooling, width = _pooling_mode(directory)
    return _OnnxEncoder(
        tokenizer,
        session,
        pooling=pooling,
        dimension=width or spec.dimension,
        pad_id=_pad_token_id(directory, tokenizer),
        normalize=spec.normalize,
    )


def _pooling_mode(directory: Path) -> tuple[str, int | None]:
    """How the checkpoint says its token vectors become a sentence vector. Read rather than
    assumed: mean-pooling a CLS-trained encoder costs accuracy and raises nothing."""
    path = directory / POOLING_CONFIG
    if not path.is_file():
        raise DependencyUnavailable(f"{path} is missing; the pooling mode cannot be guessed")
    config = json.loads(path.read_text(encoding="utf-8"))
    modes = sorted(
        key.removeprefix("pooling_mode_")
        for key, value in config.items()
        if key.startswith("pooling_mode_") and value is True
    )
    width = config.get("word_embedding_dimension")
    if modes == ["cls_token"]:
        return "cls", width if isinstance(width, int) else None
    if modes == ["mean_tokens"]:
        return "mean", width if isinstance(width, int) else None
    raise DependencyUnavailable(f"{path} asks for {modes or ['no']} pooling; this runner does one")


def _pad_token_id(directory: Path, tokenizer: Any) -> int:
    """The padding id. The attention mask makes padded positions irrelevant to the result,
    but an id outside the vocabulary is an out-of-range gather, not a harmless filler."""
    config = directory / "config.json"
    if config.is_file():
        value = json.loads(config.read_text(encoding="utf-8")).get("pad_token_id")
        if isinstance(value, int):
            return value
    for token in ("[PAD]", "<pad>"):
        found = tokenizer.token_to_id(token)
        if found is not None:
            return int(found)
    return 0


def onnx_fingerprint(spec: DenseModel, dimension: int | None = None) -> str:
    """The name of the vector space under the ONNX runner. The graph file is in it because
    an int8 graph and its fp32 parent do not produce the same vectors: over the fifty texts
    of ``docs/MEASUREMENTS.md`` section 7 the int8 graph sits at min cosine 0.9667 from the
    torch runner where the fp32 graph sits at 1.000000, and two collections that disagree by
    that much must not share a name."""
    name = (spec.model_path or spec.id).rstrip("/").split("/")[-1]
    graph = (spec.graph_file or DEFAULT_GRAPH_FILE).rsplit("/", 1)[-1].removesuffix(".onnx")
    return f"onnx-{name}-{graph}-d{dimension or spec.dimension}"


def _ort_version() -> str:
    try:
        from importlib.metadata import version

        return version("onnxruntime")
    except Exception:
        return "unknown"


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
