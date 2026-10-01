"""Late-interaction (ColBERT) encoders: one vector per token, scored by MaxSim in the store.

* ``OnnxColbert``: the frozen encoder (``constants.FROZEN_MODELS.colbert``) on an
  ``onnxruntime`` session, tokenised exactly as the checkpoint's ``onnx_config.json`` says
  PyLate tokenises it - a ``[Q]``/``[D]`` marker after ``[CLS]``, lower-cased text, queries
  kept whole, punctuation tokens dropped from documents - and normalised per token.
* ``HashLateInteraction``: a deterministic per-token hash, the hermetic suite's stand-in. It
  loads no weights and is never representative.

Both expose ``fingerprint()``, which is part of the collection names, and ``dimension``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memory_service.adapters.models._runner import SerialRunner
from memory_service.config.constants import LateInteractionModel
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo

#: the checkpoint's own description of its tokenisation, written by the publisher's export
ONNX_CONFIG = "onnx_config.json"
_TOKEN = re.compile(r"\w+", re.UNICODE)


class ColbertGraph:
    """The tensor path, and nothing else: it opens no files, so tests drive it with a fake
    session and a real tokenizer."""

    def __init__(self, tokenizer: Any, session: Any, config: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self.inputs = tuple(declared.name for declared in session.get_inputs())
        unknown = sorted(set(self.inputs) - {"input_ids", "attention_mask"})
        if unknown:
            raise DependencyUnavailable(
                f"the ColBERT graph declares inputs with no source: {unknown}"
            )
        if config.get("do_query_expansion"):
            # Expansion pads a query with mask tokens that are themselves scored; the shipped
            # checkpoint does not use it and nothing here implements it.
            raise DependencyUnavailable("query expansion is not supported by this runner")
        self.lower = bool(config.get("do_lower_case", False))
        self.query_length = int(config["query_length"])
        self.document_length = int(config["document_length"])
        self.query_prefix_id = int(config["query_prefix_id"])
        self.document_prefix_id = int(config["document_prefix_id"])
        self.pad_id = int(config.get("pad_token_id", 0))
        self.dimension = int(config["embedding_dim"])
        self.skip = frozenset(
            found
            for word in config.get("skiplist_words", [])
            if (found := tokenizer.token_to_id(word)) is not None
        )

    def encode(self, texts: Sequence[str], *, query: bool) -> list[list[list[float]]]:
        """Each text's token vectors, unit length, as lists (what the store is sent)."""
        if not texts:
            return []
        rows = self._rows(texts, query=query)
        width = max(len(row) for row in rows)
        ids = np.asarray([row + [self.pad_id] * (width - len(row)) for row in rows], dtype=np.int64)
        mask = np.asarray(
            [[1] * len(row) + [0] * (width - len(row)) for row in rows], dtype=np.int64
        )
        feed = {"input_ids": ids, "attention_mask": mask}
        hidden = np.asarray(self.session.run(None, {k: feed[k] for k in self.inputs})[0])
        if hidden.ndim != 3 or hidden.shape[:2] != ids.shape or hidden.shape[2] != self.dimension:
            raise ValueError("the ColBERT graph returned an unexpected shape")
        return [self._kept(hidden[i], row, query=query) for i, row in enumerate(rows)]

    def _rows(self, texts: Sequence[str], *, query: bool) -> list[list[int]]:
        """Token ids with the marker after ``[CLS]``, cut so the marker fits the length."""
        length = self.query_length if query else self.document_length
        prefix = self.query_prefix_id if query else self.document_prefix_id
        rows: list[list[int]] = []
        for encoded in self.tokenizer.encode_batch([t.lower() if self.lower else t for t in texts]):
            ids = list(encoded.ids)
            if len(ids) > length - 1:
                # the tokenizer's own truncation keeps [SEP] last; so does this
                ids = [*ids[: length - 2], ids[-1]]
            rows.append([ids[0], prefix, *ids[1:]])
        return rows

    def _kept(self, hidden: Any, row: list[int], *, query: bool) -> list[list[float]]:
        """A query keeps every token; a document drops its punctuation. Then unit length."""
        keep = [j for j, token in enumerate(row) if query or token not in self.skip] or [0]
        vectors = hidden[keep, :]
        norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return (vectors / norms).astype(np.float32).tolist()


class OnnxColbert:
    info: ProviderInfo

    def __init__(
        self,
        spec: LateInteractionModel,
        *,
        threads: int | None = None,
        graph: ColbertGraph | None = None,
    ) -> None:
        self.spec = spec
        self.threads = threads or spec.threads
        self._graph = graph if graph is not None else _load(spec, self.threads)
        self.dimension = self._graph.dimension
        self._runner = SerialRunner("colbert")
        name = (spec.model_path or spec.id).rstrip("/").split("/")[-1]
        graph_name = spec.graph_file.rsplit("/", 1)[-1].removesuffix(".onnx")
        self._fingerprint = f"{name}-{graph_name}-d{self.dimension}"
        self.info = ProviderInfo(
            name=spec.id,
            license=spec.license,
            origin="huggingface/" + spec.id,
            locality="local",
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[list[float]]]:
        """Encoded in batches, the model yielded between them so a query never waits for
        a whole document backlog (as ``embeddings._document_batches``)."""
        out: list[list[list[float]]] = []
        size = self.spec.batch_size
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            out.extend(await self._runner.run(self._graph.encode, batch, query=False))
        return out

    async def embed_query(self, text: str) -> list[list[float]]:
        return (await self._runner.run(self._graph.encode, [text], query=True))[0]

    def fingerprint(self) -> str:
        return self._fingerprint

    def close(self) -> None:
        self._runner.close()


def _load(spec: LateInteractionModel, threads: int) -> ColbertGraph:
    directory = Path(spec.source)
    graph = directory / spec.graph_file
    for path in (graph, directory / "tokenizer.json", directory / ONNX_CONFIG):
        if not path.is_file():
            raise DependencyUnavailable(
                f"the ColBERT encoder needs {path}; run `make models` "
                "or bake the weights under /models"
            )
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise DependencyUnavailable(
            "onnxruntime and tokenizers are required for the ColBERT encoder (install [models])"
        ) from exc
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()  # the length depends on query or document; ColbertGraph cuts
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise DependencyUnavailable(
            f"the ColBERT graph {graph} could not be loaded ({type(exc).__name__})"
        ) from exc
    config = json.loads((directory / ONNX_CONFIG).read_text(encoding="utf-8"))
    return ColbertGraph(tokenizer, session, config)


class HashLateInteraction:
    """Deterministic token vectors: each word hashed to a signed unit vector. A labelled
    non-representative stand-in, so the hermetic suite runs the late-interaction path."""

    info = ProviderInfo(
        name="hash-late-interaction",
        version="1",
        license="Apache-2.0",
        origin="internal",
        locality="local",
    )

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def _vectors(self, text: str) -> list[list[float]]:
        out = []
        for token in _TOKEN.findall(text.lower())[:64] or ["\x00"]:
            digest = hashlib.blake2b(token.encode(), digest_size=self.dimension).digest()
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) - 127.5
            out.append((vector / np.linalg.norm(vector)).tolist())
        return out

    async def embed_documents(self, texts: Sequence[str]) -> list[list[list[float]]]:
        return [self._vectors(t) for t in texts]

    async def embed_query(self, text: str) -> list[list[float]]:
        return self._vectors(text)

    def fingerprint(self) -> str:
        return f"hash-li-v1-d{self.dimension}"
