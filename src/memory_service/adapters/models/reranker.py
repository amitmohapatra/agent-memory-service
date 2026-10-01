"""Cross-encoders: the question and one candidate read together, one relevance logit each.

* ``OnnxReranker``: a frozen ``RerankerModel`` on an ``onnxruntime`` session, entered
  through its own ``SerialRunner`` so two rerankers score one pool at the same time.
* ``LexicalReranker``: the share of the question's words a candidate contains - the hermetic
  suite's stand-in, never representative.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memory_service.adapters.models._runner import SerialRunner
from memory_service.config.constants import RerankerModel
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo

_WORD = re.compile(r"\w+", re.UNICODE)


class RerankerGraph:
    """Tokenise the pairs, feed exactly the inputs the graph declares, take the logit."""

    def __init__(self, tokenizer: Any, session: Any, *, pad_id: int) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self.inputs = tuple(declared.name for declared in session.get_inputs())
        unknown = sorted(set(self.inputs) - {"input_ids", "attention_mask", "token_type_ids"})
        if unknown:
            raise DependencyUnavailable(
                f"the reranker graph declares inputs with no source: {unknown}"
            )
        self.pad_id = pad_id

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        encoded = self.tokenizer.encode_batch([(query, text) for text in texts])
        width = max(len(e.ids) for e in encoded)
        sources = {
            "input_ids": ("ids", self.pad_id),
            "attention_mask": ("attention_mask", 0),
            "token_type_ids": ("type_ids", 0),
        }
        feed = {}
        for name in self.inputs:
            attribute, padding = sources[name]
            feed[name] = np.asarray(
                [[*getattr(e, attribute), *([padding] * (width - len(e.ids)))] for e in encoded],
                dtype=np.int64,
            )
        logits = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
        logits = logits.reshape(len(texts), -1)[:, 0]
        if not np.isfinite(logits).all():
            raise ValueError("the reranker graph returned a non-finite score")
        return [float(x) for x in logits]


class OnnxReranker:
    info: ProviderInfo

    def __init__(
        self, spec: RerankerModel, *, threads: int | None = None, graph: RerankerGraph | None = None
    ) -> None:
        self.spec = spec
        self.threads = threads or spec.threads
        self._graph = graph if graph is not None else _load(spec, self.threads)
        self._runner = SerialRunner(f"rerank-{spec.local_dir}")
        self.name = spec.local_dir
        self.info = ProviderInfo(
            name=spec.id,
            license=spec.license,
            origin="huggingface/" + spec.id,
            locality="local",
        )

    def _score(self, query: str, texts: Sequence[str]) -> list[float]:
        out: list[float] = []
        for start in range(0, len(texts), self.spec.batch_size):
            out.extend(self._graph.score(query, texts[start : start + self.spec.batch_size]))
        return out

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return await self._runner.run(self._score, query, list(texts))

    def close(self) -> None:
        self._runner.close()


def _load(spec: RerankerModel, threads: int) -> RerankerGraph:
    directory = Path(spec.source)
    graph = directory / spec.graph_file
    for path in (graph, directory / "tokenizer.json"):
        if not path.is_file():
            raise DependencyUnavailable(
                f"the reranker needs {path}; run `make models` or bake the weights under /models"
            )
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise DependencyUnavailable(
            "onnxruntime and tokenizers are required for the reranker (install [models])"
        ) from exc
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=spec.max_length, strategy="longest_first")
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
            f"the reranker graph {graph} could not be loaded ({type(exc).__name__})"
        ) from exc
    pad = tokenizer.token_to_id("<pad>")
    if pad is None:
        pad = tokenizer.token_to_id("[PAD]")
    return RerankerGraph(tokenizer, session, pad_id=int(pad or 0))


class LexicalReranker:
    """The share of the question's words each candidate contains. A stand-in: it loads
    nothing and ranks nothing a model would."""

    info = ProviderInfo(
        name="lexical-reranker",
        version="1",
        license="Apache-2.0",
        origin="internal",
        locality="local",
    )

    def __init__(self, name: str = "lexical") -> None:
        self.name = name

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        wanted = set(_WORD.findall(query.lower()))
        return [
            len(wanted & set(_WORD.findall(text.lower()))) / (len(wanted) or 1) for text in texts
        ]
