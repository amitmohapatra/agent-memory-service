"""Local CPU NLI without Torch, with bounded paired tokenization and explicit label order."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memory_service.adapters.models._runner import SerialRunner
from memory_service.adapters.models.nli import BatchedNLI
from memory_service.config.constants import NLIModel
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import NLIScore, ProviderInfo


class OnnxNLI(BatchedNLI):
    def __init__(self, spec: NLIModel, *, threads: int | None = None) -> None:
        root = Path(spec.source)
        graph = root / (spec.graph_file or "onnx/model.onnx")
        paths = (graph, root / "tokenizer.json", root / "config.json")
        if not all(path.is_file() for path in paths):
            raise DependencyUnavailable(
                "ONNX NLI needs local graph, tokenizer.json and config.json"
            )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise DependencyUnavailable("ONNX NLI requires onnxruntime and tokenizers") from exc
        config = json.loads(paths[2].read_text())
        self.spec = spec
        self.threads = threads or spec.threads
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        try:
            session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
            tokenizer = Tokenizer.from_file(str(paths[1]))
            tokenizer.no_padding()
            tokenizer.enable_truncation(max_length=spec.max_length, strategy="longest_first")
            self._scorer = OnnxNLIScorer(tokenizer, session, config)
        except Exception as exc:
            raise DependencyUnavailable(
                f"ONNX NLI initialization failed ({type(exc).__name__})"
            ) from exc
        identity = hashlib.sha256(spec.model_dump_json().encode())
        for path in paths:
            with path.open("rb") as stream:
                identity.update(hashlib.file_digest(stream, "sha256").digest())
        self._fingerprint = f"nli-onnx-{identity.hexdigest()[:24]}"
        self.info = ProviderInfo(
            name=spec.id,
            license="see model card",
            origin="huggingface/" + spec.id,
            locality="local",
        )
        self._runner = SerialRunner("nli")

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        scores: list[NLIScore] = []
        for start in range(0, len(pairs), self.spec.batch_size):
            scores.extend(self._scorer.score(pairs[start : start + self.spec.batch_size]))
        return scores

    def fingerprint(self) -> str:
        return self._fingerprint


class OnnxNLIScorer:
    """Tensor-only contract; injected sessions let tests exercise actual scoring mechanics."""

    def __init__(self, tokenizer: Any, session: Any, config: dict[str, Any]) -> None:
        self.tokenizer, self.session = tokenizer, session
        self.inputs = {item.name for item in session.get_inputs()}
        if not self.inputs or self.inputs - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("Unsupported NLI graph inputs")
        labels = {str(value).lower(): int(key) for key, value in config["id2label"].items()}
        self.order = [labels[name] for name in ("entailment", "neutral", "contradiction")]
        if sorted(self.order) != [0, 1, 2]:
            raise ValueError("NLI graph must declare exactly three distinct labels")
        self.pad_id = int(config["pad_token_id"])

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        if not pairs:
            return []
        encoded = self.tokenizer.encode_batch(list(pairs))
        width = max(len(item.ids) for item in encoded)
        sources = {
            "input_ids": ("ids", self.pad_id),
            "attention_mask": ("attention_mask", 0),
            "token_type_ids": ("type_ids", 0),
        }
        feed = {}
        for name in self.inputs:
            attribute, padding = sources[name]
            feed[name] = np.asarray(
                [
                    [*getattr(item, attribute), *([padding] * (width - len(item.ids)))]
                    for item in encoded
                ],
                dtype=np.int64,
            )
        logits = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
        if logits.shape != (len(pairs), 3) or not np.isfinite(logits).all():
            raise ValueError("NLI graph returned invalid logits")
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        e, n, c = self.order
        return [
            NLIScore(entailment=float(row[e]), neutral=float(row[n]), contradiction=float(row[c]))
            for row in probabilities
        ]
