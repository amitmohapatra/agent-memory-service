"""NLI adapters: DeBERTa (transformers, CPU) and a deterministic lexical stand-in."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memory_service.adapters.models._precision import cpu_dtype_kwargs
from memory_service.adapters.models._runner import SerialRunner
from memory_service.config.constants import NLIModel
from memory_service.domain.errors import DependencyUnavailable
from memory_service.modules.grounding.lexical import conflicts, content_tokens, coverage
from memory_service.ports.models import NLIScore, ProviderInfo

# How much of a claim a premise must cover before a number or negation clash between them is
# read as a contradiction rather than a coincidence. Two sentences sharing only "EUR" and
# "million" will always disagree on some number; calling that a refutation makes a mis-cited
# claim look like a factual error. Measured over tests/eval/golden/grounding_claims.json: on the
# deciding premise, genuine contradictions score >= 0.60 coverage while coincidental clashes
# never exceed 0.40, so the bar sits in the empty band between them.
_RELATED = 0.5
_MAX_SCORE = 0.95


class LexicalNLI:
    """Token coverage with number / negation / polarity agreement mapped onto NLI scores.
    Deterministic and fast; its verdicts are NOT representative of a trained classifier
    (paraphrases score low), so reports built with it say ``representative: false``."""

    info = ProviderInfo(
        name="lexical-nli", version="1", license="Apache-2.0", origin="internal", locality="local"
    )
    representative = False

    async def entail(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]:
        return [self.score(p, hypothesis) for p in premises]

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        """Nothing to batch: this is arithmetic on token sets, with no model to enter."""
        return [[self.score(p, h) for p in premises] for premises, h in groups]

    @staticmethod
    def score(premise: str, hypothesis: str) -> NLIScore:
        if not content_tokens(hypothesis):
            return NLIScore(entailment=0.0, neutral=1.0, contradiction=0.0)
        cov = coverage(hypothesis, premise)
        if cov >= _RELATED and conflicts(hypothesis, premise):
            c = min(_MAX_SCORE, 0.6 + 0.35 * cov)
            e = round((1.0 - c) * 0.2, 4)
        else:
            e = min(_MAX_SCORE, round(cov, 4))
            c = round(0.05 * (1.0 - cov), 4)
        return NLIScore(entailment=e, neutral=round(1.0 - e - c, 4), contradiction=c)

    def fingerprint(self) -> str:
        return "lexical-nli-v1"


class TransformersNLI:
    """``AutoModelForSequenceClassification`` cross-encoder (DeBERTa-v3 MNLI/FEVER/ANLI) on
    CPU, batched, and entered one caller at a time on its own thread. Label order is read
    from the model config."""

    info: ProviderInfo
    representative = True

    def __init__(self, spec: NLIModel, *, threads: int | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise DependencyUnavailable(
                "transformers and torch are required for the NLI (install [models])"
            ) from exc
        source = spec.source
        kwargs: dict[str, Any] = {"local_files_only": True} if source != spec.id else {}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                source,
                **kwargs,
                **cpu_dtype_kwargs(),  # the NLI runs on CPU by design
            )
        except Exception as exc:
            raise DependencyUnavailable(
                f"nli model {source!r} could not be loaded ({type(exc).__name__}); "
                "run `make models` or bake the weights under /models"
            ) from exc
        self._model.eval()
        self._torch = torch
        # Process-wide, and the same number the encoder sets: two models that each fan over
        # every core are worse than two that each take two.
        self.threads = threads or spec.threads
        torch.set_num_threads(self.threads)
        self._runner = SerialRunner("nli")
        self.spec = spec
        labels = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._order = [
            next(i for i, name in labels.items() if name.startswith(prefix))
            for prefix in ("entail", "neutral", "contra")
        ]
        self.info = ProviderInfo(
            name=spec.id,
            license="MIT",
            origin="huggingface/" + spec.id,
            locality="local",
        )

    def _score(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]:
        return self._score_pairs([(p, hypothesis) for p in premises])

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        """Every (premise, hypothesis) pair, batched by the spec's batch size.

        The pairs need not share a hypothesis. Padding is per batch, so mixing claims costs
        only what the longest row in each batch costs, and one answer's premises are of
        similar length.
        """
        out: list[NLIScore] = []
        size = max(1, self.spec.batch_size)
        with self._torch.no_grad():
            for start in range(0, len(pairs), size):
                batch = list(pairs[start : start + size])
                encoded = self._tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation=True,
                    max_length=self.spec.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                probs = self._model(**encoded).logits.softmax(dim=-1)
                for row in probs.tolist():
                    e, n, c = (float(row[i]) for i in self._order)
                    out.append(NLIScore(entailment=e, neutral=n, contradiction=c))
        return out

    async def entail(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]:
        if not premises:
            return []
        return await self._runner.run(self._score, list(premises), hypothesis)

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        """One gate acquisition and one batched pass for every claim, then regrouped."""
        pairs = [(premise, hypothesis) for premises, hypothesis in groups for premise in premises]
        if not pairs:
            return [[] for _ in groups]
        flat = await self._runner.run(self._score_pairs, pairs)
        scores: list[list[NLIScore]] = []
        cut = 0
        for premises, _ in groups:
            scores.append(flat[cut : cut + len(premises)])
            cut += len(premises)
        return scores

    def close(self) -> None:
        self._runner.close()

    def fingerprint(self) -> str:
        source = self.spec.model_path or self.spec.id
        graph = f"-{self.spec.graph_file.removesuffix('.onnx')}" if self.spec.graph_file else ""
        return f"nli-{source.rstrip('/').split('/')[-1]}{graph}"
