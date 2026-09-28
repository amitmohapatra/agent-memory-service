"""The shared NLI batching contract and a deterministic lexical stand-in.

The only trained runtime is the ONNX session in ``onnx_nli.py`` (mDeBERTa, multilingual):
one model, one runtime (ADR 0024). ``LexicalNLI`` is the arithmetic stand-in the hermetic
suite and the grounding-free benchmarks select explicitly; it is never a silent fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from memory_service.adapters.models._runner import SerialRunner
from memory_service.config.constants import NLIModel
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
        name="lexical-nli", version="2", license="Apache-2.0", origin="internal", locality="local"
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
        return "lexical-nli-v2"


class BatchedNLI(ABC):
    """Ordered scoring and one bounded CPU runner shared by trained NLI runtimes."""

    info: ProviderInfo
    representative = True
    spec: NLIModel
    _runner: SerialRunner

    @abstractmethod
    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]: ...

    async def entail(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]:
        if not premises:
            return []
        pairs = [(premise, hypothesis) for premise in premises]
        return await self._runner.run(self._score_pairs, pairs)

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
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

    @abstractmethod
    def fingerprint(self) -> str: ...
