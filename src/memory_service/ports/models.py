"""Model provider ports: embeddings, reranking, sparse encoding, NLI, generative LLM.

All providers carry a :class:`ProviderInfo` (name, version, license, origin, locality,
data residency) so the provider policy can allow/deny them by configuration.
The generative LLM port is *optional*; the core service runs with ``llm.enabled=false``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memory_service.ports.search import SparseVector


class ProviderInfo(BaseModel):
    """License/origin registry entry. Every external model/provider records one."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "unknown"
    license: str = Field(..., description="SPDX identifier, e.g. Apache-2.0")
    origin: str = Field(..., description="vendor / repository")
    locality: Literal["local", "remote"]
    data_residency: str = Field(
        default="in-process", description="where data goes: in-process|region"
    )
    requires_llm: bool = False


@runtime_checkable
class EmbeddingProvider(Protocol):
    info: ProviderInfo
    dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    def fingerprint(self) -> str:
        """Model id + version + runtime; embedded in cache keys and collection names."""
        ...


@runtime_checkable
class SparseEncoder(Protocol):
    """Produces sparse vectors (BM25 term frequencies, SPLADE, miniCOIL)."""

    info: ProviderInfo

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...

    def fingerprint(self) -> str: ...


class RerankResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    info: ProviderInfo

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int
    ) -> list[RerankResult]:
        """Return the ``top_k`` best documents, best first. Bounded: len(documents) <= K."""
        ...

    def fingerprint(self) -> str: ...


class NLIScore(BaseModel):
    """Probabilities that a premise entails / is neutral to / contradicts a hypothesis."""

    model_config = ConfigDict(frozen=True)

    entailment: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    contradiction: float = Field(..., ge=0.0, le=1.0)


@runtime_checkable
class NLIProvider(Protocol):
    """Natural-language-inference classifier used by the grounding cascade. One hypothesis
    (a claim) is scored against each premise (an evidence item); the list is index-aligned
    with ``premises``. ``representative`` is False for deterministic stand-ins."""

    info: ProviderInfo
    representative: bool

    async def entail(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]: ...

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        """Several claims at once: one result list per group, index-aligned with ``groups``.

        ``entail`` takes a single hypothesis, so a caller with N claims paid N entries into
        the model - N acquisitions of the one-caller gate, N tokenizer calls and N forward
        passes of a handful of rows each - where the same work is one pass over all of them.
        With ``max_claims`` defaulting to 40 that is not a small multiple, and the verified
        context endpoint is the slowest thing the service can be asked to do.
        """
        ...

    def fingerprint(self) -> str: ...


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMCompletion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Optional generative model behind the Bifrost gateway. ``DisabledLLM`` raises
    ``ProviderNotConfigured``. ``use`` names the configured ``LLMSettings.uses`` entry making
    the call (metrics/tracing label and fast-vs-strong model selection)."""

    info: ProviderInfo
    enabled: bool

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use: str = "generic",
    ) -> LLMCompletion: ...

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        use: str = "generic",
    ) -> dict[str, Any]: ...
