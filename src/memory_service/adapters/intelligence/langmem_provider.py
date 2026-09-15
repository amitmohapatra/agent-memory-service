"""LangMem as a benchmarkable MemoryIntelligenceProvider.

LangMem's ``create_memory_manager`` runs an LLM over the new messages *and* the existing
memories and returns the updated memory list; ids that match existing memories are updates,
new ids are inserts. The manager is injectable for tests; the real one needs an LLM
(``models.llm.enabled=true``) and a LangChain model string.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import Observation
from memory_service.modules.memory.native import _evidence, normalize
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate
from memory_service.ports.models import ProviderInfo


class _Manager(Protocol):
    async def ainvoke(self, state: dict[str, Any]) -> list[Any]: ...


class LangMemIntelligence:
    info = ProviderInfo(
        name="langmem",
        license="MIT",
        origin="langchain-ai/langmem",
        locality="local",
        requires_llm=True,
    )

    def __init__(self, settings: Settings, manager: _Manager | None = None) -> None:
        self.settings = settings
        self._manager = manager
        self._updates: dict[str, str] = {}  # normalized content -> existing memory id

    def _get_manager(self) -> _Manager:
        if self._manager is None:
            try:
                from langmem import create_memory_manager
            except ImportError as exc:
                raise DependencyUnavailable(
                    "langmem is required (install [memory-providers])"
                ) from exc
            llm = self.settings.models.llm
            if not llm.enabled or not llm.model:
                raise DependencyUnavailable("langmem requires models.llm.enabled=true and a model")
            # LangMem builds a LangChain OpenAI chat model from the model string; the
            # OpenAI-compatible endpoint it talks to is the Bifrost gateway, never a provider.
            os.environ["OPENAI_BASE_URL"] = llm.base_url
            os.environ["OPENAI_API_KEY"] = llm.api_key.get_secret_value() if llm.api_key else "-"
            self._manager = create_memory_manager(
                f"openai:{llm.model}", enable_inserts=True, enable_updates=True
            )  # type: ignore[assignment]
        return self._manager  # type: ignore[return-value]

    async def extract(
        self, observation: Observation, ctx: MemoryExecutionContext
    ) -> list[MemoryCandidate]:
        # Extraction and consolidation are one LLM call in LangMem; ``existing`` is supplied
        # through ``consolidate_batch`` by the pipeline when available, else empty.
        if observation.hints.skip_extraction or not observation.content.strip():
            return []
        results = await self._get_manager().ainvoke(
            {"messages": [{"role": "user", "content": observation.content}], "existing": []}
        )
        evidence = _evidence(observation)
        out = []
        for item in results:
            content = getattr(item, "content", None)
            text = str(getattr(content, "content", content) or "").strip()
            if not text:
                continue
            ident = getattr(item, "id", None)
            out.append(
                MemoryCandidate(
                    content=text,
                    memory_type=MemoryType.SEMANTIC,
                    lifetime=Lifetime.LONG_TERM,
                    subject=f"user:{ctx.user_id}" if ctx.user_id else ctx.principal_id,
                    predicate="langmem",
                    evidence=evidence,
                    confidence=0.7,
                    provider="langmem",
                    provider_ref=str(ident) if ident else None,
                )
            )
            if ident:
                self._updates[normalize(text)] = str(ident)
        return out

    async def classify(
        self, candidate: MemoryCandidate, ctx: MemoryExecutionContext
    ) -> MemoryCandidate:
        return candidate

    async def consolidate(
        self,
        candidate: MemoryCandidate,
        existing: Sequence[CanonicalMemory],
        ctx: MemoryExecutionContext,
    ) -> ConsolidationOutcome:
        key = normalize(candidate.content)
        for m in existing:
            if normalize(m.content) == key:
                return ConsolidationOutcome(
                    decision=DedupDecision.REINFORCE,
                    candidate=candidate,
                    target_memory_id=m.memory_id,
                    score=1.0,
                    reason="langmem: identical",
                )
        ident = self._updates.pop(key, None)
        if ident:
            target = next(
                (m for m in existing if m.system_metadata.get("provider_ref") == ident), None
            )
            if target is not None:
                return ConsolidationOutcome(
                    decision=DedupDecision.SUPERSEDE,
                    candidate=candidate,
                    target_memory_id=target.memory_id,
                    score=0.9,
                    reason="langmem: update",
                )
        return ConsolidationOutcome(
            decision=DedupDecision.CREATE, candidate=candidate, reason="langmem: insert"
        )

    async def search_features(self, query: str, ctx: MemoryExecutionContext) -> dict[str, Any]:
        return {}
