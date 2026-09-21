"""Cognee as an *experimental* MemoryIntelligenceProvider.

Cognee ingests text into its own graph + vector stores (``add`` -> ``cognify``) and answers
``search`` queries; it does not expose a per-observation extraction API, so this adapter
treats each observation as a Cognee dataset addition and reads back the derived summaries
as candidates. It requires an LLM and Cognee's own databases; it is wired only when
``memory_intelligence.provider=cognee`` and is not part of the release gates.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import Observation
from memory_service.modules.memory.native import _evidence, normalize
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate
from memory_service.ports.models import ProviderInfo


class CogneeMemoryIntelligence:
    info = ProviderInfo(
        name="cognee",
        license="Apache-2.0",
        origin="topoteretes/cognee",
        locality="local",
        requires_llm=True,
    )

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def _cognee(self) -> Any:
        if self._client is None:
            llm = self.settings.models.llm
            if not llm.enabled or not llm.model:
                raise DependencyUnavailable("cognee requires models.llm.enabled=true and a model")
            # Cognee reads its LLM/embedding endpoints from the environment (OpenAI-compatible);
            # both point at the Bifrost gateway. Must be set before the package is imported.
            api_key = llm.api_key.get_secret_value() if llm.api_key else "-"
            os.environ.update(
                {
                    "LLM_PROVIDER": "openai",
                    "LLM_MODEL": llm.model,
                    "LLM_ENDPOINT": llm.base_url,
                    "LLM_API_KEY": api_key,
                    "EMBEDDING_PROVIDER": "openai",
                    "EMBEDDING_MODEL": self.settings.graph_enrichment.graphiti_embedding_model,
                    "EMBEDDING_ENDPOINT": llm.base_url,
                    "EMBEDDING_API_KEY": api_key,
                    "EMBEDDING_DIMENSIONS": str(
                        self.settings.graph_enrichment.graphiti_embedding_dim
                    ),
                }
            )
            try:
                import cognee
            except ImportError as exc:
                raise DependencyUnavailable("cognee is required (install [cognee])") from exc
            self._client = cognee
        return self._client

    async def extract(
        self, observation: Observation, ctx: MemoryExecutionContext
    ) -> list[MemoryCandidate]:
        if observation.hints.skip_extraction or not observation.content.strip():
            return []
        cognee = self._cognee()
        dataset = f"{ctx.tenant_id}__{ctx.principal_id}".replace(":", "_")
        await cognee.add(observation.content, dataset_name=dataset)
        await cognee.cognify(datasets=[dataset])
        results = await cognee.search(query_text=observation.content[:500], datasets=[dataset])
        evidence = _evidence(observation)
        out = []
        for r in results or []:
            text = str(r if isinstance(r, str) else getattr(r, "text", r)).strip()
            if text:
                out.append(
                    MemoryCandidate(
                        content=text[:2000],
                        memory_type=MemoryType.SEMANTIC,
                        lifetime=Lifetime.LONG_TERM,
                        evidence=evidence,
                        confidence=0.6,
                        provider="cognee",
                    )
                )
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
                    reason="cognee: identical",
                )
        return ConsolidationOutcome(
            decision=DedupDecision.CREATE, candidate=candidate, reason="cognee: new"
        )
