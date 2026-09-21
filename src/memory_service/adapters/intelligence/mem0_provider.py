"""Mem0 as a benchmarkable MemoryIntelligenceProvider.

Mem0 extracts and consolidates in one LLM-driven ``add`` call against its own vector store.
This adapter runs that call in an isolated, per-tenant namespace (Mem0's ``user_id`` is set
to ``<tenant>/<principal>`` so nothing crosses tenants) and translates Mem0's events into
this service's decisions:

    ADD    -> CREATE          UPDATE -> SUPERSEDE (the Mem0 memory id is kept as the slot)
    NONE   -> REINFORCE       DELETE -> IGNORE (handled as supersession of the old value)

PostgreSQL stays canonical: Mem0's store is a working copy, rebuilt by re-processing
observations. Requires an LLM (``models.llm.enabled=true``); the adapter refuses to start
otherwise. The client is injectable for tests.
"""

from __future__ import annotations

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


class _Mem0Client(Protocol):
    async def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]: ...
    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


def namespace(ctx: MemoryExecutionContext) -> str:
    return f"{ctx.tenant_id}/{ctx.principal_id}"


class Mem0MemoryIntelligence:
    info = ProviderInfo(
        name="mem0",
        license="Apache-2.0",
        origin="mem0ai/mem0",
        locality="local",
        requires_llm=True,
    )

    def __init__(self, settings: Settings, client: _Mem0Client | None = None) -> None:
        self.settings = settings
        self._client = client
        self._events: dict[str, dict[str, Any]] = {}

    async def _mem0(self) -> _Mem0Client:
        if self._client is None:
            try:
                from mem0 import AsyncMemory
            except ImportError as exc:
                raise DependencyUnavailable(
                    "mem0ai is required (install [memory-providers])"
                ) from exc
            llm = self.settings.models.llm
            if not llm.enabled:
                raise DependencyUnavailable("mem0 requires models.llm.enabled=true")
            cfg: dict[str, Any] = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm.model,
                        "openai_base_url": llm.base_url,
                        "api_key": llm.api_key.get_secret_value() if llm.api_key else None,
                    },
                },
                "vector_store": {"provider": "qdrant", "config": {"path": ":memory:"}},
            }
            emb = self.settings.models.embedding
            if emb.provider in ("sentence_transformers", "onnx", "openvino", "fastembed"):
                cfg["embedder"] = {
                    "provider": "huggingface",
                    "config": {"model": emb.model_path or emb.model},
                }
            self._client = AsyncMemory.from_config(cfg)  # type: ignore[assignment]
        return self._client  # type: ignore[return-value]

    async def extract(
        self, observation: Observation, ctx: MemoryExecutionContext
    ) -> list[MemoryCandidate]:
        if observation.hints.skip_extraction or not observation.content.strip():
            return []
        client = await self._mem0()
        result = await client.add(
            [{"role": "user", "content": observation.content}],
            user_id=namespace(ctx),
            metadata={
                "tenant_id": ctx.tenant_id,
                "thread_id": ctx.thread_id,
                "observation_id": observation.observation_id,
            },
            infer=True,
        )
        evidence = _evidence(observation)
        out: list[MemoryCandidate] = []
        for item in result.get("results", []):
            text = str(item.get("memory", "")).strip()
            if not text:
                continue
            cand = MemoryCandidate(
                content=text,
                memory_type=MemoryType.SEMANTIC,
                lifetime=Lifetime.LONG_TERM,
                subject=f"user:{ctx.user_id}" if ctx.user_id else ctx.principal_id,
                predicate="mem0",
                evidence=evidence,
                confidence=0.7,
                provider="mem0",
                category=str(item.get("event", "ADD")).lower(),
                provider_ref=str(item.get("id")) if item.get("id") else None,
            )
            self._events[normalize(text)] = {
                "event": item.get("event", "ADD"),
                "id": item.get("id"),
                "previous": item.get("previous_memory"),
            }
            out.append(cand)
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
        event = self._events.pop(normalize(candidate.content), {"event": "ADD"})
        kind = str(event.get("event", "ADD")).upper()
        same = next(
            (m for m in existing if normalize(m.content) == normalize(candidate.content)), None
        )
        if kind == "NONE" or same is not None:
            target = same or next(iter(existing), None)
            if target is not None:
                return ConsolidationOutcome(
                    decision=DedupDecision.REINFORCE,
                    candidate=candidate,
                    target_memory_id=target.memory_id,
                    score=1.0,
                    reason="mem0: NONE",
                )
        if kind in ("UPDATE", "DELETE"):
            prev = normalize(str(event.get("previous") or ""))
            target = next((m for m in existing if normalize(m.content) == prev), None)
            if target is not None:
                return ConsolidationOutcome(
                    decision=DedupDecision.SUPERSEDE,
                    candidate=candidate,
                    target_memory_id=target.memory_id,
                    score=0.9,
                    reason=f"mem0: {kind}",
                )
        return ConsolidationOutcome(
            decision=DedupDecision.CREATE, candidate=candidate, reason=f"mem0: {kind}"
        )

