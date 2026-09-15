"""``LLMAssist``: the one way modules consult the generative model.

Every native path stays deterministic and complete on its own. A module asks
``assist.wants("<use>")`` and, when true, calls ``assist.structured(...)``; any failure
(disabled provider, gateway error, invalid output, timeout) returns ``None`` and the module
continues with its native result. The model can refine a decision, never replace a path.
"""

from __future__ import annotations

import time
from typing import Any

from memory_service.config.settings import LLMSettings, LLMUse
from memory_service.modules.llm.cost import llm_tokens_used
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import llm_assist_total
from memory_service.ports.models import LLMMessage, LLMProvider

log = get_logger(__name__)


class LLMAssist:
    def __init__(self, provider: LLMProvider | None, settings: LLMSettings) -> None:
        self.provider = provider
        self.settings = settings

    @classmethod
    def disabled(cls) -> LLMAssist:
        return cls(None, LLMSettings())

    def wants(self, use: LLMUse) -> bool:
        return self.provider is not None and self.provider.enabled and self.settings.wants(use)

    @staticmethod
    def tokens_used() -> int:
        """LLM tokens (input + output) consumed so far in the current request/job."""
        return llm_tokens_used()

    async def structured(
        self,
        use: LLMUse,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 512,
    ) -> dict[str, Any] | None:
        """Schema-validated JSON from the model, or ``None`` when the model cannot help."""
        if not self.wants(use) or self.provider is None:
            return None
        started = time.perf_counter()
        try:
            out = await self.provider.structured(
                [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                schema=schema,
                max_tokens=max_tokens,
                use=use,
            )
        except Exception as exc:
            llm_assist_total.labels(use, "fallback").inc()
            log.warning(
                "llm.assist.fallback",
                use=use,
                error=type(exc).__name__,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return None
        llm_assist_total.labels(use, "used").inc()
        return out

    async def complete(
        self, use: LLMUse, *, system: str, user: str, max_tokens: int = 512
    ) -> str | None:
        if not self.wants(use) or self.provider is None:
            return None
        try:
            out = await self.provider.complete(
                [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                max_tokens=max_tokens,
                use=use,
            )
        except Exception as exc:
            llm_assist_total.labels(use, "fallback").inc()
            log.warning("llm.assist.fallback", use=use, error=type(exc).__name__)
            return None
        llm_assist_total.labels(use, "used").inc()
        return out.text.strip() or None
