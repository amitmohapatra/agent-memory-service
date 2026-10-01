"""``LLMAssist``: the one way modules consult the generative model.

Every native path stays deterministic and complete on its own. A request or job first binds
the identity that owns the work (``bound`` / ``reading``); a module then asks
``assist.wants("<use>")`` and, when true, calls ``assist.structured(...)``. Any failure
(disabled provider, gateway error, invalid output, timeout) returns ``None`` and the module
decides whether to keep native evidence or fail an explicitly assisted operation.

``wants(use)`` is the bound tenant's policy, and true only when the gateway is configured and
something can pay: the acting agent's or the tenant's registered key, or the operator.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from memory_service.config.constants import LLM
from memory_service.config.settings import LLMSettings, LLMUse
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.domain.ids import stable_key
from memory_service.modules.llm.cost import llm_tokens_used
from memory_service.modules.llm.policies import ModelPolicies
from memory_service.modules.llm.policy import (
    DEFAULT_ACCESS,
    ModelAccess,
    ModelBinding,
    bind,
    current_binding,
    identity_of,
    model_call_policy,
    model_calls_allowed,
    model_for,
)
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import llm_assist_total
from memory_service.ports.credentials import ModelIdentity
from memory_service.ports.models import LLMMessage, LLMProvider

log = get_logger(__name__)

#: Appended to every system prompt: the service stores text in any language and a model
#: must never be the step that translates it. Labels a schema fixes (query types, snake_case
#: predicates, field names) are not text and stay as specified.
SOURCE_LANGUAGE_RULE = (
    "The source may be in any language. Write every text you return in the language of "
    "the source it comes from, never translated; enumerated labels and field names stay "
    "exactly as specified."
)


def with_language_rule(system: str) -> str:
    return f"{system}\n\n{SOURCE_LANGUAGE_RULE}"


class LLMAssist:
    def __init__(
        self,
        provider: LLMProvider | None,
        settings: LLMSettings,
        policies: ModelPolicies | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.policies = policies

    @classmethod
    def disabled(cls) -> LLMAssist:
        return cls(None, LLMSettings())

    @property
    def available(self) -> bool:
        """The deployment can reach a model at all; otherwise nothing is resolved."""
        return self.provider is not None and self.provider.enabled

    @asynccontextmanager
    async def bound(self, identity: ModelIdentity) -> AsyncIterator[ModelAccess]:
        """Bind the work that follows to ``identity``: its key pays and its policy decides.
        Resolved once (no lookup at all when no model is reachable)."""
        access = (
            await self.policies.access(identity)
            if self.available and self.policies is not None
            else DEFAULT_ACCESS
        )
        with bind(ModelBinding(identity, access)):
            yield access

    @asynccontextmanager
    async def reading(self, ctx: MemoryExecutionContext) -> AsyncIterator[None]:
        """A read: bound to the caller, and model-assisted when the resolved policy's
        ``read_assist`` says so (a request does not decide it)."""
        async with self.bound(identity_of(ctx)) as access:
            with model_call_policy(access.read_assist):
                yield

    async def payable_tenants(self, use: LLMUse, only: str | None = None) -> list[str | None]:
        """The tenants background ``use`` may run for. ``[None]`` means every tenant (the
        operator pays) and ``[only]`` that one; otherwise the tenants holding a live key at
        some level, so a job never scans the memories of tenants nothing can pay for. Empty
        when the deployment does not allow ``use`` at all; each tenant's policy is still
        decided per identity (``bound``)."""
        if not (self.available and self.settings.wants(use)):
            return []
        if self.settings.operator_pays:
            return [only]
        if self.policies is None:
            return []
        keyed = await self.policies.tenants_with_keys()
        return [tenant for tenant in keyed if only is None or tenant == only]

    def wants(self, use: LLMUse) -> bool:
        if not (model_calls_allowed() and self.available and self.settings.wants(use)):
            return False
        binding = current_binding()
        if binding is None:
            return self.settings.operator_pays
        return use in binding.access.uses and (
            binding.access.has_key or self.settings.operator_pays
        )

    def cache_fingerprint(self, uses: Sequence[LLMUse]) -> str:
        """Bind cached assisted output to its gateway/model policy without exposing keys."""
        active = sorted(use for use in uses if self.wants(use))
        if not active:
            return ""
        settings = self.settings
        tuning = getattr(self.provider, "tuning", LLM)
        return stable_key(
            "bifrost-output-v2",
            (settings.base_url or "").rstrip("/"),
            ",".join(f"{use}={model_for(use, tuning)}" for use in active),
            str(tuning.max_tokens),
            settings.api_key.get_secret_value() if settings.api_key else "",
        )

    async def model(self, use: LLMUse) -> str:
        """The model ``use`` calls under the current binding: the policy's or the service's,
        with ``auto`` resolved through the gateway (still ``auto`` if it cannot be)."""
        configured = model_for(use)
        resolve = getattr(self.provider, "resolve_model", None)
        if configured != "auto" or resolve is None:
            return configured
        try:
            return await resolve(use)
        except Exception:
            return configured

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
                [
                    LLMMessage(role="system", content=with_language_rule(system)),
                    LLMMessage(role="user", content=user),
                ],
                schema=schema,
                max_tokens=max_tokens,
                use=use,
            )
        except ProviderNotConfigured:
            return None  # No credential is a normal model-free path in automatic mode.
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
                [
                    LLMMessage(role="system", content=with_language_rule(system)),
                    LLMMessage(role="user", content=user),
                ],
                max_tokens=max_tokens,
                use=use,
            )
        except ProviderNotConfigured:
            return None
        except Exception as exc:
            llm_assist_total.labels(use, "fallback").inc()
            log.warning("llm.assist.fallback", use=use, error=type(exc).__name__)
            return None
        llm_assist_total.labels(use, "used").inc()
        return out.text.strip() or None
