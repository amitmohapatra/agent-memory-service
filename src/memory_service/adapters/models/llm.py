"""Generative LLM adapter: the Bifrost gateway, and nothing else.

Bifrost (maximhq/bifrost) is an external OpenAI-compatible gateway that holds the provider
keys, does routing/fallbacks/budgets and exposes ``POST {base_url}/chat/completions``. The
service only knows a base URL, a *virtual key* and model names; no provider SDK is imported
anywhere in this code base (enforced by ``tests/unit/test_architecture.py``).

Bounded by construction: one timeout per call, ``max_retries`` retries with exponential
backoff on transient failures (429/5xx/timeouts/connection errors), and a circuit breaker
that fails fast for ``circuit_open_seconds`` after ``circuit_failure_threshold`` consecutive
failures. Every call is traced (model, use, tokens, latency), metered and logged without
prompt text unless ``log_source_text`` is on.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

import httpx

from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.modules.llm.cost import record_llm_tokens
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import llm_requests_total, llm_seconds, llm_tokens_total
from memory_service.observability.tracing import span
from memory_service.ports.models import LLMCompletion, LLMMessage, ProviderInfo

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMCallFailed(DependencyUnavailable):
    """The gateway answered but the call could not be completed (bad status or payload)."""


class LLMOutputInvalid(LLMCallFailed):
    """A structured call returned output that does not satisfy the requested schema."""

    retryable = False


class DisabledLLM:
    """The LLM port when ``models.llm.enabled=false``: every call is a ProviderNotConfigured."""

    info = ProviderInfo(
        name="disabled-llm", version="0", license="Apache-2.0", origin="internal", locality="local"
    )
    enabled = False

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use: str = "generic",
    ) -> LLMCompletion:
        raise ProviderNotConfigured("models.llm.enabled=false")

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        use: str = "generic",
    ) -> dict[str, Any]:
        raise ProviderNotConfigured("models.llm.enabled=false")

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        return None


class BifrostLLM:
    info = ProviderInfo(
        name="bifrost",
        version="openai-compatible",
        license="Apache-2.0",
        origin="maximhq/bifrost",
        locality="remote",
        data_residency="gateway",
    )
    enabled = True

    def __init__(
        self,
        settings: LLMSettings,
        *,
        log_source_text: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.provider != "bifrost" or not settings.enabled:
            raise ProviderNotConfigured("models.llm.provider must be 'bifrost' with enabled=true")
        if not settings.model:
            raise ProviderNotConfigured("models.llm.model is required")
        self.settings = settings
        self.model = settings.model
        self.fast_model = settings.fast_model or settings.model
        self.log_source_text = log_source_text
        headers = {"Content-Type": "application/json"}
        if settings.api_key is not None:
            headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(
                settings.timeout_seconds, connect=min(5.0, settings.timeout_seconds)
            ),
        )
        self._owns_client = client is None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # ------------------------------------------------------------------ port
    def model_for(self, use: str) -> str:
        return self.fast_model if use in self.settings.fast_uses else self.model

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use: str = "generic",
    ) -> LLMCompletion:
        body = {
            "model": self.model_for(use),
            "messages": [m.model_dump() for m in messages],
            "max_tokens": min(max_tokens, self.settings.max_tokens),
            "temperature": temperature,
        }
        data = await self._chat(body, use=use, messages=messages)
        return self._completion(data)

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        use: str = "generic",
    ) -> dict[str, Any]:
        body = {
            "model": self.model_for(use),
            "messages": [m.model_dump() for m in messages],
            "max_tokens": min(max_tokens, self.settings.max_tokens),
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            },
        }
        last_error: Exception | None = None
        # one bounded repair round: feed the validation error back once
        for attempt in range(2):
            data = await self._chat(body, use=use, messages=messages)
            text = self._completion(data).text
            try:
                parsed = _parse_json(text)
                _validate(parsed, schema)
                return parsed
            except LLMOutputInvalid as exc:
                last_error = exc
                if attempt == 0:
                    body = {
                        **body,
                        "messages": [
                            *body["messages"],
                            {"role": "assistant", "content": text[:4000]},
                            {
                                "role": "user",
                                "content": (
                                    "That output was invalid: "
                                    f"{exc.message}. Return only JSON matching the schema."
                                ),
                            },
                        ],
                    }
        llm_requests_total.labels(use, "invalid_output").inc()
        raise last_error or LLMOutputInvalid("structured output invalid")

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/models", timeout=3.0)
        except httpx.HTTPError:
            return False
        return resp.status_code < 500

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ internals
    async def _chat(
        self, body: dict[str, Any], *, use: str, messages: Sequence[LLMMessage]
    ) -> dict[str, Any]:
        now = time.monotonic()
        if now < self._circuit_open_until:
            llm_requests_total.labels(use, "circuit_open").inc()
            raise DependencyUnavailable(
                "llm circuit open",
                details={"retry_after_seconds": round(self._circuit_open_until - now, 1)},
            )
        attempts = self.settings.max_retries + 1
        started = time.perf_counter()
        with span("llm.chat", use=use, model=body["model"]) as current:
            for attempt in range(attempts):
                try:
                    resp = await self._client.post("/chat/completions", json=body)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    err: Exception = DependencyUnavailable(
                        f"llm gateway unreachable ({type(exc).__name__})"
                    )
                    retry = True
                else:
                    if resp.status_code < 400:
                        try:
                            data = resp.json()
                        except ValueError as exc:
                            self._failure(use, "bad_payload")
                            raise LLMCallFailed("llm gateway returned non-JSON body") from exc
                        elapsed = time.perf_counter() - started
                        self._success(use, data, elapsed, current, body, messages)
                        return data
                    retry = resp.status_code in _RETRYABLE_STATUS
                    err = LLMCallFailed(
                        f"llm gateway returned {resp.status_code}",
                        details={"status": resp.status_code, "body": resp.text[:300]},
                    )
                    if not retry:
                        self._failure(use, f"http_{resp.status_code}")
                        raise err
                if attempt + 1 < attempts and retry:
                    await asyncio.sleep(self.settings.retry_backoff_seconds * (2**attempt))
                    continue
                self._failure(use, "exhausted")
                raise err
        raise LLMCallFailed("unreachable")  # pragma: no cover

    def _success(
        self,
        use: str,
        data: dict[str, Any],
        elapsed: float,
        current: Any,
        body: dict[str, Any],
        messages: Sequence[LLMMessage],
    ) -> None:
        self._consecutive_failures = 0
        usage = data.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        llm_requests_total.labels(use, "ok").inc()
        llm_seconds.labels(use).observe(elapsed)
        if in_tok:
            llm_tokens_total.labels(use, "input").inc(int(in_tok))
        if out_tok:
            llm_tokens_total.labels(use, "output").inc(int(out_tok))
        record_llm_tokens(in_tok, out_tok)
        current.set_attribute("llm.input_tokens", int(in_tok or 0))
        current.set_attribute("llm.output_tokens", int(out_tok or 0))
        current.set_attribute("llm.latency_ms", round(elapsed * 1000, 1))
        fields: dict[str, Any] = {
            "use": use,
            "model": data.get("model") or body["model"],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": round(elapsed * 1000, 1),
        }
        if self.log_source_text:
            fields["prompt"] = [m.content for m in messages]
            fields["response"] = self._completion(data).text
        log.info("llm.call", **fields)

    def _failure(self, use: str, outcome: str) -> None:
        self._consecutive_failures += 1
        llm_requests_total.labels(use, outcome).inc()
        if self._consecutive_failures >= self.settings.circuit_failure_threshold:
            self._circuit_open_until = time.monotonic() + self.settings.circuit_open_seconds
            log.warning(
                "llm.circuit_open",
                failures=self._consecutive_failures,
                open_seconds=self.settings.circuit_open_seconds,
            )

    @staticmethod
    def _completion(data: dict[str, Any]) -> LLMCompletion:
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):  # some gateways return content parts
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMCallFailed("llm gateway returned no choices") from exc
        usage = data.get("usage") or {}
        return LLMCompletion(
            text=str(content or ""),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model=data.get("model"),
        )


# ---------------------------------------------------------------------- JSON helpers


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        parsed = json.loads(stripped)
    except ValueError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise LLMOutputInvalid("output is not JSON") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except ValueError as exc:
            raise LLMOutputInvalid("output is not JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMOutputInvalid("output is not a JSON object")
    return parsed


_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Minimal JSON-schema check (type, required, properties, items, enum) — enough to reject
    malformed model output before it reaches domain code; no external validator needed."""
    typ = schema.get("type")
    if typ:
        expected = _TYPES.get(typ) if isinstance(typ, str) else None
        if expected is not None and not isinstance(value, expected):
            raise LLMOutputInvalid(f"{path}: expected {typ}")
        if typ == "integer" and isinstance(value, bool):
            raise LLMOutputInvalid(f"{path}: expected integer")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMOutputInvalid(f"{path}: not in enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise LLMOutputInvalid(f"{path}: missing {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]")
