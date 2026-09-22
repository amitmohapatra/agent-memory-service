"""Generative LLM adapter: the Bifrost gateway, and nothing else.

Bifrost (maximhq/bifrost) is an external OpenAI-compatible gateway that holds the provider
keys, does routing/fallbacks/budgets and exposes ``POST {base_url}/chat/completions``. The
service only knows a base URL, a *virtual key* and model names; no provider SDK is imported
anywhere in this code base (enforced by ``tests/unit/test_architecture.py``).

Bounded by construction: one timeout per call, ``max_retries`` retries with exponential
backoff on transient failures (429/5xx/timeouts/connection errors), and a circuit breaker
that fails fast for ``circuit_open_seconds`` after ``circuit_failure_threshold`` consecutive
failures. All three come from the ``bifrost-sdk`` client, shared with the agent harness —
the breaker was this service's until it turned out to be the same thirty lines there, with
different defaults for the same gateway. What is this service's is the per-use model
routing, the metrics and spans, and the mapping of the client's errors into the domain's.
Every call is traced (model, use, tokens, latency), metered and logged without prompt text
unless ``log_source_text`` is on.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

import httpx
from bifrost_sdk import RETRYABLE, Bifrost, CircuitOpen, GatewayError, RateLimited, Unreachable

from memory_service.config.settings import LLMSettings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.modules.llm.cost import record_llm_tokens
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import llm_requests_total, llm_seconds, llm_tokens_total
from memory_service.observability.tracing import span
from memory_service.ports.models import LLMCompletion, LLMMessage, ProviderInfo

log = get_logger(__name__)

#: The only status for which falling back to a prompt-shaped schema is correct.
_BAD_REQUEST = 400


class LLMCallFailed(DependencyUnavailable):
    """The gateway answered but the call could not be completed (bad status or payload)."""


class LLMOutputInvalid(LLMCallFailed):
    """A structured call returned output that does not satisfy the requested schema."""

    retryable = False


def _text_or_none(data: dict[str, Any]) -> str | None:
    """The response text for a log line. Never raises: logging is not a control path."""
    try:
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content")
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else None


_NO_RESPONSE_FORMAT = re.compile(
    r"response_format|json_schema|structured output|constrained decoding", re.IGNORECASE
)


def _unsupported_response_format(exc: LLMCallFailed) -> bool:
    """Whether the gateway refused the request *because* of the JSON envelope.

    Narrow on purpose: a 400 that is not about the envelope is a real bad request, and
    retrying it without constrained decoding would hide a genuine bug behind a slower path.
    """
    if exc.details.get("status") != _BAD_REQUEST:
        return False
    return bool(_NO_RESPONSE_FORMAT.search(str(exc.details.get("body") or "")))


def _json_only_instruction(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema as a prompt, for models that cannot be constrained by the API."""
    return {
        "role": "user",
        "content": (
            "Reply with a single JSON object and nothing else - no prose, no code fence. "
            f"It must match this JSON Schema:\n{json.dumps(schema)}"
        ),
    }


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
        if not settings.enabled:
            raise ProviderNotConfigured("models.llm.enabled must be true")
        if not settings.model:
            raise ProviderNotConfigured("models.llm.model is required")
        self.settings = settings
        self.model = settings.model
        self.fast_model = settings.fast_model or settings.model
        self.log_source_text = log_source_text
        #: Transport, retries, rate-limit handling and the breaker all live in the shared
        #: client; what stays here is what is *this service's*: per-use model routing,
        #: metrics, spans and the mapping into this service's error vocabulary. The breaker
        #: was the last thing duplicated — the same thirty lines lived in the agent harness,
        #: with different defaults for the same gateway — so the thresholds are passed and
        #: the mechanism is not re-implemented.
        #: Models that answered 400 to the JSON-schema envelope, so later structured calls
        #: put the schema in the prompt from the first attempt instead of paying a wasted
        #: round trip each time (DeepSeek did, on every one of ~600 calls in one run).
        self._schema_in_prompt: set[str] = set()
        self._gateway = Bifrost(
            settings.base_url,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            backoff_seconds=settings.retry_backoff_seconds,
            max_tokens=settings.max_tokens,
            circuit_failure_threshold=settings.circuit_failure_threshold,
            circuit_open_seconds=settings.circuit_open_seconds,
            client=client,
        )

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
        return await self._chat(
            [m.model_dump() for m in messages],
            use=use,
            max_tokens=min(max_tokens, self.settings.max_tokens),
            temperature=temperature,
            source=messages,
        )

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        use: str = "generic",
    ) -> dict[str, Any]:
        turns = [m.model_dump() for m in messages]
        model = self.model_for(use)
        response_format: dict[str, Any] | None = None
        if model not in self._schema_in_prompt:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            }
        else:
            # This model already refused the envelope once; do not pay the 400 again.
            turns = [*turns, _json_only_instruction(schema)]
        last_error: Exception | None = None
        # One bounded repair round: feed the validation error back once. A refused envelope
        # is not an attempt: the fallback asks the same question again with the schema in
        # the prompt and keeps its repair round. It used to consume it, so an invalid first
        # JSON after the fallback raised where the same output from a constrained model
        # would have been repaired.
        repairs_left = 1
        for _ in range(4):  # refusal + answer + repair, with margin; never unbounded
            try:
                # Omit the key entirely on the fallback attempt. Passing
                # ``response_format=None`` still puts ``"response_format": null`` on the
                # wire, which the provider that refused the envelope refuses again.
                envelope: dict[str, Any] = (
                    {"response_format": response_format} if response_format else {}
                )
                completion = await self._chat(
                    turns,
                    use=use,
                    max_tokens=min(max_tokens, self.settings.max_tokens),
                    source=messages,
                    **envelope,
                )
            except LLMCallFailed as exc:
                # Not every model behind the gateway implements constrained decoding.
                # DeepSeek answers `400 "This response_format type is unavailable now"`,
                # and because the envelope is attached to every structured call, one
                # unsupported model turned *all* of them into hard failures — 288 of 304 in
                # one benchmark run. Ask for the JSON in the prompt instead and parse it;
                # the validate-and-repair loop below is unchanged, so the contract the
                # caller relies on is the same either way.
                if response_format is None or not _unsupported_response_format(exc):
                    raise
                log.info("llm.structured_fallback", use=use, reason="response_format_unsupported")
                self._schema_in_prompt.add(model)
                response_format = None
                turns = [*turns, _json_only_instruction(schema)]
                continue
            text = completion.text
            try:
                parsed = _parse_json(text)
                _validate(parsed, schema)
                return parsed
            except LLMOutputInvalid as exc:
                last_error = exc
                if repairs_left == 0:
                    break
                repairs_left -= 1
                turns = [
                    *turns,
                    {"role": "assistant", "content": text[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "That output was invalid: "
                            f"{exc.message}. Return only JSON matching the schema."
                        ),
                    },
                ]
        llm_requests_total.labels(use, "invalid_output").inc()
        raise last_error or LLMOutputInvalid("structured output invalid")

    async def ping(self) -> bool:
        return await self._gateway.ping()

    async def close(self) -> None:
        await self._gateway.aclose()

    # ------------------------------------------------------------------ internals
    async def _chat(
        self,
        turns: list[dict[str, Any]],
        *,
        use: str,
        max_tokens: int,
        temperature: float = 0.0,
        source: Sequence[LLMMessage],
        **extra: Any,
    ) -> LLMCompletion:
        """One gateway call, in this service's error vocabulary.

        Returns the extracted completion rather than the raw payload: both callers want the
        text, and the outcome recorded here depends on whether there is any.
        """
        model = self.model_for(use)
        started = time.perf_counter()
        with span("llm.chat", use=use, model=model) as current:
            try:
                data = await self._gateway.complete(
                    turns,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
            except CircuitOpen as exc:
                # Nothing was sent: the shared client's breaker is open after consecutive
                # failures, so this call fails immediately instead of paying the timeout.
                llm_requests_total.labels(use, "circuit_open").inc()
                log.warning("llm.circuit_open", use=use, retry_after=exc.retry_after)
                raise DependencyUnavailable("llm circuit open", details=exc.details) from exc
            except RateLimited as exc:
                # Backpressure, not brokenness. The retries are already spent by the time
                # this surfaces; the shared breaker deliberately does not count it.
                self._failure(use, "exhausted")
                # Carry how long to wait. The SDK parses it — Gemini puts the delay in the
                # response body rather than a Retry-After header — and stores it on the
                # exception, but `details` is only whatever the gateway literally said, so
                # forwarding that alone dropped it. Nothing above this adapter could tell
                # "come back in 30 seconds" from "this is broken", which is the difference
                # between backpressure and an outage. The key matches the one the circuit
                # breaker already uses above.
                raise LLMCallFailed(
                    "llm gateway rate limited the request",
                    details={
                        **exc.details,
                        **(
                            {"retry_after_seconds": round(exc.retry_after, 1)}
                            if exc.retry_after is not None
                            else {}
                        ),
                    },
                ) from exc
            except Unreachable as exc:
                self._failure(use, "exhausted")
                raise DependencyUnavailable(str(exc)) from exc
            except GatewayError as exc:
                status = exc.details.get("status")
                # A retryable status only reaches here once the client has given up on it;
                # anything else was refused on the first attempt and is named by its status.
                if status is None:
                    outcome = "bad_payload"
                elif status in RETRYABLE:
                    outcome = "exhausted"
                else:
                    outcome = f"http_{status}"
                self._failure(use, outcome)
                raise LLMCallFailed(str(exc), details=exc.details) from exc
            # Extract before recording, because the outcome is not known until the content
            # is. A reasoning model that spends its whole budget thinking answers 200 with
            # an empty string, and counting that ok — which is what recording here used to
            # do — made the error rate say the feature was healthy while every caller saw
            # it fail.
            try:
                completion = self._completion(data)
            except LLMCallFailed:
                # Not a gateway problem and not transient: the budget is too small for this
                # model and will be next time too. The shared client already recorded the
                # call as a success, so this does not reach the breaker — which is right:
                # tripping on it would take out the uses that are working.
                self._failure(use, "empty_output")
                raise
            self._success(use, data, time.perf_counter() - started, current, model, source)
        return completion

    def _success(
        self,
        use: str,
        data: dict[str, Any],
        elapsed: float,
        current: Any,
        model: str,
        messages: Sequence[LLMMessage],
    ) -> None:
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
            "model": data.get("model") or model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": round(elapsed * 1000, 1),
        }
        if self.log_source_text:
            # Read defensively rather than through _completion(): that raises on an
            # exhausted output budget, and raising *here* would mean the failure came out
            # of the success path — after the metrics above, outside every except clause
            # in _chat, so none of the failure accounting ran. Turning logging on to
            # investigate a problem must not change how that problem presents.
            fields["prompt"] = [m.content for m in messages]
            fields["response"] = _text_or_none(data)
        log.info("llm.call", **fields)

    def _failure(self, use: str, outcome: str) -> None:
        """Record a failed call under the outcome that caused it.

        Whether it also opens the circuit is the shared client's decision now, and the rule
        it applies is the one this service learned: a 429 is the gateway healthy and asking
        for less, not the gateway broken. Counting it converted "slow down" into "stop" —
        measured on a real run, 17 rate limits tripped the breaker and the next 62 calls
        failed with "circuit open" having sent nothing at all.
        """
        llm_requests_total.labels(use, outcome).inc()

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
        # Reasoning models spend the output budget on thinking before they emit anything, so
        # a too-small `max_tokens` comes back 200 OK with an empty string and a `length`
        # finish reason. Measured against gemini-3.6-flash: "Reply with exactly: OK" burned 57
        # reasoning tokens, so max_tokens=16 produced no content at all. Returning "" here let
        # that reach every call site as a silently degraded answer. Say what happened instead.
        if not content and choice.get("finish_reason") == "length":
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            raise LLMCallFailed(
                "llm returned no content: the output budget was exhausted before any text "
                "was produced (raise models.llm.max_tokens)",
                details={
                    "finish_reason": "length",
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": reasoning,
                },
            )
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
