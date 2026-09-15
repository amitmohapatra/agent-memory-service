"""HTTP transport: headers, retries for retryable errors, error envelope mapping."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from universal_memory.errors import MemoryError, error_from_envelope
from universal_memory.models import Scope

HEADER_TENANT = "X-Memory-Tenant"
HEADER_WORKSPACE = "X-Memory-Workspace"
HEADER_USER = "X-Memory-User"
HEADER_GROUPS = "X-Memory-Groups"
HEADER_API_KEY = "X-API-Key"
HEADER_IDEMPOTENCY = "Idempotency-Key"
HEADER_TRACE = "X-Trace-ID"
HEADER_CORRELATION = "X-Correlation-ID"


class Transport:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "universal-memory-python",
    ) -> None:
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        if api_key:
            headers[HEADER_API_KEY] = api_key
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
            )
            owns = True
        else:
            client.headers.update(headers)
            owns = False
        self._client = client
        self._owns_client = owns
        self.max_retries = max_retries

    @staticmethod
    def scope_headers(scope: Scope) -> dict[str, str]:
        h = {HEADER_TENANT: scope.tenant_id}
        if scope.workspace_id:
            h[HEADER_WORKSPACE] = scope.workspace_id
        if scope.user_id:
            h[HEADER_USER] = scope.user_id
        if scope.group_ids:
            h[HEADER_GROUPS] = ",".join(scope.group_ids)
        if scope.trace_id:
            h[HEADER_TRACE] = scope.trace_id
        if scope.correlation_id:
            h[HEADER_CORRELATION] = scope.correlation_id
        return h

    @staticmethod
    def scope_params(scope: Scope) -> dict[str, str]:
        """Lineage for body-less (GET/DELETE) routes; security fields stay in headers."""
        fields = (
            "thread_id",
            "work_id",
            "task_id",
            "agent_id",
            "agent_group_id",
            "agent_run_id",
            "parent_agent_run_id",
        )
        return {f: v for f in fields if (v := getattr(scope, f, None))}

    async def request(
        self,
        method: str,
        path: str,
        *,
        scope: Scope | None = None,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        files: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        headers: dict[str, str] = {}
        if scope is not None:
            headers.update(self.scope_headers(scope))
            if json is None and method.upper() in ("GET", "DELETE"):
                params = {**self.scope_params(scope), **(params or {})}
        if idempotency_key:
            headers[HEADER_IDEMPOTENCY] = idempotency_key
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._client.request(
                    method, path, json=json, params=params, headers=headers, files=files, data=data
                )
            except httpx.TransportError as exc:
                if attempt > self.max_retries:
                    raise MemoryError(
                        str(exc), code="DEPENDENCY_UNAVAILABLE", status=0, retryable=True
                    ) from exc
                await asyncio.sleep(_backoff(attempt))
                continue
            if response.status_code < 400:
                if response.status_code == 204 or not response.content:
                    return None
                return response.json()
            body = _safe_json(response)
            err = error_from_envelope(response.status_code, body)
            safe_to_retry = idempotency_key is not None or method.upper() == "GET"
            if err.retryable and safe_to_retry and attempt <= self.max_retries:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise err

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _backoff(attempt: int) -> float:
    return min(2.0, 0.1 * (2 ** (attempt - 1))) + random.uniform(0, 0.05)  # noqa: S311


def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
    try:
        return response.json()
    except ValueError:
        return None
