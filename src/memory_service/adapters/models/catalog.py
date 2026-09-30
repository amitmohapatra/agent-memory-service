"""Credential-scoped discovery of eligible text models from the configured gateway.

Discovery never invokes a model. Only recognized original model families are selected;
an opaque gateway alias is not evidence of model lineage. Explicit model selection remains
available to operators. This preference policy is not a claim of benchmark superiority.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict

import httpx
from bifrost_sdk import Options

from memory_service.domain.errors import ProviderNotConfigured
from memory_service.domain.provenance import permitted_model
from memory_service.ports.credentials import ModelIdentity

_FAMILIES = re.compile(
    r"^(?:openai/gpt-[45][\w.:-]*|anthropic/claude-(?:haiku|sonnet|opus)-[\w.:-]+"
    r"|(?:google|gemini)/gemini-[\d.]+-(?:flash|pro)(?:-[\w.]+)?)$"
)
CatalogKey = tuple[ModelIdentity, int | None] | None


def choose_model(models: tuple[str, ...], *, fast: bool) -> str:
    eligible = []
    for name in models:
        canonical = name.removeprefix("openrouter/")
        if not _FAMILIES.fullmatch(canonical) or not permitted_model(canonical):
            continue
        # By name part, not substring: every "gemini" contains "mini", which made each Gemini
        # model compact and a strong use pick whichever sorted last.
        parts = set(re.split(r"[/_.:-]", canonical))
        compact = bool(parts & {"flash", "haiku", "mini", "nano"})
        # Prefer a compact model for bounded extraction/expansion, and a full model for
        # synthesis. Within a tier use deterministic IDs; do not infer speed from size.
        eligible.append((compact == fast, name))
    if not eligible:
        raise ProviderNotConfigured("The virtual key exposes no recognized eligible text model")
    return max(eligible)[1]


class ModelCatalog:
    """Bounded cache and shared discovery per credential revision; no keys stored here."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self._entries: OrderedDict[CatalogKey, tuple[float, tuple[str, ...]]] = OrderedDict()
        self._pending: dict[CatalogKey, asyncio.Task[tuple[str, ...]]] = {}

    async def resolve(self, key: CatalogKey, options: Options, *, fast: bool) -> str:
        cached = self._entries.get(key)
        if cached is not None and cached[0] > time.monotonic():
            self._entries.move_to_end(key)
            return choose_model(cached[1], fast=fast)
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= 32:
                raise ProviderNotConfigured("Model discovery is at its concurrency limit")
            task = asyncio.create_task(self._discover(options))
            self._pending[key] = task
            task.add_done_callback(lambda finished: self._finished(key, finished))
        try:
            models = await asyncio.shield(task)
            self._entries[key] = (time.monotonic() + 300, models)
            self._entries.move_to_end(key)
            while len(self._entries) > 128:
                self._entries.popitem(last=False)
            return choose_model(models, fast=fast)
        finally:
            if task.done() and self._pending.get(key) is task:
                del self._pending[key]

    def _finished(self, key: CatalogKey, task: asyncio.Task[tuple[str, ...]]) -> None:
        if self._pending.get(key) is task:
            del self._pending[key]
        # All waiters may have disconnected. Consume an orphaned failure and release its
        # slot; it must not permanently exhaust discovery for unrelated agents.
        if not task.cancelled():
            task.exception()

    async def _discover(self, options: Options) -> tuple[str, ...]:
        try:
            response = await self.client.get("/models", headers=options.headers(), timeout=5)
            response.raise_for_status()
            data = response.json().get("data")
            if not isinstance(data, list) or len(data) > 4096:
                raise ValueError("Invalid model catalogue")
            return tuple(
                row["id"]
                for row in data
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            )
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise ProviderNotConfigured("Gateway model discovery failed") from exc

    async def close(self) -> None:
        tasks = tuple(self._pending.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
        self._entries.clear()
