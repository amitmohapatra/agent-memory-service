"""``auto``: a model chosen from an endpoint's ``/models``, by what the listing says about it.

Discovery never invokes a model and never reads a model's name for meaning: a row is a
candidate when its own metadata says it writes text (``architecture.output_modalities``,
``architecture.modality``, or a ``type``/``mode`` of ``chat``), as OpenRouter, LiteLLM and
vLLM list them. Among candidates the cheapest by the listed completion price is chosen (the
service's uses are bounded extraction, judging and summaries), then the id. A listing with no
metadata at all is usable only when it holds one model (a llama.cpp or vLLM server); anything
else is a clear error asking for an explicit model, never a guess. The provenance policy
(``domain.provenance``) applies to every candidate. The row's profile facts (whether it
reasons, takes a temperature, its output ceiling) shape the request where the listing has
them.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from memory_service.domain.errors import ProviderNotConfigured
from memory_service.domain.provenance import permitted_model
from trellis.contracts import ModelProfile

#: What to set when ``auto`` cannot choose.
EXPLICIT_MODEL = (
    "name the model: the harness agent's model (its registered endpoint carries it), "
    "MEMORY__MODELS__LLM__MODEL for the service's own endpoint, or the tenant's policy "
    "(PUT /v1/model-key/policy)"
)
_TEXT_TYPES = frozenset({"chat", "completion", "text", "llm", "language"})


@dataclass(frozen=True)
class ListedModel:
    """One ``/models`` row, as far as its metadata says anything."""

    id: str
    #: True/False when the row says whether it writes text; None when it says nothing
    text: bool | None = None
    #: the listed price of a completion token; None when not listed
    price: float | None = None
    #: the profile facts the row gives, as ``ModelProfile`` keys (only those it gives)
    profile: Mapping[str, Any] | None = None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _writes_text(row: Mapping[str, Any]) -> bool | None:
    """Whether the row says the model writes text: its output modalities (OpenRouter), its
    ``input->output`` modality, or a ``type``/``mode`` (vLLM, LiteLLM); None when silent."""
    architecture = row.get("architecture")
    if isinstance(architecture, Mapping):
        outputs = architecture.get("output_modalities")
        modality = architecture.get("modality")
        if isinstance(outputs, list):
            return "text" in outputs
        if isinstance(modality, str) and "->" in modality:
            return "text" in modality.split("->", 1)[1]
    kind = row.get("type") or row.get("mode")
    return kind.lower() in _TEXT_TYPES if isinstance(kind, str) else None


def _profile_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ModelProfile`` keys the row gives: what its supported parameters include, and
    its output ceiling."""
    facts: dict[str, Any] = {}
    supported = row.get("supported_parameters")
    if isinstance(supported, list):
        facts["temperature"] = "temperature" in supported
        facts["reasoning_output"] = "reasoning" in supported
    provider = row.get("top_provider")
    ceiling = provider.get("max_completion_tokens") if isinstance(provider, Mapping) else None
    ceiling = ceiling if ceiling is not None else row.get("max_output_tokens")
    if isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0:
        facts["max_output_tokens"] = ceiling
    return facts


def listed(row: Mapping[str, Any]) -> ListedModel | None:
    """A ``/models`` row read for what it states about itself."""
    model_id = row.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    pricing = row.get("pricing")
    price = _number(pricing.get("completion")) if isinstance(pricing, Mapping) else None
    return ListedModel(model_id, _writes_text(row), price, _profile_facts(row) or None)


def choose_model(models: Sequence[ListedModel]) -> ListedModel:
    """The listed model ``auto`` calls, or ``ProviderNotConfigured`` saying what to set."""
    permitted = [m for m in models if permitted_model(m.id)]
    described = [m for m in permitted if m.text is not None]
    if described:
        candidates = [m for m in described if m.text]
    else:
        # Nothing says what any model is: only a one-model listing is unambiguous.
        candidates = permitted if len(permitted) == 1 else []
    if not candidates:
        raise ProviderNotConfigured(
            f"auto found no model it can tell writes text among {len(models)} listed: "
            f"{EXPLICIT_MODEL}"
        )
    if len(candidates) > 1 and all(m.price is None for m in candidates):
        raise ProviderNotConfigured(
            f"auto cannot choose between {len(candidates)} listed text models with no price "
            f"listed: {EXPLICIT_MODEL}"
        )
    return min(candidates, key=lambda m: (m.price is None, m.price or 0.0, m.id))


def profile_of(model: ListedModel, base: ModelProfile) -> ModelProfile:
    """``base`` (the endpoint's profile) with what the listing says about ``model``."""
    return base.model_copy(update=dict(model.profile)) if model.profile else base


class ModelCatalog:
    """Bounded cache and shared discovery per endpoint revision; no keys stored here."""

    def __init__(self) -> None:
        self._entries: OrderedDict[Hashable, tuple[float, ListedModel]] = OrderedDict()
        self._pending: dict[Hashable, asyncio.Task[ListedModel]] = {}

    async def resolve(self, key: Hashable, client: httpx.AsyncClient) -> ListedModel:
        """The model ``auto`` means for the endpoint ``client`` reaches; ``key`` names that
        endpoint's revision, so a rotation discovers again."""
        cached = self._entries.get(key)
        if cached is not None and cached[0] > time.monotonic():
            self._entries.move_to_end(key)
            return cached[1]
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= 32:
                raise ProviderNotConfigured("Model discovery is at its concurrency limit")
            task = asyncio.create_task(self._discover(client))
            self._pending[key] = task
            task.add_done_callback(lambda finished: self._finished(key, finished))
        try:
            chosen = await asyncio.shield(task)
            self._entries[key] = (time.monotonic() + 300, chosen)
            self._entries.move_to_end(key)
            while len(self._entries) > 128:
                self._entries.popitem(last=False)
            return chosen
        finally:
            if task.done() and self._pending.get(key) is task:
                del self._pending[key]

    def _finished(self, key: Hashable, task: asyncio.Task[ListedModel]) -> None:
        if self._pending.get(key) is task:
            del self._pending[key]
        # All waiters may have disconnected. Consume an orphaned failure and release its
        # slot; it must not permanently exhaust discovery for unrelated agents.
        if not task.cancelled():
            task.exception()

    @staticmethod
    async def _discover(client: httpx.AsyncClient) -> ListedModel:
        try:
            response = await client.get("/models", timeout=5)
            response.raise_for_status()
            data = response.json().get("data")
            if not isinstance(data, list) or len(data) > 4096:
                raise ValueError("Invalid model catalogue")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise ProviderNotConfigured("Model discovery (GET /models) failed") from exc
        rows = [listed(row) for row in data if isinstance(row, Mapping)]
        return choose_model([row for row in rows if row is not None])

    async def close(self) -> None:
        tasks = tuple(self._pending.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
        self._entries.clear()
