"""Provider registry: Open/Closed extension point.

Adapters register factories under (port, provider_name). Configuration selects one.
Adding a provider means registering a factory, never editing a switch statement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memory_service.domain.errors import ProviderNotConfigured
from memory_service.ports.models import ProviderInfo

type Factory[T] = Callable[..., T]


class ProviderRegistry[T]:
    def __init__(self, port_name: str):
        self.port_name = port_name
        self._factories: dict[str, Factory[T]] = {}
        self._infos: dict[str, ProviderInfo] = {}

    def register(self, name: str, factory: Factory[T], info: ProviderInfo | None = None) -> None:
        self._factories[name] = factory
        if info is not None:
            self._infos[name] = info

    def names(self) -> list[str]:
        return sorted(self._factories)

    def info(self, name: str) -> ProviderInfo | None:
        return self._infos.get(name)

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ProviderNotConfigured(
                f"{self.port_name} provider {name!r} is not registered; known: {self.names()}"
            ) from exc
        return factory(*args, **kwargs)


class Registries:
    """All registries in one place so the composition root can wire providers."""

    def __init__(self) -> None:
        self.cache: ProviderRegistry[Any] = ProviderRegistry("cache")
        self.search: ProviderRegistry[Any] = ProviderRegistry("search")
        self.blob: ProviderRegistry[Any] = ProviderRegistry("blob")
        self.tasks: ProviderRegistry[Any] = ProviderRegistry("tasks")
        self.authorization: ProviderRegistry[Any] = ProviderRegistry("authorization")
        self.embedding: ProviderRegistry[Any] = ProviderRegistry("embedding")
        self.sparse: ProviderRegistry[Any] = ProviderRegistry("sparse")
        self.reranker: ProviderRegistry[Any] = ProviderRegistry("reranker")
        self.llm: ProviderRegistry[Any] = ProviderRegistry("llm")
        self.memory_intelligence: ProviderRegistry[Any] = ProviderRegistry("memory_intelligence")
        self.graph_store: ProviderRegistry[Any] = ProviderRegistry("graph_store")
        self.graph_enrichment: ProviderRegistry[Any] = ProviderRegistry("graph_enrichment")
        self.document_parser: ProviderRegistry[Any] = ProviderRegistry("document_parser")
        self.retrieval_strategy: ProviderRegistry[Any] = ProviderRegistry("retrieval_strategy")

    def all(self) -> dict[str, ProviderRegistry[Any]]:
        return {k: v for k, v in vars(self).items() if isinstance(v, ProviderRegistry)}


_registries: Registries | None = None


def get_registries() -> Registries:
    global _registries  # noqa: PLW0603 - process-wide registry populated by adapters
    if _registries is None:
        _registries = Registries()
    return _registries
