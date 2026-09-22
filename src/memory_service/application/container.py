"""Composition root.

The container wires configured providers to ports. It is the *only* place that knows
which adapter backs which port. Application services receive ports, never adapters.

Providers are attached milestone by milestone; a ``None`` port means the capability is
not configured, and readiness distinguishes mandatory from optional dependencies.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from memory_service.config.registry import Registries, get_registries
from memory_service.config.settings import Settings
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

Pinger = Callable[[], Awaitable[bool]]


@dataclass
class Dependency:
    name: str
    mandatory: bool
    ping: Pinger
    close: Callable[[], Awaitable[None]] | None = None


@dataclass
class Container:
    settings: Settings
    version: str
    registries: Registries = field(default_factory=get_registries)
    dependencies: dict[str, Dependency] = field(default_factory=dict)

    # ports (populated by milestones; typed as Any to avoid import cycles in M0)
    cache: Any = None
    search: Any = None
    blob: Any = None
    tasks: Any = None
    authorization: Any = None
    embedding: Any = None
    sparse: Any = None
    reranker: Any = None
    nli: Any = None
    llm: Any = None
    memory_intelligence: Any = None
    graph_store: Any = None
    graph_enrichment: Any = None
    document_parser: Any = None
    database: Any = None
    services: dict[str, Any] = field(default_factory=dict)

    def add_dependency(self, dep: Dependency) -> None:
        self.dependencies[dep.name] = dep

    async def readiness(self) -> dict[str, dict[str, Any]]:
        """Ping every dependency. Optional dependencies never fail readiness when disabled."""
        from memory_service.observability.metrics import dependency_up

        results: dict[str, dict[str, Any]] = {}
        for name, dep in self.dependencies.items():
            try:
                ok = bool(await dep.ping())
            except Exception as exc:
                ok = False
                results[name] = {
                    "ok": False,
                    "mandatory": dep.mandatory,
                    "error": type(exc).__name__,
                }
            else:
                results[name] = {"ok": ok, "mandatory": dep.mandatory}
            dependency_up.labels(name).set(1 if ok else 0)
        return results

    async def close(self) -> None:
        for name, dep in reversed(list(self.dependencies.items())):
            if dep.close is None:
                continue
            try:
                await dep.close()
            except Exception:
                log.warning("dependency.close_failed", dependency=name)


async def build_container(settings: Settings, version: str) -> Container:
    """Wire providers for the configured environment. Extended in later milestones."""
    from memory_service.adapters import wire_adapters

    container = Container(settings=settings, version=version)
    await wire_adapters(container)
    return container
