"""Composition root.

The container wires configured providers to ports. It is the *only* place that knows
which adapter backs which port. Application services receive ports, never adapters.

Providers are attached milestone by milestone; a ``None`` port means the capability is
not configured, and readiness distinguishes mandatory from optional dependencies.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from memory_service.config import constants
from memory_service.config.constants import (
    ArchiveSettings,
    ContextSettings,
    CrossEncoderModel,
    DenseModel,
    DocumentSettings,
    GraphSettings,
    MemoryIntelligenceSettings,
    NLISettings,
    RetrievalSettings,
)
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


@dataclass(frozen=True)
class Overrides:
    """In-process stand-ins for the backing stores and models, and replacements for the
    frozen tuning, for tests and benchmarks.

    None of these is reachable from the environment. They used to be provider values in
    ``Settings`` (``cache.provider=memory``, ``models.embedding.provider=hash``,
    ``tasks.provider=inline``), which made the test suite's stand-ins part of the operator's
    configuration surface: a deployment could be pointed at an in-memory queue by a typo in
    an env file. The shipped service has exactly one implementation per port; a test that
    needs something else says so here, in code, when it builds its container.

    ``None`` means "the real adapter, from ``Settings``" or "the constant from
    ``config/constants.py``".
    """

    #: ``memory``: a dict-backed cache. ``disabled``: no cache at all, which the service
    #: must degrade under (every read falls through to the canonical store).
    cache: Literal["memory", "disabled"] | None = None
    #: ``memory``: qdrant-client local mode (``:memory:``) in this process. Exact brute-force
    #: scan, no HNSW: fine for a fixture-sized corpus, quietly O(n) beyond that.
    search: Literal["memory"] | None = None
    #: A directory for qdrant-client local mode when the vectors must outlive the process.
    search_local_path: str | None = None
    #: ``inline``: run each job as soon as the outbox relay dispatches it. ``memory``: record
    #: jobs and run them on ``drain()``.
    tasks: Literal["inline", "memory"] | None = None
    #: an in-process authorization model instead of OpenFGA
    authorization: Literal["memory"] | None = None
    #: a dict-backed blob store instead of the filesystem / GCS
    blob: Literal["memory"] | None = None
    #: the in-memory knowledge-graph store instead of PostgreSQL
    graph_store: Literal["memory"] | None = None
    #: ``hash``: a deterministic feature-hashed embedding - a labelled *non-representative*
    #: stand-in that loads no weights
    embedding: Literal["hash"] | None = None
    embedding_dimension: int = 64
    #: a specific dense encoder in place of the frozen one (benchmark challengers only)
    dense_model: DenseModel | None = None
    #: ``lexical``: BM25-style overlap; ``cross_encoder``: the given model, loaded whatever
    #: ``retrieval.rerank`` says (the benchmark that measures it); ``disabled``: none
    reranker: Literal["lexical", "cross_encoder", "disabled"] | None = None
    reranker_model: CrossEncoderModel | None = None
    #: ``lexical``: token coverage mapped onto NLI scores (never representative)
    nli: Literal["lexical", "disabled"] | None = None
    #: the text parser instead of docling
    document_parser: Literal["builtin"] | None = None
    #: no graph enrichment at all (memories must still land without it)
    graph_enrichment: Literal["disabled"] | None = None
    # ---- frozen tuning replaced for one container --------------------------------------
    retrieval: RetrievalSettings | None = None
    context: ContextSettings | None = None
    memory_intelligence: MemoryIntelligenceSettings | None = None
    documents: DocumentSettings | None = None
    graph: GraphSettings | None = None
    archive: ArchiveSettings | None = None
    nli_settings: NLISettings | None = None

    def summary(self) -> dict[str, str]:
        """The stand-ins in force, for the startup log and /version."""
        pairs = (
            ("cache", self.cache),
            ("search", self.search or self.search_local_path),
            ("tasks", self.tasks),
            ("authorization", self.authorization),
            ("blob", self.blob),
            ("graph_store", self.graph_store),
            ("embedding", self.embedding or (self.dense_model and self.dense_model.id)),
            ("reranker", self.reranker),
            ("nli", self.nli),
            ("document_parser", self.document_parser),
            ("graph_enrichment", self.graph_enrichment),
        )
        return {name: str(value) for name, value in pairs if value}


@dataclass(frozen=True)
class Tuning:
    """The stage tuning a container runs with: the constants, unless an override replaced
    one. Read this, never ``constants`` directly, wherever a test may want a different value."""

    retrieval: RetrievalSettings
    context: ContextSettings
    memory_intelligence: MemoryIntelligenceSettings
    documents: DocumentSettings
    graph: GraphSettings
    archive: ArchiveSettings
    nli: NLISettings

    @classmethod
    def resolve(cls, overrides: Overrides) -> Tuning:
        return cls(
            retrieval=overrides.retrieval or constants.RETRIEVAL,
            context=overrides.context or constants.CONTEXT,
            memory_intelligence=overrides.memory_intelligence or constants.MEMORY_INTELLIGENCE,
            documents=overrides.documents or constants.DOCUMENTS,
            graph=overrides.graph or constants.GRAPH,
            archive=overrides.archive or constants.ARCHIVE,
            nli=overrides.nli_settings or constants.NLI,
        )


@dataclass
class Container:
    settings: Settings
    version: str
    overrides: Overrides = field(default_factory=Overrides)
    tuning: Tuning = field(default_factory=lambda: Tuning.resolve(Overrides()))
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
        self._close_models()

    def _close_models(self) -> None:
        """In-process models are not dependencies — there is nothing to ping — but each one
        owns a thread. The API builds one container per process and would not notice; the
        benchmarks build one per candidate, and the threads would accumulate across the run.
        """
        for name in ("embedding", "reranker", "nli"):
            model = getattr(self, name, None)
            closer = getattr(model, "close", None)
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                log.warning("model.close_failed", model=name)


async def build_container(
    settings: Settings, version: str, *, overrides: Overrides | None = None
) -> Container:
    """Wire providers for the configured environment.

    ``overrides`` swaps backing stores and models for in-process stand-ins and replaces frozen
    tuning; production never passes it.
    """
    from memory_service.adapters import wire_adapters

    overrides = overrides or Overrides()
    container = Container(
        settings=settings,
        version=version,
        overrides=overrides,
        tuning=Tuning.resolve(overrides),
    )
    await wire_adapters(container)
    return container
