"""Provider wiring: one implementation per port, from ``Settings`` and ``constants``.

Every ``if`` here that is not about ``Settings`` is about ``container.overrides``: the
in-process stand-ins a test or benchmark asked for. Production builds a container with no
overrides and takes the first branch of nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memory_service.application.container import Dependency
from memory_service.config import constants
from memory_service.config.constants import FROZEN_MODELS
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.observability.logging import get_logger

if TYPE_CHECKING:
    from memory_service.application.container import Container

log = get_logger(__name__)


async def wire_all(container: Container) -> None:
    settings = container.settings
    log.info(
        "wiring.start",
        environment=settings.service.environment,
        blob=settings.blob.provider,
        llm_enabled=settings.models.llm.enabled,
        stand_ins=container.overrides.summary(),
    )
    await _wire_cache(container)
    await _wire_database(container)
    await _wire_tasks(container)
    _wire_uow(container)
    await _wire_authorization(container)
    _wire_services(container)
    _wire_llm(container)
    _wire_conversation(container)
    await _wire_blob(container)
    _wire_archive(container)
    _wire_ingestion(container)
    await _wire_search(container)
    _wire_models(container)
    _wire_nli(container)
    _wire_retrieval(container)
    _wire_memory(container)
    _wire_tools(container)
    _wire_graph(container)
    _wire_context_preservation(container)
    _register_jobs(container)
    log.info(
        "wiring.done",
        dependencies=sorted(container.dependencies),
        # The parser actually in use, not the one configured: an image built without the
        # docling extra falls back to the builtin parser, and an operator should be able to
        # see that from the startup log rather than from a document that parsed poorly.
        parser=getattr(container.document_parser, "info", None)
        and container.document_parser.info.name,
    )


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


async def _wire_cache(container: Container) -> None:
    stand_in = container.overrides.cache
    if stand_in == "disabled":
        container.cache = None
        return
    if stand_in == "memory":
        from memory_service.adapters.cache.memory_cache import MemoryCache

        container.cache = MemoryCache()
    else:
        from memory_service.adapters.cache.redis_cache import RedisCache

        container.cache = RedisCache(container.settings.cache)
    cache = container.cache
    container.add_dependency(
        Dependency(name="cache", mandatory=False, ping=cache.ping, close=cache.close)
    )


async def _wire_database(container: Container) -> None:
    from memory_service.adapters.db.engine import Database

    db = Database(container.settings.database)
    container.database = db
    container.add_dependency(
        Dependency(name="postgres", mandatory=True, ping=db.ping, close=db.close)
    )


async def _wire_tasks(container: Container) -> None:
    stand_in = container.overrides.tasks
    if stand_in == "inline":
        from memory_service.adapters.tasks.inline_queue import InlineTaskQueue

        container.tasks = InlineTaskQueue()
    elif stand_in == "memory":
        from memory_service.adapters.tasks.inline_queue import RecordingTaskQueue

        container.tasks = RecordingTaskQueue()
    else:
        from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

        queue = ProcrastinateTaskQueue(
            container.settings.database.procrastinate_dsn,
            default_retries=constants.TASKS.default_retries,
            job_timeout_seconds=constants.TASKS.job_timeout_seconds,
        )
        container.tasks = queue
        container.add_dependency(
            Dependency(name="task_queue", mandatory=True, ping=queue.ping, close=queue.close)
        )


def _wire_uow(container: Container) -> None:
    from memory_service.adapters.db.uow import OutboxRelay, SqlUnitOfWorkFactory

    relay = OutboxRelay(container.database.session_factory, container.tasks)
    container.services["outbox_relay"] = relay
    container.services["uow_factory"] = SqlUnitOfWorkFactory(
        container.database.session_factory, relay
    )


async def _wire_authorization(container: Container) -> None:
    if container.overrides.authorization == "memory":
        from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider

        provider = MemoryAuthorizationProvider(
            max_listed_objects=constants.AUTHORIZATION.max_listed_objects
        )
    else:
        from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider

        provider = OpenFGAAuthorizationProvider(container.settings.authorization)
        container.add_dependency(
            Dependency(name="openfga", mandatory=True, ping=provider.ping, close=provider.close)
        )
    container.authorization = provider


def _wire_services(container: Container) -> None:
    from memory_service.modules.auth.authentication import ServiceAuthenticator
    from memory_service.modules.authz.service import AuthorizationService
    from memory_service.modules.idempotency.service import IdempotencyService

    settings = container.settings
    container.services["idempotency"] = IdempotencyService(container.cache)
    container.services["authenticator"] = ServiceAuthenticator(settings.authentication)
    container.services["authz"] = AuthorizationService(
        container.authorization,
        container.cache,
        max_listed_objects=constants.AUTHORIZATION.max_listed_objects,
        cache_ttl_seconds=constants.CACHE.authz_ttl_seconds,
        decision_cache=constants.AUTHORIZATION.decision_cache,
    )


def _wire_conversation(container: Container) -> None:
    from memory_service.modules.conversation.service import ConversationService
    from memory_service.modules.working_memory.hot_thread import HotThreadCache, WorkingMemory

    hot = HotThreadCache(
        container.cache,
        max_messages=constants.CACHE.hot_thread_max_messages,
        ttl_seconds=constants.CACHE.hot_thread_ttl_seconds,
    )
    container.services["hot_thread"] = hot
    container.services["working_memory"] = WorkingMemory(
        container.cache, ttl_seconds=constants.CACHE.working_memory_ttl_seconds
    )
    container.services["conversation"] = ConversationService(
        container.services["authz"], hot, archive_enabled=container.tuning.archive.enabled
    )


def _register_jobs(container: Container) -> None:
    from memory_service.modules.jobs.registry import register_handlers

    register_handlers(container)


async def _wire_blob(container: Container) -> None:
    cfg = container.settings.blob
    if container.overrides.blob == "memory":
        from memory_service.adapters.blob.memory import MemoryBlobStore

        store = MemoryBlobStore()
    elif cfg.provider == "gcs":
        from memory_service.adapters.blob.gcs import GCSBlobStore

        store = GCSBlobStore(cfg)
        container.add_dependency(Dependency(name="blob", mandatory=True, ping=store.ping))
    else:
        from memory_service.adapters.blob.filesystem import FilesystemBlobStore

        store = FilesystemBlobStore(cfg.filesystem_root)
        container.add_dependency(Dependency(name="blob", mandatory=True, ping=store.ping))
    container.blob = store


def _wire_archive(container: Container) -> None:
    from memory_service.modules.archive.service import ArchiveService

    container.services["archive_service"] = ArchiveService(
        container.services["uow_factory"],
        container.blob,
        archive=container.tuning.archive,
        blob_settings=container.settings.blob,
    )


def _wire_ingestion(container: Container) -> None:
    from memory_service.adapters.parsers import register_parsers
    from memory_service.adapters.parsers.builtin import BuiltinParser
    from memory_service.modules.ingestion.service import IngestionService

    cfg = container.tuning.documents
    # The registry is the extension point: a new parser is one entry in adapters/parsers,
    # not a branch here. This used to be a switch statement, which is precisely what
    # config/registry.py says a provider must never require.
    register_parsers(container.registries.document_parser)
    builtin = BuiltinParser()
    wanted = "builtin" if container.overrides.document_parser == "builtin" else cfg.parser
    parser = container.registries.document_parser.create(wanted)
    if parser is None:
        # The chosen parser cannot run here (an image built without the docling extra).
        # Falling back to the builtin is the designed behaviour, not an accident, and
        # /version reports the difference.
        parser = builtin
    container.document_parser = parser
    container.services["ingestion"] = IngestionService(
        container.services["uow_factory"],
        container.services["authz"],
        parser,
        container.blob,
        settings=cfg,
        file_bucket=container.settings.blob.file_bucket,
        tenant_shards=container.tuning.archive.tenant_shards,
        fallback_parser=builtin,
        assist=container.services["llm_assist"],
    )


async def _wire_search(container: Container) -> None:
    from memory_service.adapters.search.qdrant_store import QdrantSearchStore

    stand_in = container.overrides
    local = ":memory:" if stand_in.search == "memory" else stand_in.search_local_path
    store = QdrantSearchStore(container.settings.search, local_path=local)
    container.search = store
    if local is None:
        container.add_dependency(
            Dependency(name="qdrant", mandatory=True, ping=store.ping, close=store.close)
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _wire_models(container: Container) -> None:
    from memory_service.adapters.models.embeddings import (
        HashEmbedding,
        SentenceTransformersEmbedding,
    )
    from memory_service.adapters.models.rerankers import CrossEncoderReranker, LexicalReranker
    from memory_service.adapters.models.sparse import Bm25SparseEncoder

    stand_in = container.overrides
    if stand_in.embedding == "hash":
        container.embedding = HashEmbedding(stand_in.embedding_dimension)
    else:
        container.embedding = SentenceTransformersEmbedding(
            stand_in.dense_model or FROZEN_MODELS.dense,
            threads=container.settings.models.embedding.threads,
        )
    container.sparse = Bm25SparseEncoder()

    reranker: Any = None
    if stand_in.reranker == "lexical":
        # Free to construct, so it is not behind the flag below: a test that turns
        # `retrieval.rerank` on after wiring still has a reranker to exercise.
        reranker = LexicalReranker()
    elif stand_in.reranker == "cross_encoder":
        model = stand_in.reranker_model or FROZEN_MODELS.reranker
        if model is None:
            raise ProviderNotConfigured("reranker=cross_encoder needs a reranker_model")
        reranker = CrossEncoderReranker(model)
    elif stand_in.reranker == "disabled" or not container.tuning.retrieval.rerank:
        # Guarded by its own flag. Without this the cross-encoder was constructed whatever
        # `retrieval.rerank` said — 566 MB of weights loaded into both the API and the
        # worker at startup, reported on /version as an active provider, and never called,
        # because engine.py guards the only call site on `cfg.rerank`. Reranking is off on
        # measured evidence (constants.RetrievalSettings.rerank).
        reranker = None
    elif FROZEN_MODELS.reranker is None:
        log.warning("reranker.no_model", note="retrieval.rerank is on but no reranker is frozen")
    else:
        reranker = CrossEncoderReranker(FROZEN_MODELS.reranker)
    container.reranker = reranker


def _wire_llm(container: Container) -> None:
    """The generative model is optional and reachable only through the Bifrost gateway."""
    from memory_service.adapters.models.llm import BifrostLLM, DisabledLLM
    from memory_service.modules.llm.assist import LLMAssist

    cfg = container.settings.models.llm
    if not cfg.enabled:
        container.llm = DisabledLLM()
        container.services["llm_assist"] = LLMAssist.disabled()
        return
    llm = BifrostLLM(cfg, log_source_text=constants.LOG_SOURCE_TEXT)
    container.llm = llm
    container.services["llm_assist"] = LLMAssist(llm, cfg)
    container.add_dependency(
        Dependency(name="llm", mandatory=False, ping=llm.ping, close=llm.close)
    )


def _wire_nli(container: Container) -> None:
    """Claim-support classifier + grounding cascade. Like the parser, the model degrades to
    the deterministic stand-in with a warning when its weights cannot be loaded; reports
    then say ``representative: false``."""
    from memory_service.adapters.models.nli import LexicalNLI, TransformersNLI
    from memory_service.domain.errors import DependencyUnavailable
    from memory_service.modules.grounding.cascade import GroundingCascade

    stand_in = container.overrides.nli
    if stand_in == "disabled":
        container.nli = None
        return
    nli: Any = LexicalNLI()
    if stand_in != "lexical":
        try:
            nli = TransformersNLI(FROZEN_MODELS.nli)
        except DependencyUnavailable as exc:
            log.warning("nli.unavailable", error=exc.message, fallback="lexical")
    container.nli = nli
    container.services["grounding"] = GroundingCascade(
        nli, settings=container.tuning.nli, assist=container.services["llm_assist"]
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _wire_retrieval(container: Container) -> None:
    from memory_service.modules.context.builder import ContextBuilder
    from memory_service.modules.memory.ephemeral import EphemeralMemory
    from memory_service.modules.rag.indexer import Indexer
    from memory_service.modules.retrieval.engine import RetrievalEngine

    tuning = container.tuning
    dense = container.overrides.dense_model or FROZEN_MODELS.dense
    indexer = Indexer(
        container.services["uow_factory"],
        container.search,
        container.embedding,
        container.sparse,
        container.cache,
        batch_size=dense.batch_size,
        embedding_cache_ttl=constants.CACHE.embedding_ttl_seconds,
        assist=container.services["llm_assist"],
    )
    container.services["indexer"] = indexer
    engine = RetrievalEngine(
        container.services["uow_factory"],
        container.services["authz"],
        container.search,
        indexer,
        container.reranker,
        settings=tuning.retrieval,
        rerank_k=tuning.retrieval.rerank_k,
        assist=container.services["llm_assist"],
    )
    container.services["retrieval"] = engine
    working = EphemeralMemory(
        container.cache, ttl_seconds=constants.CACHE.working_memory_ttl_seconds
    )
    container.services["ephemeral_memory"] = working
    container.services["context_builder"] = ContextBuilder(
        container.services["uow_factory"],
        engine,
        container.services["conversation"],
        container.cache,
        settings=tuning.context,
        retrieval=tuning.retrieval,
        cache_ttl_seconds=constants.CACHE.context_bundle_ttl_seconds,
        working=working,
        assist=container.services["llm_assist"],
    )


def _wire_memory(container: Container) -> None:
    """Memory intelligence: the native provider + observation pipeline + service."""
    from memory_service.modules.memory.forgetting import ForgettingService
    from memory_service.modules.memory.native import NativeMemoryIntelligence
    from memory_service.modules.memory.pipeline import ObservationPipeline
    from memory_service.modules.memory.reflection import ReflectionService
    from memory_service.modules.memory.service import MemoryService
    from memory_service.ports.intelligence import MemoryIntelligenceProvider

    cfg = container.tuning.memory_intelligence
    provider: MemoryIntelligenceProvider = NativeMemoryIntelligence(
        cfg, container.embedding, assist=container.services["llm_assist"]
    )
    container.services["memory_provider"] = provider
    container.services["observation_pipeline"] = ObservationPipeline(
        container.services["uow_factory"],
        provider,
        settings=cfg,
        working=container.services.get("ephemeral_memory"),
    )
    container.services["memory"] = MemoryService(container.services["authz"])
    container.services["forgetting"] = ForgettingService(
        container.services["uow_factory"],
        settings=cfg,
        cache=container.cache,
        working_ttl_seconds=constants.CACHE.working_memory_ttl_seconds,
    )
    container.services["reflection"] = ReflectionService(
        container.services["uow_factory"], assist=container.services["llm_assist"]
    )


def _wire_tools(container: Container) -> None:
    """Tool memory: registry, invocation records, output cache, chains and procedures."""
    from memory_service.modules.tools.service import ToolMemoryService

    container.services["tool_memory"] = ToolMemoryService(
        container.services["uow_factory"],
        container.services["authz"],
        blob=container.blob,
        blob_bucket=container.settings.blob.file_bucket,
        indexer=container.services.get("indexer"),
    )


def _wire_graph(container: Container) -> None:
    """Knowledge graph: store (postgres | memory), native enrichment, service, retrieval stage."""
    from memory_service.modules.graph.native import NativeGraphEnrichment
    from memory_service.modules.graph.retrieval import GraphStage
    from memory_service.modules.graph.service import GraphService

    stand_in = container.overrides
    if stand_in.graph_store == "memory":
        from memory_service.adapters.graph.memory_store import MemoryGraphStore

        container.graph_store = MemoryGraphStore()
    else:
        from memory_service.adapters.graph.postgres_store import PostgresGraphStore

        container.graph_store = PostgresGraphStore(container.database.engine)
    if stand_in.graph_enrichment == "disabled":
        container.graph_enrichment = None
        return
    assist = container.services["llm_assist"]
    provider = NativeGraphEnrichment(assist=assist)
    container.graph_enrichment = provider
    graph = GraphService(
        container.services["uow_factory"],
        container.graph_store,
        provider,
        container.services["authz"],
        settings=container.tuning.graph,
        assist=assist,
    )
    container.services["graph"] = graph
    if container.tuning.retrieval.graph:
        engine = container.services["retrieval"]
        engine.post_stages["graph"] = GraphStage(
            graph,
            container.services["uow_factory"],
            max_facts=container.tuning.context.graph_facts_max,
        )


def _wire_context_preservation(container: Container) -> None:
    """M9: expansion over the Document Context Graph, then evidence-group verification."""
    from memory_service.modules.context.evidence import VerificationStage
    from memory_service.modules.context.expansion import ExpansionStage

    settings = container.tuning.retrieval
    engine = container.services["retrieval"]
    expansion = ExpansionStage(container.services["uow_factory"], settings=settings)
    if settings.parent_expansion or settings.neighbor_expansion or settings.definition_expansion:
        engine.post_stages["expansion"] = expansion
    if settings.evidence_verification:
        engine.post_stages["verify"] = VerificationStage(
            container.services["uow_factory"], expansion, settings=settings
        )
    container.services["expansion"] = expansion
