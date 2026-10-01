"""Provider wiring: one implementation per port, from ``Settings`` and ``constants``.

Every ``if`` here that is not about ``Settings`` is about ``container.overrides``: the
in-process stand-ins a test or benchmark asked for. Production builds a container with no
overrides and takes the first branch of nothing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from memory_service.application.container import Dependency
from memory_service.config import constants
from memory_service.config.constants import FROZEN_MODELS
from memory_service.domain.errors import DependencyUnavailable
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
        llm_enabled=settings.llm.enabled,
        stand_ins=container.overrides.summary(),
    )
    await _wire_cache(container)
    await _wire_database(container)
    await _wire_tasks(container)
    _wire_uow(container)
    await _wire_authorization(container)
    _wire_services(container)
    await _prime_tenant_registry(container)
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
    _wire_graph(container)
    _wire_tools(container)
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


async def _prime_tenant_registry(container: Container) -> None:
    """Load quotas and suspensions before the first request; the loop keeps them current."""
    try:
        await container.services["tenant_registry"].refresh()
    except Exception as exc:  # the store may still be migrating; the loop retries
        log.warning("tenant_registry.prime_failed", error=str(exc))


def _wire_services(container: Container) -> None:
    from memory_service.modules.audit.service import ReadAudit
    from memory_service.modules.auth.authentication import ServiceAuthenticator
    from memory_service.modules.auth.keys import ApiKeyVerifier
    from memory_service.modules.authz.service import AuthorizationService
    from memory_service.modules.idempotency.service import IdempotencyService
    from memory_service.modules.tenancy.registry import TenantRegistry
    from memory_service.modules.tenancy.retention import RetentionService
    from memory_service.modules.tenancy.service import TenancyService

    settings = container.settings
    uow_factory = container.services["uow_factory"]
    container.services["idempotency"] = IdempotencyService(container.cache)
    registry = TenantRegistry(uow_factory)
    registry.start()
    container.services["tenant_registry"] = registry
    container.add_closer("tenant_registry", registry.close)
    keys = ApiKeyVerifier(uow_factory, container.cache, known=registry.knows_key)
    container.services["api_keys"] = keys
    container.services["authenticator"] = ServiceAuthenticator(settings.authentication, keys=keys)
    authz = AuthorizationService(
        container.authorization,
        container.cache,
        max_listed_objects=constants.AUTHORIZATION.max_listed_objects,
        cache_ttl_seconds=constants.CACHE.authz_ttl_seconds,
        decision_cache=constants.AUTHORIZATION.decision_cache,
    )
    container.services["authz"] = authz
    container.services["tenancy"] = TenancyService(authz)
    container.services["retention"] = RetentionService(uow_factory)
    audit = ReadAudit(uow_factory)
    audit.start()
    container.services["read_audit"] = audit
    container.add_closer("read_audit", audit.close)


def _wire_conversation(container: Container) -> None:
    from memory_service.modules.conversation.service import ConversationService
    from memory_service.modules.conversation.summary import ThreadSummaries
    from memory_service.modules.profile.service import ProfileService
    from memory_service.modules.working_memory.hot_thread import HotThreadCache

    hot = HotThreadCache(
        container.cache,
        max_messages=constants.CACHE.hot_thread_max_messages,
        ttl_seconds=constants.CACHE.hot_thread_ttl_seconds,
    )
    container.services["hot_thread"] = hot
    container.services["conversation"] = ConversationService(
        container.services["authz"], hot, archive_enabled=container.tuning.archive.enabled
    )
    container.services["thread_summaries"] = ThreadSummaries(
        container.services["uow_factory"], container.services["llm_assist"]
    )
    container.services["profile"] = ProfileService(
        container.services["uow_factory"],
        container.services["authz"],
        container.services["llm_assist"],
        reader=lambda: container.services["context_builder"],
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


def _model_threads(container: Container) -> int:
    """Intra-op threads every in-process model shares: this worker's share of the cores.

    ``torch.set_num_threads`` is process-wide, so the last model to load decides it for all of
    them; one resolved number, given to each, is the only way it means anything. The measured
    8-vCPU box with three workers resolves to the 2 the models were frozen with.
    """
    return max(1, (os.cpu_count() or 1) // container.settings.service.workers)


def _wire_models(container: Container) -> None:
    from memory_service.adapters.models.embeddings import HashEmbedding, load_dense
    from memory_service.adapters.models.sparse import Bm25SparseEncoder
    from memory_service.domain.script import Script
    from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
    from memory_service.ports.search import VectorName

    stand_in = container.overrides
    if stand_in.embedding == "hash":
        # the stand-in stands in for the space every query searches
        spaces = DenseSpaces.single(HashEmbedding(stand_in.embedding_dimension))
    else:
        dense = stand_in.dense_model or FROZEN_MODELS.dense
        threads = _model_threads(container)
        english = load_dense(dense, threads=threads)
        if stand_in.multilingual_dense == "disabled":
            # the single-encoder arm: the English specialist answers every script
            spaces = DenseSpaces.single(english, VectorName.DENSE_EN)
        else:
            spaces = DenseSpaces(
                [
                    DenseSpace(
                        VectorName.DENSE_EN, english, query_scripts=frozenset({Script.LATIN})
                    ),
                    DenseSpace(
                        VectorName.DENSE_ML, load_dense(FROZEN_MODELS.dense_ml, threads=threads)
                    ),
                ]
            )
    container.dense_spaces = spaces
    container.embedding = spaces.primary
    container.sparse = Bm25SparseEncoder()
    _wire_late_and_rerankers(container)


def _wire_late_and_rerankers(container: Container) -> None:
    """The late-interaction arm and the memories' two cross-encoders (ADR 0025). The real
    graphs must load; the stand-ins are chosen only by an override, and the hash embedding
    implies them, so a hermetic container runs the same path with nothing to load."""
    from memory_service.adapters.models.late_interaction import HashLateInteraction, OnnxColbert
    from memory_service.adapters.models.reranker import LexicalReranker, OnnxReranker

    stand_in = container.overrides
    hashed = stand_in.embedding == "hash"
    threads = _model_threads(container)
    late = stand_in.late_interaction or ("hash" if hashed else None)
    if late == "hash":
        container.late = HashLateInteraction()
    elif late is None:
        container.late = OnnxColbert(FROZEN_MODELS.colbert, threads=threads)
    rerankers = stand_in.rerankers or ("lexical" if hashed else None)
    if rerankers == "lexical":
        container.rerankers = tuple(
            LexicalReranker(spec.local_dir) for spec in FROZEN_MODELS.rerankers
        )
    elif rerankers is None:
        container.rerankers = tuple(
            OnnxReranker(spec, threads=threads) for spec in FROZEN_MODELS.rerankers
        )


def _wire_llm(container: Container) -> None:
    """The generative model is optional and reachable only through the Bifrost gateway.

    Model keys, per-level policies and the daily usage ledger are wired whatever the
    configuration, so a tenant can register a key or a policy before the operator turns the
    model on."""
    from memory_service.adapters.models.credential_cipher import AesCredentialCipher
    from memory_service.adapters.models.llm import BifrostLLM, DisabledLLM
    from memory_service.modules.llm.assist import LLMAssist
    from memory_service.modules.llm.credentials import ModelCredentials
    from memory_service.modules.llm.policies import LLMUsage, ModelPolicies

    uow_factory = container.services["uow_factory"]
    cipher = AesCredentialCipher(container.settings.agent_credentials)
    credentials = ModelCredentials(uow_factory, cipher)
    policies = ModelPolicies(uow_factory)
    usage = LLMUsage(uow_factory)
    container.services["model_credentials"] = credentials
    container.services["model_policies"] = policies
    container.services["llm_usage"] = usage
    cfg = container.settings.llm
    if not cfg.enabled:
        container.llm = DisabledLLM()
        container.services["llm_assist"] = LLMAssist.disabled()
        return
    llm = BifrostLLM(
        cfg,
        log_source_text=constants.LOG_SOURCE_TEXT,
        credentials=credentials,
        usage=usage,
        tuning=container.tuning.llm,
    )
    container.llm = llm
    container.services["llm_assist"] = LLMAssist(llm, cfg, policies)
    if not cfg.operator_pays:
        # Agent credentials are resolved only in their authenticated request/job scope.
        # An unauthenticated background health probe cannot represent their gateway access.
        container.add_closer("llm", llm.close)
        return
    container.add_dependency(
        Dependency(name="llm", mandatory=False, ping=llm.ping, close=llm.close)
    )


def _wire_nli(container: Container) -> None:
    """Claim-support classifier + grounding cascade.

    The frozen graph must load: a lexical substitute cannot provide the multilingual
    verification contract of a trained model, so a missing graph is a startup error rather
    than a report that quietly says ``representative: false``. The stand-in is chosen only
    by the ``nli="lexical"`` override (the hermetic suite, the benchmarks that do not score
    grounding).
    """
    from memory_service.adapters.models.nli import LexicalNLI
    from memory_service.adapters.models.onnx_nli import OnnxNLI
    from memory_service.modules.grounding.cascade import GroundingCascade

    stand_in = container.overrides.nli
    if stand_in == "disabled":
        container.nli = None
        return
    nli: Any = LexicalNLI()
    if stand_in != "lexical":
        nli = OnnxNLI(FROZEN_MODELS.nli, threads=_model_threads(container))
    container.nli = nli
    container.services["grounding"] = GroundingCascade(
        nli, settings=container.tuning.nli, assist=container.services["llm_assist"]
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _wire_retrieval(container: Container) -> None:
    from memory_service.modules.context.builder import ContextBuilder
    from memory_service.modules.context.sections import ContextSections
    from memory_service.modules.memory.ephemeral import EphemeralMemory
    from memory_service.modules.rag.indexer import Indexer
    from memory_service.modules.retrieval.engine import RetrievalEngine
    from memory_service.modules.retrieval.search import Searcher

    tuning = container.tuning
    dense = container.overrides.dense_model or FROZEN_MODELS.dense
    indexer = Indexer(
        container.services["uow_factory"],
        container.search,
        container.dense_spaces,
        container.sparse,
        container.cache,
        late=container.late,
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
        settings=tuning.retrieval,
        assist=container.services["llm_assist"],
        rerankers=container.rerankers,
    )
    container.services["retrieval"] = engine
    working = EphemeralMemory(
        container.cache, ttl_seconds=constants.CACHE.working_memory_ttl_seconds
    )
    container.services["ephemeral_memory"] = working
    builder = ContextBuilder(
        container.services["uow_factory"],
        engine,
        container.services["conversation"],
        container.cache,
        settings=tuning.context,
        retrieval=tuning.retrieval,
        cache_ttl_seconds=constants.CACHE.context_bundle_ttl_seconds,
        working=working,
        assist=container.services["llm_assist"],
        sections=ContextSections(container.services["uow_factory"], container.services),
    )
    container.services["context_builder"] = builder
    container.services["bundle_records"] = builder.records
    container.services["search"] = Searcher(
        container.services["uow_factory"],
        engine,
        container.services["conversation"],
        container.services["llm_assist"],
    )
    # The builder buffers served-memory ids for up to access_flush_seconds and writes bundles
    # to the cache in the background. Without this, SIGTERM drops a whole window of both, per
    # worker, on every rolling deploy - for the counter the forgetting policy reads.
    container.add_closer("context_builder", builder.close)


def _wire_memory(container: Container) -> None:
    """Memory intelligence: the native provider + observation pipeline + service."""
    from memory_service.modules.feedback.service import FeedbackService
    from memory_service.modules.memory.connections import ConnectionService
    from memory_service.modules.memory.forgetting import ForgettingService
    from memory_service.modules.memory.native import NativeMemoryIntelligence
    from memory_service.modules.memory.pipeline import ObservationPipeline
    from memory_service.modules.memory.reflection import ReflectionService
    from memory_service.modules.memory.service import MemoryService
    from memory_service.ports.intelligence import MemoryIntelligenceProvider

    cfg = container.tuning.memory_intelligence
    extractor = None
    if container.settings.hindsight.base_url is not None:
        from memory_service.adapters.models.hindsight import HindsightExtractor

        try:
            extractor = HindsightExtractor(container.settings.hindsight)
        except DependencyUnavailable as exc:
            # the optional extra is absent: extraction stays on the native path
            log.warning("hindsight.unavailable", reason=str(exc))
        else:
            container.add_closer("hindsight_extractor", extractor.close)
    provider: MemoryIntelligenceProvider = NativeMemoryIntelligence(
        cfg,
        container.embedding,
        assist=container.services["llm_assist"],
        contextual_extractor=extractor,
    )
    container.services["memory_provider"] = provider
    container.services["observation_pipeline"] = ObservationPipeline(
        container.services["uow_factory"],
        provider,
        settings=cfg,
        working=container.services.get("ephemeral_memory"),
        assist=container.services["llm_assist"],
    )
    container.services["memory"] = MemoryService(container.services["authz"])
    container.services["feedback"] = FeedbackService(
        container.services["uow_factory"], container.services["authz"], container.services["memory"]
    )
    container.services["forgetting"] = ForgettingService(
        container.services["uow_factory"],
        settings=cfg,
        cache=container.cache,
        working_ttl_seconds=constants.CACHE.working_memory_ttl_seconds,
    )
    container.services["reflection"] = ReflectionService(
        container.services["uow_factory"], assist=container.services["llm_assist"]
    )
    container.services["connections"] = ConnectionService(
        container.services["uow_factory"], assist=container.services["llm_assist"]
    )


def _wire_tools(container: Container) -> None:
    """Tool memory: the catalog and call records, tool search, hints and the learning job."""
    from memory_service.modules.agent_tools.service import AgentTools
    from memory_service.modules.tools.hints import ToolHintsService
    from memory_service.modules.tools.index import ToolIndex
    from memory_service.modules.tools.learning import ToolLearning
    from memory_service.modules.tools.service import ToolMemoryService

    uow_factory = container.services["uow_factory"]
    container.services["tool_memory"] = ToolMemoryService(
        container.services["authz"],
        blob=container.blob,
        blob_bucket=container.settings.blob.file_bucket,
    )
    index = ToolIndex(uow_factory, container.services["indexer"], container.search)
    container.services["tool_index"] = index
    container.services["tool_hints"] = ToolHintsService(uow_factory, index, container.graph_store)
    container.services["tool_learning"] = ToolLearning(
        uow_factory, container.services["llm_assist"], container.graph_store
    )
    container.services["agent_tools"] = AgentTools(uow_factory, container.services)


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

        store = PostgresGraphStore(
            container.database.engine, budget_ms=container.tuning.graph.prefetch_budget_ms
        )
        container.graph_store = store
        container.add_closer("graph_store", store.close)
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
        stage = GraphStage(
            graph,
            container.services["uow_factory"],
            max_facts=container.tuning.context.graph_facts_max,
        )
        engine.post_stages["graph"] = stage


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
            container.services["uow_factory"], settings=settings
        )
    container.services["expansion"] = expansion
