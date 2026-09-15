"""Provider wiring. Grows milestone by milestone."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory_service.application.container import Dependency
from memory_service.observability.logging import get_logger

if TYPE_CHECKING:
    from memory_service.application.container import Container

log = get_logger(__name__)


async def wire_all(container: Container) -> None:
    settings = container.settings
    log.info(
        "wiring.start",
        environment=settings.service.environment,
        cache=settings.cache.provider,
        search=settings.search.provider,
        blob=settings.blob.provider,
        tasks=settings.tasks.provider,
        authorization=settings.authorization.provider,
        llm_enabled=settings.models.llm.enabled,
    )
    await _wire_cache(container)
    await _wire_database(container)
    await _wire_tasks(container)
    _wire_uow(container)
    await _wire_authorization(container)
    _wire_services(container)
    _wire_conversation(container)
    await _wire_blob(container)
    _wire_archive(container)
    _wire_ingestion(container)
    await _wire_search(container)
    _wire_models(container)
    _wire_retrieval(container)
    _wire_memory(container)
    _register_jobs(container)
    log.info("wiring.done", dependencies=sorted(container.dependencies))


# ---------------------------------------------------------------------------
# M1 wiring
# ---------------------------------------------------------------------------


async def _wire_cache(container: Container) -> None:
    cfg = container.settings.cache
    if cfg.provider == "disabled":
        container.cache = None
        return
    if cfg.provider == "memory":
        from memory_service.adapters.cache.memory_cache import MemoryCache

        container.cache = MemoryCache()
    else:
        from memory_service.adapters.cache.redis_cache import RedisCache

        container.cache = RedisCache(cfg)
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
    cfg = container.settings.tasks
    if cfg.provider == "procrastinate":
        from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

        queue = ProcrastinateTaskQueue(
            container.settings.database.procrastinate_dsn,
            default_retries=cfg.default_retries,
            job_timeout_seconds=cfg.job_timeout_seconds,
        )
        container.tasks = queue
        container.add_dependency(
            Dependency(name="task_queue", mandatory=True, ping=queue.ping, close=queue.close)
        )
    elif cfg.provider == "inline":
        from memory_service.adapters.tasks.inline_queue import InlineTaskQueue

        container.tasks = InlineTaskQueue()
    else:
        from memory_service.adapters.tasks.inline_queue import RecordingTaskQueue

        container.tasks = RecordingTaskQueue()


def _wire_uow(container: Container) -> None:
    from memory_service.adapters.db.uow import OutboxRelay, SqlUnitOfWorkFactory

    relay = OutboxRelay(container.database.session_factory, container.tasks)
    container.services["outbox_relay"] = relay
    container.services["uow_factory"] = SqlUnitOfWorkFactory(
        container.database.session_factory, relay
    )


async def _wire_authorization(container: Container) -> None:
    cfg = container.settings.authorization
    if cfg.provider == "openfga":
        from memory_service.adapters.authz.openfga_provider import OpenFGAAuthorizationProvider

        provider = OpenFGAAuthorizationProvider(cfg)
        container.add_dependency(
            Dependency(name="openfga", mandatory=True, ping=provider.ping, close=provider.close)
        )
    else:
        from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider

        provider = MemoryAuthorizationProvider(max_listed_objects=cfg.max_listed_objects)
    container.authorization = provider


def _wire_services(container: Container) -> None:
    from memory_service.modules.auth.authentication import ServiceAuthenticator
    from memory_service.modules.authz.service import AuthorizationService
    from memory_service.modules.idempotency.service import IdempotencyService

    settings = container.settings
    container.services["idempotency"] = IdempotencyService(container.cache)
    from memory_service.adapters.auth.gcp_id_token import verify_google_id_token

    container.services["authenticator"] = ServiceAuthenticator(
        settings.authentication, gcp_verifier=verify_google_id_token
    )
    container.services["authz"] = AuthorizationService(
        container.authorization,
        container.cache,
        max_listed_objects=settings.authorization.max_listed_objects,
        cache_ttl_seconds=settings.cache.authz_ttl_seconds,
        decision_cache=settings.authorization.decision_cache,
    )


def _wire_conversation(container: Container) -> None:
    from memory_service.modules.conversation.service import ConversationService
    from memory_service.modules.working_memory.hot_thread import HotThreadCache, WorkingMemory

    cache_cfg = container.settings.cache
    hot = HotThreadCache(
        container.cache,
        max_messages=cache_cfg.hot_thread_max_messages,
        ttl_seconds=cache_cfg.hot_thread_ttl_seconds,
    )
    container.services["hot_thread"] = hot
    container.services["working_memory"] = WorkingMemory(
        container.cache, ttl_seconds=cache_cfg.working_memory_ttl_seconds
    )
    container.services["conversation"] = ConversationService(
        container.services["authz"], hot, archive_enabled=container.settings.archive.enabled
    )


def _register_jobs(container: Container) -> None:
    from memory_service.modules.jobs.registry import register_handlers

    register_handlers(container)


async def _wire_blob(container: Container) -> None:
    cfg = container.settings.blob
    if cfg.provider == "gcs":
        from memory_service.adapters.blob.gcs import GCSBlobStore

        store = GCSBlobStore(cfg)
        container.add_dependency(Dependency(name="blob", mandatory=True, ping=store.ping))
    elif cfg.provider == "memory":
        from memory_service.adapters.blob.memory import MemoryBlobStore

        store = MemoryBlobStore()
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
        archive=container.settings.archive,
        blob_settings=container.settings.blob,
    )


def _wire_ingestion(container: Container) -> None:
    from memory_service.adapters.parsers.builtin import BuiltinParser
    from memory_service.modules.ingestion.service import IngestionService

    cfg = container.settings.documents
    builtin = BuiltinParser()
    parser = builtin
    if cfg.parser == "docling":
        try:
            from memory_service.adapters.parsers.docling_parser import DoclingParser

            parser = DoclingParser()
        except Exception as exc:
            log.warning("docling.unavailable", error=str(exc))
    container.document_parser = parser
    container.services["ingestion"] = IngestionService(
        container.services["uow_factory"],
        container.services["authz"],
        parser,
        container.blob,
        settings=cfg,
        file_bucket=container.settings.blob.file_bucket,
        tenant_shards=container.settings.archive.tenant_shards,
        fallback_parser=builtin,
    )


async def _wire_search(container: Container) -> None:
    from memory_service.adapters.search.qdrant_store import QdrantSearchStore

    cfg = container.settings.search
    if cfg.provider == "memory":
        cfg = cfg.model_copy(update={"qdrant_local_path": ":memory:"})
    store = QdrantSearchStore(cfg)
    container.search = store
    if cfg.qdrant_local_path is None:
        container.add_dependency(
            Dependency(name="qdrant", mandatory=True, ping=store.ping, close=store.close)
        )


def _wire_models(container: Container) -> None:
    from memory_service.adapters.models.embeddings import (
        FastEmbedEmbedding,
        HashEmbedding,
        SentenceTransformersEmbedding,
    )
    from memory_service.adapters.models.rerankers import CrossEncoderReranker, LexicalReranker
    from memory_service.adapters.models.sparse import Bm25SparseEncoder
    from memory_service.config.registry import check_provider_policy

    settings = container.settings
    emb_cfg = settings.models.embedding
    if emb_cfg.provider == "hash":
        embedding = HashEmbedding(emb_cfg.dimension)
    elif emb_cfg.provider == "fastembed":
        embedding = FastEmbedEmbedding(emb_cfg)
    elif emb_cfg.provider in ("sentence_transformers", "onnx", "openvino"):
        embedding = SentenceTransformersEmbedding(emb_cfg)
    else:
        raise NotImplementedError(f"embedding provider {emb_cfg.provider} not implemented yet")
    check_provider_policy(
        embedding.info,
        [*settings.provider_policy.allowed_licenses, "see model card"],
        settings.provider_policy.allow_remote_models,
    )
    container.embedding = embedding
    container.sparse = Bm25SparseEncoder()
    rr_cfg = settings.models.reranker
    if rr_cfg.provider == "disabled":
        container.reranker = None
    elif rr_cfg.provider == "lexical":
        container.reranker = LexicalReranker()
    else:
        container.reranker = CrossEncoderReranker(rr_cfg)


def _wire_retrieval(container: Container) -> None:
    from memory_service.modules.context.builder import ContextBuilder
    from memory_service.modules.memory.ephemeral import EphemeralMemory
    from memory_service.modules.rag.indexer import Indexer
    from memory_service.modules.retrieval.engine import RetrievalEngine

    settings = container.settings
    indexer = Indexer(
        container.services["uow_factory"],
        container.search,
        container.embedding,
        container.sparse,
        container.cache,
        batch_size=settings.models.embedding.batch_size,
        embedding_cache_ttl=settings.cache.embedding_ttl_seconds,
    )
    container.services["indexer"] = indexer
    engine = RetrievalEngine(
        container.services["uow_factory"],
        container.services["authz"],
        container.search,
        indexer,
        container.reranker,
        settings=settings.retrieval,
        rerank_k=settings.models.reranker.candidate_k,
    )
    container.services["retrieval"] = engine
    working = EphemeralMemory(
        container.cache, ttl_seconds=settings.cache.working_memory_ttl_seconds
    )
    container.services["ephemeral_memory"] = working
    container.services["context_builder"] = ContextBuilder(
        container.services["uow_factory"],
        engine,
        container.services["conversation"],
        container.cache,
        settings=settings.context,
        retrieval=settings.retrieval,
        cache_ttl_seconds=settings.cache.context_bundle_ttl_seconds,
        working=working,
    )


def _wire_memory(container: Container) -> None:
    """Memory intelligence: provider (native by default) + observation pipeline + service."""
    from memory_service.config.registry import check_provider_policy
    from memory_service.modules.memory.native import NativeMemoryIntelligence
    from memory_service.modules.memory.pipeline import ObservationPipeline
    from memory_service.modules.memory.service import MemoryService
    from memory_service.ports.intelligence import MemoryIntelligenceProvider

    settings = container.settings
    cfg = settings.memory_intelligence
    provider: MemoryIntelligenceProvider
    if cfg.provider == "native":
        provider = NativeMemoryIntelligence(cfg, container.embedding)
    elif cfg.provider == "mem0":
        from memory_service.adapters.intelligence.mem0_provider import Mem0MemoryIntelligence

        provider = Mem0MemoryIntelligence(settings)
    elif cfg.provider == "langmem":
        from memory_service.adapters.intelligence.langmem_provider import LangMemIntelligence

        provider = LangMemIntelligence(settings)
    elif cfg.provider == "cognee":
        from memory_service.adapters.intelligence.cognee_provider import CogneeMemoryIntelligence

        provider = CogneeMemoryIntelligence(settings)
    else:  # pragma: no cover - settings Literal guards this
        raise NotImplementedError(cfg.provider)
    check_provider_policy(
        provider.info,
        [*settings.provider_policy.allowed_licenses, "see model card"],
        settings.provider_policy.allow_remote_models,
    )
    if provider.info.requires_llm and not settings.models.llm.enabled:
        raise NotImplementedError(
            f"memory intelligence provider {cfg.provider!r} requires an LLM; "
            "set MEMORY__MODELS__LLM__ENABLED=true and configure the model"
        )
    container.services["memory_provider"] = provider
    container.services["observation_pipeline"] = ObservationPipeline(
        container.services["uow_factory"],
        provider,
        settings=cfg,
        working=container.services.get("ephemeral_memory"),
    )
    container.services["memory"] = MemoryService(container.services["authz"])
