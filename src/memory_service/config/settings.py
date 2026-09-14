"""Typed configuration. Sources, highest precedence first:

1. environment variables (``MEMORY__SECTION__KEY``; nested with ``__``)
2. a YAML file named by ``MEMORY_CONFIG_FILE`` (default ``config/memory.yaml`` if present)
3. defaults below

Every provider is chosen here. Domain code never reads environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class ServiceSettings(BaseModel):
    name: str = "memory-service"
    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    host: str = "0.0.0.0"  # noqa: S104 - container default
    port: int = 8080
    log_level: str = "INFO"
    log_json: bool = True
    log_source_text: bool = Field(default=False, description="never log raw source text by default")
    api_version: str = "v1"
    request_timeout_seconds: float = 30.0
    max_body_bytes: int = 25 * 1024 * 1024


class DatabaseSettings(BaseModel):
    url: str = "postgresql+psycopg://memory:memory@localhost:5432/memory"
    pool_size: int = 10
    max_overflow: int = 10
    pool_timeout_seconds: float = 5.0
    statement_timeout_ms: int = 15_000
    echo: bool = False

    @property
    def sync_url(self) -> str:
        """SQLAlchemy URL for sync use (Alembic). psycopg3 serves both sync and async."""
        if "+psycopg" in self.url or "+" not in self.url.split("://", 1)[0]:
            return (
                self.url
                if "+psycopg" in self.url
                else self.url.replace("postgresql://", "postgresql+psycopg://", 1)
            )
        return self.url

    @property
    def procrastinate_dsn(self) -> str:
        """Plain libpq DSN for Procrastinate's psycopg connector."""
        return self.url.replace("postgresql+psycopg://", "postgresql://", 1)


class CacheSettings(BaseModel):
    provider: Literal["dragonfly", "valkey", "redis", "memory", "disabled"] = "dragonfly"
    url: str = "redis://localhost:6379/0"
    default_ttl_seconds: int = 3600
    hot_thread_ttl_seconds: int = 6 * 3600
    hot_thread_max_messages: int = 200
    working_memory_ttl_seconds: int = 1800
    embedding_ttl_seconds: int = 7 * 24 * 3600
    context_bundle_ttl_seconds: int = 300
    authz_ttl_seconds: int = 60
    connect_timeout_seconds: float = 0.5
    socket_timeout_seconds: float = 0.5


class TaskSettings(BaseModel):
    provider: Literal["procrastinate", "inline", "memory"] = "procrastinate"
    worker_concurrency: int = 4
    default_retries: int = 5
    job_timeout_seconds: int = 600
    periodic_reconcile_seconds: int = 300


class AuthenticationSettings(BaseModel):
    mode: Literal["trusted_dev", "jwt", "gcp_iam", "mtls"] = "trusted_dev"
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_url: str | None = None
    jwt_hs256_secret: SecretStr | None = Field(
        default=None, description="dev/test symmetric key only"
    )
    trusted_dev_api_keys: list[str] = Field(default_factory=lambda: ["dev-key"])
    gcp_allowed_service_accounts: list[str] = Field(default_factory=list)
    header_tenant: str = "X-Memory-Tenant"
    header_workspace: str = "X-Memory-Workspace"
    header_user: str = "X-Memory-User"
    header_groups: str = "X-Memory-Groups"
    header_api_key: str = "X-API-Key"


class AuthorizationSettings(BaseModel):
    provider: Literal["openfga", "memory"] = "openfga"
    openfga_api_url: str = "http://localhost:8081"
    openfga_store_id: str | None = None
    openfga_model_id: str | None = None
    openfga_api_token: SecretStr | None = None
    max_listed_objects: int = 2000
    decision_cache: bool = True


class PolicySettings(BaseModel):
    provider: Literal["disabled", "opa", "static"] = "static"
    opa_url: str = "http://localhost:8181"
    static_allow_external_models: bool = False
    static_default_retention_days: int = 3650


class BlobSettings(BaseModel):
    provider: Literal["gcs", "filesystem", "s3", "memory"] = "filesystem"
    chat_bucket: str = "memory-chat-archive"
    file_bucket: str = "memory-file-archive"
    benchmark_bucket: str = "memory-benchmarks"
    ingest_bucket: str = "memory-ingest-tmp"
    filesystem_root: str = "./.blob"
    gcs_project: str | None = None
    s3_endpoint_url: str | None = None
    lifecycle_policy: Literal["autoclass", "explicit"] = "autoclass"
    explicit_lifecycle_days_nearline: int = 60
    explicit_lifecycle_days_coldline: int = 180
    explicit_lifecycle_days_archive: int = 365


class ArchiveSettings(BaseModel):
    enabled: bool = True
    segment_target_bytes: int = Field(
        default=4 * 1024 * 1024, description="compressed; benchmark 1-8MB"
    )
    segment_max_messages: int = 5000
    segment_max_age_seconds: int = 900
    compression: Literal["zstd"] = "zstd"
    zstd_level: int = 6
    purge_grace_seconds: int = 24 * 3600
    purge_min_payload_bytes: int = Field(
        default=4096, description="only payloads larger than this are purged from hot DB"
    )
    tenant_shards: int = 64


class SearchSettings(BaseModel):
    provider: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_local_path: str | None = Field(
        default=None,
        description="use qdrant-client local mode (':memory:' or a path) instead of a server",
    )
    collection_prefix: str = "mem"
    on_disk_payload: bool = True
    timeout_seconds: float = 5.0


class GraphSettings(BaseModel):
    store: Literal["postgres", "memory"] = "postgres"
    max_visited: int = 200
    default_hops: int = 1


class EmbeddingSettings(BaseModel):
    provider: Literal[
        "sentence_transformers",
        "fastembed",
        "onnx",
        "openvino",
        "vertex",
        "openai",
        "hash",
        "disabled",
    ] = "sentence_transformers"
    model: str = "ibm-granite/granite-embedding-small-english-r2"
    model_path: str | None = Field(default=None, description="local directory with model files")
    dimension: int = 384
    batch_size: int = 32
    max_tokens: int = 512
    normalize: bool = True
    device: str = "cpu"
    threads: int | None = None


class RerankerSettings(BaseModel):
    provider: Literal["sentence_transformers", "onnx", "disabled", "lexical"] = (
        "sentence_transformers"
    )
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model_path: str | None = None
    candidate_k: int = Field(default=20, description="bounded rerank K; benchmark 15-25")
    batch_size: int = 16


class LLMSettings(BaseModel):
    enabled: bool = False
    provider: Literal["disabled", "local", "vertex", "openai", "custom"] = "disabled"
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    uses: list[
        Literal[
            "ambiguous_extraction",
            "ambiguous_worthiness",
            "relation_extraction",
            "entity_resolution",
            "conflict_adjudication",
            "summaries",
            "reflection",
            "query_expansion",
            "chunk_context",
        ]
    ] = Field(default_factory=list)


class ModelSettings(BaseModel):
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    llm: LLMSettings = LLMSettings()


class MemoryIntelligenceSettings(BaseModel):
    provider: Literal["native", "mem0", "cognee", "langmem"] = "native"
    challengers: list[Literal["mem0", "cognee", "langmem"]] = Field(default_factory=list)
    dedup_lexical_threshold: float = 0.92
    dedup_dense_threshold: float = 0.90
    dedup_candidate_k: int = 20
    false_merge_rate_max: float = Field(default=0.01, description="release gate threshold")


class GraphEnrichmentSettings(BaseModel):
    provider: Literal["native", "graphiti", "docling_graph", "cognee", "disabled"] = "native"
    graphiti_neo4j_url: str | None = None
    graphiti_neo4j_user: str | None = None
    graphiti_neo4j_password: SecretStr | None = None


class DocumentSettings(BaseModel):
    parser: Literal["docling", "builtin"] = "docling"
    fallback_parser: Literal["builtin"] = "builtin"
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 40
    chunk_overlap_tokens: int = 40
    contextual_chunks: bool = True
    keep_tables_intact: bool = True
    keep_code_intact: bool = True
    max_file_bytes: int = 100 * 1024 * 1024


class RetrievalSettings(BaseModel):
    exact: bool = True
    bm25: bool = True
    dense: bool = True
    graph: bool = True
    fusion: Literal["rrf", "dbsf", "none"] = "rrf"
    rrf_k: int = 60
    prefetch_k: int = Field(default=50, description="per-retriever candidates before fusion")
    fused_k: int = Field(default=40, description="candidates after fusion")
    rerank: bool = True
    final_k: int = 20
    contextual_chunks: bool = True
    parent_expansion: bool = True
    neighbor_expansion: bool = True
    definition_expansion: bool = True
    expansion_budget_items: int = 8
    evidence_verification: bool = True
    escalation_max_rounds: int = 2
    abstain_when_insufficient: bool = True
    # benchmark-gated; default off
    splade: bool = False
    minicoil: bool = False
    colbert: bool = False
    pageindex: bool = False
    raptor: bool = False
    graph_ppr: bool = False
    graphrag_global: bool = False
    late_chunking: bool = False


class ContextSettings(BaseModel):
    token_budget: int = 6000
    conversation_max_messages: int = 20
    conversation_token_budget: int = 2000
    memories_max: int = 12
    knowledge_max: int = 12
    graph_facts_max: int = 12
    summaries_max: int = 4


class EvaluationSettings(BaseModel):
    enabled: bool = True
    run_async: bool = True
    deepeval_enabled: bool = False
    ragas_enabled: bool = False
    judge_model: str | None = None
    critical_recall_k: int = 20
    sample_rate: float = 0.0


class ObservabilitySettings(BaseModel):
    otel_enabled: bool = True
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str | None = None
    metrics_enabled: bool = True
    openlineage_enabled: bool = False
    openlineage_url: str | None = None
    openlineage_namespace: str = "memory-service"


class ProviderPolicySettings(BaseModel):
    allowed_licenses: list[str] = Field(
        default_factory=lambda: [
            "Apache-2.0",
            "MIT",
            "BSD-3-Clause",
            "BSD-2-Clause",
            "BSL-1.1",
            "LGPL-3.0-only",
            "MPL-2.0",
        ]
    )
    allow_remote_models: bool = False
    allowed_regions: list[str] = Field(default_factory=list)


class PerformanceBudgets(BaseModel):
    """Benchmark targets in milliseconds (p95). Targets, not promises."""

    chat_accept_p95_ms: float = 100
    cached_context_p95_ms: float = 75
    recall_p95_ms: float = 300
    context_bundle_p95_ms: float = 400
    file_accept_p95_ms: float = 200


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


def _yaml_source_factory(settings_cls: type[BaseSettings]) -> PydanticBaseSettingsSource:
    class YamlSource(PydanticBaseSettingsSource):
        def __init__(self, cls: type[BaseSettings]):
            super().__init__(cls)
            path = os.environ.get("MEMORY_CONFIG_FILE")
            candidates = [Path(path)] if path else [Path("config/memory.yaml"), Path("memory.yaml")]
            self._data: dict[str, Any] = {}
            for candidate in candidates:
                if candidate.is_file():
                    with candidate.open("r", encoding="utf-8") as fh:
                        self._data = yaml.safe_load(fh) or {}
                    break

        def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
            return self._data.get(field_name), field_name, False

        def __call__(self) -> dict[str, Any]:
            return dict(self._data)

    return YamlSource(settings_cls)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORY__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service: ServiceSettings = ServiceSettings()
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    tasks: TaskSettings = TaskSettings()
    authentication: AuthenticationSettings = AuthenticationSettings()
    authorization: AuthorizationSettings = AuthorizationSettings()
    policy: PolicySettings = PolicySettings()
    blob: BlobSettings = BlobSettings()
    archive: ArchiveSettings = ArchiveSettings()
    search: SearchSettings = SearchSettings()
    graph: GraphSettings = GraphSettings()
    models: ModelSettings = ModelSettings()
    memory_intelligence: MemoryIntelligenceSettings = Field(
        default_factory=MemoryIntelligenceSettings
    )
    graph_enrichment: GraphEnrichmentSettings = GraphEnrichmentSettings()
    documents: DocumentSettings = DocumentSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    context: ContextSettings = ContextSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    provider_policy: ProviderPolicySettings = ProviderPolicySettings()
    budgets: PerformanceBudgets = PerformanceBudgets()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _yaml_source_factory(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.service.environment == "prod":
            if self.authentication.mode == "trusted_dev":
                raise ValueError("authentication.mode=trusted_dev is not allowed in prod")
            if self.authorization.provider == "memory":
                raise ValueError("authorization.provider=memory is not allowed in prod")
            if self.blob.provider in ("memory", "filesystem"):
                raise ValueError("blob.provider must be gcs or s3 in prod")
        if self.models.llm.enabled and self.models.llm.provider == "disabled":
            raise ValueError("llm.enabled=true requires a provider")
        return self

    def redacted(self) -> dict[str, Any]:
        """Config snapshot safe to expose on /version (secrets already redacted by SecretStr)."""
        return self.model_dump(mode="json", exclude={"authentication": {"trusted_dev_api_keys"}})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
