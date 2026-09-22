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
    max_body_bytes: int = 25 * 1024 * 1024
    rate_limit_per_minute: int = Field(
        default=1200,
        description="Requests per tenant per minute (0 disables); counted in the cache",
    )
    rate_limit_burst: int = Field(
        default=200, description="Extra requests tolerated above the rate"
    )


class DatabaseSettings(BaseModel):
    url: str = "postgresql+psycopg://memory:memory@localhost:5432/memory"
    pool_size: int = 10
    max_overflow: int = 10
    pool_timeout_seconds: float = 5.0
    statement_timeout_ms: int = 15_000
    #: libpq gives up on opening a connection after this long.
    #:
    #: Without it libpq waits indefinitely, and "indefinitely" is reachable: a PostgreSQL
    #: container whose port is still published but whose server has stopped answering
    #: completes the TCP handshake and then never replies to the startup packet. The
    #: readiness ping then hangs instead of reporting not-ready — the one failure a
    #: readiness probe exists to report is the one it cannot survive. Bounded here so
    #: /health/ready always answers.
    connect_timeout_seconds: int = 5
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
    stalled_after_seconds: float = Field(
        default=120,
        description="A job whose worker stopped heartbeating for this long is re-queued",
    )


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


class BlobSettings(BaseModel):
    provider: Literal["gcs", "filesystem", "memory"] = "filesystem"
    chat_bucket: str = "memory-chat-archive"
    file_bucket: str = "memory-file-archive"
    filesystem_root: str = "./.blob"
    gcs_project: str | None = None
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
    #: Only values that wiring can actually build. "vertex", "openai" and "disabled" were
    #: declared here but had no branch in wire_models: setting one passed validation and then
    #: killed startup with NotImplementedError. A configuration surface that advertises what
    #: it cannot build is worse than a smaller one.
    provider: Literal[
        "sentence_transformers",
        "fastembed",
        "onnx",
        "openvino",
        "hash",
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
    #: The dial that decides capacity: one request costs this many cross-encoder pairs, so
    #: target RPS x candidate_k is the throughput the reranker tier has to sustain.
    candidate_k: int = Field(default=20, description="bounded rerank K; benchmark 15-25")
    batch_size: int = 16


class NLISettings(BaseModel):
    """Claim-support classifier for the grounding cascade. ``transformers`` runs the DeBERTa
    NLI cross-encoder on CPU; ``lexical`` is a deterministic stand-in (never representative)."""

    provider: Literal["transformers", "lexical", "disabled"] = "transformers"
    model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    model_path: str | None = Field(default=None, description="local directory with model files")
    batch_size: int = 16
    max_length: int = 512
    supported_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="entailment (or contradiction) score to decide"
    )
    borderline_band: tuple[float, float] = Field(
        default=(0.3, 0.7), description="entailment band in which the LLM judge is consulted"
    )
    premises_per_claim: int = Field(
        default=5, ge=1, description="evidence items scored per claim (best lexical overlap)"
    )
    max_claims: int = Field(default=40, ge=1)


LLMUse = Literal[
    "ambiguous_extraction",
    "ambiguous_worthiness",
    "relation_extraction",
    "entity_resolution",
    "conflict_adjudication",
    "summaries",
    "reflection",
    "query_expansion",
    "chunk_context",
    "grounding_judge",
]


class LLMSettings(BaseModel):
    """Generative model access. The only provider is the Bifrost gateway (OpenAI-compatible);
    provider keys live in Bifrost, the service holds a Bifrost *virtual key*."""

    #: The gateway is the only provider there is, so ``enabled`` says everything a second
    #: ``provider`` field could. It used to be both, as ``enabled: bool`` and
    #: ``provider: Literal["disabled", "bifrost"]``, kept in agreement by a validator whose
    #: entire job was to reject the two spellings of "off" that disagreed. Two fields that
    #: must always agree are one field; /version still reports "bifrost" or "disabled",
    #: derived from this.
    enabled: bool = False
    base_url: str = Field(
        default="http://localhost:8090/v1", description="Bifrost OpenAI-compatible endpoint"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Bifrost virtual key (MEMORY__MODELS__LLM__API_KEY or secrets.env)",
    )
    model: str | None = Field(
        default="anthropic/claude-sonnet-5",
        description="strong model (Bifrost provider/model name) for complex uses",
    )
    fast_model: str | None = Field(
        default="anthropic/claude-haiku-4-5-20251001",
        description="cheap model for fast_uses (classification-sized calls)",
    )
    fast_uses: list[LLMUse] = Field(
        default_factory=lambda: ["ambiguous_worthiness", "query_expansion", "chunk_context"]
    )
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    max_retries: int = Field(default=2, description="retries on 429/5xx/timeouts, bounded")
    retry_backoff_seconds: float = 0.5
    circuit_failure_threshold: int = Field(
        default=5, description="consecutive failures that open the circuit"
    )
    circuit_open_seconds: float = 30.0
    uses: list[LLMUse] = Field(
        default_factory=list,
        description="which deterministic paths may consult the model; each falls back natively",
    )

    def wants(self, use: LLMUse) -> bool:
        return self.enabled and use in self.uses


class ModelSettings(BaseModel):
    """Embedding / reranker / LLM providers plus the benchmark-gated model names (M10)."""

    sparse_model: str = "prithivida/Splade_PP_en_v1"
    sparse_model_path: str | None = None
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    nli: NLISettings = NLISettings()
    llm: LLMSettings = LLMSettings()


class MemoryIntelligenceSettings(BaseModel):
    dedup_lexical_threshold: float = 0.92
    dedup_dense_threshold: float = 0.90
    dedup_candidate_k: int = 20
    false_merge_rate_max: float = Field(default=0.01, description="release gate threshold")
    # admission gate (worthiness x novelty x confidence x expected utility -> admit/defer/reject)
    admission_worthiness_min: float = Field(default=0.35, ge=0.0, le=1.0)
    admission_confidence_min: float = Field(default=0.3, ge=0.0, le=1.0)
    admission_score_min: float = Field(default=0.4, ge=0.0, le=1.0)
    admission_defer_band: float = Field(
        default=0.08, ge=0.0, le=1.0, description="score band below the minimum that defers"
    )
    #: Keep the turn itself, not only what the rules could parse out of it.
    #:
    #: Extraction is rule-based and first-person ("I work at X"). A conversation *about*
    #: someone — third-person narrative, which is most real dialogue — matches nothing, and
    #: `_from_sentence` returns None. Measured on LoCoMo: 452 of 788 turns (57.4%) produced
    #: no candidate at all, so the text was never indexed and no retriever could reach it.
    #: The perfect-retrieval ceiling under that regime is 0.098; keeping the turn verbatim
    #: raises it to 0.685.
    #:
    #: The verbatim copy is an OBSERVATION, which is in DERIVED_MEMORY_TYPES, so it is
    #: excluded from supersession and reflection (landing.py:63, :77) and cannot disturb the
    #: fact machinery or the false-merge gate. It augments the rule output; it never
    #: replaces it. Applies to user-authored messages outside a thread only: a thread's
    #: turns are kept by the hot-thread cache and the archive, and an agent's messages are
    #: working chatter that must not inherit a shared visibility.
    keep_verbatim_turns: bool = True
    #: Longest turn kept verbatim. Beyond this the turn is truncated rather than dropped.
    verbatim_max_chars: int = Field(default=2000, ge=200)
    # landing reflection and derived memories
    landing_reflection_k: int = Field(default=8, ge=0, le=8)
    belief_min_support: int = Field(default=2, ge=2)
    entity_summary_min_facts: int = Field(default=2, ge=1)
    # forgetting: importance x recency x access decay
    forgetting_half_life_days: float = Field(default=30.0, gt=0.0)
    forgetting_archive_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    forgetting_min_idle_days: float = Field(default=30.0, ge=0.0)
    forgetting_batch: int = Field(default=500, ge=1)


class GraphEnrichmentSettings(BaseModel):
    provider: Literal["native", "disabled"] = "native"


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
    #: Longest query text that is retrieved on. Anything beyond this is cut.
    #:
    #: A query is not free: it is embedded, run through the sparse model, and then paired with
    #: every reranked candidate — and a cross-encoder pair is as expensive as its longest
    #: side. Measured on the degenerate-input benchmark, a 2,000-character wall of noise cost
    #: 20 seconds against 1 second for an ordinary question, on the same corpus. Nothing is
    #: lost by cutting: the embedding models truncate at 512 tokens regardless, so the text
    #: past this point never reached the model — it was only ever paid for.
    max_query_chars: int = Field(
        default=2048, ge=64, description="query text beyond this is truncated before retrieval"
    )
    exact: bool = True
    bm25: bool = True
    dense: bool = True
    graph: bool = True
    fusion: Literal["rrf", "dbsf", "none"] = "rrf"
    rrf_k: int = 60
    prefetch_k: int = Field(default=100, description="per-retriever candidates before fusion")
    fused_k: int = Field(default=100, description="candidates after fusion")
    #: Cross-encoder reranking. **Off by default, on measured evidence.**
    #:
    #: BeIR/SciFact, 1,000 documents, 70 paired queries, clean vector store, real models:
    #:
    #:              recall@10   nDCG@10   p50
    #:   rerank on     97.14%    79.33%   11,151 ms
    #:   rerank off    98.57%    84.51%      533 ms
    #:
    #: Paired, the reranker rescued *zero* queries the first stage missed and lost one. nDCG
    #: better on 4 queries, worse on 16, identical on 50 — exact sign test p = 0.012, mean
    #: delta -0.0518 with a 95% interval of [-0.0917, -0.0120] that excludes zero. It is
    #: significantly worse here, not merely not better.
    #:
    #: The likely mechanism is domain mismatch: ms-marco-MiniLM-L6-v2 is trained on web
    #: passage ranking, and reordering scientific claim-evidence pairs is not that task. A
    #: reranker trained in-domain may well earn its place — this setting is about *this*
    #: default, not about reranking as an idea.
    #:
    #: The cost decides it either way. At 11.2 s per query against 0.53 s, 20 RPS needs ~161
    #: cores with it and ~8 without. A component that is 21x the latency has to buy something,
    #: and this one is buying a loss.
    rerank: bool = False
    final_k: int = 50
    contextual_chunks: bool = True
    parent_expansion: bool = True
    neighbor_expansion: bool = True
    definition_expansion: bool = True
    expansion_budget_items: int = 8
    evidence_verification: bool = True
    escalation_max_rounds: int = 2
    abstain_when_insufficient: bool = True
    #: On conversational (memory-only) bundles, require that some retrieved memory *about
    #: the person the question names* shares a content term with the rest of the question.
    #: The plain overlap rule above cannot see a wrong-person presupposition — "what was
    #: grandma's gift to Melanie?" when it was Caroline's grandma — because a two-person
    #: conversation shares terms with any question about either of them. Measured on LoCoMo:
    #: every one of 304 bundles, 71 of them unanswerable by construction, reported COMPLETE.
    subject_evidence_check: bool = True
    # benchmark-gated; default off
    splade: bool = False


class ContextSettings(BaseModel):
    token_budget: int = 8000
    conversation_max_messages: int = 20
    conversation_token_budget: int = 2000
    #: Keep what the engine ranked, and let the token budget do the bounding.
    #:
    #: This was 12 against RetrievalSettings.final_k = 20, so eight already-retrieved,
    #: already-ranked candidates were dropped for free — and the bundles that survived used
    #: 5.8% of the 6000-token budget (measured: median 1380 rendered chars, ~345 tokens).
    #: Two numbers that must agree were written down twice and drifted. For reference, Mem0
    #: publishes 6,956 tokens per retrieval at 92.5 on LoCoMo; a bundle spending 345 is not
    #: competing on the same axis. token_budget stays the real constraint.
    memories_max: int = 50
    knowledge_max: int = 12
    graph_facts_max: int = 12
    summaries_max: int = 4


class EvaluationSettings(BaseModel):
    enabled: bool = True
    critical_recall_k: int = 20


class ObservabilitySettings(BaseModel):
    otel_enabled: bool = True
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str | None = None


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
        env_file=(".env", "secrets.env"),  # secrets.env is git-ignored; later files win
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service: ServiceSettings = ServiceSettings()
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    tasks: TaskSettings = TaskSettings()
    authentication: AuthenticationSettings = AuthenticationSettings()
    authorization: AuthorizationSettings = AuthorizationSettings()
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
                raise ValueError("blob.provider must be gcs in prod")
        if self.models.llm.enabled and not self.models.llm.model:
            raise ValueError("llm.enabled=true requires models.llm.model")
        if self.models.llm.enabled and not self.models.llm.uses:
            # Opting in per use is the design — each path falls back natively, so a use you
            # have not enabled is a deterministic answer, not a broken one. What is not the
            # design is the silence: with uses empty the service starts clean, reports
            # "llm": "bifrost" on /version, and sends the gateway nothing at all. Every one
            # of the ten paths quietly takes its fallback, and the only way to find out
            # is to notice that the token metrics never move.
            raise ValueError(
                "llm.enabled=true with models.llm.uses empty: nothing would call the model. "
                "List the paths that may consult it, e.g. "
                'MEMORY__MODELS__LLM__USES=["query_expansion","summaries"], or set '
                "llm.enabled=false."
            )
        if self.models.llm.enabled and self.models.llm.fast_uses:
            # fast_model is a separate credential in practice: it defaults to a different
            # provider than model does. `self.fast_model or settings.model` in the adapter
            # does not rescue a mismatch because the default is truthy — so setting MODEL
            # and forgetting FAST_MODEL sends exactly the fast_uses to whatever the default
            # happens to be, which on this deployment is a provider with no credit.
            fast = set(self.models.llm.fast_uses) & set(self.models.llm.uses)
            if fast and not self.models.llm.fast_model:
                raise ValueError(
                    f"models.llm.fast_uses {sorted(fast)} are enabled but fast_model is "
                    "unset: those uses would silently route somewhere else"
                )
        return self

    def redacted(self) -> dict[str, Any]:
        """Config snapshot safe to expose on /version (secrets already redacted by SecretStr)."""
        return self.model_dump(mode="json", exclude={"authentication": {"trusted_dev_api_keys"}})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
