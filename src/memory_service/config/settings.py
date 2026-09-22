"""Typed configuration: the operator's surface, and nothing else.

Sources, highest precedence first:

1. environment variables (``MEMORY__SECTION__KEY``; nested with ``__``)
2. ``.env`` then ``secrets.env`` in the working directory
3. defaults below

Only topology and credentials live here - where the stores are, how to authenticate, whether
the generative model is on and where its gateway is. Everything that makes the product what
it is (models, retrieval depth, budgets, timeouts, thresholds) is a constant in
``config/constants.py``; the test suite's in-process stand-ins are
``application.container.Overrides``. Domain code never reads environment variables.

Every credential is a ``SecretStr`` and ``Settings.redacted()`` masks all of them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class ServiceSettings(BaseModel):
    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    rate_limit_per_minute: int = Field(
        default=1200,
        description="Requests per tenant per minute (0 disables); counted in the cache",
    )


class DatabaseSettings(BaseModel):
    url: SecretStr = SecretStr("postgresql+psycopg://memory:memory@localhost:5432/memory")
    pool_size: int = 10
    max_overflow: int = 10

    @property
    def dsn(self) -> str:
        return self.url.get_secret_value()

    @property
    def sync_url(self) -> str:
        """SQLAlchemy URL for sync use (Alembic). psycopg3 serves both sync and async."""
        url = self.dsn
        if "+psycopg" in url:
            return url
        if "+" not in url.split("://", 1)[0]:
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    @property
    def procrastinate_dsn(self) -> str:
        """Plain libpq DSN for Procrastinate's psycopg connector."""
        return self.dsn.replace("postgresql+psycopg://", "postgresql://", 1)


class CacheSettings(BaseModel):
    """One redis-protocol cache (the dev stack runs Dragonfly)."""

    url: SecretStr = SecretStr("redis://localhost:6379/0")


class TaskSettings(BaseModel):
    """Procrastinate on the application database."""

    worker_concurrency: int = Field(default=4, ge=1, le=8)


class AuthenticationSettings(BaseModel):
    mode: Literal["trusted_dev", "jwt"] = "trusted_dev"
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_url: str | None = None
    trusted_dev_api_keys: list[SecretStr] = Field(default_factory=lambda: [SecretStr("dev-key")])


class AuthorizationSettings(BaseModel):
    """OpenFGA. The in-memory provider is a test seam on ``Overrides``."""

    openfga_api_url: str = "http://localhost:8081"
    openfga_store_id: str | None = None
    openfga_model_id: str | None = None
    openfga_api_token: SecretStr | None = None


class BlobSettings(BaseModel):
    provider: Literal["gcs", "filesystem"] = "filesystem"
    chat_bucket: str = "memory-chat-archive"
    file_bucket: str = "memory-file-archive"
    filesystem_root: str = "./.blob"
    gcs_project: str | None = None


class SearchSettings(BaseModel):
    """A Qdrant server. qdrant-client local mode is a test seam on ``Overrides``."""

    qdrant_url: str = "http://localhost:6333"
    #: Qdrant's second port. The client talks gRPC to it for every query; the URL above stays
    #: the REST endpoint, which is what the dashboard and the snapshot API speak.
    qdrant_grpc_port: int = 6334
    qdrant_api_key: SecretStr | None = None


class EmbeddingSettings(BaseModel):
    """The encoder itself is ``constants.FROZEN_MODELS.dense``; the only thing a deployment
    says about it is how many torch threads it may use (unset: torch's own default)."""

    threads: int | None = Field(default=None, ge=1)


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
    uses: list[LLMUse] = Field(
        default_factory=list,
        description="which deterministic paths may consult the model; each falls back natively",
    )
    fast_uses: list[LLMUse] = Field(
        default_factory=lambda: ["ambiguous_worthiness", "query_expansion", "chunk_context"]
    )
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, description="retries on 429/5xx/timeouts, bounded")

    def wants(self, use: LLMUse) -> bool:
        return self.enabled and use in self.uses


class ModelSettings(BaseModel):
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()


class ObservabilitySettings(BaseModel):
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str | None = None

    @property
    def otel_enabled(self) -> bool:
        """Tracing is on exactly when something receives the spans. There used to be a
        separate ``otel_enabled`` flag beside the exporter; on by default with the exporter
        off, it installed a tracer provider that dropped every span."""
        return self.otel_exporter != "none"


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


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
    search: SearchSettings = SearchSettings()
    models: ModelSettings = ModelSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.service.environment == "prod":
            if self.authentication.mode == "trusted_dev":
                raise ValueError("authentication.mode=trusted_dev is not allowed in prod")
            if self.blob.provider == "filesystem":
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
        """Config snapshot safe to expose on /version: every SecretStr renders as asterisks."""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
