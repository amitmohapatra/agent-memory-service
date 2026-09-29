"""Typed configuration: the operator's surface, and nothing else.

Sources, highest precedence first:

1. environment variables (``MEMORY__SECTION__KEY``; nested with ``__``)
2. ``.env`` then ``secrets.env`` in the working directory
3. defaults below

``WEB_CONCURRENCY`` is the one variable read outside that scheme: it is the ecosystem's name
for ``service.workers``, and it supplies that field's default (see ``_workers_default``).

Only topology and credentials live here - where the stores are, how to authenticate, whether
the generative model is on and where its gateway is. Everything that makes the product what
it is (models, retrieval depth, budgets, timeouts, thresholds) is a constant in
``config/constants.py``; the test suite's in-process stand-ins are
``application.container.Overrides``. Domain code never reads environment variables.

Every credential is a ``SecretStr`` and ``Settings.redacted()`` masks all of them.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memory_service.domain.provenance import require_permitted_model

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


#: Worker processes when nothing says otherwise. One process is one GIL, and one GIL does
#: not serve the target rate; three is what the 8 vCPU target has room for beside two math
#: threads each.
_DEFAULT_WORKERS = 3


def _workers_default() -> int:
    """``WEB_CONCURRENCY`` is this number under the name the rest of the ecosystem uses.

    Gunicorn, uvicorn and every PaaS that sizes a container read it, so an operator who sets
    it expects to be obeyed - and the image sets it rather than a second name of our own. It
    is only the *default* here: ``MEMORY__SERVICE__WORKERS`` still wins where someone wants
    the two to differ. A value that is not a number at all is ignored rather than fatal
    (an empty ``WEB_CONCURRENCY=`` is a common way for a platform to say "unset"); a number
    outside 1-8 is not ignored, it fails validation below, because silently serving three
    workers to someone who asked for sixteen is the drift this is meant to remove.
    """
    raw = os.environ.get("WEB_CONCURRENCY")
    if raw is None:
        return _DEFAULT_WORKERS
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_WORKERS


class ServiceSettings(BaseModel):
    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    rate_limit_per_minute: int = Field(
        default=6000,
        description="Requests per tenant per minute (0 disables); counted in the cache",
    )
    workers: int = Field(
        default_factory=_workers_default,
        # the factory reads the environment, so the bounds below have to apply to what it
        # returns as well as to what someone passes; pydantic does not check defaults unless
        # it is told to
        validate_default=True,
        ge=1,
        le=8,
        description="uvicorn worker processes (defaults to WEB_CONCURRENCY); "
        "one container, one model set and one pool each",
    )


class DatabaseSettings(BaseModel):
    #: Per *process*, not per service: with three API workers and a worker container the
    #: pools add up, so 8+8 each keeps the total inside a default max_connections while
    #: leaving every request the two or three checkouts it takes.
    url: SecretStr = SecretStr("postgresql+psycopg://memory:memory@localhost:5432/memory")
    pool_size: int = 8
    max_overflow: int = 8

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
    mode: Literal["trusted_dev", "jwt", "api_key"] = "trusted_dev"
    #: The platform operator in ``api_key`` mode: the one secret that may onboard tenants and
    #: issue their first admin key. It acts for no tenant. Unset, nobody can onboard, which
    #: is the safe state for a deployment that has finished onboarding.
    bootstrap_admin_key: SecretStr | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_url: str | None = None
    trusted_dev_api_keys: list[SecretStr] = Field(default_factory=lambda: [SecretStr("dev-key")])
    #: Claim on the credential naming the tenant it may act for. Unset, a credential may
    #: assert ANY tenant - the tenant arrives in a header and nothing checks it against who
    #: is calling, so one key reaches every tenant on the deployment by changing a header.
    #: That is only safe where the tenant is a constant stamped by a gateway, i.e. one
    #: deployment per customer. Set it for a SHARED deployment and the assertion becomes a
    #: proof: the header must equal this claim or the request is refused. Fails closed - a
    #: credential carrying no such claim is refused rather than trusted.
    tenant_claim: str | None = None


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
    says about it is how many intra-op threads the in-process models may use.

    Unset means the count frozen with the encoder, not torch's own default - both this
    docstring and ``.env.example`` used to say otherwise. ``torch.set_num_threads`` is
    process-wide, so the encoder and the NLI head share whatever this resolves to; giving
    them separate numbers only meant the one that loaded last won, which is how this field
    came to have no effect at all on the encoder it names.
    """

    threads: int | None = Field(default=None, ge=1)


LLMUse = Literal[
    "contextual_extraction",
    "ambiguous_extraction",
    "ambiguous_worthiness",
    "relation_extraction",
    "entity_resolution",
    "conflict_adjudication",
    "summaries",
    "reflection",
    #: Typed connections between memories that already exist (supersedes / contradicts /
    #: relates). Separate from ``reflection``, which writes a new insight instead of an edge,
    #: so an operator can run one without the other.
    "memory_connections",
    "briefs",
    "query_expansion",
    "chunk_context",
    "grounding_judge",
]


class AgentCredentialSettings(BaseModel):
    """Operator-managed envelope keys, never agent keys or model-provider credentials."""

    active_key_id: str | None = Field(default=None, min_length=1, max_length=100)
    encryption_keys: dict[str, SecretStr] = Field(default_factory=dict)


class WebhookSettings(BaseModel):
    """Outbound webhooks (ADR 0023). Delivery tuning lives in ``config/constants.py``; the one
    deployment fact is whether receivers may live on the local network (docker-compose
    development), which is refused in deployed environments."""

    allow_local_targets: bool = Field(
        default=False,
        description="permit http and loopback, private and link-local targets: development only",
    )


class HindsightSettings(BaseModel):
    """Integrated knowledge-processing service topology and deployment quota."""

    base_url: str = "http://localhost:8888"
    api_key: SecretStr | None = None
    bank_id: str = Field(default="extraction-preview", min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_concurrency: int = Field(default=1, ge=1, le=16)


class LLMSettings(BaseModel):
    """Generative access gates. Native calls use Bifrost; Hindsight extraction
    uses its server's model configuration. Model provider keys stay outside this service.
    """

    #: Auto requires a registered agent key or an operator key at call time. False is
    #: a deployment-wide prohibition. True retains explicit operator use selection.
    enabled: bool | Literal["auto"] = "auto"
    base_url: str = Field(
        default="http://localhost:8090/v1", description="Bifrost OpenAI-compatible endpoint"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Bifrost virtual key (MEMORY__MODELS__LLM__API_KEY or secrets.env)",
    )
    model: str | None = Field(
        default="auto",
        description="strong model (Bifrost provider/model name) for complex uses",
    )
    fast_model: str | None = Field(
        default="auto",
        description="cheap model for fast_uses (classification-sized calls)",
    )
    uses: list[LLMUse] = Field(
        default_factory=list,
        description="which uses may consult the model; explicitly assisted briefs require "
        "a valid model result, while native paths remain available",
    )
    fast_uses: list[LLMUse] = Field(
        default_factory=lambda: [
            "ambiguous_worthiness",
            "contextual_extraction",
            "query_expansion",
            "chunk_context",
        ]
    )
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, description="retries on 429/5xx/timeouts, bounded")

    @field_validator("model", "fast_model")
    @classmethod
    def _permitted_model(cls, value: str | None) -> str | None:
        """An operator naming a model is bound by the same provenance rule as discovery."""
        if value and value != "auto":
            require_permitted_model(value)
        return value

    def wants(self, use: LLMUse) -> bool:
        uses = self.uses
        if self.enabled == "auto" and "uses" not in self.model_fields_set:
            uses = ["contextual_extraction", "reflection", "briefs", "query_expansion"]
        return self.enabled is not False and use in uses


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

    #: built per Settings() rather than once at import, so ``workers`` reads the
    #: environment the process actually has (see ``_workers_default``)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    tasks: TaskSettings = TaskSettings()
    authentication: AuthenticationSettings = AuthenticationSettings()
    authorization: AuthorizationSettings = AuthorizationSettings()
    blob: BlobSettings = BlobSettings()
    search: SearchSettings = SearchSettings()
    models: ModelSettings = ModelSettings()
    hindsight: HindsightSettings = HindsightSettings()
    agent_credentials: AgentCredentialSettings = AgentCredentialSettings()
    webhooks: WebhookSettings = WebhookSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    #: Environments that are *deployed*, and so may not run the laptop defaults. ``test`` is
    #: absent on purpose: the suite and the benchmarks run under it with ``trusted_dev`` and a
    #: filesystem blob store, which is what they are for. ``staging`` used not to be here, so a
    #: deployment set to it started with header-trust authentication and the single API key
    #: whose value is the default in this file and is printed in ``.env.example`` - anyone who
    #: could reach the port and had read the repository could authenticate as any tenant by
    #: setting three headers. ``.env.example`` told the operator that staging refused exactly
    #: that, so the misconfiguration was invisible: the service started clean and readiness
    #: went green.
    DEPLOYED_ENVIRONMENTS: ClassVar[frozenset[str]] = frozenset({"staging", "prod"})

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.service.environment in self.DEPLOYED_ENVIRONMENTS:
            where = self.service.environment
            if self.authentication.mode == "trusted_dev":
                raise ValueError(f"authentication.mode=trusted_dev is not allowed in {where}")
            if self.blob.provider == "filesystem":
                raise ValueError(f"blob.provider must be gcs in {where}")
            bootstrap = self.authentication.bootstrap_admin_key
            if bootstrap is not None and len(bootstrap.get_secret_value()) < 32:
                # The one credential that onboards tenants and can administer any of them;
                # a short operator-chosen value is guessable online.
                raise ValueError(
                    f"authentication.bootstrap_admin_key must be at least 32 characters in "
                    f"{where} (e.g. `openssl rand -base64 32`)"
                )
            if self.webhooks.allow_local_targets:
                raise ValueError(
                    f"webhooks.allow_local_targets is a development flag and is not allowed "
                    f"in {where}"
                )
        if self.models.llm.enabled and not self.models.llm.model:
            raise ValueError("llm.enabled=true requires models.llm.model")
        if self.models.llm.enabled is True and not self.models.llm.uses:
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
