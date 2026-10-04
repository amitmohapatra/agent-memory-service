"""Typed configuration: the operator's surface, and nothing else.

Sources, highest precedence first:

1. environment variables (``MEMORY__SECTION__KEY``; nested with ``__``)
2. ``.env`` then ``secrets.env`` in the working directory
3. defaults below

Four variables are read outside that scheme, under the names the rest of the platform uses:
``WEB_CONCURRENCY`` (the default of ``service.workers``), ``BIFROST_URL`` and
``BIFROST_VIRTUAL_KEY`` (the model gateway, and the operator's key on it) and
``OTEL_EXPORTER_OTLP_ENDPOINT`` (where traces go; tracing is on exactly when it is set).

Only deployment facts live here - where the stores and the gateway are, the secrets, ports
and worker counts. Everything that makes the product what it is (models, retrieval depth,
budgets, timeouts, retries, thresholds, rate limits) is a constant in ``config/constants.py``;
the test suite's in-process stand-ins are ``application.container.Overrides``. What is
derived is derived: the model is available when the gateway is configured, the
authentication mode follows from the credentials configured, tracing from its endpoint.
Domain code never reads environment variables.

Every credential is a ``SecretStr`` and ``Settings.redacted()`` masks all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal, get_args

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memory_service.domain.fiscal import parse_calendar

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


#: The most processes (API workers) or concurrent jobs a default ever picks: past eight,
#: one container's models, pools and caches are better spent on a second container.
_MAX_DEFAULT_PARALLELISM = 8


def available_cpus() -> int:
    """The CPUs this process may actually run on, not the host's.

    ``os.cpu_count()`` is the host's count inside a container: a pod limited to two CPUs on a
    64-core node would start eight workers and eight jobs, each with its model threads. The
    cgroup v2 quota (``cpu.max``) is the limit a container is actually held to, the
    scheduler affinity the cores it may use; the smallest of what can be read wins.
    """
    counts: list[int] = []
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        counts.append(len(affinity(0)))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]  # noqa: SIM115
        if quota != "max":
            counts.append(max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    if not counts:
        counts.append(os.cpu_count() or 1)
    return max(1, min(counts))


def default_parallelism() -> int:
    """``available_cpus()`` clamped to 1-8: the default API worker count and job concurrency
    when nothing sets them (``WEB_CONCURRENCY``, ``MEMORY__SERVICE__WORKERS``,
    ``MEMORY__TASKS__WORKER_CONCURRENCY``)."""
    return max(1, min(_MAX_DEFAULT_PARALLELISM, available_cpus()))


def _workers_default() -> int:
    """``WEB_CONCURRENCY`` is this number under the name the rest of the ecosystem uses.

    Gunicorn, uvicorn and every PaaS that sizes a container read it, so an operator who sets
    it expects to be obeyed - and the image sets it rather than a second name of our own. It
    is only the *default* here: ``MEMORY__SERVICE__WORKERS`` still wins where someone wants
    the two to differ. A value that is not a number at all is ignored rather than fatal
    (an empty ``WEB_CONCURRENCY=`` is a common way for a platform to say "unset"); a number
    outside 1-8 is not ignored, it fails validation below, because silently serving three
    workers to someone who asked for sixteen is the drift this is meant to remove. Unset,
    the machine decides: one worker per available CPU, clamped to 1-8
    (``default_parallelism``). It was a constant three, sized for the 8 vCPU target, which
    oversubscribed a two-CPU pod and left most of a sixteen-CPU one idle.
    """
    raw = os.environ.get("WEB_CONCURRENCY")
    if raw is None:
        return default_parallelism()
    try:
        return int(raw)
    except ValueError:
        return default_parallelism()


class ServiceSettings(BaseModel):
    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
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


#: Connections one process opens when no budget is set: 8+8 for requests, 4+4 for the
#: graph traversal, 4 for the task queue - what the pools were before the budget existed.
_CONNECTIONS_PER_PROCESS = 28


@dataclass(frozen=True)
class PoolPlan:
    """One process's share of the pod's connection budget, per pool (``pool_plan``)."""

    #: the request pool through ``url``: SQLAlchemy ``pool_size`` + ``max_overflow``
    main_size: int
    main_overflow: int
    #: the graph traversal's budgeted pool through ``direct_url``
    graph_size: int
    graph_overflow: int
    #: Procrastinate's psycopg pool through ``direct_url`` (max; min is 1)
    queue_max: int

    @property
    def total(self) -> int:
        return (
            self.main_size + self.main_overflow + self.graph_size + self.graph_overflow
        ) + self.queue_max


class DatabaseSettings(BaseModel):
    """Where PostgreSQL is, and how many connections a pod may hold to it.

    ``url`` is the request path's connection and may be a transaction-mode pooler
    (PgBouncer ``pool_mode=transaction``; set ``transaction_pooler``). ``direct_url`` is for
    the work that needs a session of its own - Procrastinate's LISTEN/NOTIFY and its job
    locks, and the graph traversal's session ``statement_timeout`` and prepared plan - and
    defaults to ``url``. See docs/deploy/database.md.
    """

    url: SecretStr = SecretStr("postgresql+psycopg://memory:memory@localhost:5432/memory")
    direct_url: SecretStr | None = None
    #: ``url`` is a transaction-mode pooler: no server-side prepared statements, and no
    #: session parameters in the startup packet (the pooler sets ``statement_timeout`` on its
    #: server connections, ``connect_query`` in deploy/pgbouncer/pgbouncer.ini).
    transaction_pooler: bool = False
    #: Connections one pod (container) may open to PostgreSQL or to the pooler, across all
    #: its processes and all their pools. Each process takes ``budget // processes`` and
    #: splits it 4:2:1 between the request pool, the graph traversal and the task queue
    #: (``pool_plan``). Unset: 28 per process, the sizes the pools had before.
    connection_budget: int | None = Field(default=None, ge=6)

    @property
    def dsn(self) -> str:
        return self.url.get_secret_value()

    @property
    def direct_dsn(self) -> str:
        """The session-capable connection (``direct_url``, else ``url``)."""
        return (self.direct_url or self.url).get_secret_value()

    def pool_plan(self, processes: int) -> PoolPlan:
        """One process's pools, from the pod's budget::

            per_process = connection_budget // processes      (unset: 28)
            queue       = max(2, per_process // 7)
            graph       = max(2, 2 * per_process // 7)        split ceil(g/2) + floor(g/2)
            main        = max(2, per_process - graph - queue) split ceil(m/2) + floor(m/2)

        A pool's overflow is the burst it may open past its steady size and closes again;
        the budget counts both, because a burst is when the budget matters.
        """
        processes = max(1, processes)
        if self.connection_budget is None:
            per_process = _CONNECTIONS_PER_PROCESS
        else:
            per_process = max(6, self.connection_budget // processes)
        queue = max(2, per_process // 7)
        graph = max(2, 2 * per_process // 7)
        main = max(2, per_process - graph - queue)
        return PoolPlan(
            main_size=-(-main // 2),
            main_overflow=main // 2,
            graph_size=-(-graph // 2),
            graph_overflow=graph // 2,
            queue_max=queue,
        )

    @property
    def sync_url(self) -> str:
        """SQLAlchemy URL for sync use (Alembic). psycopg3 serves both sync and async.
        Migrations take locks and set session parameters, so they go direct."""
        url = self.direct_dsn
        if "+psycopg" in url:
            return url
        if "+" not in url.split("://", 1)[0]:
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    @property
    def procrastinate_dsn(self) -> str:
        """Plain libpq DSN for Procrastinate's psycopg connector: always the session-capable
        connection, since its worker LISTENs and holds locks across statements."""
        return self.direct_dsn.replace("postgresql+psycopg://", "postgresql://", 1)


class CacheSettings(BaseModel):
    """One redis-protocol cache (the dev stack runs Dragonfly)."""

    url: SecretStr = SecretStr("redis://localhost:6379/0")


class TaskSettings(BaseModel):
    """Procrastinate on the application database."""

    #: jobs one worker process runs at once; unset, one per available CPU clamped to 1-8
    worker_concurrency: int = Field(default_factory=default_parallelism, ge=1, le=8)
    #: where the worker serves its Prometheus metrics and healthcheck (``None``: not at all)
    metrics_port: int | None = Field(default=9464, ge=1, le=65535)


AuthenticationMode = Literal["trusted_dev", "jwt", "api_key"]


class AuthenticationSettings(BaseModel):
    """How callers authenticate. The mode is not a setting: it follows from what is
    configured (``mode``)."""

    #: The platform operator in ``api_key`` mode: the one secret that may onboard tenants and
    #: issue their first admin key. It acts for no tenant. Unset, nobody can onboard, which
    #: is the safe state for a deployment that has finished onboarding.
    bootstrap_admin_key: SecretStr | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_url: str | None = None
    #: Development keys that trust the context headers as given (refused when deployed).
    trusted_dev_api_keys: list[SecretStr] = Field(default_factory=list)
    #: Claim on the credential naming the tenant it may act for. Unset, a credential may
    #: assert ANY tenant - the tenant arrives in a header and nothing checks it against who
    #: is calling, so one key reaches every tenant on the deployment by changing a header.
    #: That is only safe where the tenant is a constant stamped by a gateway, i.e. one
    #: deployment per customer. Set it for a SHARED deployment and the assertion becomes a
    #: proof: the header must equal this claim or the request is refused. Fails closed - a
    #: credential carrying no such claim is refused rather than trusted.
    tenant_claim: str | None = None

    @property
    def mode(self) -> AuthenticationMode:
        """``jwt`` with an issuer's JWKS; ``api_key`` with the bootstrap key; ``trusted_dev``
        with development keys; ``api_key`` otherwise (the service's own issued keys)."""
        if self.jwt_jwks_url:
            return "jwt"
        if self.bootstrap_admin_key is None and self.trusted_dev_api_keys:
            return "trusted_dev"
        return "api_key"


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
    #: How a new collection is laid out on a Qdrant cluster: shards to spread its points
    #: over, copies of each shard, and how many copies must acknowledge a write. Facts about
    #: the cluster, so they are settings; they apply when a collection is created, and an
    #: existing collection takes new values only through a rebuild
    #: (``tools/reindex.py --drop``, docs/deploy/search.md). The defaults are one node.
    shard_number: int = Field(default=1, ge=1)
    replication_factor: int = Field(default=1, ge=1)
    write_consistency_factor: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _consistency_within_replicas(self) -> SearchSettings:
        if self.write_consistency_factor > self.replication_factor:
            raise ValueError("search.write_consistency_factor cannot exceed replication_factor")
        return self


LLMUse = Literal[
    "contextual_extraction",
    "relation_extraction",
    "entity_resolution",
    "conflict_adjudication",
    "summaries",
    "reflection",
    #: Typed connections between memories that already exist (supersedes / contradicts /
    #: relates). Separate from ``reflection``, which writes a new insight instead of an edge,
    #: so an operator can run one without the other.
    "memory_connections",
    "query_expansion",
    "chunk_context",
    #: Restate a conversation turn so it stands on its own, appended to its index key
    #: (``modules.memory.restatement``, ADR 0027).
    "memory_restatement",
    "grounding_judge",
    #: Distil a stored procedure's title and strategy from the runs that followed it (the
    #: tool learning job); the miner's own rendering is kept without it.
    "procedure_abstraction",
]
#: Every use: the default tenant policy.
ALL_LLM_USES: tuple[LLMUse, ...] = get_args(LLMUse)
#: Uses a tenant turns on in its policy rather than gets by default: each costs a model call
#: per conversation message, which registering a key must not start on its own.
OPT_IN_LLM_USES: frozenset[LLMUse] = frozenset({"memory_restatement"})


class AgentCredentialSettings(BaseModel):
    """Operator-managed envelope keys, never agent keys or model-provider credentials."""

    active_key_id: str | None = Field(default=None, min_length=1, max_length=100)
    encryption_keys: dict[str, SecretStr] = Field(default_factory=dict)


class HindsightSettings(BaseModel):
    """Where the Hindsight extraction service is, when a deployment runs one: non-agent
    contextual extraction goes through it (with the ``[hindsight]`` extra installed). Its
    tuning is ``constants.HINDSIGHT``."""

    base_url: str | None = None
    api_key: SecretStr | None = None


class LLMSettings(BaseModel):
    """The model gateway. The model is available exactly when ``base_url`` is set; a call runs
    only when a key can pay for it - the acting agent's or its tenant's registered key
    (``PUT /v1/agents/model-key``, ``PUT /v1/model-key``) or the operator's ``api_key`` - and
    the tenant's policy allows the use. Models, budgets and retries are
    ``constants.LLM``; the tenant's policy may name the model a use calls."""

    base_url: str | None = Field(
        default=None, description="Bifrost OpenAI-compatible endpoint, e.g. http://bifrost:8080/v1"
    )
    api_key: SecretStr | None = Field(
        default=None, description="the operator's Bifrost virtual key: pays for tenants without one"
    )

    @property
    def enabled(self) -> bool:
        return self.base_url is not None

    def wants(self, use: LLMUse) -> bool:
        """Whether the deployment can call the model for ``use`` at all (before any policy)."""
        return self.enabled

    @property
    def operator_pays(self) -> bool:
        """A call may proceed without a registered key: the operator's key pays."""
        return self.api_key is not None


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
    #: the model gateway and the operator's key on it (the platform's names, unprefixed)
    bifrost_url: str | None = Field(
        default=None, validation_alias=AliasChoices("BIFROST_URL", "bifrost_url")
    )
    bifrost_virtual_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("BIFROST_VIRTUAL_KEY", "bifrost_virtual_key"),
    )
    #: where traces go (OTLP over HTTP); tracing is on exactly when it is set
    otel_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT", "otel_endpoint"),
    )
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    #: built per Settings() too: the job concurrency's default reads the machine
    tasks: TaskSettings = Field(default_factory=TaskSettings)
    authentication: AuthenticationSettings = AuthenticationSettings()
    authorization: AuthorizationSettings = AuthorizationSettings()
    blob: BlobSettings = BlobSettings()
    search: SearchSettings = SearchSettings()
    hindsight: HindsightSettings = HindsightSettings()
    agent_credentials: AgentCredentialSettings = AgentCredentialSettings()
    #: A retailer's fiscal calendar (``MEMORY__RETAIL_CALENDAR=454``): ``454``, ``445`` or
    #: ``544``, then optionally the month the year ends in and ``end`` when a year is named by
    #: the calendar year it ends in (``445-12``, ``454-01-end``); NRF's is ``454``. A fact about
    #: the customer, not a tuning: with it, "last week", "LY", "wk 32", "Q3" and "FW26" in a
    #: memory resolve to days in that calendar, and a query's planning shorthand ("WOS",
    #: "ST%") is searched with its expansion (``domain.fiscal``, ``domain.glossary``).
    retail_calendar: str | None = None

    @field_validator("retail_calendar")
    @classmethod
    def _valid_calendar(cls, value: str | None) -> str | None:
        if value is not None:
            parse_calendar(value)
        return value

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(base_url=self.bifrost_url, api_key=self.bifrost_virtual_key)

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
        return self

    def redacted(self) -> dict[str, Any]:
        """Config snapshot safe to expose on /version: every SecretStr renders as asterisks."""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
