# Configuration reference

Every setting the service reads, with its default, a working example, and whether the service
works it out for itself when you leave it unset. The settings are the operator's surface and
nothing else: where the stores and the gateway are, the secrets, ports and worker counts.
Everything that makes the product what it is (models, retrieval depth, budgets, timeouts,
retries, thresholds, rate limits) is a constant in `src/memory_service/config/constants.py`,
changed by a reviewed code change, never by an environment variable
([chapter 11](guide/11-operations.md#what-is-deliberately-a-constant) lists them).

`tests/unit/test_configuration_doc.py` keeps this page honest: every field of `Settings` must
have a row here and a line in [`.env.example`](../.env.example), every default below must be
the code's, and the **Example** column, set all at once, must build a valid production
`Settings`.

## How it is read

Sources, highest precedence first (`src/memory_service/config/settings.py`):

1. environment variables;
2. `.env`, then `secrets.env` in the working directory (both git-ignored; the later file wins);
3. the defaults below.

Service settings are named `MEMORY__<SECTION>__<FIELD>` (the `__` nests). Four names are the
platform's own and carry no prefix: `WEB_CONCURRENCY`, `BIFROST_URL`, `BIFROST_VIRTUAL_KEY` and
`OTEL_EXPORTER_OTLP_ENDPOINT`. A list or a map is written as JSON
(`MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS=["dev-key"]`). Every credential is a
`SecretStr`: `GET /version` shows a redacted snapshot of the settings, with each secret as
asterisks. Domain code never reads the environment.

Start from [`.env.example`](../.env.example): it lists every variable once, commented where the
default is right for a laptop. `docker compose` reads the same file.

## What is automatic

**Auto** in the tables means the service decides the value when you set nothing, and how:

- **Worker counts follow the CPUs the container may use** (the cgroup quota and the CPU
  affinity, not the host's count), clamped to 1-8: API processes and concurrent jobs.
- **The authentication mode follows from the credentials** configured: a JWKS URL means
  `jwt`; development keys and no bootstrap key mean `trusted_dev`; otherwise `api_key`.
- **The model is available exactly when `BIFROST_URL` is set.** Unset, no model call is ever
  made and every path runs its deterministic form.
- **Tracing is on exactly when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.**
- **The connection pools are sized from one budget** per pod (`CONNECTION_BUDGET`), split per
  process; unset, 28 connections per process.
- **The session connection defaults to the request connection** (`DIRECT_URL` to `URL`).
- **Model-key encryption gets a development key in `dev` and `test`** when no envelope keys
  are set (derived, never stored, logged as a warning). It protects nothing; `staging` and
  `prod` never get one.

## Service

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__SERVICE__ENVIRONMENT` | `dev` | `prod` | no | `dev`, `test`, `staging` or `prod`. `staging` and `prod` are deployed: the [production guards](#production-guards) apply |
| `MEMORY__SERVICE__PORT` | `8080` | `8080` | no | the port the API listens on |
| `MEMORY__SERVICE__LOG_LEVEL` | `INFO` | `WARNING` | no | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `MEMORY__SERVICE__LOG_JSON` | `true` | `true` | no | JSON lines for a log collector; `false` for the readable console renderer |
| `MEMORY__SERVICE__WORKERS` | auto | `3` | yes: `WEB_CONCURRENCY`, else one per available CPU, 1-8 | uvicorn worker processes, 1-8. Each loads one model set and opens its own pools |
| `WEB_CONCURRENCY` | unset | `3` | n/a | the platform's name for the worker count: the default of `MEMORY__SERVICE__WORKERS`, which wins when both are set. The image sets `3`. A value that is not a number is ignored |

## Database (PostgreSQL)

PostgreSQL is the source of truth and the job queue. [deploy/database.md](deploy/database.md)
covers the connection budget, PgBouncer and online migrations.

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__DATABASE__URL` | `postgresql+psycopg://memory:memory@localhost:5432/memory` | `postgresql+psycopg://memory:<password>@pgbouncer:6432/memory` | no | the request path's connection; may be a transaction-mode PgBouncer |
| `MEMORY__DATABASE__DIRECT_URL` | unset | `postgresql+psycopg://memory:<password>@postgres:5432/memory` | yes: `URL` | PostgreSQL itself, for the job queue (LISTEN/NOTIFY, job locks), the graph traversal and migrations |
| `MEMORY__DATABASE__TRANSACTION_POOLER` | `false` | `true` | no | `true` when `URL` is a transaction-mode pooler: no server-side prepared statements, no session parameters at connect |
| `MEMORY__DATABASE__CONNECTION_BUDGET` | unset | `84` | yes: 28 per process | connections one pod may open, all processes and pools together (at least 6). Each process takes `budget // processes` and splits it 4:2:1 between requests, the graph traversal and the queue |

## Cache

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__CACHE__URL` | `redis://localhost:6379/0` | `redis://dragonfly:6379/0` | no | any Redis-protocol cache (the dev stack runs Dragonfly). Never load-bearing: the service degrades to PostgreSQL when it is down |

## Background jobs

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__TASKS__WORKER_CONCURRENCY` | auto | `4` | yes: one per available CPU, 1-8 | jobs one worker process runs at once, 1-8 |
| `MEMORY__TASKS__METRICS_PORT` | `9464` | `9464` | no | where the job worker serves its Prometheus series and healthcheck (1-65535). It cannot be switched off from the environment |

## Authentication

The mode is not a setting: it follows from what is set ([What is automatic](#what-is-automatic)).
Chapter 7 explains the modes; [api/admin.md](api/admin.md) onboarding.

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__AUTHENTICATION__BOOTSTRAP_ADMIN_KEY` | unset | `replace-with-32-or-more-random-characters` | no | the platform operator's secret: onboards tenants and issues their first admin key (`POST /v1/admin/tenants`). Setting it selects `api_key` mode. Unset it once onboarding is done; issued keys keep working. At least 32 characters when deployed (`openssl rand -base64 32`) |
| `MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS` | `[]` | `["dev-key"]` | no | development keys that trust the scope headers as given. Alone, they select `trusted_dev` mode; refused in `staging` and `prod` |
| `MEMORY__AUTHENTICATION__TRUSTED_DEV_TENANT` | `default` | `default` | no | the tenant a development key acts in when a request names none, and the one `GET /v1/keys/self` reports for it |
| `MEMORY__AUTHENTICATION__JWT_JWKS_URL` | unset | `https://issuer.example.com/.well-known/jwks.json` | no | an issuer's JWKS: selects `jwt` mode (per-customer deployments behind a gateway) |
| `MEMORY__AUTHENTICATION__JWT_ISSUER` | unset | `https://issuer.example.com` | no | the `iss` a token must carry |
| `MEMORY__AUTHENTICATION__JWT_AUDIENCE` | unset | `memory-service` | no | the `aud` a token must carry |
| `MEMORY__AUTHENTICATION__TENANT_CLAIM` | unset | `tenant` | no | the claim naming the tenant a credential may act for. Set it on a **shared** deployment: the tenant header must equal the claim, and a credential without it is refused. Unset, the header is trusted (one deployment per customer) |

## Authorization (OpenFGA)

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__AUTHORIZATION__OPENFGA_API_URL` | `http://localhost:8081` | `http://openfga:8080` | no | the OpenFGA API |
| `MEMORY__AUTHORIZATION__OPENFGA_STORE_ID` | unset | `01J8ZK7Q9V3W2X1Y0ZABCDEFGH` | yes: the store this build writes | pin a store; only one this build wrote (the `openfga.model_written` log names it) |
| `MEMORY__AUTHORIZATION__OPENFGA_MODEL_ID` | unset | `01J8ZK7QA0B1C2D3E4F5G6H7J8` | yes: this build's model | pin an authorization model; one that is not this build's stops the service at start (ADR 0021) |
| `MEMORY__AUTHORIZATION__OPENFGA_API_TOKEN` | unset | `<openfga-preshared-key>` | no | OpenFGA's pre-shared key, when it requires one |

## Blob storage (archives, documents, large tool outputs)

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__BLOB__PROVIDER` | `filesystem` | `gcs` | no | `gcs` or `filesystem`; `filesystem` is refused in `staging` and `prod` |
| `MEMORY__BLOB__CHAT_BUCKET` | `memory-chat-archive` | `acme-memory-chat-archive` | no | where conversation archive segments go |
| `MEMORY__BLOB__FILE_BUCKET` | `memory-file-archive` | `acme-memory-file-archive` | no | where uploaded files and large payloads go |
| `MEMORY__BLOB__FILESYSTEM_ROOT` | `./.blob` | `/var/lib/memory/blob` | no | the directory the `filesystem` provider writes under |
| `MEMORY__BLOB__GCS_PROJECT` | unset | `acme-prod` | yes: the credentials' project | the GCP project of the buckets |

## Search (Qdrant)

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__SEARCH__QDRANT_URL` | `http://localhost:6333` | `http://qdrant:6333` | no | Qdrant's REST endpoint (the dashboard and the snapshot API) |
| `MEMORY__SEARCH__QDRANT_GRPC_PORT` | `6334` | `6334` | no | Qdrant's gRPC port: every query goes over gRPC |
| `MEMORY__SEARCH__QDRANT_API_KEY` | unset | `<qdrant-api-key>` | no | Qdrant's API key, when it requires one |
| `MEMORY__SEARCH__SHARD_NUMBER` | `1` | `3` | no | shards a **new** collection is spread over |
| `MEMORY__SEARCH__REPLICATION_FACTOR` | `1` | `2` | no | copies of each shard of a new collection |
| `MEMORY__SEARCH__WRITE_CONSISTENCY_FACTOR` | `1` | `2` | no | copies that must acknowledge a write; at most `REPLICATION_FACTOR`. The three layout values apply when a collection is created; an existing one takes them only through a rebuild ([deploy/search.md](deploy/search.md)) |

## The model gateway and model keys

The service never talks to a model provider; it calls a [Bifrost](https://github.com/maximhq/bifrost)
gateway you run ([chapter 8](guide/08-models.md)). Which uses run and which model each calls is
the tenant's policy, set through the API, not a setting.

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `BIFROST_URL` | unset | `http://bifrost:8080/v1` | n/a | the gateway's OpenAI-compatible endpoint. Set, the model is available; unset, no model call is ever made |
| `BIFROST_VIRTUAL_KEY` | unset | `<operator-virtual-key>` | no | the operator's virtual key: pays for tenants and agents that registered none. Unset, only registered keys pay |
| `MEMORY__AGENT_CREDENTIALS__ACTIVE_KEY_ID` | unset | `v1` | yes in `dev`/`test`: a development key | the envelope key that encrypts newly registered model keys |
| `MEMORY__AGENT_CREDENTIALS__ENCRYPTION_KEYS` | `{}` | `{"v1":"<base64 of 32 random bytes>"}` | yes in `dev`/`test`: a development key | the keyring, by version (keep old versions while rotating). Required in `staging` and `prod`, where registration is refused without it |
| `MEMORY__HINDSIGHT__BASE_URL` | unset | `http://hindsight:8888` | no | an optional Hindsight extraction service (install the `[hindsight]` extra): non-agent contextual extraction goes through it |
| `MEMORY__HINDSIGHT__API_KEY` | unset | `<hindsight-token>` | no | Hindsight's server authentication token |

## Observability

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | `http://otel-collector:4318/v1/traces` | n/a | where traces go (OTLP over HTTP). Set, tracing is on |

Metrics need no setting: the API serves `/metrics`, the job worker serves its own on
`MEMORY__TASKS__METRICS_PORT`.

## The customer's calendar

| Variable | Default | Example | Auto | What it does |
|---|---|---|---|---|
| `MEMORY__RETAIL_CALENDAR` | unset | `454` | no | a retailer's fiscal calendar: `454`, `445` or `544`, then optionally the month the year ends in and `end` when a year is named by the calendar year it ends in (`445-12`, `454-01-end`). With it, "last week", "LY", "wk 32", "Q3" and "FW26" resolve to days in that calendar, and planning shorthand ("WOS", "ST%") is searched with its expansion ([chapter 3](guide/03-time.md)) |

## Production guards

When `MEMORY__SERVICE__ENVIRONMENT` is `staging` or `prod`, the service refuses to start if:

- the authentication mode is `trusted_dev` (development keys and no bootstrap key);
- `MEMORY__BLOB__PROVIDER` is `filesystem`;
- `MEMORY__AUTHENTICATION__BOOTSTRAP_ADMIN_KEY` is set and shorter than 32 characters.

Registering a model key is refused there until the envelope keys are set. `test` is not
deployed on purpose: the suite and the benchmarks run under it with development keys and a
filesystem blob store.

## Other environment variables

These are read outside `Settings`, by a tool or a client rather than the service:

| Variable | Read by | Default | What it does |
|---|---|---|---|
| `MEMORY_MIGRATION_LOCK_TIMEOUT` | `migrations/env.py` | `5s` | how long a migration waits for a lock before it fails, to be retried |
| `MEMORY_MIGRATION_STATEMENT_TIMEOUT` | `migrations/env.py` | `15min` | the bound on one migration statement; raise it for a maintenance window |
| `MEMORY_DOCLING_ARTIFACTS` | the Docling parser | `docling` under a model root (`/models`, then `./models`) | where Docling's layout and table models are |
| `PROMETHEUS_MULTIPROC_DIR` | `memory-api` | a temporary directory | where the API's worker processes write the metrics `/metrics` sums |
| `MEMORY_URL` | the SDK (`MemoryClient()`) | `http://localhost:8080` | the service the SDK talks to |
| `TRELLIS_API_KEY` | the SDK (`MemoryClient()`) | unset | the key the SDK sends |
| `MEMORY_EXAMPLES_DATABASE_URL`, `EXAMPLES_LIVE`, `TRELLIS_BOOTSTRAP_KEY` | `examples/` | see [examples/README.md](../examples/README.md) | the examples' own database, and running them against a live service |

The model weights are not configured either: the service reads the frozen set from `/models`
or `./models` (`make models`), and a missing model is a startup error. The test suite's own
variables (`MEMORY_TEST_*`) are described in [chapter 12](guide/12-testing-and-gates.md).
