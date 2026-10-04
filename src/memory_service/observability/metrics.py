"""Prometheus metrics, exposed on /metrics (the API) and on the worker's metrics port.

The API runs ``service.workers`` uvicorn processes behind one socket, and prometheus_client
keeps values in process memory: a scrape answered by one worker used to show that worker's
share of the traffic, about 1/N of it and not the same worker twice. When there is more than
one worker the entrypoint now sets ``PROMETHEUS_MULTIPROC_DIR`` (emptied at start,
``memory_service.__main__``) before the workers import this module; each process then writes
its values to files there, and ``render_metrics`` sums them with ``MultiProcessCollector``.
Gauges declare how processes combine (``multiprocess_mode``): a queue depth or a checked-out
connection count is a sum over the live processes, a dependency's state the most recent
reading. A worker that exits marks itself dead so its live gauges stop counting
(``mark_process_dead``); its counters stay, as counters should.

With one worker, or when the directory is not set (tests, the job worker), the registry is
the process's own and is the whole.
"""

from __future__ import annotations

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0)

http_requests_total = Counter(
    "memory_http_requests_total",
    "HTTP requests",
    ["method", "route", "status"],
    registry=REGISTRY,
)
http_request_seconds = Histogram(
    "memory_http_request_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
stage_seconds = Histogram(
    "memory_stage_seconds",
    "Latency of internal stages (auth, authz, db, cache, retrieval.*, context, archive)",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
jobs_total = Counter(
    "memory_jobs_total",
    "Background jobs by task and outcome",
    ["task", "outcome"],
    registry=REGISTRY,
)
cache_ops_total = Counter(
    "memory_cache_ops_total", "Cache operations", ["cache", "outcome"], registry=REGISTRY
)
authz_denials_total = Counter(
    "memory_authz_denials_total", "Authorization denials", ["relation"], registry=REGISTRY
)
read_audit_dropped_total = Counter(
    "memory_read_audit_dropped_total",
    "Read-audit entries lost: the queue was full or a flush failed",
    registry=REGISTRY,
)
archive_bytes_total = Counter(
    "memory_archive_bytes_total", "Bytes archived", ["kind", "stage"], registry=REGISTRY
)
reconciler_repairs_total = Counter(
    "memory_reconciler_repairs_total", "Reconciler repairs", ["issue"], registry=REGISTRY
)
dependency_up = Gauge(
    "memory_dependency_up",
    "1 when a backing store answered ping",
    ["dependency"],
    registry=REGISTRY,
    multiprocess_mode="mostrecent",
)
evidence_status_total = Counter(
    "memory_evidence_status_total", "Evidence verification outcomes", ["status"], registry=REGISTRY
)
memory_decisions_total = Counter(
    "memory_decisions_total",
    "Consolidation decisions by outcome",
    ["decision"],
    registry=REGISTRY,
)
llm_requests_total = Counter(
    "memory_llm_requests_total",
    "LLM gateway calls by use and outcome (ok, http_<status>, exhausted, circuit_open, ...)",
    ["use", "outcome"],
    registry=REGISTRY,
)
llm_seconds = Histogram(
    "memory_llm_seconds",
    "LLM gateway call latency by use",
    ["use"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0),
    registry=REGISTRY,
)
llm_assist_total = Counter(
    "memory_llm_assist_total",
    "Module-level LLM assistance by use and outcome (used|fallback)",
    ["use", "outcome"],
    registry=REGISTRY,
)
llm_tokens_total = Counter(
    "memory_llm_tokens_total",
    "LLM tokens by tenant, use and direction (input|output); tenant is 'operator' for "
    "calls no request or job identity owns",
    ["tenant", "use", "direction"],
    registry=REGISTRY,
)
grounding_claims_total = Counter(
    "memory_grounding_claims_total",
    "Grounding cascade claim verdicts (supported|unsupported|contradicted|borderline)",
    ["verdict"],
    registry=REGISTRY,
)

model_queue_waiters = Gauge(
    "memory_model_queue_waiters",
    "Callers queued for an in-process model (SerialRunner), by runner",
    ["runner"],
    registry=REGISTRY,
    multiprocess_mode="livesum",
)
model_queue_rejected_total = Counter(
    "memory_model_queue_rejected_total",
    "Callers refused with 503 because a model's queue was full, by runner",
    ["runner"],
    registry=REGISTRY,
)
request_deadline_exceeded_total = Counter(
    "memory_request_deadline_exceeded_total",
    "Requests answered 504 because they ran past their route class's deadline",
    ["route_class"],
    registry=REGISTRY,
)


db_pool_checked_out = Gauge(
    "memory_db_pool_checked_out",
    "Database connections checked out of a pool right now, by pool",
    ["pool"],
    registry=REGISTRY,
    multiprocess_mode="livesum",
)
db_pool_capacity = Gauge(
    "memory_db_pool_capacity",
    "Connections a pool may hold (size + overflow), by pool, summed over processes",
    ["pool"],
    registry=REGISTRY,
    multiprocess_mode="livesum",
)

# -- the job worker's own series (served on its metrics port) ----------------------------
queue_depth = Gauge(
    "memory_queue_depth",
    "Jobs waiting (todo) per Procrastinate queue",
    ["queue"],
    registry=REGISTRY,
)
queue_oldest_lag_seconds = Gauge(
    "memory_queue_oldest_lag_seconds",
    "Age of the oldest job that is due and not yet started, per queue",
    ["queue"],
    registry=REGISTRY,
)
outbox_backlog = Gauge(
    "memory_outbox_backlog",
    "Outbox rows committed but not yet dispatched to the queue",
    registry=REGISTRY,
)
jobs_failed_terminal = Gauge(
    "memory_jobs_failed_terminal",
    "Jobs that exhausted their retries (Procrastinate status failed), per queue",
    ["queue"],
    registry=REGISTRY,
)
worker_last_sample_timestamp = Gauge(
    "memory_worker_last_sample_timestamp_seconds",
    "When the worker last read its queue gauges; a stale value is a stuck worker",
    registry=REGISTRY,
)
worker_running_jobs = Gauge(
    "memory_worker_running_jobs",
    "Jobs this worker process is running right now",
    registry=REGISTRY,
)


api_workers = Gauge(
    "memory_api_workers",
    "API worker processes in this container",
    registry=REGISTRY,
    multiprocess_mode="max",
)


def multiprocess_dir() -> str | None:
    """The directory the API's processes share their values through, when there is one."""
    return os.environ.get(MULTIPROC_ENV) or None


def mark_process_dead() -> None:
    """Called by an exiting API worker: its live gauges stop counting toward the sum."""
    if multiprocess_dir():
        multiprocess.mark_process_dead(os.getpid())


def render_metrics(workers: int = 1) -> tuple[bytes, str]:
    """The exposition, preceded by whose numbers it is.

    In multiprocess mode the series are every API worker's, summed; otherwise they are this
    process's, which is the whole only when it is the only worker. ``memory_api_workers``
    carries the count, and a comment block says which case a human reading ``curl | grep``
    is looking at (a ``#`` line that is not HELP or TYPE is a comment to any parser).
    """
    api_workers.set(workers)
    if multiprocess_dir():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        banner = (
            f"# SCOPE: all {workers} API worker processes of this container, aggregated\n"
            "# (PROMETHEUS_MULTIPROC_DIR); counters and histograms are totals.\n"
        )
        return banner.encode() + generate_latest(registry), CONTENT_TYPE_LATEST
    if workers > 1:
        banner = (
            f"# SCOPE: this is one API worker process (pid {os.getpid()}) of {workers}, and\n"
            "# PROMETHEUS_MULTIPROC_DIR is not set, so every series below is roughly\n"
            f"# 1/{workers} of the traffic. Start the API through `memory-api`, which sets it.\n"
        )
    else:
        banner = "# SCOPE: this is the only API worker process; the series below are the whole.\n"
    return banner.encode() + generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def serve_worker_metrics(port: int) -> None:
    """The job worker's exposition on its own port (it has no HTTP server of its own)."""
    from prometheus_client import start_http_server

    start_http_server(port, registry=REGISTRY)
