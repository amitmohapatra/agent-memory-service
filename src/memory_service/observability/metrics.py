"""Prometheus metrics. One registry per process; exposed on /metrics.

One registry per process is the whole story on a multi-worker container, and it is a story
the endpoint has to tell. prometheus_client keeps counters in process memory; the API runs
``service.workers`` uvicorn processes sharing one listening socket, so a scrape is answered
by whichever worker accepts that connection. Every counter here is therefore one worker's
share of the traffic - about 1/N of it, and not the same worker twice in a row, so a counter
read across two scrapes can go down.

Multiprocess collection (``PROMETHEUS_MULTIPROC_DIR`` plus ``MultiProcessCollector``) is the
fix, and it is a change with its own weight: every metric must be constructed after the
directory exists, gauges need a declared aggregation mode, histograms lose their per-process
``_created`` series, and every worker exit needs ``mark_process_dead``. Until that is done,
``render_metrics`` says plainly what the numbers are, and docs/MEASUREMENTS.md records that
a gate reading this endpoint must scrape with one worker.
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
)

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
    "Latency of internal stages (auth, authz, db, cache, retrieval.*, rerank, context, archive)",
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
    "LLM tokens by use and direction (input|output)",
    ["use", "direction"],
    registry=REGISTRY,
)
grounding_claims_total = Counter(
    "memory_grounding_claims_total",
    "Grounding cascade claim verdicts (supported|unsupported|contradicted|borderline)",
    ["verdict"],
    registry=REGISTRY,
)


api_workers = Gauge(
    "memory_api_workers",
    "API worker processes in this container; the series here belong to one of them",
    registry=REGISTRY,
)


def render_metrics(workers: int = 1) -> tuple[bytes, str]:
    """The exposition, preceded by whose numbers it is.

    ``workers`` is ``service.workers`` as this process was configured with. It is exposed as
    a series (``memory_api_workers``) so a dashboard can see the divisor, and repeated in a
    comment block so a human running ``curl | grep`` sees it too - the comment lines are
    ignored by any Prometheus parser, which treats a ``#`` that is not HELP or TYPE as a
    comment.
    """
    api_workers.set(workers)
    if workers > 1:
        banner = (
            f"# SCOPE: this is one API worker process (pid {os.getpid()}) of {workers}.\n"
            f"# Counters and histograms live in process memory and the listening socket hands\n"
            f"# each scrape to whichever worker accepts it, so every series below is roughly\n"
            f"# 1/{workers} of the traffic and jumps between scrapes rather than rising.\n"
            "# To read a service-wide total: scrape with WEB_CONCURRENCY=1, or run the client\n"
            "# in multiprocess mode (PROMETHEUS_MULTIPROC_DIR).\n"
        )
    else:
        banner = "# SCOPE: this is the only API worker process; the series below are the whole.\n"
    return banner.encode() + generate_latest(REGISTRY), CONTENT_TYPE_LATEST
