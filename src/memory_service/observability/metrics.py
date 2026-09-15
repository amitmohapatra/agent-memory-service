"""Prometheus metrics. One registry per process; exposed on /metrics."""

from __future__ import annotations

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


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
