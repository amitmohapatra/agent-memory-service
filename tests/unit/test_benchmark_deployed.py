"""Pure parts of the network-hop gate producers: acknowledgement verification and report
shaping shared through ``benchmark.harness``, the deployed-run helpers, and the Locust CSV
reduction. No network, no subprocesses."""

from __future__ import annotations

import sys

import pytest
from benchmark import deployed, harness
from benchmark.load import run as load_run

pytestmark = pytest.mark.unit


# -- harness: verification --------------------------------------------------------


def test_message_listing_detects_missing_and_altered_messages() -> None:
    acked = {
        "m1": {"thread_id": "t1", "content": "one"},
        "m2": {"thread_id": "t1", "content": "two"},
        "m3": {"thread_id": "t1", "content": "three"},
        "other": {"thread_id": "t2", "content": "elsewhere"},
    }
    listing = {
        "messages": [
            {"message_id": "m1", "content": "one"},
            {"message_id": "m2", "content": "TWO"},
        ]
    }
    lost = harness.check_message_listing(listing, 200, "t1", acked)
    assert len(lost) == 2
    assert any("m2 content mismatch" in x for x in lost)
    assert any("m3 missing from listing (200)" in x for x in lost)
    assert harness.check_message_listing({}, 503, "t2", acked) == [
        "message other missing from listing (503)"
    ]


def test_document_must_be_ready_and_archived() -> None:
    assert (
        harness.check_document("d1", 200, {"status": "READY", "archive_status": "ARCHIVED"}) == []
    )
    assert harness.check_document("d1", 404, None) == ["document d1: 404"]
    assert harness.check_document("d1", 200, {"status": "READY", "archive_status": "STAGED"}) == [
        "document d1: READY/STAGED"
    ]
    assert harness.check_document(
        "d1", 200, {"status": "FAILED", "archive_status": "ARCHIVED"}
    ) == ["document d1: FAILED/ARCHIVED"]


def test_duplicate_memories_counts_extra_copies_per_fact() -> None:
    memories = [
        {"predicate": "timezone", "object": "Europe/Berlin"},
        {"predicate": "timezone", "object": "Europe/Berlin"},
        {"predicate": "timezone", "object": "Europe/Berlin"},
        {"predicate": "works_at", "object": "ACME"},
        {"predicate": None, "object": None},
    ]
    assert harness.duplicate_memories(memories) == 2
    assert harness.duplicate_memories([]) == 0


def test_acked_counts() -> None:
    acked = harness.Acked()
    acked.messages["m"] = {}
    acked.uploads["d"] = {}
    assert acked.counts() == {"messages": 1, "observations": 0, "files": 1}
    assert acked.refused == {"messages": 0, "observations": 0, "files": 0}


# -- harness: latency report --------------------------------------------------------


def test_latency_report_has_the_performance_json_shape() -> None:
    lat = {
        "chat": [10.0, 20.0, 30.0],
        "cached": [1.0, 2.0],
        "recall": [50.0],
        "context": [80.0, 90.0],
        "file": [5.0],
    }
    statuses = {k: {202: len(v)} for k, v in lat.items()}
    budgets = {
        "chat_accept_p95_ms": 100,
        "cached_context_p95_ms": 75,
        "recall_p95_ms": 300,
        "context_bundle_p95_ms": 400,
        "file_accept_p95_ms": 200,
    }
    out = harness.latency_report(lat, statuses, budgets)
    assert set(harness.BUDGET_KEYS) <= set(out)
    assert out["chat_accept_p95_ms"] == 30.0 and out["context_bundle_p95_ms"] == 90.0
    assert out["budgets_ms"] == budgets and out["within_budget"] is True
    assert out["detail"]["chat"] == {
        "p50": 20.0,
        "p95": 30.0,
        "p99": 30.0,
        "max": 30.0,
        "samples": 3,
        "status_codes": {202: 3},
    }
    out = harness.latency_report(lat, statuses, {**budgets, "recall_p95_ms": 40})
    assert out["within_budget"] is False


def test_headers_and_scope() -> None:
    assert harness.headers("k", "t", "u") == {
        "X-API-Key": "k",
        "X-Memory-Tenant": "t",
        "X-Memory-User": "u",
    }
    scope = harness.new_scope()
    assert set(scope) == {"thread_id", "session_id", "turn_id"}


# -- deployed: helpers ---------------------------------------------------------------


def test_resolve_cmd_falls_back_to_the_interpreter_entrypoint(monkeypatch) -> None:
    monkeypatch.setattr(deployed.shutil, "which", lambda _name: None)
    argv = deployed.resolve_cmd("memory-worker", python="py")
    assert argv[:2] == ["py", "-c"] and "run_worker()" in argv[2]
    argv = deployed.resolve_cmd("memory-api --flag", python="py")
    assert "run_api()" in argv[2] and argv[-1] == "--flag"
    assert deployed.resolve_cmd("uv run memory-worker") == ["uv", "run", "memory-worker"]
    monkeypatch.setattr(deployed.shutil, "which", lambda _name: "/bin/memory-worker")
    assert deployed.resolve_cmd("memory-worker") == ["memory-worker"]
    assert deployed.resolve_cmd("memory-worker", python=sys.executable) == ["memory-worker"]


def test_network_providers_representativeness() -> None:
    version = {"providers": {"embedding": "hash:hash", "search": "qdrant", "cache": "dragonfly"}}
    out = deployed.network_providers(version)
    assert out["representative"] is False and out["search"] == "qdrant"
    version = {"providers": {"embedding": "sentence_transformers:granite", "search": "qdrant"}}
    assert deployed.network_providers(version)["representative"] is True
    version = {"providers": {"embedding": "sentence_transformers:granite", "search": "memory"}}
    assert deployed.network_providers(version)["representative"] is False
    assert deployed.network_providers({})["representative"] is False


def test_recovery_pending_until_every_condition_is_zero() -> None:
    done = {
        "jobs_open": 0,
        "observations_unprocessed": 0,
        "documents_pending": 0,
        "messages_unarchived": 0,
    }
    assert deployed.recovery_pending(done) is False
    assert deployed.recovery_pending({**done, "messages_unarchived": 3}) is True
    assert deployed.recovery_pending({}) is True


# -- load runner: CSV reduction --------------------------------------------------------

STATS = """Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
POST,POST /v1/messages,120,2,14,15.5,9,80,210.0,2.01,0.03,14,16,18,19,25,31,45,60,80,80,80
POST,POST /v1/recall,60,0,70,72.0,40,150,900.0,1.0,0.0,70,75,80,85,95,110,130,140,150,150,150
,Aggregated,180,2,20,34.3,9,150,440.0,3.01,0.03,20,40,60,70,90,105,130,140,150,150,150
"""
FAILURES = """Method,Name,Error,Occurrences
POST,POST /v1/messages,"HTTPError('429 Client Error: Too Many Requests')",2
"""


def test_parse_stats_separates_endpoints_and_aggregate() -> None:
    stats = load_run.parse_stats(STATS)
    assert set(stats["endpoints"]) == {"POST /v1/messages", "POST /v1/recall"}
    msgs = stats["endpoints"]["POST /v1/messages"]
    assert msgs["requests"] == 120 and msgs["failures"] == 2 and msgs["rps"] == 2.01
    assert (msgs["p50_ms"], msgs["p95_ms"], msgs["p99_ms"]) == (14.0, 31.0, 60.0)
    assert stats["aggregated"]["requests"] == 180 and stats["aggregated"]["p95_ms"] == 105.0
    assert load_run.parse_stats("") == {"endpoints": {}, "aggregated": {}}


def test_parse_failures() -> None:
    failures = load_run.parse_failures(FAILURES)
    assert failures == [
        {
            "method": "POST",
            "name": "POST /v1/messages",
            "error": "HTTPError('429 Client Error: Too Many Requests')",
            "occurrences": 2,
        }
    ]


def test_load_report_shape() -> None:
    report = load_run.load_report(
        STATS,
        FAILURES,
        base_url="http://api:8080",
        users=20,
        spawn_rate=5,
        run_time="60s",
        exit_status=0,
    )
    assert report["transport"] == "tcp" and report["base_url"] == "http://api:8080"
    assert report["total_requests"] == 180 and report["total_failures"] == 2
    assert report["failure_ratio"] == round(2 / 180, 4)
    assert report["endpoints"]["POST /v1/recall"]["p95_ms"] == 110.0
    assert report["failures"][0]["occurrences"] == 2
    empty = load_run.load_report(
        "", "", base_url="x", users=1, spawn_rate=1, run_time="1s", exit_status=1
    )
    assert empty["failure_ratio"] is None and empty["total_requests"] == 0
