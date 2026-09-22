"""The load artifact has to carry the one number that travels between machines.

A throughput figure is about the box it was measured on: ten requests a second on four
cores says nothing about eight. Core-seconds per request does travel - multiply it by a
target rate and it sizes any machine - so the runner samples the container's own CPU
during the run and divides. These tests pin that arithmetic and the parsing around it,
because the sampler is best-effort by design and must degrade to "absent", never to zero.
"""

from __future__ import annotations

from benchmark.load.run import _ResourceSampler, _run_seconds, parse_stats

STATS_CSV = """Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
POST,POST /v1/context,600,0,140,150,90,800,2048,10.0,0.0,140,160,170,180,220,260,300,320,700,800,800
,Aggregated,900,0,120,130,40,800,1800,15.0,0.0,120,150,160,170,210,250,290,310,700,800,800
"""


def test_percentiles_and_rates_are_read_per_endpoint_and_in_aggregate() -> None:
    parsed = parse_stats(STATS_CSV)
    context = parsed["endpoints"]["POST /v1/context"]
    assert context["requests"] == 600 and context["failures"] == 0
    assert context["p50_ms"] == 140 and context["p95_ms"] == 260 and context["p99_ms"] == 320
    assert parsed["aggregated"]["rps"] == 15.0


def test_core_seconds_per_request_is_cpu_time_over_requests() -> None:
    """200% of one core for 300 s over 600 requests is one core-second per request."""
    sampler = _ResourceSampler([])
    sampler.samples = {"memory-api": [(200.0, 500.0)] * 10}
    report = sampler.report(requests=600, seconds=300.0)["memory-api"]
    assert report["core_seconds_per_request"] == 1.0
    assert report["cpu_percent_mean"] == 200.0
    assert report["rss_mib_max"] == 500.0


def test_memory_drift_is_reported_as_a_percentage_of_the_first_sample() -> None:
    sampler = _ResourceSampler([])
    sampler.samples = {"memory-api": [(10.0, 1000.0), (10.0, 1100.0)]}
    assert sampler.report(requests=1, seconds=1.0)["memory-api"]["rss_drift_percent"] == 10.0


def test_a_container_that_could_not_be_sampled_is_absent_not_zero() -> None:
    """A box with no docker socket must not publish 0 % CPU as if it had measured it."""
    sampler = _ResourceSampler([])
    sampler.samples = {"memory-api": []}
    report = sampler.report(requests=100, seconds=60.0)["memory-api"]
    assert report == {"samples": 0}
    assert "core_seconds_per_request" not in report


def test_no_requests_means_no_per_request_cost() -> None:
    sampler = _ResourceSampler([])
    sampler.samples = {"memory-api": [(100.0, 10.0)]}
    assert "core_seconds_per_request" not in sampler.report(requests=0, seconds=60.0)["memory-api"]


def test_run_time_accepts_the_units_locust_does() -> None:
    assert _run_seconds("300s") == 300
    assert _run_seconds("5m") == 300
    assert _run_seconds("1h") == 3600
    assert _run_seconds("90") == 90
    assert _run_seconds("nonsense") == 0.0
