"""Overload cannot masquerade as sustained throughput or disappear from the denominator."""

import asyncio

import pytest
from benchmark.cpu_capacity import cpu_projection, fixed_arrivals

pytestmark = pytest.mark.unit


async def test_full_queue_counts_dropped_arrivals_and_keeps_work_bounded():
    active = peak = 0

    async def slow(_):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.04)
        active -= 1

    result = await fixed_arrivals(slow, rate=1000, requests=10, max_pending=2, deadline_seconds=1)
    assert peak <= 2
    assert len(result["rows"]) == 10
    assert any(row["status"] == "overload" for row in result["rows"])
    assert result["successful_rps_including_drain"] < 1000
    assert result["failure_ratio"] > 0


async def test_errors_and_timeouts_are_not_successful_capacity():
    async def call(index):
        if index == 0:
            raise RuntimeError("fixture")
        await asyncio.sleep(0.03)

    result = await fixed_arrivals(
        call, rate=1000, requests=2, max_pending=2, deadline_seconds=0.005
    )
    assert [row["status"] for row in result["rows"]] == ["error", "timeout"]
    assert result["successful_requests"] == 0
    assert result["successful_latency_ms"] is None
    assert result["cpu_seconds_per_success"] is None


def test_projection_uses_cpu_demand_and_headroom_without_certifying_endpoint():
    result = cpu_projection(0.02, rate=20, cores=8)
    assert result["encoder_core_demand"] == pytest.approx(0.4)
    assert result["cores_remaining_with_headroom"] == pytest.approx(5.2)
    assert result["endpoint_capacity_established"] is False
    assert cpu_projection(0.3, rate=20, cores=8)["cores_remaining_with_headroom"] < 0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_cpu_measurement_cannot_produce_a_capacity_claim(value):
    with pytest.raises(ValueError):
        cpu_projection(value, rate=20, cores=8)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
async def test_invalid_arrival_rate_fails_before_starting_work(rate):
    async def must_not_run(_):
        pytest.fail("Invalid workload must not start inference")

    with pytest.raises(ValueError):
        await fixed_arrivals(
            must_not_run, rate=rate, requests=1, max_pending=1, deadline_seconds=1
        )
    with pytest.raises(ValueError):
        cpu_projection(0.01, rate=rate, cores=8)
