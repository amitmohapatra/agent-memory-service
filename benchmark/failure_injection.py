"""Failure-injection gate evidence: runs ``tests/failure`` and records, per scenario,
whether every test of that scenario passed. A scenario with no test, a skipped test or
any failure is recorded as ``fail`` — the release gate needs an explicit ``pass``.

    uv run python -m benchmark.failure_injection
"""

from __future__ import annotations

import sys
from typing import Any

from benchmark.common import provenance, write_result
from benchmark.pytest_results import run_suite

SCENARIOS = ("worker_kill", "cache_flush", "blob_outage", "search_rebuild", "authz_denial")


def reduce(by_test: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario in SCENARIOS:
        tests = {n: o for n, o in by_test.items() if f"::test_{scenario}" in n}
        if not tests:
            out[scenario] = "fail"
            out[f"{scenario}_detail"] = "no test found for this scenario"
            continue
        out[scenario] = "pass" if all(o == "passed" for o in tests.values()) else "fail"
        out[f"{scenario}_detail"] = {n.rsplit("::", 1)[-1]: o for n, o in tests.items()}
    return out


def main() -> int:
    code, collector = run_suite(["tests/failure", "-m", "failure"])
    payload = {
        **reduce(collector.by_test),
        "tests": collector.counts(),
        "pytest_exit_status": code,
        "provenance": provenance(),
    }
    path = write_result("failure_injection.json", payload)
    print(f"wrote {path}")
    for scenario in SCENARIOS:
        print(f"{scenario:16} {payload[scenario]}")
    return 0 if all(payload[s] == "pass" for s in SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(main())
