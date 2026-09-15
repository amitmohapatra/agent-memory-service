"""Unauthorized-retrieval gate evidence: runs ``tests/security`` (property-based and
exhaustive isolation oracles, retrieval- and graph-level matrices) and records the leak
counters. The suites assert *zero* leaks; here a failing or missing suite is recorded as
``1`` (unknown) for the corresponding counter, never as 0.

    uv run python -m benchmark.security
"""

from __future__ import annotations

import sys
from typing import Any

from benchmark.common import provenance, write_result
from benchmark.pytest_results import run_suite

# gate counter -> the tests that establish it (substrings of node ids)
COUNTERS = {
    "cross_tenant_unauthorized": (
        "test_exhaustive_cross_tenant_never_leaks",
        "test_scope_resolution_is_tenant_bound",
        "test_every_reader_gets_only_authorized_records",
        "test_graph",
    ),
    "cross_user_unauthorized": (
        "test_visibility_matches_independent_oracle",
        "test_every_reader_gets_only_authorized_records",
        "test_isolation_holds_per_retriever_and_with_document_filter",
        "test_empty_visibility_returns_nothing",
    ),
    "private_agent_leakage": (
        "test_private_agent_memory_invisible_to_everyone_else",
        "test_agent_inherits_user_access_but_not_user_private_memories",
        "test_every_reader_gets_only_authorized_records",
    ),
}


def reduce(by_test: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for counter, needles in COUNTERS.items():
        relevant = {n: o for n, o in by_test.items() if any(s in n for s in needles)}
        missing = [s for s in needles if not any(s in n for n in relevant)]
        ok = bool(relevant) and not missing and all(o == "passed" for o in relevant.values())
        out[counter] = 0 if ok else 1
        out[f"{counter}_evidence"] = {
            "tests": {n.rsplit("::", 1)[-1]: o for n, o in relevant.items()},
            "missing": missing,
        }
    return out


def main() -> int:
    code, collector = run_suite(["tests/security", "-m", "security"])
    payload = {
        **reduce(collector.by_test),
        "tests": collector.counts(),
        "pytest_exit_status": code,
        "provenance": provenance(),
    }
    path = write_result("security.json", payload)
    print(f"wrote {path}")
    for counter in COUNTERS:
        print(f"{counter:28} {payload[counter]}")
    return 0 if all(payload[c] == 0 for c in COUNTERS) else 1


if __name__ == "__main__":
    sys.exit(main())
