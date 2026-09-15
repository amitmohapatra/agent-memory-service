"""pytest plugin + helpers that turn test outcomes into gate evidence.

As a plugin (``pytest -p benchmark.pytest_results``) it writes ``tests.json`` — the
"all tests pass" gate — at session end. As a library (``run_suite``) it runs a marked
suite in-process and returns the outcome per test, which ``failure_injection.py`` and
``security.py`` reduce to their scenario/leak counters. Outcomes are recorded, never
asserted into shape: a failing or *missing* scenario is a failed gate.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from benchmark.common import provenance, write_result

OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")


class OutcomeCollector:
    """Records the terminal outcome of every test (setup errors count as ``error``)."""

    def __init__(self) -> None:
        self.by_test: dict[str, str] = {}
        self.deselected = 0
        self.started = time.time()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" or (report.when != "call" and report.outcome != "passed"):
            outcome = report.outcome
            if report.when != "call" and outcome == "failed":
                outcome = "error"
            if hasattr(report, "wasxfail"):
                outcome = "xpassed" if outcome == "passed" else "xfailed"
            self.by_test[report.nodeid] = outcome

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected += len(items)

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(OUTCOMES, 0)
        for outcome in self.by_test.values():
            out[outcome] = out.get(outcome, 0) + 1
        out["errors"] = out.pop("error")
        out["deselected"] = self.deselected
        out["total"] = len(self.by_test)
        return out


def run_suite(args: list[str]) -> tuple[int, OutcomeCollector]:
    """Run pytest in-process with the collector attached. Returns (exit code, collector)."""
    collector = OutcomeCollector()
    code = pytest.main([*args, "-p", "no:cacheprovider", "-q", "-q"], plugins=[collector])
    return int(code), collector


# -- plugin mode: `pytest -p benchmark.pytest_results` ------------------------------
_collector: OutcomeCollector | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _collector  # noqa: PLW0603
    if _collector is None and not hasattr(config, "workerinput"):
        _collector = OutcomeCollector()
        config.pluginmanager.register(_collector, "memory-service-outcomes")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _collector is None or hasattr(session.config, "workerinput"):
        return
    counts = _collector.counts()
    write_result(
        "tests.json",
        {
            **counts,
            "exit_status": int(exitstatus),
            "duration_seconds": round(time.time() - _collector.started, 1),
            "args": list(session.config.invocation_params.args),
            "failed_tests": sorted(
                n for n, o in _collector.by_test.items() if o in ("failed", "error")
            ),
            "provenance": provenance(),
        },
    )
