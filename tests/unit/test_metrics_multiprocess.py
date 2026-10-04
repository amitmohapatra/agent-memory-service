"""/metrics sums every API worker process (ADR 0031).

prometheus_client decides at import time whether values live in process memory or in files
under ``PROMETHEUS_MULTIPROC_DIR``, so this runs real processes: two "workers" count
requests and a third renders, as a scrape answered by any one worker does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory_service.__main__ import MULTIPROC_ENV, prepare_multiprocess_metrics

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

_WORKER = """
from memory_service.observability.metrics import http_requests_total, model_queue_waiters
for _ in range({n}):
    http_requests_total.labels("POST", "/v1/context", "200").inc()
model_queue_waiters.labels("encoder").set({n})
"""
_SCRAPE = """
import sys
from memory_service.observability.metrics import render_metrics
sys.stdout.buffer.write(render_metrics(2)[0])
"""


def _run(code: str, directory: Path) -> str:
    env = {**os.environ, MULTIPROC_ENV: str(directory), "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    ).stdout


def test_a_scrape_sums_every_worker_process(tmp_path: Path) -> None:
    _run(_WORKER.format(n=3), tmp_path)
    _run(_WORKER.format(n=4), tmp_path)
    body = _run(_SCRAPE, tmp_path)
    assert body.startswith("# SCOPE: all 2 API worker processes")
    line = next(
        row
        for row in body.splitlines()
        if row.startswith("memory_http_requests_total{") and 'route="/v1/context"' in row
    )
    assert float(line.rsplit(" ", 1)[1]) == 7.0


def test_the_entrypoint_empties_the_directory_and_one_worker_needs_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "metrics"
    stale.mkdir()
    (stale / "counter_1.db").write_bytes(b"from the previous run")
    monkeypatch.setenv(MULTIPROC_ENV, str(stale))
    assert prepare_multiprocess_metrics(3) == str(stale)
    assert list(stale.iterdir()) == [], "a restart must not add the last run's counters"
    monkeypatch.delenv(MULTIPROC_ENV)
    assert prepare_multiprocess_metrics(1) is None
    assert MULTIPROC_ENV not in os.environ
