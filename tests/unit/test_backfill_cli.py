"""The backfill command line (``python -m memory_service.tools.backfill``): its arguments
reach the run, its report is printed as JSON, a failure is a non-zero exit, and the container
is closed whatever happens."""

from __future__ import annotations

import json
import sys

import pytest

from memory_service.tools import backfill as module
from memory_service.tools.backfill import BackfillReport

pytestmark = pytest.mark.unit


class _Container:
    closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def wired(monkeypatch):
    container = _Container()
    seen: dict = {}

    async def build(settings, version):
        return container

    async def run(container_, **kwargs):
        seen.update(kwargs)
        report = BackfillReport()
        report.failures = list(seen.pop("fail", []))
        return report

    import memory_service.application.container as app
    import memory_service.modules.jobs.registry as registry

    monkeypatch.setattr(app, "build_container", build)
    monkeypatch.setattr(registry, "register_handlers", lambda c: None)
    monkeypatch.setattr(module, "backfill", run)
    return container, seen


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["backfill", *argv])
    return module.main()


def test_arguments_reach_the_run_and_the_report_is_printed(wired, monkeypatch, capsys, tmp_path):
    container, seen = wired
    state = tmp_path / "b.json"
    code = _run(
        monkeypatch, "--state", str(state), "--tenant", "acme", "--user", "u1", "--batch", "7"
    )
    assert code == 0 and container.closed
    assert seen == {"state": state, "tenant": "acme", "user": "u1", "batch": 7}
    assert json.loads(capsys.readouterr().out)["failures"] == []


def test_without_a_state_file_the_run_has_none(wired, monkeypatch, capsys):
    _, seen = wired
    assert _run(monkeypatch) == 0
    assert seen["state"] is None and seen["batch"] == module.BATCH
    capsys.readouterr()


def test_reindex_all_adds_the_rebuild_failures_and_fails(wired, monkeypatch, capsys):
    container, _ = wired

    class _Rebuilt:
        failures = ["document acme/d1: boom"]

    async def rebuild(container_, tenant_id=None):
        assert tenant_id == "acme"
        return _Rebuilt()

    import memory_service.tools.reindex as reindex

    monkeypatch.setattr(reindex, "rebuild_search_index", rebuild)
    assert _run(monkeypatch, "--tenant", "acme", "--reindex-all") == 1
    assert json.loads(capsys.readouterr().out)["failures"] == ["document acme/d1: boom"]
    assert container.closed


def test_the_container_is_closed_when_the_run_raises(wired, monkeypatch):
    container, _ = wired

    async def boom(container_, **kwargs):
        raise RuntimeError("db gone")

    monkeypatch.setattr(module, "backfill", boom)
    with pytest.raises(RuntimeError):
        _run(monkeypatch)
    assert container.closed
