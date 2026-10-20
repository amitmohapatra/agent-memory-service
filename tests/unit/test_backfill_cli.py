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


# ------------------------------------------------------------------ stale rule labels


def _stored(content: str, **meta):
    from datetime import UTC, datetime

    from memory_service.domain.enums import MemoryType
    from memory_service.domain.memory import CanonicalMemory

    return CanonicalMemory.model_validate(
        {
            "tenant_id": "acme",
            "content": content,
            "memory_type": MemoryType.USER,
            "lifetime": "LONG_TERM",
            "visibility": "USER",
            "scope": {"level": "USER", "tenant_id": "acme", "user_id": "u1"},
            "owner_principal": "user:u1",
            "normalized_hash": content,
            "temporal": {"observed_at": datetime.now(UTC)},
            "evidence": [
                {"source_type": "message", "source_id": "msg_1", "observed_at": datetime.now(UTC)}
            ],
            "system_metadata": meta,
        }
    )


@pytest.mark.parametrize(
    "meta",
    [
        {"category": "rule", "statement_kind": "RULE", "standing_instruction": "yes"},
        {"category": "verbatim_turn", "statement_kind": "RULE"},
    ],
)
def test_a_rule_label_this_lexicon_no_longer_reads_becomes_a_fact(meta):
    from memory_service.tools.backfill import relabelled

    fixed = relabelled(_stored("आज से ऑन-कॉल पेजर Ugo Mbeki की जगह Zora Zabala संभालेंगे।", **meta))
    assert fixed is not None
    assert fixed.system_metadata["statement_kind"] == "FACT"
    assert "standing_instruction" not in fixed.system_metadata
    assert fixed.system_metadata["category"] == (
        "fact" if meta["category"] == "rule" else meta["category"]
    )


@pytest.mark.parametrize(
    ("content", "meta"),
    [
        (
            "Always answer stock audits as a markdown table.",
            {"category": "rule", "statement_kind": "RULE"},
        ),
        (
            "Never book deliveries on Sundays.",
            {"category": "verbatim_turn", "statement_kind": "RULE"},
        ),
        ("Our hero style is SL641114.", {"category": "verbatim_turn", "statement_kind": "FACT"}),
    ],
)
def test_a_rule_this_lexicon_reads_or_a_fact_is_left_alone(content, meta):
    from memory_service.tools.backfill import relabelled

    assert relabelled(_stored(content, **meta)) is None


def test_only_a_deployment_without_a_model_relabels():
    from types import SimpleNamespace

    from memory_service.tools.backfill import without_a_model

    def container(url):
        return SimpleNamespace(settings=SimpleNamespace(llm=SimpleNamespace(base_url=url)))

    assert without_a_model(container(None)) and not without_a_model(container("http://gw/v1"))
    assert not without_a_model(SimpleNamespace())
