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
        seen["built"] = True
        return container

    async def pending(settings, *, state, tenant, user):
        return (1, 1)

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
    monkeypatch.setattr(module, "pending_users", pending)
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
    assert seen == {"built": True, "state": state, "tenant": "acme", "user": "u1", "batch": 7}
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


def test_with_no_user_left_it_exits_0_before_loading_a_model(wired, monkeypatch, capsys):
    """A fresh install (no user), or every user finished at this revision: compose runs the
    backfill on every ``up``, and it must not load the models to find there is nothing."""
    _, seen = wired

    async def nobody(settings, *, state, tenant, user):
        return (3, 0)

    monkeypatch.setattr(module, "pending_users", nobody)
    assert _run(monkeypatch, "--state", "/nonexistent/state.json") == 0
    assert "built" not in seen and "state" not in seen
    report = json.loads(capsys.readouterr().out)
    assert report["users"] == 0 and report["skipped_users"] == 3 and report["failures"] == []


def test_reindex_all_runs_even_with_no_user_left(wired, monkeypatch, capsys):
    _, seen = wired

    async def nobody(settings, *, state, tenant, user):
        return (0, 0)

    async def rebuild(container_, tenant_id=None):
        return BackfillReport()

    import memory_service.tools.reindex as reindex

    monkeypatch.setattr(module, "pending_users", nobody)
    monkeypatch.setattr(reindex, "rebuild_search_index", rebuild)
    assert _run(monkeypatch, "--reindex-all") == 0
    assert seen["built"]
    capsys.readouterr()


# ------------------------------------------------------------------ the checkpoint's revision


def test_a_state_from_another_revision_starts_over(tmp_path):
    from memory_service.tools.backfill import BACKFILL_REVISION, Checkpoint

    state = tmp_path / "state.json"
    old = Checkpoint(state, revision=BACKFILL_REVISION - 1)
    old.finish("acme", "user:u1")
    old.progress("acme", "user:u2", "2026-10-01T00:00:00+00:00")
    reread = Checkpoint(state)
    assert not reread.finished("acme", "user:u1")
    assert reread.resume_after("acme", "user:u2") is None
    reread.finish("acme", "user:u3")
    assert json.loads(state.read_text()) == {
        "revision": BACKFILL_REVISION,
        "done": [["acme", "user:u3"]],
        "current": None,
    }


def test_a_state_written_before_revisions_were_recorded_starts_over(tmp_path):
    from memory_service.tools.backfill import Checkpoint

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"done": [["acme", "user:u1"]], "current": None}))
    assert not Checkpoint(state).finished("acme", "user:u1")


def test_a_state_from_this_revision_skips_the_users_it_finished(tmp_path):
    from memory_service.tools.backfill import Checkpoint

    state = tmp_path / "nested" / "state.json"  # a directory not made yet (a fresh volume)
    first = Checkpoint(state)
    first.finish("acme", "user:u1")
    first.progress("acme", "user:u2", "2026-10-01T00:00:00+00:00")
    again = Checkpoint(state)
    assert again.finished("acme", "user:u1") and not again.finished("acme", "user:u2")
    assert again.resume_after("acme", "user:u2") == "2026-10-01T00:00:00+00:00"


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
