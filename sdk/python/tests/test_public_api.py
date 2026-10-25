"""The SDK's public API is a contract the harness and other clients build against: a change to
any exported signature, model field or enum member fails here until the snapshot is
regenerated — which ``scripts/sdk_api.py --update`` does only under a new SDK version that the
CHANGELOG names (``scripts/sdk_api.py --check-base`` holds the same against the base branch in
CI)."""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from scripts import sdk_api as public_api


def test_the_public_api_is_the_committed_snapshot() -> None:
    kept = public_api.stored()
    now = public_api.snapshot()
    differs = public_api.changes(kept["api"], now)
    assert not differs, (
        "the public API of trellis-memory changed: "
        + "; ".join(differs)
        + ". Bump sdk/python/pyproject.toml, add a CHANGELOG entry naming the new version, then "
        "run: python scripts/sdk_api.py --update"
    )


def test_the_snapshot_version_is_released_and_named_in_the_changelog() -> None:
    kept = public_api.stored()
    assert kept["version"] == public_api.version(), (
        "the snapshot was taken under another SDK version: run python scripts/sdk_api.py --update"
    )
    assert public_api.named_in_changelog(public_api.CHANGELOG.read_text(), kept["version"])


def test_every_export_is_described() -> None:
    import trellis.memory as sdk

    assert set(public_api.snapshot()) == set(sdk.__all__)


# --------------------------------------------------------------------------- the description


class _Colour(enum.Enum):
    RED = "red"


class _Model(BaseModel):
    name: str
    count: int = 3
    tags: list[str] = []


def test_each_kind_of_export_is_described_by_what_callers_depend_on() -> None:
    assert public_api.describe(_Colour) == {"kind": "enum", "members": {"RED": "'red'"}}
    model = public_api.describe(_Model)
    assert model["kind"] == "model"
    assert model["fields"] == {
        "name": "str (required)",
        "count": "int = 3",
        "tags": "list[str] = []",
    }
    assert public_api.describe(len) == {"kind": "function", "signature": "(obj, /)"}
    assert public_api.describe(42) == {"kind": "value", "value": "42"}


def test_a_signature_change_is_a_change() -> None:
    def before(a: int) -> None: ...

    def after(a: int, b: int = 0) -> None: ...

    old = {"f": public_api.describe(before)}
    new = {"f": public_api.describe(after), "g": {"kind": "value", "value": "1"}}
    assert public_api.changes(old, new) == ["changed: f", "added: g"]
    assert public_api.changes(new, old) == ["changed: f", "removed: g"]
    assert public_api.changes(old, old) == []


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("## 2026-10-04: SDK `trellis-memory` 0.4.0 (service unchanged)", True),
        ("- SDK 0.5.0: a new verb", True),
        ("- trellis-memory 0.5.0", True),
        ("- SDK 0.4.01 is something else", False),
        ("the service 0.5.0", False),
        ("", False),
    ],
)
def test_the_changelog_names_a_version_as_the_sdks(text: str, named: bool) -> None:
    version = "0.4.0" if "0.4.0" in text or "0.4.01" in text else "0.5.0"
    assert public_api.named_in_changelog(text, version) is named


def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, api: Any, version: str) -> Path:
    snapshot = tmp_path / "public_api.json"
    pyproject = tmp_path / "pyproject.toml"
    changelog = tmp_path / "CHANGELOG.md"
    pyproject.write_text(f'[project]\nname = "trellis-memory"\nversion = "{version}"\n')
    changelog.write_text("# Changelog\n")
    monkeypatch.setattr(public_api, "SNAPSHOT", snapshot)
    monkeypatch.setattr(public_api, "PYPROJECT", pyproject)
    monkeypatch.setattr(public_api, "CHANGELOG", changelog)
    monkeypatch.setattr(public_api, "snapshot", lambda: api)
    return snapshot


def test_update_refuses_a_changed_api_under_the_same_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _isolated(monkeypatch, tmp_path, api={"f": {"kind": "value"}}, version="0.4.0")
    snapshot.write_text(json.dumps({"version": "0.4.0", "api": {}}))
    with pytest.raises(SystemExit, match=r"still 0\.4\.0"):
        public_api.update()
    assert json.loads(snapshot.read_text())["api"] == {}


def test_update_refuses_a_version_the_changelog_does_not_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _isolated(monkeypatch, tmp_path, api={"f": {"kind": "value"}}, version="0.5.0")
    snapshot.write_text(json.dumps({"version": "0.4.0", "api": {}}))
    with pytest.raises(SystemExit, match=r"names no trellis-memory 0\.5\.0"):
        public_api.update()


def test_update_writes_a_new_version_the_changelog_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = {"f": {"kind": "value", "value": "1"}}
    snapshot = _isolated(monkeypatch, tmp_path, api=api, version="0.5.0")
    snapshot.write_text(json.dumps({"version": "0.4.0", "api": {}}))
    public_api.CHANGELOG.write_text("## SDK `trellis-memory` 0.5.0\n")
    assert public_api.update() == "written: the public API of trellis-memory 0.5.0"
    assert json.loads(snapshot.read_text()) == {"version": "0.5.0", "api": api}
    assert public_api.update() == "unchanged: the public API of trellis-memory 0.5.0"


def test_a_first_snapshot_needs_its_version_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _isolated(monkeypatch, tmp_path, api={}, version="1.0.0")
    with pytest.raises(SystemExit):
        public_api.update()
    public_api.CHANGELOG.write_text("- trellis-memory 1.0.0\n")
    public_api.update()
    assert json.loads(snapshot.read_text())["version"] == "1.0.0"


def _based(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    base: dict[str, Any] | None,
    base_version: str = "0.4.0",
    version: str = "0.4.0",
    named: bool = False,
) -> None:
    """The committed snapshot (``{"f": 2}`` under ``version``) against a base branch's."""
    snapshot = _isolated(monkeypatch, tmp_path, api={}, version=version)
    snapshot.write_text(json.dumps({"version": version, "api": {"f": 2}}))
    if named:
        public_api.CHANGELOG.write_text(f"- SDK {version}: f changed\n")
    files = {
        public_api.SNAPSHOT: None if base is None else json.dumps(base),
        public_api.PYPROJECT: f'[project]\nversion = "{base_version}"\n',
    }
    monkeypatch.setattr(public_api, "_at", lambda ref, path: files[path])


def test_the_base_branch_without_a_snapshot_is_not_compared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _based(monkeypatch, tmp_path, base=None)
    assert public_api.check_base("origin/main") == "no snapshot at origin/main: nothing to compare"


def test_an_unchanged_api_passes_against_the_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _based(monkeypatch, tmp_path, base={"version": "0.4.0", "api": {"f": 2}})
    assert public_api.check_base("origin/main") == "the public API is origin/main's"


def test_a_snapshot_regenerated_under_the_base_version_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _based(monkeypatch, tmp_path, base={"version": "0.4.0", "api": {"f": 1}}, named=True)
    with pytest.raises(SystemExit, match=r"changed: f.*still 0\.4\.0"):
        public_api.check_base("origin/main")


def test_a_new_version_needs_its_changelog_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = {"version": "0.4.0", "api": {"f": 1}}
    _based(monkeypatch, tmp_path, base=base, version="0.5.0")
    with pytest.raises(SystemExit, match=r"names no trellis-memory 0\.5\.0"):
        public_api.check_base("origin/main")
    _based(monkeypatch, tmp_path, base=base, version="0.5.0", named=True)
    assert public_api.check_base("origin/main").endswith("trellis-memory 0.5.0: allowed")


def test_a_ref_git_does_not_have_reads_as_nothing() -> None:
    assert public_api._at("no-such-ref-anywhere", public_api.SNAPSHOT) is None


def test_an_annotation_reads_the_same_before_and_after_its_references_resolve() -> None:
    from typing import ForwardRef

    assert public_api._annotation(ForwardRef("list[SkillView]")) == "list[SkillView]"
    assert public_api._annotation(list[Path]) == "list[Path]"
    assert public_api._annotation(int) == "int"
