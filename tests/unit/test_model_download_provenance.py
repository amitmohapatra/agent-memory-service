"""Cached model bytes cannot acquire the identity of a newer upstream checkpoint."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memory_service.tools import download_models as downloader

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("pin,expected_cached", [(None, True), ("old", True), ("new", False)])
def test_cached_weights_keep_recorded_revision_or_fetch_requested_pin(
    tmp_path, monkeypatch, pin, expected_cached
):
    model = downloader.Model("fixture", "test/model", "embedding", "fixture", revision=pin)
    monkeypatch.setattr(downloader, "MODELS", (model,))
    target = tmp_path / model.directory
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"fixture")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({model.directory: {"repo": model.repo, "revision": "old", "role": "embedding"}})
    )
    api = SimpleNamespace(model_info=Mock(return_value=SimpleNamespace(sha="immutable-new")))
    download = Mock()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: api,
            snapshot_download=download,
        ),
    )
    result = downloader.fetch(model, tmp_path)
    assert result["cached"] == expected_cached
    if expected_cached:
        assert result["revision"] == "old"
        api.model_info.assert_not_called()
        download.assert_not_called()
    else:
        assert result["revision"] == "immutable-new"
        assert download.call_args.kwargs["revision"] == result["revision"]
        api.model_info.assert_called_once_with(model.repo, revision="new")


def test_unidentified_existing_weights_are_not_claimed_as_latest(tmp_path, monkeypatch):
    model = downloader.Model("fixture", "test/model", "embedding", "fixture")
    monkeypatch.setattr(downloader, "MODELS", (model,))
    target = tmp_path / model.directory
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"unknown")
    api = SimpleNamespace(model_info=Mock(return_value=SimpleNamespace(sha="fetched")))
    download = Mock()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: api,
            snapshot_download=download,
        ),
    )
    assert downloader.fetch(model, tmp_path)["cached"] is False
    assert download.call_args.kwargs["revision"] == "fetched"


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_refresh_invalidates_exported_graph_before_download(
    tmp_path, monkeypatch, force
):
    model = downloader.Model("fixture", "test/model", "embedding", "fixture", revision="new")
    monkeypatch.setattr(downloader, "MODELS", (model,))
    target = tmp_path / model.directory
    graph_dir = target / "onnx"
    graph_dir.mkdir(parents=True)
    for name in ("model.onnx", "model_qint8.onnx", "model.onnx_data"):
        (graph_dir / name).write_bytes(b"old graph")
    (target / "model.safetensors").write_bytes(b"old weights")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({model.directory: {"repo": model.repo, "revision": "new" if force else "old"}})
    )

    def download(*args, **kwargs):
        assert not list(graph_dir.glob("*.onnx*"))
        raise OSError("interrupted checkpoint download")

    api = SimpleNamespace(model_info=Mock(return_value=SimpleNamespace(sha="new")))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: api,
            snapshot_download=download,
        ),
    )
    with pytest.raises(OSError, match="interrupted"):
        downloader.fetch(model, tmp_path, force=force)
    assert not list(graph_dir.glob("*.onnx*"))


def test_interrupted_refresh_is_retried_even_when_old_manifest_matches(tmp_path, monkeypatch):
    model = downloader.Model("fixture", "test/model", "embedding", "fixture")
    monkeypatch.setattr(downloader, "MODELS", (model,))
    target = tmp_path / model.directory
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"possibly mixed weights")
    (target / ".download-in-progress").write_text("new")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({model.directory: {"repo": model.repo, "revision": "old"}})
    )
    api = SimpleNamespace(model_info=Mock(return_value=SimpleNamespace(sha="new")))
    download = Mock()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: api,
            snapshot_download=download,
        ),
    )
    result = downloader.fetch(model, tmp_path)
    assert result["revision"] == "new" and not result["cached"]
    download.assert_called_once()
    assert (target / ".download-in-progress").exists()
    downloader.commit_manifest(tmp_path, {model.directory: result}, model)
    assert not (target / ".download-in-progress").exists()


def test_manifest_commit_failure_keeps_old_provenance_and_incomplete_marker(tmp_path, monkeypatch):
    model = downloader.Model("fixture", "test/model", "embedding", "fixture")
    target = tmp_path / model.directory
    target.mkdir()
    marker = target / ".download-in-progress"
    marker.write_text("new")
    prior = {model.directory: {"repo": model.repo, "revision": "old"}}
    current = {model.directory: {"repo": model.repo, "revision": "new"}}
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(prior))

    def interrupted(*args):
        raise OSError("interrupted publication")

    monkeypatch.setattr(downloader.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        downloader.commit_manifest(tmp_path, current, model)
    assert json.loads(path.read_text()) == prior
    assert marker.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["MANIFEST.json", "fixture"]
