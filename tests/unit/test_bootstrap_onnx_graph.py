"""The bootstrap must leave behind everything the frozen runtime needs to start.

The hub publishes no ONNX graph for granite-embedding-small-english-r2, so the graph the ONNX
runner loads is exported from the downloaded checkpoint. That export used to be a separate
command nothing called: the ``--dir`` bootstrap that docker-compose runs downloaded the
weights and returned. Flipping ``DenseModel.runtime`` to ``onnx`` would therefore have left a
fresh ``docker compose up`` raising DependencyUnavailable on a file nothing ever wrote -
which is exactly the difference between one package that starts by plain docker compose and
one that does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.config.constants import FROZEN_MODELS
from memory_service.tools import download_models

pytestmark = pytest.mark.unit


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record what would have been exported; the real export needs torch and onnx."""
    calls: list[Path] = []

    def _fake(directory: Path) -> int:
        calls.append(directory)
        return 0

    monkeypatch.setattr(download_models, "export_onnx", _fake)
    return calls


def _spec(**overrides: object):
    return FROZEN_MODELS.dense.model_copy(update=overrides)


def test_the_torch_runtime_exports_nothing(tmp_path: Path, exported: list[Path]) -> None:
    """The flip has not happened yet, so this must be inert on today's default."""
    assert download_models.ensure_dense_graph(tmp_path, _spec(runtime="torch")) == 0
    assert exported == []


def test_the_onnx_runtime_exports_the_graph_the_runner_will_open(
    tmp_path: Path, exported: list[Path]
) -> None:
    spec = _spec(runtime="onnx")
    assert download_models.ensure_dense_graph(tmp_path, spec) == 0
    assert exported == [tmp_path / spec.local_dir], "the bootstrap left no graph to load"


def test_an_existing_graph_is_not_exported_twice(tmp_path: Path, exported: list[Path]) -> None:
    """The bootstrap runs on every `docker compose up`; it must be idempotent."""
    spec = _spec(runtime="onnx")
    graph = tmp_path / spec.local_dir / "onnx" / "model.onnx"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"a graph")
    assert download_models.ensure_dense_graph(tmp_path, spec) == 0
    assert exported == []


def test_a_named_graph_file_is_the_one_checked(tmp_path: Path, exported: list[Path]) -> None:
    """int8 and fp32 are different vector spaces, so presence of one is not the other."""
    spec = _spec(runtime="onnx", graph_file="onnx/model_qint8.onnx")
    fp32 = tmp_path / spec.local_dir / "onnx" / "model.onnx"
    fp32.parent.mkdir(parents=True)
    fp32.write_bytes(b"the wrong graph")
    assert download_models.ensure_dense_graph(tmp_path, spec) == 0
    assert exported == [tmp_path / spec.local_dir], "an fp32 graph does not satisfy int8"


def test_a_failed_export_is_reported_to_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap that could not write the graph must not exit zero and look successful."""
    monkeypatch.setattr(download_models, "export_onnx", lambda directory: 2)
    assert download_models.ensure_dense_graph(tmp_path, _spec(runtime="onnx")) == 2
