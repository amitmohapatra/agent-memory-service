"""Where the local model weights are, and what can actually run here.

Two different reasons a model test cannot run, which were previously reported as one:

* the **weights** are not downloaded (``make models``), or
* the **runtime** cannot be installed on this machine — torch and onnxruntime publish no
  macOS x86_64 wheels, so on an Intel Mac the ``[models]`` extra is unavailable at any
  version. Those tests run in the Linux runtime image instead: ``make model-test``.

fastembed-based adapters (SPLADE, ColBERT) need only onnxruntime and do run natively, so
they should not be swept into the same skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def models_dir() -> Path | None:
    """The weights directory: ``MEMORY_MODELS_DIR`` if set, else ``./models`` if present."""
    if env := os.environ.get("MEMORY_MODELS_DIR"):
        return Path(env)
    default = _REPO / "models"
    return default if default.is_dir() else None


MODELS_DIR = models_dir()

NO_WEIGHTS = "no model weights — run `make models` or set MEMORY_MODELS_DIR"
NO_RUNTIME = (
    "the 'models' extra is not installed (torch/onnxruntime publish no macOS x86_64 wheels) "
    "— run `make model-test` to run this in the Linux runtime image"
)


def requires_weights(*names: str) -> Path:
    """Skip unless every named model directory is present; returns the weights root."""
    if MODELS_DIR is None:
        pytest.skip(NO_WEIGHTS)
    for name in names:
        if not (MODELS_DIR / name).exists():
            pytest.skip(f"{NO_WEIGHTS} ({name} missing)")
    return MODELS_DIR


def requires_torch() -> None:
    """Skip unless the torch-based runtime is importable here."""
    pytest.importorskip("torch", reason=NO_RUNTIME)
    pytest.importorskip("transformers", reason=NO_RUNTIME)


def requires_sentence_transformers() -> None:
    requires_torch()
    pytest.importorskip("sentence_transformers", reason=NO_RUNTIME)
