"""One thread count, shared by every in-process model: this worker's share of the cores.

``torch.set_num_threads`` is process-wide, so the last model to load decides it for all of
them; one resolved number, given to each, is the only way it means anything. It is derived,
not configured: the cores the host has, divided among the worker processes it runs.
"""

from __future__ import annotations

import os

import pytest

from memory_service.adapters.wiring import _model_threads
from memory_service.config.constants import FROZEN_MODELS

pytestmark = pytest.mark.unit


class _Container:
    def __init__(self, workers: int) -> None:
        class _Service:
            pass

        class _Settings:
            pass

        service, settings = _Service(), _Settings()
        service.workers = workers  # type: ignore[attr-defined]
        settings.service = service  # type: ignore[attr-defined]
        self.settings = settings


def test_each_worker_gets_its_share_of_the_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _model_threads(_Container(3)) == 2  # type: ignore[arg-type]
    assert _model_threads(_Container(1)) == 8  # type: ignore[arg-type]


def test_the_measured_box_resolves_to_the_frozen_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The encoder was frozen with two threads on the 8-vCPU, three-worker target."""
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _model_threads(_Container(3)) == FROZEN_MODELS.dense.threads  # type: ignore[arg-type]


def test_never_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _model_threads(_Container(8)) == 1  # type: ignore[arg-type]
