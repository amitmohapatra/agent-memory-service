"""The examples launcher: its stand-ins are ``Overrides`` on the container, never settings,
and the model stand-ins follow the weights."""

from __future__ import annotations

from pathlib import Path

import pytest
from examples import serve

from memory_service.config import constants

pytestmark = pytest.mark.unit


def test_stand_ins_replace_the_stores_a_single_process_never_talks_to() -> None:
    with_weights = serve.stand_ins(weights=True)
    assert (with_weights.search, with_weights.tasks, with_weights.authorization) == (
        "memory",
        "inline",
        "memory",
    )
    assert with_weights.document_parser == "builtin"
    assert with_weights.embedding is None and with_weights.nli is None
    assert with_weights.reranker is None
    # the tuning is the shipped tuning: an example must not run a different retriever
    assert with_weights.retrieval is None and with_weights.context is None


def test_model_stand_ins_only_without_the_weights() -> None:
    without = serve.stand_ins(weights=False)
    assert without.embedding == "hash" and without.embedding_dimension == 64
    assert without.reranker == "lexical" and without.nli == "lexical"
    assert without.search == "memory" and without.tasks == "inline"


def test_weights_present_needs_the_encoder_and_the_nli_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(constants, "MODEL_ROOTS", (tmp_path,))
    assert serve.weights_present() is False
    (tmp_path / constants.FROZEN_MODELS.dense.local_dir).mkdir()
    assert serve.weights_present() is False
    (tmp_path / constants.FROZEN_MODELS.nli.local_dir).mkdir()
    assert serve.weights_present() is True


def test_no_stand_in_is_reachable_from_the_environment() -> None:
    """The launcher sets only real settings; every stand-in goes through ``Overrides``."""
    assert all(name.startswith("MEMORY__") for name in serve.DEFAULT_ENV)
    assert not any("PROVIDER" in name for name in serve.DEFAULT_ENV)
