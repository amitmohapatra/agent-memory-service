"""Contract tests for the M10 model adapters. They run only with local weights
(``MEMORY_MODELS_DIR``); without them they verify the *honest failure*: a missing model is a
DependencyUnavailable at construction, never a silent fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.contract


def test_a_missing_sparse_model_fails_loudly(tmp_path: Path) -> None:
    """SPLADE is the one remaining model-backed retrieval experiment.

    The colbert and late-chunking halves of this test went with their flags. What must not go
    is the rule they shared: a strategy switched on without its weights fails at construction
    rather than falling back to something else and reporting a number for it.
    """
    from memory_service.adapters.models.advanced import FastEmbedSparseEncoder

    # Two honest failures, not one: with the [models] extra installed the runtime is there and
    # the *weights* are missing; without it the import fails first. Which one a machine gets is
    # not the rule under test — that a construction without weights raises instead of quietly
    # returning an encoder is, and matching only the first spelling failed on any machine that
    # had not installed the extra.
    with pytest.raises(DependencyUnavailable, match="could not be loaded|fastembed is required"):
        FastEmbedSparseEncoder("prithivida/Splade_PP_en_v1", model_path=str(tmp_path / "none"))


async def test_wiring_refuses_splade_without_weights(make_settings) -> None:
    """A deployment that turns splade on without the weights must not start degraded."""
    from memory_service.__about__ import __version__
    from memory_service.application.container import build_container

    settings = make_settings(
        retrieval={"splade": True}, models={"sparse_model_path": "/nonexistent"}
    )
    with pytest.raises(DependencyUnavailable):
        await build_container(settings, __version__)
