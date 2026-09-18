"""Contract tests for the M10 model adapters. They run only with local weights
(``MEMORY_MODELS_DIR``); without them they verify the *honest failure*: a missing model is a
DependencyUnavailable at construction, never a silent fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memory_service.config.settings import EmbeddingSettings
from memory_service.domain.errors import DependencyUnavailable
from tests.support_models import MODELS_DIR as MODELS
from tests.support_models import requires_sentence_transformers, requires_weights

pytestmark = pytest.mark.contract


def test_missing_sparse_and_late_models_fail_loudly(tmp_path: Path) -> None:
    from memory_service.adapters.models.advanced import (
        FastEmbedLateInteraction,
        FastEmbedSparseEncoder,
        LateChunkingEmbedding,
    )

    with pytest.raises(DependencyUnavailable, match="could not be loaded"):
        FastEmbedSparseEncoder("prithivida/Splade_PP_en_v1", model_path=str(tmp_path / "none"))
    with pytest.raises(DependencyUnavailable, match="could not be loaded"):
        FastEmbedLateInteraction(
            "answerdotai/answerai-colbert-small-v1", model_path=str(tmp_path / "none")
        )
    with pytest.raises(DependencyUnavailable):
        LateChunkingEmbedding(
            EmbeddingSettings(
                provider="sentence_transformers", model="x/y", model_path=str(tmp_path / "none")
            )
        )


async def test_wiring_refuses_model_backed_flags_without_weights(make_settings) -> None:
    """A deployment that turns a model-backed strategy on without the weights must not start
    with a degraded configuration."""
    from memory_service.__about__ import __version__
    from memory_service.application.container import build_container

    settings = make_settings(
        retrieval={"colbert": True}, models={"late_interaction_model_path": "/nonexistent"}
    )
    with pytest.raises(DependencyUnavailable):
        await build_container(settings, __version__)


@pytest.mark.models
async def test_splade_and_colbert_with_local_weights() -> None:
    requires_weights("Splade_PP_en_v1", "answerai-colbert-small-v1")
    from memory_service.adapters.models.advanced import (
        FastEmbedLateInteraction,
        FastEmbedSparseEncoder,
    )

    splade = MODELS / "Splade_PP_en_v1"
    colbert = MODELS / "answerai-colbert-small-v1"
    if splade.exists():
        enc = FastEmbedSparseEncoder("prithivida/Splade_PP_en_v1", model_path=str(splade))
        doc = enc.encode_documents(["Adjusted EBITDA increased despite lower revenue"])[0]
        q = enc.encode_query("why did ebitda increase")
        assert doc.indices and q.indices and set(doc.indices) & set(q.indices)
        assert enc.server_side_idf is False and enc.fingerprint().startswith("sparse-")
    if colbert.exists():
        li = FastEmbedLateInteraction(
            "answerdotai/answerai-colbert-small-v1", model_path=str(colbert)
        )
        vecs = await li.embed_documents_multi(["Adjusted EBITDA increased"])
        assert vecs and len(vecs[0]) > 1 and len(vecs[0][0]) == li.dimension
        assert len(await li.embed_query_multi("ebitda")) >= 1


@pytest.mark.models
async def test_late_chunking_with_local_weights() -> None:
    requires_weights("granite-embedding-small-english-r2")
    requires_sentence_transformers()
    from memory_service.adapters.models.advanced import LateChunkingEmbedding

    emb = LateChunkingEmbedding(
        EmbeddingSettings(
            provider="sentence_transformers",
            model="ibm-granite/granite-embedding-small-english-r2",
            model_path=str(MODELS / "granite-embedding-small-english-r2"),
        )
    )
    doc = "Adjusted EBITDA means earnings before interest. It increased to EUR 98 million."
    spans = [(0, 45), (46, len(doc))]
    vecs = await emb.embed_spans(doc, spans, [doc[:45], doc[46:]])
    assert len(vecs) == 2 and all(len(v) == emb.dimension for v in vecs)
    assert emb.fingerprint().endswith("-late")
