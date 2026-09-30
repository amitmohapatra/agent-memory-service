"""No Chinese-origin model, nor a derivative of one, anywhere in the stack.

The rule is about provenance, not language: every script is supported and the multilingual
encoder is chosen for that. It is stated once (``domain/provenance.py``) and asserted here on
every surface that names a model - the frozen set, the download catalogue, the benchmark
challengers, the build's defaults, gateway discovery, operator configuration and every
``ProviderInfo`` an adapter constructs - so it cannot hold in one place and lapse in another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.adapters.models.catalog import choose_model
from memory_service.config.constants import FROZEN_MODELS
from memory_service.domain.errors import ProviderNotConfigured
from memory_service.domain.provenance import EXCLUDED_MODEL_ORIGINS, permitted_model
from memory_service.ports.models import ProviderInfo
from memory_service.tools.download_models import MODELS, load_challengers

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

EXCLUDED = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "Alibaba-NLP/gte-multilingual-base",
    "deepseek/deepseek-flash",
    "openrouter/moonshotai/kimi-k2",
    "zai-org/glm-4.6",
    "01-ai/yi-34b",
    "openai/gpt-4o-qwen-distill",  # a derivative keeps the family in its name
    "microsoft/qwen-derivative",
]
PERMITTED = [
    "ibm-granite/granite-embedding-small-english-r2",
    "hotchpotch/bekko-embedding-v1-a8m",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "intfloat/multilingual-e5-small",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "gemini/gemini-3.8-flash",
    "openrouter/openai/gpt-4.1-mini",
    "anthropic/claude-haiku-4-5-20251001",
    "bm25-sparse",
    "hash-embedding",
]

#: Files whose non-comment lines choose a model for a build, a benchmark or a deployment.
#: Comments may record history; ``benchmark/results`` is evidence and is not scanned.
BUILD_SURFACES = [
    "Makefile",
    ".env.example",
    "docker-compose.yml",
    "benchmark/challengers.txt",
    "benchmark/embedding.py",
]


@pytest.mark.parametrize("name", EXCLUDED)
def test_excluded_families_are_recognised(name: str) -> None:
    assert not permitted_model(name)


@pytest.mark.parametrize("name", PERMITTED)
def test_permitted_models_are_not_false_positives(name: str) -> None:
    assert permitted_model(name)


def test_the_frozen_set_is_permitted() -> None:
    frozen = [FROZEN_MODELS.dense.id, FROZEN_MODELS.dense_ml.id, FROZEN_MODELS.nli.id]
    assert all(permitted_model(name) for name in frozen), frozen


def test_the_download_catalogue_is_permitted() -> None:
    offending = [m.repo for m in MODELS if not permitted_model(m.repo)]
    assert not offending, offending


def test_the_benchmark_challengers_are_permitted() -> None:
    challengers = load_challengers(ROOT / "benchmark" / "challengers.txt")
    offending = [m.repo for m in challengers if not permitted_model(m.repo)]
    assert not offending, offending


@pytest.mark.parametrize("path", BUILD_SURFACES)
def test_build_surfaces_name_no_excluded_model(path: str) -> None:
    hits = [
        line
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().lstrip("@").startswith("#") and EXCLUDED_MODEL_ORIGINS.search(line)
    ]
    assert not hits, hits


def test_a_provider_cannot_be_constructed_from_an_excluded_origin() -> None:
    with pytest.raises(ValueError, match="excluded origin"):
        ProviderInfo(
            name="BAAI/bge-m3", license="MIT", origin="huggingface/BAAI/bge-m3", locality="local"
        )


def test_a_tenant_policy_refuses_an_excluded_model() -> None:
    from memory_service.api.routers.v1.model_keys import ModelPolicyRequest

    with pytest.raises(ValueError, match="excluded origin"):
        ModelPolicyRequest(
            uses=["summaries"], read_assist=True, models={"summaries": "deepseek/deepseek-flash"}
        )
    ok = ModelPolicyRequest(
        uses=["summaries"], read_assist=True, models={"summaries": "gemini/gemini-3.8-flash"}
    )
    assert ok.models == {"summaries": "gemini/gemini-3.8-flash"}


def test_gateway_discovery_never_selects_an_excluded_family() -> None:
    with pytest.raises(ProviderNotConfigured):
        choose_model(("openrouter/deepseek/deepseek-chat", "openrouter/qwen/qwen3-235b"), fast=True)
    chosen = choose_model(
        ("openrouter/deepseek/deepseek-chat", "openrouter/openai/gpt-4.1-mini"), fast=True
    )
    assert chosen.endswith("gpt-4.1-mini")
