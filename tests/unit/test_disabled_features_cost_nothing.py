"""A feature that is off must not be built, loaded, or advertised.

Retrieval capabilities that were measured and rejected are removed rather than left behind a
flag: cross-encoder reranking (worse *and* twelve times slower on document RAG —
docs/MEASUREMENTS.md §3b), SPLADE (a BERT-sized pass per document at ingest for an
English-only vocabulary) and ColBERT late interaction (ADR 0012).
"""

from __future__ import annotations

import pathlib

from memory_service.api.routers.ops import _active
from memory_service.config.constants import RETRIEVAL
from memory_service.config.settings import Settings

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_a_disabled_component_reports_disabled_rather_than_its_configured_name() -> None:
    """/version used to name a cross-encoder that had never been loaded."""
    assert _active(None, "sentence_transformers") == "disabled"


def test_colbert_left_no_configuration_behind() -> None:
    """ADR 0012 removed late interaction. The env var outlived it in two files, mapping to
    no setting at all — pydantic's extra="ignore" swallowed it silently in both."""
    from tests.unit.test_settings import _leaves

    leaves = _leaves(Settings)
    assert not [leaf for leaf in leaves if "late_interaction" in leaf]
    assert not [leaf for leaf in leaves if "sparse_model" in leaf], "SPLADE went the same way"
    for name in ("docker-compose.yml", ".env.example"):
        path = ROOT / name
        if path.exists():
            assert "LATE_INTERACTION" not in path.read_text(), (
                f"{name} sets a variable that no setting reads"
            )


def test_every_memory_env_var_maps_to_a_real_setting() -> None:
    """The general form of the bug above: a variable nobody reads looks configured and is
    inert, and nothing warns because extra="ignore" is what lets unrelated env vars coexist.

    Every file that sets one, not just compose. `.env` is the file a developer actually
    edits, and it was the one carrying MEMORY__MODELS__LLM__PROVIDER after the field was
    removed — checking only compose would have called that clean.
    """
    import re

    settings = Settings(_env_file=None)
    unknown: dict[str, list[str]] = {}
    for name in ("docker-compose.yml", ".env", ".env.example", "Makefile"):
        path = ROOT / name
        if not path.exists():
            continue
        for var in sorted(set(re.findall(r"MEMORY__[A-Z0-9_]+", path.read_text()))):
            node, parts = settings, var.removeprefix("MEMORY__").lower().split("__")
            for part in parts:
                fields = getattr(type(node), "model_fields", {})
                if part not in fields:
                    unknown.setdefault(name, []).append(var)
                    break
                node = getattr(node, part)
    assert not unknown, f"variables no setting reads: {unknown}"


def test_reranking_left_no_configuration_behind() -> None:
    """The reranker was measured worse and removed: no flag, no depth, no frozen model. Two
    cross-encoders were measured again as features of the memories' learned ranking (ADR
    0025): +2.5 points of recall@10 for 2.4 CPU-seconds a query, which the 20 requests a
    second the service is sized for cannot afford on CPU, so they stayed out."""
    assert "rerank" not in type(RETRIEVAL).model_fields
    assert "rerank_k" not in type(RETRIEVAL).model_fields
    wiring = (ROOT / "src/memory_service/adapters/wiring.py").read_text()
    assert "rerank" not in wiring
    assert "rerank" not in (ROOT / "src/memory_service/config/constants.py").read_text()
