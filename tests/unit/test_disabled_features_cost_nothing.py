"""A feature that is off must not be built, loaded, or advertised.

Three retrieval capabilities were measured and left off: reranking (worse *and* twelve
times slower on document RAG — docs/MEASUREMENTS.md §3b), SPLADE (benchmark-gated), and
ColBERT late interaction (removed outright, ADR 0012). Off should mean off all the way
down, and for the reranker it did not: `_wire_models` branched on the model provider and
never on `retrieval.rerank`, so a cross-encoder was constructed in both the API and the
worker whatever the flag said, and /version reported it as an active provider.

Nothing called it. `RetrievalEngine` guards its only call site on `cfg.rerank`.
"""

from __future__ import annotations

import pathlib

from memory_service.api.routers.ops import _active
from memory_service.config.settings import Settings

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_the_reranker_is_off_by_default() -> None:
    """On measured evidence, not preference. Changing this needs a new measurement."""
    assert Settings(_env_file=None).retrieval.rerank is False


def test_a_disabled_component_reports_disabled_rather_than_its_configured_name() -> None:
    """/version used to name a cross-encoder that had never been loaded."""
    assert _active(None, "sentence_transformers") == "disabled"


def test_the_reranker_is_guarded_by_its_own_flag_like_splade_is() -> None:
    """The asymmetry this test exists to prevent: splade's model load sits behind
    `if settings.retrieval.splade:` and the reranker's sat behind nothing."""
    wiring = (ROOT / "src/memory_service/adapters/wiring.py").read_text()
    assert "if settings.retrieval.splade:" in wiring
    assert "if not settings.retrieval.rerank:" in wiring, (
        "the reranker must not be constructed when retrieval.rerank is false — that loads "
        "566 MB of weights into every process for something nothing will call"
    )


def test_colbert_left_no_configuration_behind() -> None:
    """ADR 0012 removed late interaction. The env var outlived it in two files, mapping to
    no setting at all — pydantic's extra="ignore" swallowed it silently in both."""
    models = type(Settings(_env_file=None).models)
    assert "late_interaction_model_path" not in models.model_fields
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
    for name in ("docker-compose.yml", ".env", ".env.example"):
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
