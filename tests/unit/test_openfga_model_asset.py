"""The authorization model is an asset the deployment cannot do without.

A deployment writes it to OpenFGA the first time it finds no model there. Leaving it out of
the runtime image therefore works on every machine where some earlier run already created
the model, and fails only on a genuinely fresh install — which is the one case nobody tests
by hand. These two checks are cheap and would have caught it.
"""

from __future__ import annotations

from pathlib import Path

from memory_service.adapters.authz.openfga_provider import MODEL_PATH, _dsl_to_json

REPO = Path(__file__).resolve().parents[2]


def test_the_model_file_is_where_the_provider_looks_for_it() -> None:
    assert MODEL_PATH.is_file(), MODEL_PATH
    model = _dsl_to_json(MODEL_PATH.read_text(encoding="utf-8"))
    assert model["type_definitions"], "the DSL must parse into a writable model"


def test_the_runtime_image_copies_it() -> None:
    dockerfile = (REPO / "deploy" / "Dockerfile").read_text()
    stages = {
        stage.splitlines()[0].rsplit(" AS ", 1)[-1].strip(): stage
        for stage in dockerfile.split("\nFROM ")
        if " AS " in stage.splitlines()[0]
    }
    assert "COPY deploy/openfga" in stages["runtime"], (
        "the runtime stage must copy deploy/openfga, or a fresh deployment cannot write its "
        "authorization model"
    )
