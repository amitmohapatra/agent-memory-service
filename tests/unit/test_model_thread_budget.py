"""One thread count, shared by every in-process model, and it is the one a deployment set.

``torch.set_num_threads`` is process-wide, so the last model to load decides it for all of
them. The encoder was given ``models.embedding.threads`` and the NLI head was given its own
frozen default, and the NLI head loads second - so the one environment knob over the
component that is 61% of query p99 was inert in every process that loads both, which is every
API and worker process. Both the field's docstring and .env.example also described a default
("torch's own default (every core)") that the wiring never implemented.
"""

from __future__ import annotations

import pytest

from memory_service.adapters.wiring import _model_threads
from memory_service.config.constants import FROZEN_MODELS

pytestmark = pytest.mark.unit


class _Container:
    def __init__(self, threads: int | None) -> None:
        class _Embedding:
            pass

        class _Models:
            pass

        class _Settings:
            pass

        embedding, models, settings = _Embedding(), _Models(), _Settings()
        embedding.threads = threads  # type: ignore[attr-defined]
        models.embedding = embedding  # type: ignore[attr-defined]
        settings.models = models  # type: ignore[attr-defined]
        self.settings = settings


def test_an_unset_count_is_the_one_frozen_with_the_encoder() -> None:
    """Not "torch's own default", which is what the docs claimed and the code never did."""
    assert _model_threads(_Container(None)) == FROZEN_MODELS.dense.threads  # type: ignore[arg-type]


def test_a_deployment_that_sets_the_count_gets_it() -> None:
    assert _model_threads(_Container(4)) == 4  # type: ignore[arg-type]


def test_every_model_resolves_to_the_same_number() -> None:
    """The defect: two models each resolving their own, process-wide, last-loader-wins."""
    container = _Container(3)
    encoder = _model_threads(container, FROZEN_MODELS.dense)  # type: ignore[arg-type]
    nli = _model_threads(container)  # type: ignore[arg-type]
    assert encoder == nli == 3
