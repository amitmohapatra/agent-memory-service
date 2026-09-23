"""The first inference is paid at startup, and a cold model does not stop the service.

The first encode through an ONNX session or a torch module allocates the runtime's arenas
and selects its kernels, so it costs several times what the ones after it cost: the
development box measured a 481 ms slowest encode against a 98 ms median. A load generator
ramps the instant the port opens, so without a warm-up that difference lands inside the p99
the load test exists to measure and is read as service latency.

The second half matters as much. Warming must never be able to stop the service coming up -
readiness is what reports a model that cannot serve, and a failed warm-up that refused to
boot would replace a degraded service with no service at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory_service.api.app import _warm_encoders

pytestmark = pytest.mark.unit


class _Encoder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("model is not loaded")
        return [0.0]


class _Indexer:
    def __init__(self, embedding: Any = None, sparse: Any = None) -> None:
        self.embedding = embedding
        self.sparse = sparse


class _Container:
    def __init__(self, indexer: Any) -> None:
        self.services: dict[str, Any] = {} if indexer is None else {"indexer": indexer}


async def test_both_encoders_are_warmed_before_the_first_request() -> None:
    dense, sparse = _Encoder(), _Encoder()
    await _warm_encoders(_Container(_Indexer(dense, sparse)))  # type: ignore[arg-type]
    assert dense.calls and sparse.calls, "a cold encoder pays its first inference in a request"


async def test_a_failing_encoder_does_not_stop_the_service_starting() -> None:
    dense, sparse = _Encoder(fail=True), _Encoder()
    await _warm_encoders(_Container(_Indexer(dense, sparse)))  # type: ignore[arg-type]
    assert sparse.calls, "one cold model must not stop the others warming"


async def test_a_container_without_an_indexer_warms_nothing_and_raises_nothing() -> None:
    await _warm_encoders(_Container(None))  # type: ignore[arg-type]


async def test_an_encoder_that_cannot_embed_a_query_is_skipped() -> None:
    """A sparse implementation need not expose the query path; it must not be called blindly."""
    sparse = _Encoder()
    await _warm_encoders(_Container(_Indexer(object(), sparse)))  # type: ignore[arg-type]
    assert sparse.calls
