"""The waits between retries are recorded, not slept: the suite checks how long the client
would wait without spending it."""

from collections.abc import Iterator

import pytest

from trellis.memory import transport as transport_module


@pytest.fixture(autouse=True)
def slept(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(transport_module, "_sleep", record)
    monkeypatch.delenv("MEMORY_URL", raising=False)
    monkeypatch.delenv("TRELLIS_API_KEY", raising=False)
    yield waits
