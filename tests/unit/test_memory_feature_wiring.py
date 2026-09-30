"""Write features can be measured independently; disabled modes build no SDK client."""

from unittest.mock import AsyncMock, Mock

import pytest

from memory_service.adapters.wiring import _wire_memory
from memory_service.application.container import Container
from memory_service.modules.llm.assist import LLMAssist

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("hindsight_url", [None, "http://hindsight.test"])
async def test_the_extraction_client_is_built_only_where_one_runs(
    make_settings, monkeypatch, hindsight_url
):
    """Hindsight is a service a deployment runs or does not: the client is built exactly
    when its URL is configured."""
    client = Mock(close=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr("memory_service.adapters.models.hindsight.HindsightExtractor", factory)
    settings = make_settings(hindsight={"base_url": hindsight_url})
    container = Container(settings, "test")
    container.services.update(uow_factory=Mock(), authz=Mock(), llm_assist=LLMAssist.disabled())
    _wire_memory(container)
    expected = hindsight_url is not None
    assert factory.call_count == int(expected)
    assert (container.services["memory_provider"].contextual_extractor is not None) == expected
    await container.close()
    assert client.close.await_count == int(expected)


def test_defaults_keep_unmeasured_features_off(make_settings):
    settings = make_settings()
    assert not settings.llm.wants("contextual_extraction")
    assert settings.hindsight.base_url is None
