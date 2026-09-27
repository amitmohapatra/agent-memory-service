"""Write features can be measured independently; disabled modes build no SDK client."""

from unittest.mock import AsyncMock, Mock

import pytest

from memory_service.adapters.wiring import _wire_memory
from memory_service.application.container import Container, Overrides, Tuning
from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.modules.llm.assist import LLMAssist

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("consolidate", [False, True])
@pytest.mark.parametrize("llm_enabled", [False, True])
@pytest.mark.parametrize("use", ["contextual_extraction", "reflection"])
async def test_extraction_and_consolidation_are_independent(
    make_settings, monkeypatch, consolidate, llm_enabled, use
):
    client = Mock(close=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr("memory_service.adapters.models.hindsight.HindsightExtractor", factory)
    settings = make_settings(
        models={
            "llm": {
                "enabled": llm_enabled,
                "uses": [use],
            }
        }
    )
    container = Container(settings, "test")
    container.tuning = Tuning.resolve(
        Overrides(memory_intelligence=MemoryIntelligenceSettings(consolidation_enabled=consolidate))
    )
    container.services.update(uow_factory=Mock(), authz=Mock(), llm_assist=LLMAssist.disabled())
    _wire_memory(container)
    assert (container.services["observation_pipeline"].landing is not None) == consolidate
    expected = llm_enabled and use == "contextual_extraction"
    assert factory.call_count == int(expected)
    assert (container.services["memory_provider"].contextual_extractor is not None) == expected
    await container.close()
    assert client.close.await_count == int(expected)


def test_defaults_keep_unmeasured_features_off(make_settings):
    settings = make_settings()
    assert not settings.models.llm.wants("contextual_extraction")
    assert not MemoryIntelligenceSettings().consolidation_enabled
