"""``Container.services``: one typed attribute per service, also readable by name.

It was a ``dict[str, Any]`` of about fifty string keys, so a misspelt key was a ``KeyError``
in a request and nothing a service returned had a type. The names are declared once now; the
by-name access stays for the callers still written against the dictionary.
"""

from __future__ import annotations

import pytest

from memory_service.application.services import Services
from memory_service.config.settings import Settings
from memory_service.modules.llm.assist import LLMAssist

pytestmark = pytest.mark.unit


def test_a_service_is_one_attribute_read_by_name_too() -> None:
    services = Services()
    assist = LLMAssist.disabled()
    services.llm_assist = assist
    assert services["llm_assist"] is assist and services.get("llm_assist") is assist
    services["llm_assist"] = other = LLMAssist.disabled()
    assert services.llm_assist is other and "llm_assist" in services


def test_a_service_not_wired_is_absent_not_none() -> None:
    services = Services()
    assert "graph" not in services and services.get("graph") is None
    assert services.get("extra_reconcilers", []) == []
    with pytest.raises(KeyError, match="graph"):
        services["graph"]
    with pytest.raises(AttributeError, match="graph"):
        _ = services.graph


def test_a_name_that_is_not_a_service_is_refused_where_it_is_written() -> None:
    services = Services()
    with pytest.raises(KeyError, match="memroy"):
        services["memroy"] = object()
    with pytest.raises(KeyError, match="memroy"):
        services.update(memroy=object())
    assert "memroy" not in services and services.get("memroy") is None


def test_a_container_starts_with_no_services() -> None:
    from memory_service.application.container import Container

    container = Container(settings=Settings(_env_file=None), version="test")  # type: ignore[call-arg]
    assert isinstance(container.services, Services) and "memory" not in container.services


def test_every_service_wiring_names_is_declared() -> None:
    """The names wiring assigns and the attributes ``Services`` declares are one list."""
    import re
    from pathlib import Path

    import memory_service

    wiring = (Path(memory_service.__file__).parent / "adapters" / "wiring.py").read_text()
    assigned = set(re.findall(r"container\.services(?:\[\"|\.)(\w+)(?:\"\])? = ", wiring))
    assert assigned and assigned <= Services.names(), sorted(assigned - Services.names())
    unwired = Services.names() - assigned - {"extra_reconcilers", "extra_task_registrars"}
    assert not unwired, f"declared and never wired: {sorted(unwired)}"
