"""Adapters: concrete implementations of ports. Only this package imports provider SDKs.

``wire_adapters`` is called by the composition root and attaches configured providers to
the container. Each milestone adds wiring here; unconfigured capabilities stay ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_service.application.container import Container


async def wire_adapters(container: Container) -> None:
    from memory_service.adapters.wiring import wire_all

    await wire_all(container)
