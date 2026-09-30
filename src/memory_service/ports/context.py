"""Evidence context consumed by derived views, independent of the retrieval implementation."""

from typing import Protocol

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ContextBundle


class ContextReader(Protocol):
    async def build(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        *,
        token_budget: int | None = None,
        window: bool = True,
    ) -> ContextBundle: ...
