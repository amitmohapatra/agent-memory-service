"""Optional LangGraph adapter for the ``universal-memory`` SDK (milestone M12).

LangGraph is never a dependency of the Memory Service core; this package depends on the
SDK only and imports LangGraph lazily where it reads the running config.
"""

from universal_memory_langgraph.adapter import LangGraphMemory, NodeResult
from universal_memory_langgraph.lineage import (
    Lineage,
    Segment,
    lineage_from_config,
    safe_id,
    scope_fields,
)
from universal_memory_langgraph.messages import MessageView, as_view, new_messages, trailing_human

__version__ = "0.1.0"
__all__ = [
    "LangGraphMemory",
    "Lineage",
    "MessageView",
    "NodeResult",
    "Segment",
    "__version__",
    "as_view",
    "lineage_from_config",
    "new_messages",
    "safe_id",
    "scope_fields",
    "trailing_human",
]
