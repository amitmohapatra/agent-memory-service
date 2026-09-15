"""Duck-typed view of LangChain/LangGraph chat messages (no langchain_core import).

A node that follows the ``messages`` convention returns ``{"messages": [...]}`` holding
only the *new* messages (the ``add_messages`` reducer appends them). Those are what get
recorded in the thread. Accepted shapes: LangChain ``BaseMessage`` objects (``type``,
``content``, ``id``), ``{"role": ..., "content": ...}`` dicts, and ``(role, content)``
tuples. Multi-part content keeps its text blocks only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_ROLE_TO_TYPE = {
    "user": "human",
    "human": "human",
    "assistant": "ai",
    "ai": "ai",
    "tool": "tool",
    "system": "system",
    "function": "tool",
}


@dataclass(frozen=True)
class MessageView:
    type: str  # human | ai | tool | system
    content: str
    id: str | None = None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type", "text") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def as_view(message: Any) -> MessageView | None:
    if isinstance(message, MessageView):
        return message
    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
        t = _ROLE_TO_TYPE.get(str(role).lower())
        return MessageView(t, _text(content)) if t else None
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
        t = _ROLE_TO_TYPE.get(str(role).lower()) if role else None
        return MessageView(t, _text(message.get("content")), message.get("id")) if t else None
    t = _ROLE_TO_TYPE.get(str(getattr(message, "type", "")).lower())
    if t is None:
        return None
    return MessageView(t, _text(getattr(message, "content", "")), getattr(message, "id", None))


def new_messages(state: Any, result: Any) -> list[MessageView]:
    """Messages a node *added*: the ``messages`` entry of its result (a list or one)."""
    if not isinstance(result, Mapping):
        return []
    raw = result.get("messages")
    if raw is None:
        return []
    items: Iterable[Any] = raw if isinstance(raw, list | tuple) else [raw]
    out: list[MessageView] = []
    for item in items:
        view = as_view(item)
        if view is not None and view.content.strip():
            out.append(view)
    return out


def trailing_human(state: Any) -> list[MessageView]:
    """The human messages at the end of ``state["messages"]`` — the turn not yet answered."""
    raw = state.get("messages") if isinstance(state, Mapping) else getattr(state, "messages", None)
    if not isinstance(raw, list | tuple):
        return []
    out: list[MessageView] = []
    for item in reversed(raw):
        view = as_view(item)
        if view is None or view.type != "human":
            break
        if view.content.strip():
            out.append(view)
    out.reverse()
    return out
