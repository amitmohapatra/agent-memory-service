"""Server configuration: the service endpoint, credentials and the *authorized scope*.

The tenant is fixed by the API key; the configured user / workspace are the defaults and
the allow-lists bound what a client may ask for per call (``MEMORY_ALLOWED_USERS``,
``MEMORY_ALLOWED_WORKSPACES``; ``*`` allows any). Everything comes from the environment so
the server runs as ``uvx universal-memory-mcp`` with no config file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

ENV_PREFIX = "MEMORY_"


def _split(value: str | None) -> tuple[str, ...]:
    return tuple(v.strip() for v in (value or "").split(",") if v.strip())


@dataclass(frozen=True)
class ServerConfig:
    url: str = "http://localhost:8080"
    api_key: str | None = None
    bearer_token: str | None = None
    tenant_id: str = ""
    workspace_id: str | None = None
    user_id: str | None = None
    group_ids: tuple[str, ...] = ()
    thread_id: str | None = None
    agent_id: str | None = None
    agent_run_id: str | None = None
    agent_group_id: str | None = None
    allowed_users: tuple[str, ...] = ()
    allowed_workspaces: tuple[str, ...] = ()
    token_budget: int | None = None
    timeout: float = 30.0
    name: str = "universal-memory"
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerConfig:
        env = dict(os.environ if environ is None else environ)

        def get(key: str, default: str | None = None) -> str | None:
            return env.get(f"{ENV_PREFIX}{key}", default)

        tenant = get("TENANT_ID") or ""
        if not tenant:
            raise ValueError(f"{ENV_PREFIX}TENANT_ID is required")
        budget = get("TOKEN_BUDGET")
        return cls(
            url=get("URL", "http://localhost:8080") or "http://localhost:8080",
            api_key=get("API_KEY"),
            bearer_token=get("BEARER_TOKEN"),
            tenant_id=tenant,
            workspace_id=get("WORKSPACE_ID"),
            user_id=get("USER_ID"),
            group_ids=_split(get("GROUP_IDS")),
            thread_id=get("THREAD_ID"),
            agent_id=get("AGENT_ID"),
            agent_run_id=get("AGENT_RUN_ID"),
            agent_group_id=get("AGENT_GROUP_ID"),
            allowed_users=_split(get("ALLOWED_USERS")),
            allowed_workspaces=_split(get("ALLOWED_WORKSPACES")),
            token_budget=int(budget) if budget else None,
            timeout=float(get("TIMEOUT", "30") or 30),
            name=get("SERVER_NAME", "universal-memory") or "universal-memory",
        )

    @property
    def defaults(self) -> dict[str, Any]:
        """Adapter-level identity for :func:`universal_memory.integrations.scope_fields`."""
        return {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "group_ids": list(self.group_ids),
            "agent_group_id": self.agent_group_id,
        }

    def user_allowed(self, user_id: str) -> bool:
        return user_id == self.user_id or "*" in self.allowed_users or user_id in self.allowed_users

    def workspace_allowed(self, workspace_id: str) -> bool:
        return (
            workspace_id == self.workspace_id
            or "*" in self.allowed_workspaces
            or workspace_id in self.allowed_workspaces
        )
