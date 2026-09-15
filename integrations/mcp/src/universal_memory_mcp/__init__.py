"""MCP server over the ``universal-memory`` SDK (INTEGRATIONS_PLAN §28).

The ``mcp`` package is never a dependency of the Memory Service core; this package depends
on the SDK and ``mcp`` only.
"""

from universal_memory_mcp.config import ServerConfig
from universal_memory_mcp.server import ScopeArgs, build_server, main, resolve_scope

__version__ = "0.1.0"
__all__ = ["ScopeArgs", "ServerConfig", "__version__", "build_server", "main", "resolve_scope"]
