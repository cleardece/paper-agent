"""Unified MCP client for external paper services."""

import logging
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger("paper-agent")


class MCPClient:
    """Manage MCP connections for arXiv, Semantic Scholar, and similar services."""

    def __init__(self):
        self.sessions: dict[str, Any] = {}
        self.exit_stacks: dict[str, AsyncExitStack] = {}
        self.config = self._load_config()

    def _load_config(self) -> dict:
        from config import MCP_ARXIV_URL, MCP_SS_URL

        return {
            "arxiv": {
                "url": MCP_ARXIV_URL,
                "transport": "http" if MCP_ARXIV_URL.rstrip("/").endswith("/mcp") else "sse",
            },
            "semantic_scholar": {
                "url": MCP_SS_URL,
                "transport": "http" if MCP_SS_URL.rstrip("/").endswith("/mcp") else "sse",
            },
        }

    def _normalize_url(self, url: str, transport: str) -> str:
        url = url.rstrip("/")
        if transport == "sse" and not url.endswith("/sse"):
            return f"{url}/sse"
        if transport == "http" and not url.endswith("/mcp"):
            return f"{url}/mcp"
        return url

    async def connect_sse(self, server_name: str) -> bool:
        """Connect to a configured MCP server.

        The method name is kept for compatibility with existing callers. It now
        supports both legacy SSE (`/sse`) and Streamable HTTP (`/mcp`).
        """
        if server_name in self.sessions:
            return True

        config = self.config.get(server_name)
        if not config or not config.get("url"):
            logger.error("[MCP] missing config or URL: %s", server_name)
            return False

        transport = config.get("transport", "sse")
        url = self._normalize_url(config["url"], transport)

        try:
            from mcp import ClientSession

            stack = AsyncExitStack()
            if transport == "http":
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await stack.enter_async_context(streamablehttp_client(url))
            else:
                from mcp.client.sse import sse_client

                read, write = await stack.enter_async_context(sse_client(url))

            session = ClientSession(read, write)
            await stack.enter_async_context(session)
            await session.initialize()

            self.sessions[server_name] = session
            self.exit_stacks[server_name] = stack
            logger.info("[MCP] connected %s via %s: %s", server_name, transport, url)
            return True
        except Exception as exc:
            logger.error("[MCP] connect failed %s: %s", server_name, exc)
            return False

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        session = self.sessions.get(server_name)
        if not session:
            if not await self.connect_sse(server_name):
                return None
            session = self.sessions[server_name]

        try:
            return await session.call_tool(tool_name, arguments)
        except Exception as exc:
            logger.error("[MCP] tool call failed %s.%s: %s", server_name, tool_name, exc)
            return None

    async def list_tools(self, server_name: str) -> list:
        session = self.sessions.get(server_name)
        if not session:
            if not await self.connect_sse(server_name):
                return []
            session = self.sessions[server_name]

        try:
            return await session.list_tools()
        except Exception as exc:
            logger.error("[MCP] list tools failed %s: %s", server_name, exc)
            return []

    async def disconnect(self, server_name: str):
        self.sessions.pop(server_name, None)
        stack = self.exit_stacks.pop(server_name, None)
        if stack:
            try:
                await stack.aclose()
            except Exception:
                pass

    async def disconnect_all(self):
        for name in list(self.sessions.keys()):
            await self.disconnect(name)


mcp_client = MCPClient()
