"""
Paper Agent - MCP Client
连接外部 MCP Server（ArXiv, Semantic Scholar 等）
"""

import os
import json
import logging
import asyncio
from typing import Optional, Any

logger = logging.getLogger("paper-agent")


class MCPClient:
    """MCP 客户端 - 统一管理外部 MCP 连接"""

    def __init__(self):
        self.sessions = {}  # server_name -> session
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """加载 MCP 配置"""
        from config import (
            MCP_ARXIV_COMMAND, MCP_ARXIV_ARGS,
            MCP_SS_COMMAND, MCP_SS_ARGS
        )

        config = {
            "arxiv": {
                "command": MCP_ARXIV_COMMAND,
                "args": MCP_ARXIV_ARGS.split(),
            },
            "semantic_scholar": {
                "command": MCP_SS_COMMAND,
                "args": MCP_SS_ARGS.split(),
            },
        }
        return config

    async def connect(self, server_name: str) -> bool:
        """连接到 MCP Server"""
        if server_name in self.sessions:
            return True

        config = self.config.get(server_name)
        if not config:
            logger.error(f"[MCP] 未找到配置: {server_name}")
            return False

        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client

            command = config["command"]
            args = config["args"]

            logger.info(f"[MCP] 连接 {server_name}: {command} {' '.join(args)}")

            # 创建连接
            read, write = await stdio_client(command, *args).__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            self.sessions[server_name] = session
            logger.info(f"[MCP] 连接成功: {server_name}")
            return True

        except Exception as e:
            logger.error(f"[MCP] 连接失败 {server_name}: {e}")
            return False

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 工具"""
        session = self.sessions.get(server_name)
        if not session:
            # 尝试连接
            if not await self.connect(server_name):
                return None
            session = self.sessions[server_name]

        try:
            result = await session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"[MCP] 调用失败 {server_name}.{tool_name}: {e}")
            return None

    async def list_tools(self, server_name: str) -> list:
        """列出 MCP Server 的所有工具"""
        session = self.sessions.get(server_name)
        if not session:
            if not await self.connect(server_name):
                return []
            session = self.sessions[server_name]

        try:
            tools = await session.list_tools()
            return tools
        except Exception as e:
            logger.error(f"[MCP] 列出工具失败 {server_name}: {e}")
            return []

    async def disconnect(self, server_name: str):
        """断开连接"""
        session = self.sessions.pop(server_name, None)
        if session:
            try:
                await session.__aexit__(None, None, None)
            except:
                pass

    async def disconnect_all(self):
        """断开所有连接"""
        for name in list(self.sessions.keys()):
            await self.disconnect(name)


# 全局实例
mcp_client = MCPClient()
