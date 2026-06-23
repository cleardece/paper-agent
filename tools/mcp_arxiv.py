"""
Paper Agent - ArXiv via MCP
通过 MCP 调用 ArXiv 服务
"""

import logging
from typing import Optional

logger = logging.getLogger("paper-agent")


class ArxivMCP:
    """ArXiv MCP 客户端"""

    def __init__(self):
        from tools.mcp_client import mcp_client
        self.client = mcp_client
        self.server_name = "arxiv"

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """搜索论文"""
        result = await self.client.call_tool(
            self.server_name,
            "search_papers",
            {"query": query, "max_results": max_results}
        )

        if not result:
            logger.warning("[ArxivMCP] 搜索返回空结果")
            return []

        # 解析结果
        papers = []
        try:
            content = result.content[0].text if hasattr(result, 'content') else str(result)
            import json
            data = json.loads(content) if isinstance(content, str) else content

            for item in data.get("papers", data) if isinstance(data, dict) else data:
                papers.append({
                    "arxiv_id": item.get("id", item.get("arxiv_id", "")),
                    "title": item.get("title", ""),
                    "url": item.get("url", f"https://arxiv.org/abs/{item.get('id', '')}"),
                    "abstract": item.get("abstract", ""),
                    "authors": item.get("authors", []),
                    "pdf_url": item.get("pdf_url", f"https://arxiv.org/pdf/{item.get('id', '')}"),
                })
        except Exception as e:
            logger.error(f"[ArxivMCP] 解析结果失败: {e}")

        logger.info(f"[ArxivMCP] 搜索到 {len(papers)} 篇论文")
        return papers

    async def get_paper(self, paper_id: str) -> Optional[dict]:
        """获取论文详情"""
        result = await self.client.call_tool(
            self.server_name,
            "get_paper",
            {"paper_id": paper_id}
        )

        if not result:
            return None

        try:
            content = result.content[0].text if hasattr(result, 'content') else str(result)
            import json
            item = json.loads(content) if isinstance(content, str) else content

            return {
                "arxiv_id": item.get("id", item.get("arxiv_id", "")),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "authors": item.get("authors", []),
                "pdf_url": item.get("pdf_url", f"https://arxiv.org/pdf/{item.get('id', '')}"),
            }
        except Exception as e:
            logger.error(f"[ArxivMCP] 解析论文失败: {e}")
            return None


class ArxivAPI:
    """ArXiv API - 兼容原有接口，优先使用 MCP，降级到直接 API"""

    def __init__(self, use_mcp: bool = True):
        self.use_mcp = use_mcp
        self._mcp_client = None
        self._direct_client = None

        if use_mcp:
            try:
                self._mcp_client = ArxivMCP()
                logger.info("[ArxivAPI] 使用 MCP 模式")
            except Exception as e:
                logger.warning(f"[ArxivAPI] MCP 初始化失败，降级到直接 API: {e}")
                self.use_mcp = False

        if not self.use_mcp:
            from tools.arxiv_api import ArxivAPI as DirectArxivAPI
            self._direct_client = DirectArxivAPI()
            logger.info("[ArxivAPI] 使用直接 API 模式")

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """搜索论文（同步接口）"""
        if self.use_mcp and self._mcp_client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在异步环境中，创建新任务
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run,
                            self._mcp_client.search(query, max_results)
                        ).result()
                    return result
                else:
                    return loop.run_until_complete(
                        self._mcp_client.search(query, max_results)
                    )
            except Exception as e:
                logger.error(f"[ArxivAPI] MCP 调用失败: {e}")

        # 降级到直接 API
        if self._direct_client:
            return self._direct_client.search(query, max_results)

        return []


import asyncio
