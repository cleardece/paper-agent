"""
Paper Agent - Semantic Scholar via MCP
通过 MCP 调用 Semantic Scholar 服务
"""

import logging
from typing import Optional

logger = logging.getLogger("paper-agent")


class SemanticScholarMCP:
    """Semantic Scholar MCP 客户端"""

    def __init__(self):
        from tools.mcp_client import mcp_client
        self.client = mcp_client
        self.server_name = "semantic_scholar"

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """搜索论文"""
        result = await self.client.call_tool(
            self.server_name,
            "search_papers",
            {"query": query, "limit": max_results}
        )

        if not result:
            return []

        papers = []
        try:
            content = result.content[0].text if hasattr(result, 'content') else str(result)
            import json
            data = json.loads(content) if isinstance(content, str) else content

            for item in data.get("papers", data) if isinstance(data, dict) else data:
                # 提取 ArXiv ID
                arxiv_id = None
                external_ids = item.get("externalIds", {})
                if external_ids and external_ids.get("ArXiv"):
                    arxiv_id = external_ids["ArXiv"]

                # 获取 PDF URL
                pdf_url = None
                if item.get("openAccessPdf"):
                    pdf_url = item["openAccessPdf"].get("url")
                if not pdf_url and arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

                papers.append({
                    "arxiv_id": arxiv_id or item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "abstract": item.get("abstract", ""),
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "pdf_url": pdf_url,
                    "year": item.get("year"),
                    "citation_count": item.get("citationCount"),
                })
        except Exception as e:
            logger.error(f"[SemanticScholarMCP] 解析结果失败: {e}")

        logger.info(f"[SemanticScholarMCP] 搜索到 {len(papers)} 篇论文")
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
                "paperId": item.get("paperId", ""),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "citationCount": item.get("citationCount"),
                "references": item.get("references", []),
                "citations": item.get("citations", []),
            }
        except Exception as e:
            logger.error(f"[SemanticScholarMCP] 解析论文失败: {e}")
            return None

    async def get_citations(self, paper_id: str) -> list[dict]:
        """获取引用论文"""
        result = await self.client.call_tool(
            self.server_name,
            "get_citations",
            {"paper_id": paper_id}
        )

        if not result:
            return []

        try:
            content = result.content[0].text if hasattr(result, 'content') else str(result)
            import json
            data = json.loads(content) if isinstance(content, str) else content
            return data.get("citations", data) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"[SemanticScholarMCP] 获取引用失败: {e}")
            return []


class SemanticScholarAPI:
    """Semantic Scholar API - 兼容原有接口，优先使用 MCP，降级到直接 API"""

    def __init__(self, api_key: str = None, use_mcp: bool = True):
        self.api_key = api_key
        self.use_mcp = use_mcp
        self._mcp_client = None
        self._direct_client = None

        if use_mcp:
            try:
                self._mcp_client = SemanticScholarMCP()
                logger.info("[SemanticScholarAPI] 使用 MCP 模式")
            except Exception as e:
                logger.warning(f"[SemanticScholarAPI] MCP 初始化失败，降级到直接 API: {e}")
                self.use_mcp = False

        if not self.use_mcp:
            from tools.semantic_scholar import SemanticScholarAPI as DirectSSAPI
            self._direct_client = DirectSSAPI(api_key=api_key)
            logger.info("[SemanticScholarAPI] 使用直接 API 模式")

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """搜索论文（同步接口）"""
        if self.use_mcp and self._mcp_client:
            try:
                import asyncio
                import concurrent.futures

                async def _search():
                    return await self._mcp_client.search(query, max_results)

                # 检查是否有运行中的事件循环
                try:
                    loop = asyncio.get_running_loop()
                    # 在异步环境中，使用线程池
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(asyncio.run, _search()).result()
                    return result
                except RuntimeError:
                    # 没有运行中的事件循环
                    return asyncio.run(_search())

            except Exception as e:
                logger.error(f"[SemanticScholarAPI] MCP 调用失败: {e}")

        # 降级到直接 API
        if self._direct_client:
            return self._direct_client.search(query, max_results)

        return []
