"""arXiv access through MCP, with direct API fallback."""

import json
import logging

logger = logging.getLogger("paper-agent")


class ArxivMCP:
    """MCP-backed arXiv client."""

    def __init__(self):
        from tools.mcp_client import mcp_client

        self.client = mcp_client
        self.server_name = "arxiv"

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            result = await self.client.call_tool(
                self.server_name,
                "search_arxiv",
                {"query": query, "limit": max_results},
            )
            if not result:
                logger.warning("[ArxivMCP] empty search result")
                return []

            content = result.content[0].text if hasattr(result, "content") else str(result)
            data = json.loads(content) if isinstance(content, str) else content
            # 返回格式可能是列表或字典
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("papers", data.get("results", []))
            else:
                items = []

            papers = []
            if isinstance(items, list):
                for item in items:
                    paper_id = item.get("id", item.get("arxiv_id", item.get("paper_id", "")))
                    papers.append(
                        {
                            "arxiv_id": paper_id,
                            "title": item.get("title", ""),
                            "url": item.get("url", item.get("link", f"https://arxiv.org/abs/{paper_id}")),
                            "abstract": item.get("abstract", item.get("summary", "")),
                            "authors": item.get("authors", []),
                            "pdf_url": item.get("pdf_url", item.get("pdf_link", f"https://arxiv.org/pdf/{paper_id}")),
                        }
                    )
            logger.info("[ArxivMCP] found %s papers", len(papers))
            return papers
        except Exception as exc:
            logger.error("[ArxivMCP] search failed: %s", exc)
            return []
        finally:
            await self.client.disconnect(self.server_name)


class ArxivAPI:
    """Synchronous arXiv API facade used by FetcherAgent."""

    def __init__(self, use_mcp: bool = True):
        self.use_mcp = use_mcp
        self._mcp_client = None
        self._direct_client = None

        if use_mcp:
            try:
                from config import MCP_ARXIV_URL

                if MCP_ARXIV_URL:
                    self._mcp_client = ArxivMCP()
                    logger.info("[ArxivAPI] using MCP mode")
                else:
                    self.use_mcp = False
            except Exception as exc:
                logger.warning("[ArxivAPI] MCP init failed: %s", exc)
                self.use_mcp = False

        if not self.use_mcp:
            from tools.arxiv_api import ArxivAPI as DirectArxivAPI

            self._direct_client = DirectArxivAPI()
            logger.info("[ArxivAPI] using direct API mode")

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        if self.use_mcp and self._mcp_client:
            try:
                import asyncio
                import concurrent.futures

                async def _search():
                    return await self._mcp_client.search(query, max_results)

                try:
                    asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, _search()).result()
                except RuntimeError:
                    return asyncio.run(_search())
            except Exception as exc:
                logger.error("[ArxivAPI] MCP call failed: %s", exc)

        if self._direct_client:
            return self._direct_client.search(query, max_results)

        return []
