"""
Paper Agent - Semantic Scholar API 工具
替代 arXiv API，限流更宽松
"""

import logging
import time
import httpx

logger = logging.getLogger("paper-agent")

# Semantic Scholar API 基础 URL
BASE_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarAPI:
    """Semantic Scholar 论文搜索 API"""

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: API Key（可选，有 Key 限制更宽松）
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        搜索论文

        Args:
            query: 搜索关键词
            max_results: 最大返回数量

        Returns:
            论文列表
        """
        logger.info(f"[SemanticScholar] 搜索: {query[:50]}...")

        url = f"{BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,abstract,authors,url,openAccessPdf,year,citationCount",
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()

            papers = []
            for item in data.get("data", []):
                # 提取 arxiv_id（如果有的话）
                paper_id = item.get("paperId", "")
                arxiv_id = self._extract_arxiv_id(paper_id)

                # 获取 PDF 链接
                pdf_url = None
                if item.get("openAccessPdf"):
                    pdf_url = item["openAccessPdf"].get("url")

                papers.append({
                    "arxiv_id": arxiv_id or paper_id,
                    "title": item.get("title", ""),
                    "url": item.get("url", f"https://www.semanticscholar.org/paper/{paper_id}"),
                    "abstract": item.get("abstract", ""),
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "pdf_url": pdf_url,
                    "year": item.get("year"),
                    "citation_count": item.get("citationCount"),
                })

            logger.info(f"[SemanticScholar] 找到 {len(papers)} 篇论文")
            return papers

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("[SemanticScholar] 速率限制，等待后重试...")
                time.sleep(5)
                return self.search(query, max_results)
            logger.error(f"[SemanticScholar] 请求失败: {e}")
            return []
        except Exception as e:
            logger.error(f"[SemanticScholar] 搜索失败: {e}")
            return []

    def _extract_arxiv_id(self, paper_id: str) -> str | None:
        """从 paper_id 提取 arxiv_id"""
        if "ArXiv:" in paper_id:
            return paper_id.replace("ArXiv:", "")
        if paper_id.startswith("10.") and "arxiv" in paper_id.lower():
            # 可能是 DOI
            return None
        return None

    def get_paper_details(self, paper_id: str) -> dict | None:
        """获取论文详情"""
        url = f"{BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,abstract,authors,url,openAccessPdf,year,citationCount,references",
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"[SemanticScholar] 获取详情失败: {e}")
            return None
