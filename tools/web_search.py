"""
Paper Agent - Web搜索工具
基于Tavily API的网络搜索，用于补充论文信息
"""

import os
import json
import urllib.request


TAVILY_API_URL = "https://api.tavily.com/search"


class WebSearch:
    """Tavily网络搜索"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> dict:
        """
        搜索网络内容
        query: 搜索关键词
        search_depth: "basic" 或 "advanced"
        返回: {"answer": str, "results": [{"title", "url", "content"}, ...]}
        """
        if not self.api_key:
            return {"answer": "", "results": [], "error": "TAVILY_API_KEY未设置"}

        payload = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }).encode("utf-8")

        req = urllib.request.Request(
            TAVILY_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            return {
                "answer": data.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in data.get("results", [])
                ],
            }
        except Exception as e:
            return {"answer": "", "results": [], "error": str(e)}

    def search_paper(self, paper_title: str) -> dict:
        """搜索论文相关信息（项目页、代码仓库、讨论等）"""
        return self.search(
            query=f"{paper_title} paper code implementation",
            max_results=5,
            search_depth="advanced",
        )
