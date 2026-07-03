"""
Paper Agent - arXiv API工具
搜索论文、获取论文内容
"""

import arxiv
import logging
import time

logger = logging.getLogger("paper-agent")

def search_papers(query: str, max_results: int = 5, retries: int = 3):
    """搜索arXiv论文，带重试机制"""
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,  # 增加请求间隔到3秒
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for attempt in range(retries):
        try:
            logger.info(f"[arXiv] 搜索关键词: '{query}' (第 {attempt + 1} 次, max_results={max_results})")
            results = []
            for r in client.results(search):
                # 从 entry_id 提取 arxiv_id
                entry_id = r.entry_id
                arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id
                results.append({
                    "arxiv_id": arxiv_id,
                    "title": r.title,
                    "url": r.entry_id,
                    "abstract": r.summary,
                    "authors": [a.name for a in r.authors],
                    "pdf_url": r.pdf_url,
                })
            logger.info(f"[arXiv] 找到 {len(results)} 篇论文")
            if results:
                for i, r in enumerate(results[:3]):
                    logger.info(f"[arXiv]   {i+1}. {r['title'][:60]}")
            return results

        except arxiv.HTTPError as e:
            if "429" in str(e) and attempt < retries - 1:
                wait_time = (attempt + 1) * 10  # 递增等待：10s, 20s
                logger.warning(f"[arXiv] 速率限制，等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"[arXiv] 搜索失败: {e}")
                raise

    return []


def fetch_paper_content(url: str) -> str:
    """下载论文PDF并提取文本"""
    import pdfplumber
    import requests
    from io import BytesIO

    if "arxiv.org" in url:
        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
    else:
        pdf_url = url

    resp = requests.get(pdf_url)
    with pdfplumber.open(BytesIO(resp.content)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text


class ArxivAPI:
    """封装为类，供FetcherAgent使用"""
    def search(self, query, max_results=5):
        return search_papers(query, max_results)

    def fetch(self, url):
        return fetch_paper_content(url)