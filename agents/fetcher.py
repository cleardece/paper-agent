"""
Paper Agent - Fetcher Agent
从arXiv抓取论文，解析PDF，分块入库
"""

import logging
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage

from state.graph_state import AgentState
from core.cache import cache

logger = logging.getLogger("paper-agent")


class FetcherAgent:
    """论文抓取入库Agent"""

    def __init__(self, arxiv_api, pdf_parser, mongodb_client, embedding_service, milvus_client):
        self.arxiv = arxiv_api
        self.parser = pdf_parser
        self.mongo = mongodb_client
        self.embedder = embedding_service
        self.milvus = milvus_client

    def invoke(self, state: AgentState) -> dict:
        """
        抓取流程：
        1. arXiv搜索论文（带缓存）
        2. 逐篇：下载PDF → 解析 → 分块
        3. 存MongoDB（元数据+分块）
        4. Embedding → 存Milvus
        5. 更新state
        """
        # 优先使用 Supervisor 提取的 search_query，否则用 user_query
        query = state.get("search_query") or state["user_query"]
        logger.info(f"[Fetcher] 开始搜索论文: {query[:50]}...")

        # 1. 搜索（带缓存）
        cache_key = f"arxiv_search:{query}"
        cached_papers = cache.get(cache_key)
        if cached_papers:
            logger.info(f"[Fetcher] 使用缓存，找到 {len(cached_papers)} 篇论文")
            papers = cached_papers
        else:
            logger.info("[Fetcher] 正在调用 arXiv API...")
            papers = self.arxiv.search(query, max_results=5)
            logger.info(f"[Fetcher] 找到 {len(papers)} 篇论文")
            # 缓存搜索结果
            if papers:
                cache.set(cache_key, papers)

        # 2-4. 逐篇处理
        fetched = []
        for paper_meta in papers:
            arxiv_id = paper_meta["arxiv_id"]

            # 跳过已入库的
            if self.mongo.paper_exists(arxiv_id):
                continue

            try:
                self._process_paper(paper_meta)
                fetched.append(paper_meta)
            except Exception as e:
                logger.error(f"[Fetcher] 处理失败 {arxiv_id}: {e}")

        # 5. 返回结果给Supervisor
        if fetched:
            titles = "\n".join(f"- {p['title']}" for p in fetched)
            return {
                "target_papers": fetched,
                "next_agent": "retriever",
                "error": None,
            }
        else:
            return {
                "target_papers": [],
                "error": "未找到新论文，可能已全部入库。",
            }

    def _process_paper(self, paper_meta: dict):
        """单篇论文完整处理链路"""
        arxiv_id = paper_meta["arxiv_id"]
        title = paper_meta.get('title', '未知')[:40]
        logger.info(f"[Fetcher] 正在处理论文: {title}...")

        # 1. 入MongoDB（状态: pending）
        paper_meta["status"] = "pending"
        self.mongo.upsert_paper(paper_meta)
        logger.info(f"[Fetcher] 论文元数据已存入 MongoDB")

        # 2. 下载并解析PDF
        if not paper_meta.get("pdf_url"):
            logger.info(f"[Fetcher] 论文无 PDF 链接，跳过")
            return

        logger.info(f"[Fetcher] 正在下载 PDF...")
        pdf_path = self._download_pdf(paper_meta["pdf_url"], arxiv_id)
        logger.info(f"[Fetcher] 正在解析 PDF...")
        parsed = self.parser.parse(pdf_path)

        # 3. 分块
        logger.info(f"[Fetcher] 正在分块...")
        chunks = self.parser.chunk(parsed["sections"])

        # 4. 更新MongoDB状态 + 存分块
        self.mongo.update_paper_status(arxiv_id, "parsed")
        mongo_chunks = [
            {
                "paper_arxiv_id": arxiv_id,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "metadata": c["metadata"],
            }
            for c in chunks
        ]
        self.mongo.insert_chunks(mongo_chunks)
        self.mongo.update_paper_status(arxiv_id, "chunked")
        logger.info(f"[Fetcher] 已生成 {len(chunks)} 个分块并存入 MongoDB")

        # 5. Embedding
        logger.info(f"[Fetcher] 正在生成 Embedding...")
        texts = [c["content"] for c in chunks]
        vectors = self.embedder.embed_texts(texts)
        logger.info(f"[Fetcher] Embedding 完成")

        # 6. 存Milvus
        import json
        milvus_records = [
            {
                "paper_arxiv_id": arxiv_id,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "embedding": vectors[i],
                "metadata_json": json.dumps(c["metadata"]),
            }
            for i, c in enumerate(chunks)
        ]
        self.milvus.insert(milvus_records)
        self.mongo.update_paper_status(arxiv_id, "indexed")
        logger.info(f"[Fetcher] 论文处理完成: {title}")

    def _download_pdf(self, url: str, arxiv_id: str) -> str:
        """下载PDF到本地临时目录"""
        import os
        import httpx

        tmp_dir = os.path.join(os.getcwd(), "tmp_pdfs")
        os.makedirs(tmp_dir, exist_ok=True)

        pdf_path = os.path.join(tmp_dir, f"{arxiv_id.replace('/', '_')}.pdf")

        if not os.path.exists(pdf_path):
            logger.info(f"[Fetcher] 正在下载 PDF: {url[:60]}...")
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
            logger.info(f"[Fetcher] PDF 下载完成: {pdf_path}")

        return pdf_path