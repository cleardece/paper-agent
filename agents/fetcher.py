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
        cached_papers = cache.get(state.get("session_id"), cache_key)
        if cached_papers:
            logger.info(f"[Fetcher] 使用缓存，找到 {len(cached_papers)} 篇论文")
            papers = cached_papers
        else:
            logger.info("[Fetcher] 正在调用 arXiv API...")
            papers = self.arxiv.search(query, max_results=5)
            logger.info(f"[Fetcher] 找到 {len(papers)} 篇论文")
            # 缓存搜索结果
            if papers:
                cache.set(state.get("session_id"), cache_key, papers)

        # 2-4. 逐篇处理（串行，避免资源爆炸）
        fetched = []
        already_exists = []
        failed = []
        no_pdf = []

        for i, paper_meta in enumerate(papers):
            arxiv_id = paper_meta["arxiv_id"]

            # 跳过已入库的（任何状态）
            if self.mongo.paper_exists(arxiv_id):
                already_exists.append(paper_meta)
                continue

            # 检查是否有 PDF
            if not paper_meta.get("pdf_url"):
                no_pdf.append(paper_meta)
                continue

            try:
                logger.info(f"[Fetcher] 处理论文 {i+1}/{len(papers)}: {paper_meta.get('title', '')[:40]}...")
                self._process_paper(paper_meta)
                fetched.append(paper_meta)

                # 仅在 GPU 模式下清理显存（CPU 模式无需）
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            except Exception as e:
                logger.error(f"[Fetcher] 处理失败 {arxiv_id}: {e}")
                failed.append({"title": paper_meta.get("title", ""), "error": str(e)})

        # 5. 构建详细的反馈信息
        total = len(papers)
        summary_parts = []
        summary_parts.append(f"搜索到 {total} 篇论文：")

        if fetched:
            summary_parts.append(f"\n[成功入库 {len(fetched)} 篇]")
            for p in fetched:
                summary_parts.append(f"  - {p['title'][:60]}")

        if already_exists:
            summary_parts.append(f"\n[已存在 {len(already_exists)} 篇，跳过]")
            for p in already_exists:
                summary_parts.append(f"  - {p['title'][:60]}")

        if no_pdf:
            summary_parts.append(f"\n[无PDF链接 {len(no_pdf)} 篇，无法入库]")
            for p in no_pdf:
                summary_parts.append(f"  - {p['title'][:60]}")

        if failed:
            summary_parts.append(f"\n[处理失败 {len(failed)} 篇]")
            for f in failed:
                summary_parts.append(f"  - {f['title'][:40]}: {f['error'][:40]}")

        summary = "\n".join(summary_parts)
        logger.info(f"[Fetcher] 结果汇总:\n{summary}")

        # 返回结果
        if fetched:
            return {
                "target_papers": fetched,
                "next_agent": "retriever",
                "error": None,
                "answer": summary,
            }
        else:
            return {
                "target_papers": [],
                "next_agent": "presenter",
                "error": None,
                "answer": summary + "\n\n建议：稍后重试，或手动从 arXiv 搜索。",
            }

    def _process_paper(self, paper_meta: dict, session_id: str = None):
        """单篇论文完整处理链路"""
        arxiv_id = paper_meta["arxiv_id"]
        title = paper_meta.get('title', '未知')[:40]
        logger.info(f"[Fetcher] 正在处理论文: {title}...")

        # 检查缓存（会话级）
        cache_key = f"paper:{arxiv_id}"
        cached = None
        if session_id:
            cached = cache.get(session_id, cache_key)
        if cached:
            logger.info(f"[Fetcher] 使用缓存: {arxiv_id}")
            chunks = cached["chunks"]
        else:
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

            # 缓存分块结果（会话级）
            if session_id:
                cache.set(session_id, cache_key, {"chunks": chunks})

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
        try:
            logger.info(f"[Fetcher] 正在生成 Embedding...")
            texts = [c["content"] for c in chunks]
            vectors = self.embedder.embed_texts(texts)
            logger.info(f"[Fetcher] Embedding 完成")

            # 同时计算论文级 embedding（标题+摘要），持久化到 MongoDB + Milvus
            paper_text = f"{paper_meta.get('title', '')} {paper_meta.get('abstract', '')}"
            if paper_text.strip():
                paper_emb = self.embedder.embed_texts([paper_text])[0]
                self.mongo.update_paper_status(arxiv_id, "indexed", title_embedding=paper_emb)
                # 写入 Milvus 论文级 collection（用于 Stage 1 快速排序）
                self.milvus.insert_paper_embedding(
                    arxiv_id=arxiv_id,
                    title=paper_meta.get("title", ""),
                    embedding=paper_emb,
                )
        except Exception as e:
            logger.error(f"[Fetcher] Embedding 生成失败: {e}")
            self.mongo.update_paper_status(arxiv_id, "embed_failed")
            raise

        # 6. 存Milvus
        try:
            milvus_records = [
                {
                    "paper_arxiv_id": arxiv_id,
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "embedding": vectors[i],
                    "section": c.get("metadata", {}).get("section", ""),
                    "page": c.get("metadata", {}).get("page", 0),
                    "heading": c.get("metadata", {}).get("heading", ""),
                }
                for i, c in enumerate(chunks)
            ]
            self.milvus.insert(milvus_records)
            self.mongo.update_paper_status(arxiv_id, "indexed")
            logger.info(f"[Fetcher] 论文处理完成: {title}")
        except Exception as e:
            logger.error(f"[Fetcher] Milvus 存储失败: {e}")
            self.mongo.update_paper_status(arxiv_id, "milvus_failed")
            raise

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