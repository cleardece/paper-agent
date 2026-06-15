"""
Paper Agent - Retriever Agent
从Milvus语义检索相关论文分块
"""

import json
import logging
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage

from state.graph_state import AgentState, RetrievedChunk
from core.cache import cache

logger = logging.getLogger("paper-agent")


class RetrieverAgent:
    """语义检索Agent"""

    def __init__(self, embedding_service, milvus_client, mongodb_client):
        self.embedder = embedding_service
        self.milvus = milvus_client
        self.mongo = mongodb_client

    def _expand_query(self, query: str) -> list[str]:
        """Query expansion: 对模糊查询生成多个变体"""
        import re
        expanded = [query]

        # "这篇论文" 类模糊查询 → 追加通用变体
        if re.search(r"这篇论文|该论文|本文", query):
            expanded.append(query.replace("这篇论文", "论文"))
            # 追加更具体的变体
            core = re.sub(r"(这篇论文|该论文|本文)(的|主要|核心)?", "", query).strip()
            if core:
                expanded.append(f"论文 {core}")

        # "主要贡献是什么" → 追加 "创新点 方法 贡献"
        if "主要贡献" in query:
            expanded.append(query.replace("主要贡献", "创新点和方法"))

        return expanded

    def invoke(self, state: AgentState) -> dict:
        """
        检索流程：
        1. query → expansion → embedding
        2. Milvus语义检索 → top-k chunks
        3. 去重 + 按score排序
        4. MongoDB补全论文标题等元数据
        5. 更新state
        """
        query = state["user_query"]
        logger.info(f"[Retriever] 开始检索: {query[:50]}...")

        # 1. Query expansion
        expanded_queries = self._expand_query(query)
        logger.info(f"[Retriever] Query expansion: {len(expanded_queries)} 个变体")

        # 2. 对每个变体做 embedding + 检索，合并去重
        all_hits = {}  # key: (arxiv_id, chunk_index) -> hit
        for eq in expanded_queries:
            cache_key = f"embedding:{eq}"
            cached_vector = cache.get(state.get("session_id"), cache_key)
            if cached_vector:
                query_vector = cached_vector
            else:
                query_vector = self.embedder.embed_query(eq)
                cache.set(state.get("session_id"), cache_key, query_vector)

            hits = self.milvus.search(
                query_embedding=query_vector,
                top_k=12,
                output_fields=["paper_arxiv_id", "chunk_index", "content", "metadata_json"],
            )

            for h in hits:
                key = (h["paper_arxiv_id"], h["chunk_index"])
                if key not in all_hits or h["score"] > all_hits[key]["score"]:
                    all_hits[key] = h

        # 3. 按 score 降序排列，取 top-15
        hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)[:15]
        logger.info(f"[Retriever] 合并去重后: {len(hits)} 个分块")

        # 3. 补全元数据 + 构造RetrievedChunk
        retrieved = []
        seen_papers = set()

        for hit in hits:
            arxiv_id = hit["paper_arxiv_id"]

            # 查论文标题
            if arxiv_id not in seen_papers:
                paper = self.mongo.get_paper(arxiv_id)
                seen_papers.add(arxiv_id)
            else:
                paper = None  # 已查过，跳过重复查询

            # 解析 metadata_json
            import json
            metadata = {}
            if hit.get("metadata_json"):
                try:
                    metadata = json.loads(hit["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            retrieved.append(RetrievedChunk(
                paper_arxiv_id=arxiv_id,
                paper_title=paper["title"] if paper else arxiv_id,
                chunk_index=hit["chunk_index"],
                content=hit["content"],
                score=hit["score"],
                metadata=metadata,
            ))

        # 4. 如果知识库为空，提示用户
        if not retrieved:
            return {
                "retrieved_chunks": [],
                "error": "知识库中未找到相关论文，请先使用fetcher入库论文。",
            }

        return {"retrieved_chunks": retrieved, "error": None}