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

    def invoke(self, state: AgentState) -> dict:
        """
        检索流程：
        1. query → embedding
        2. Milvus语义检索 → top-k chunks
        3. MongoDB补全论文标题等元数据
        4. 更新state
        """
        query = state["user_query"]
        logger.info(f"[Retriever] 开始检索: {query[:50]}...")

        # 1. Embedding（带缓存）
        cache_key = f"embedding:{query}"
        cached_vector = cache.get(state.get("session_id"), cache_key)
        if cached_vector:
            logger.info("[Retriever] 使用缓存向量")
            query_vector = cached_vector
        else:
            logger.info("[Retriever] 正在生成查询向量...")
            query_vector = self.embedder.embed_query(query)
            cache.set(state.get("session_id"), cache_key, query_vector)
            logger.info("[Retriever] 向量生成完成")

        # 2. Milvus检索
        logger.info("[Retriever] 正在 Milvus 中检索...")
        hits = self.milvus.search(
            query_embedding=query_vector,
            top_k=10,
            output_fields=["paper_arxiv_id", "chunk_index", "content", "metadata_json"],
        )
        logger.info(f"[Retriever] 找到 {len(hits)} 个相关分块")

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