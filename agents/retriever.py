"""
Paper Agent - Retriever Agent
从Milvus语义检索相关论文分块
"""

from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage

from state.graph_state import AgentState, RetrievedChunk


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

        # 1. Embedding
        query_vector = self.embedder.encode(query)

        # 2. Milvus检索
        hits = self.milvus.search(
            query_embedding=query_vector,
            top_k=10,
            output_fields=["paper_arxiv_id", "chunk_index", "content", "metadata_json"],
        )

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

            retrieved.append(RetrievedChunk(
                paper_arxiv_id=arxiv_id,
                paper_title=paper["title"] if paper else arxiv_id,
                chunk_index=hit["chunk_index"],
                content=hit["content"],
                score=hit["score"],
                metadata={},
            ))

        # 4. 如果知识库为空，提示用户
        if not retrieved:
            return {
                "retrieved_chunks": [],
                "error": "知识库中未找到相关论文，请先使用fetcher入库论文。",
            }

        return {"retrieved_chunks": retrieved, "error": None}