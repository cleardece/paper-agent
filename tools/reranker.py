"""
Paper Agent - Reranker
Fast Path: 用 Cross-Encoder 对检索结果重排序
"""

import logging
from typing import Optional

logger = logging.getLogger("paper-agent")


class Reranker:
    """重排序器 - 用 Cross-Encoder 提升精度"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Args:
            model_name: Cross-Encoder 模型名
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """懒加载模型"""
        if self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"[Reranker] 加载模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("[Reranker] 模型加载完成")
        except Exception as e:
            logger.error(f"[Reranker] 模型加载失败: {e}")
            self.model = None

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        重排序

        Args:
            query: 查询
            chunks: 候选 chunks
            top_k: 返回数量

        Returns:
            重排序后的 chunks
        """
        if not chunks:
            return []

        self.load_model()
        if self.model is None:
            # 模型不可用，直接返回原始顺序
            logger.warning("[Reranker] 模型不可用，跳过重排序")
            return chunks[:top_k]

        # 构建 query-document 对
        pairs = [(query, c.get("content", "")) for c in chunks]

        # 计算相关性分数
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"[Reranker] 预测失败: {e}")
            return chunks[:top_k]

        # 按分数排序
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k
        results = []
        for chunk, score in scored_chunks[:top_k]:
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(score)
            results.append(chunk_copy)

        logger.info(f"[Reranker] 重排序完成: {len(chunks)} → {top_k}")
        return results
