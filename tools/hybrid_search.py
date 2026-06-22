"""
Paper Agent - Hybrid Search
Fast Path: BM25 + Vector 混合检索，RRF 融合
"""

import logging
import math
from collections import Counter
from typing import Optional

logger = logging.getLogger("paper-agent")


class BM25:
    """BM25 关键词检索"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0
        self.doc_freqs = {}  # term -> doc_count
        self.doc_lengths = []  # 每个文档的长度
        self.documents = []  # 原始文档
        self.term_freqs = []  # 每个文档的词频

    def fit(self, documents: list[str]):
        """构建 BM25 索引"""
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        for doc in documents:
            terms = doc.lower().split()
            self.doc_lengths.append(len(terms))

            # 统计词频
            tf = Counter(terms)
            self.term_freqs.append(tf)

            # 统计文档频率
            for term in set(terms):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        logger.info(f"[BM25] 索引构建完成: {self.doc_count} 文档, 平均长度 {self.avg_dl:.0f}")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """搜索，返回 (doc_index, score) 列表"""
        query_terms = query.lower().split()
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            dl = self.doc_lengths[i]
            tf = self.term_freqs[i]

            for term in query_terms:
                if term not in tf:
                    continue

                # IDF
                df = self.doc_freqs.get(term, 0)
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

                # TF
                term_tf = tf[term]
                tf_norm = (term_tf * (self.k1 + 1)) / (
                    term_tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )

                score += idf * tf_norm

            if score > 0:
                scores.append((i, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridSearch:
    """混合检索：BM25 + Vector，RRF 融合"""

    def __init__(self, bm25_weight: float = 0.3, vector_weight: float = 0.7, rrf_k: int = 60):
        """
        Args:
            bm25_weight: BM25 权重
            vector_weight: 向量检索权重
            rrf_k: RRF 常数（越大越平滑）
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25()

    def build_bm25_index(self, chunks: list[dict]):
        """构建 BM25 索引"""
        documents = [c.get("content", "") for c in chunks]
        self.bm25.fit(documents)

    def bm25_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """BM25 搜索"""
        return self.bm25.search(query, top_k)

    def rrf_fusion(self, bm25_results: list[tuple[int, float]],
                   vector_results: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF)
        将两路结果融合

        公式: score = sum(1 / (k + rank)) for each ranking
        """
        scores = {}

        # BM25 结果
        for rank, (doc_idx, _) in enumerate(bm25_results):
            if doc_idx not in scores:
                scores[doc_idx] = 0
            scores[doc_idx] += self.bm25_weight * (1 / (self.rrf_k + rank + 1))

        # Vector 结果
        for rank, (doc_idx, _) in enumerate(vector_results):
            if doc_idx not in scores:
                scores[doc_idx] = 0
            scores[doc_idx] += self.vector_weight * (1 / (self.rrf_k + rank + 1))

        # 按融合分数排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(self, query: str, chunks: list[dict],
               vector_results: list[tuple[int, float]],
               top_k: int = 10) -> list[dict]:
        """
        混合检索

        Args:
            query: 查询
            chunks: 所有 chunks
            vector_results: 向量检索结果 [(chunk_index, score), ...]
            top_k: 返回数量

        Returns:
            融合后的结果
        """
        # 1. BM25 搜索
        bm25_results = self.bm25_search(query, top_k=top_k * 2)

        # 2. RRF 融合
        fused = self.rrf_fusion(bm25_results, vector_results)

        # 3. 返回 top_k 结果
        results = []
        for doc_idx, score in fused[:top_k]:
            if doc_idx < len(chunks):
                chunk = chunks[doc_idx].copy()
                chunk["hybrid_score"] = round(score, 6)
                results.append(chunk)

        return results
