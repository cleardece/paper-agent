"""
Paper Agent - Retriever Agent
Section-aware + MultiQuery + 两层检索（paper→chunk）
"""

import json
import logging
import re
import numpy as np
from typing import Optional

from state.graph_state import AgentState, RetrievedChunk
from core.cache import cache

logger = logging.getLogger("paper-agent")

# Section 意图映射：查询关键词 → 目标 section
SECTION_INTENT_MAP = {
    "贡献/创新/提出/新方法/ novelty": ["introduction", "conclusion", "abstract"],
    "方法/算法/模型/框架/ approach": ["methodology", "method", "approach"],
    "实验/结果/性能/准确率/ accuracy": ["results", "experiment", "evaluation"],
    "背景/动机/问题/现状": ["introduction", "background", "related work"],
    "结论/总结/未来工作": ["conclusion", "summary", "discussion"],
}


class RetrieverAgent:
    """语义检索 Agent - Section-aware + MultiQuery + 两层检索"""

    def __init__(self, embedding_service, milvus_client, mongodb_client, llm=None):
        self.embedder = embedding_service
        self.milvus = milvus_client
        self.mongo = mongodb_client
        self.llm = llm  # 用于 MultiQuery 生成

    def _detect_section_intent(self, query: str) -> Optional[list[str]]:
        """根据查询内容推断应该检索哪些 section"""
        query_lower = query.lower()

        # 检查是否包含特定 section 意图关键词
        for patterns, sections in SECTION_INTENT_MAP.items():
            for pattern in patterns.split("/"):
                if pattern in query_lower:
                    logger.info(f"[Retriever] Section 意图: {pattern} → {sections}")
                    return sections

        # 如果没有明确意图，不做过滤（返回 None）
        return None

    def _multi_query(self, query: str, session_id: str = None) -> list[str]:
        """MultiQuery: 用 LLM 生成多个查询变体"""
        # 如果没有 LLM，用规则降级
        if not self.llm:
            return self._rule_based_expand(query)

        # 检查缓存
        cache_key = f"multi_query:{query}"
        cached = cache.get(session_id, cache_key) if session_id else None
        if cached:
            return cached

        prompt = f"""你是一个查询优化专家。给定用户查询，生成3个不同角度的搜索变体，用于检索学术论文。

用户查询: {query}

要求:
1. 保持语义一致，但用不同的表述
2. 一个变体用更学术的表达
3. 一个变体用更具体的关键词
4. 一个变体用英文（如果原查询是中文）

只输出JSON数组，不要其他内容:
["变体1", "变体2", "变体3"]"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # 提取 JSON 数组
            import re
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                # 始终包含原始查询
                all_queries = [query] + [v for v in variants if v != query]
                logger.info(f"[Retriever] MultiQuery 生成 {len(all_queries)} 个变体")

                if session_id:
                    cache.set(session_id, cache_key, all_queries)
                return all_queries
        except Exception as e:
            logger.warning(f"[Retriever] MultiQuery LLM 降级: {e}")

        return self._rule_based_expand(query)

    def _rule_based_expand(self, query: str) -> list[str]:
        """规则降级的 query expansion"""
        expanded = [query]

        # "这篇论文" 类模糊查询
        if re.search(r"这篇论文|该论文|本文", query):
            core = re.sub(r"(这篇论文|该论文|本文)(的|主要|核心)?", "", query).strip()
            if core:
                expanded.append(f"论文 {core}")

        # "主要贡献是什么" → 追加变体
        if "主要贡献" in query:
            expanded.append(query.replace("主要贡献", "创新点和方法"))
            expanded.append(query.replace("主要贡献", "novelty and contribution"))

        # "区别/差异" → 追加对比变体
        if "区别" in query or "差异" in query:
            expanded.append(query.replace("区别", "对比"))
            expanded.append(query.replace("区别", "difference"))

        return expanded

    def _two_level_retrieval(self, query: str, query_vector, session_id: str = None) -> list[dict]:
        """两层检索：先找论文，再在论文内找 chunk"""

        # ===== Stage 1: 论文级检索 =====
        # 用查询找最相关的论文（通过标题和摘要的向量）
        logger.info("[Retriever] Stage 1: 论文级检索...")

        # 获取所有已入库论文
        all_papers = self.mongo.list_papers(limit=50)
        if not all_papers:
            return []

        # 对每篇论文计算与查询的相似度
        paper_scores = []
        for paper in all_papers:
            # 用论文标题 + 摘要生成向量
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            if not paper_text.strip():
                continue

            cache_key = f"paper_emb:{paper.get('arxiv_id', '')}"
            paper_vector = cache.get(session_id, cache_key) if session_id else None
            if not paper_vector:
                paper_vector = self.embedder.embed_query(paper_text)
                if session_id:
                    cache.set(session_id, cache_key, paper_vector)

            # 计算余弦相似度
            score = float(np.dot(query_vector, paper_vector) /
                         (np.linalg.norm(query_vector) * np.linalg.norm(paper_vector)))
            paper_scores.append({
                "arxiv_id": paper.get("arxiv_id"),
                "title": paper.get("title", ""),
                "score": score,
            })

        # 按分数排序，取 top-5 论文
        paper_scores.sort(key=lambda x: x["score"], reverse=True)
        top_papers = paper_scores[:5]
        top3_info = [(p['title'][:30], f"{p['score']:.3f}") for p in top_papers[:3]]
        logger.info(f"[Retriever] Top-5 论文 (前3): {top3_info}")

        # ===== Stage 2: Chunk级检索（限定在 top-5 论文内）=====
        logger.info("[Retriever] Stage 2: Chunk级检索（限定 top-5 论文）...")
        top_paper_ids = [p["arxiv_id"] for p in top_papers]

        hits = self.milvus.search(
            query_embedding=query_vector,
            top_k=10,
            paper_ids=top_paper_ids,
            output_fields=["paper_arxiv_id", "chunk_index", "content", "metadata_json"],
        )

        return hits

    def invoke(self, state: AgentState) -> dict:
        """
        检索流程：
        1. MultiQuery 生成变体
        2. Section 意图检测
        3. 两层检索：论文级 → Chunk级
        4. 补全元数据
        5. 更新state
        """
        query = state["user_query"]
        session_id = state.get("session_id")
        logger.info(f"[Retriever] 开始检索: {query[:50]}...")

        # 1. MultiQuery 生成变体
        expanded_queries = self._multi_query(query, session_id)
        logger.info(f"[Retriever] MultiQuery: {len(expanded_queries)} 个变体")

        # 2. Section 意图检测
        target_sections = self._detect_section_intent(query)

        # 3. 对每个变体做 embedding + 检索，合并去重
        all_hits = {}  # key: (arxiv_id, chunk_index) -> hit

        for eq in expanded_queries:
            # Embedding
            cache_key = f"embedding:{eq}"
            cached_vector = cache.get(session_id, cache_key) if session_id else None
            if cached_vector:
                query_vector = cached_vector
            else:
                query_vector = self.embedder.embed_query(eq)
                if session_id:
                    cache.set(session_id, cache_key, query_vector)

            # 两层检索
            hits = self._two_level_retrieval(eq, query_vector, session_id)

            # 合并结果
            for h in hits:
                key = (h["paper_arxiv_id"], h["chunk_index"])
                if key not in all_hits or h["score"] > all_hits[key]["score"]:
                    all_hits[key] = h

        # 4. 按 score 降序排列，取 top-15
        hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)[:15]
        logger.info(f"[Retriever] 合并去重后: {len(hits)} 个分块")

        # 5. Section 过滤（仅对明确意图生效，且过滤后至少保留3个结果）
        if target_sections and hits and len(hits) > 3:
            filtered = []
            for h in hits:
                metadata = {}
                if h.get("metadata_json"):
                    try:
                        metadata = json.loads(h["metadata_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                chunk_section = metadata.get("section", "").lower()
                # 如果 chunk 有 section 信息且匹配目标，优先保留
                if chunk_section and any(s in chunk_section for s in target_sections):
                    h["_section_match"] = True
                    filtered.append(h)
                elif not chunk_section:
                    # 没有 section 信息的 chunk 也保留（兜底）
                    filtered.append(h)

            # 只有过滤后有足够结果时才使用过滤结果，否则用原始结果
            if len(filtered) >= 3:
                hits = filtered[:15]
                logger.info(f"[Retriever] Section 过滤后: {len(hits)} 个分块")
            else:
                logger.info(f"[Retriever] Section 过滤结果不足({len(filtered)})，使用原始结果")

        # 6. 补全元数据 + 构造 RetrievedChunk
        retrieved = []
        seen_papers = set()

        for hit in hits:
            arxiv_id = hit["paper_arxiv_id"]

            # 查论文标题
            if arxiv_id not in seen_papers:
                paper = self.mongo.get_paper(arxiv_id)
                seen_papers.add(arxiv_id)
            else:
                paper = None

            # 解析 metadata_json
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

        # 7. 如果知识库为空，提示用户
        if not retrieved:
            return {
                "retrieved_chunks": [],
                "error": "知识库中未找到相关论文，请先使用fetcher入库论文。",
            }

        return {"retrieved_chunks": retrieved, "error": None}
