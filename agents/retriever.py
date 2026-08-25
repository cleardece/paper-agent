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
    """语义检索 Agent - Section-aware + MultiQuery + 两层检索 + 混合搜索 + 重排序"""

    def __init__(self, embedding_service, milvus_client, mongodb_client, llm=None,
                 hybrid_search=None, research_graph=None):
        self.embedder = embedding_service
        self.milvus = milvus_client
        self.mongo = mongodb_client
        self.llm = llm  # 用于 MultiQuery 生成
        self.hybrid_search = hybrid_search
        self.research_graph = research_graph

    @staticmethod
    def _has_graph_intent(query: str) -> bool:
        return any(token in query.lower() for token in (
            "哪些论文", "共同", "比较", "对比", "关系", "关联", "使用", "uses", "compare",
        ))

    def _graph_paper_ids(self, query: str) -> list[str]:
        """图谱只提供候选范围；事实仍由后续 chunk 检索提供。"""
        if not self.research_graph or not self._has_graph_intent(query):
            return []
        phrases = re.findall(r"[A-Za-z][A-Za-z0-9_-]*(?:\\s+[A-Za-z][A-Za-z0-9_-]*){0,5}", query)
        for phrase in sorted(phrases, key=len, reverse=True):
            paper_ids = self.research_graph.find_related_paper_ids(phrase)
            if paper_ids:
                logger.info("[Retriever] 图谱定位到 %d 篇候选论文: %s", len(paper_ids), phrase[:40])
                return paper_ids
        return []

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

    def _multi_query(self, query: str, session_id: str = None, user_interests: list[str] = None) -> list[str]:
        """MultiQuery: 用 LLM 生成多个查询变体"""
        # 如果没有 LLM，用规则降级
        if not self.llm:
            return self._rule_based_expand(query)

        # 检查缓存
        cache_key = f"multi_query:{query}"
        cached = cache.get(session_id, cache_key) if session_id else None
        if cached:
            return cached

        # 构建用户上下文
        user_context = ""
        if user_interests:
            user_context = f"\n用户研究兴趣: {', '.join(user_interests[:5])}"

        prompt = f"""你是一个查询优化专家。给定用户查询，生成3个不同角度的搜索变体，用于检索学术论文。

用户查询: {query}{user_context}

要求:
1. 保持语义一致，但用不同的表述
2. 一个变体用更学术的表达
3. 一个变体用更具体的关键词
4. 一个变体用英文（如果原查询是中文）
5. 考虑用户的研究兴趣

只输出JSON数组，不要其他内容:
["变体1", "变体2", "变体3"]"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # 提取 JSON 数组
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

    def _two_level_retrieval(self, query: str, query_vector, session_id: str = None,
                             target_paper: str = None, graph_paper_ids: list[str] | None = None) -> list[dict]:
        """两层检索：先找论文，再在论文内找 chunk"""

        # ===== Stage 1: 论文级检索（Milvus HNSW 向量搜索）=====
        logger.info("[Retriever] Stage 1: 论文级检索（Milvus）...")

        # 图谱命中只限制候选论文，不绕过 chunk 检索和重排序。
        top_papers = None
        if graph_paper_ids:
            top_papers = []
            for paper_id in graph_paper_ids:
                paper = self.mongo.get_paper(paper_id)
                if paper:
                    top_papers.append({"arxiv_id": paper_id, "title": paper.get("title", ""), "score": 1.0})

        # 优先用目标论文直接匹配
        if target_paper:
            # 从 MongoDB 查论文 ID
            all_papers = self.mongo.list_papers(
                limit=50, projection={"arxiv_id": 1, "title": 1},
            )
            target_lower = target_paper.lower()
            for paper in all_papers:
                title = paper.get("title", "").lower()
                if any(kw in title for kw in target_lower.split() if len(kw) > 3):
                    logger.info(f"[Retriever] 直接匹配目标论文: {paper.get('title', '')[:50]}")
                    hits = self.milvus.search(
                        query_embedding=query_vector,
                        top_k=15,
                        paper_ids=[paper["arxiv_id"]],
                        output_fields=["paper_arxiv_id", "chunk_index", "content", "section", "page", "heading"],
                    )
                    return hits

        # Milvus 论文级向量搜索（HNSW 近似最近邻，替代 Python 逐篇计算）
        if top_papers is None:
            top_papers = self.milvus.search_papers(query_embedding=query_vector, top_k=5)
        if not top_papers:
            logger.info("[Retriever] Milvus 论文级搜索无结果，尝试 MongoDB 兜底...")
            # 兜底：如果 paper_embeddings collection 为空（旧数据未迁移），走 MongoDB
            all_papers = self.mongo.list_papers(
                limit=50, projection={"arxiv_id": 1, "title": 1, "title_embedding": 1},
            )
            all_papers = [p for p in all_papers if p.get("title_embedding")]
            if not all_papers:
                return []
            # 用 numpy 计算（仅兜底路径）
            paper_scores = []
            for paper in all_papers:
                paper_vector = paper["title_embedding"]
                score = float(np.dot(query_vector, paper_vector) /
                             (np.linalg.norm(query_vector) * np.linalg.norm(paper_vector)))
                paper_scores.append({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper.get("title", ""),
                    "score": score,
                })
            paper_scores.sort(key=lambda x: x["score"], reverse=True)
            top_papers = paper_scores[:5]

        top3_info = [(p['title'][:30], f"{p['score']:.3f}") for p in top_papers[:3]]
        logger.info(f"[Retriever] Top-5 论文 (前3): {top3_info}")

        # ===== Stage 2: Chunk级检索（限定在 top-5 论文内）=====
        logger.info("[Retriever] Stage 2: Chunk级检索（限定 top-5 论文）...")
        top_paper_ids = [p["arxiv_id"] for p in top_papers]

        vector_hits = self.milvus.search(
            query_embedding=query_vector,
            top_k=10,
            paper_ids=top_paper_ids,
            output_fields=["paper_arxiv_id", "chunk_index", "content", "metadata_json"],
        )

        # ===== Stage 3: 混合搜索（BM25 + Vector RRF 融合）=====
        if self.hybrid_search and vector_hits:
            logger.info("[Retriever] Stage 3: BM25 + Vector 混合搜索...")
            # 获取这些论文的所有 chunks 用于 BM25
            all_chunks = []
            for pid in top_paper_ids:
                chunks = list(self.mongo.get_chunks_by_paper(pid))
                all_chunks.extend(chunks)

            if all_chunks:
                # BM25 索引缓存：用论文 ID 列表的 hash 作为 key
                import hashlib
                bm25_cache_key = hashlib.md5(str(sorted(top_paper_ids)).encode()).hexdigest()
                if getattr(self.hybrid_search, '_cache_key', None) != bm25_cache_key:
                    self.hybrid_search.build_bm25_index(all_chunks)
                    self.hybrid_search._cache_key = bm25_cache_key
                    logger.info(f"[BM25] 索引重建: {len(all_chunks)} chunks")
                else:
                    logger.info(f"[BM25] 使用缓存索引")

                # BM25 搜索
                bm25_results = self.hybrid_search.bm25_search(query, top_k=10)

                # 向量结果转为 (doc_idx, score) 格式（用字典 O(1) 查找）
                chunk_index_map = {}
                for idx, c in enumerate(all_chunks):
                    key = (c.get("paper_arxiv_id"), c.get("chunk_index"))
                    chunk_index_map[key] = idx

                vector_results = []
                for h in vector_hits:
                    key = (h["paper_arxiv_id"], h.get("chunk_index"))
                    idx = chunk_index_map.get(key)
                    if idx is not None:
                        vector_results.append((idx, h.get("score", 0)))

                # RRF 融合
                fused = self.hybrid_search.rrf_fusion(bm25_results, vector_results)

                # 转换回 hit 格式
                fused_hits = []
                for doc_idx, score in fused[:10]:
                    if doc_idx < len(all_chunks):
                        c = all_chunks[doc_idx]
                        meta = c.get("metadata", {})
                        fused_hits.append({
                            "paper_arxiv_id": c.get("paper_arxiv_id"),
                            "chunk_index": c.get("chunk_index", 0),
                            "content": c.get("content", ""),
                            "section": meta.get("section", ""),
                            "page": meta.get("page", 0),
                            "heading": meta.get("heading", ""),
                            "score": score,
                        })

                logger.info(f"[Retriever] 混合搜索完成: BM25 {len(bm25_results)} + Vector {len(vector_hits)} → 融合 {len(fused_hits)}")
                return fused_hits

        return vector_hits

    def _extract_target_paper(self, query: str, context: str = "") -> str:
        """从查询和对话上下文中提取用户指代的论文标题"""
        # 检查是否包含跟随意图词
        # 注意："它" 会误匹配 "它们"，用 "它的" / "这篇" 等更精确的模式
        followup_patterns = ["这篇论文", "该论文", "它的", "上一篇", "刚才的", "之前"]
        is_followup = any(p in query for p in followup_patterns)
        if not is_followup:
            return None

        # 从对话上下文中提取最近讨论的论文标题
        if context:
            # 匹配 "助手: ..." 中提到的论文标题（英文标题）
            titles = re.findall(r'[A-Z][a-zA-Z\s\-:]{5,80}', context)
            if titles:
                # 返回最后一个出现的英文标题（最近讨论的）
                return titles[-1].strip()

        # 从知识库中匹配最近的论文
        try:
            papers = self.mongo.list_papers(limit=5, projection={"title": 1})
            if papers:
                return papers[-1].get("title", "")
        except Exception:
            pass

        return None

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
        user_id = state.get("user_id", "default")
        context = state.get("conversation_context", "")
        logger.info(f"[Retriever] 开始检索: {query[:50]}...")

        # 提取用户指代的论文
        target_paper = self._extract_target_paper(query, context)
        if target_paper:
            logger.info(f"[Retriever] 识别到目标论文: {target_paper[:50]}")
        graph_paper_ids = self._graph_paper_ids(query)

        # 获取用户兴趣
        user_interests = self.mongo.user_memory.get_interests(user_id, top_k=5)

        # 1. MultiQuery 生成变体
        expanded_queries = self._multi_query(query, session_id, user_interests)
        logger.info(f"[Retriever] MultiQuery: {len(expanded_queries)} 个变体")

        # 处理用户交互记忆
        self.mongo.user_memory.process_interaction(user_id, query)

        # 2. Section 意图检测
        target_sections = self._detect_section_intent(query)

        # 3. 批量 embedding 所有变体，再逐个检索
        all_hits = {}  # key: (arxiv_id, chunk_index) -> hit

        # 先批量编码所有变体（一次 GPU 调用）
        uncached_queries = []
        uncached_indices = []
        query_vectors = [None] * len(expanded_queries)

        for i, eq in enumerate(expanded_queries):
            cache_key = f"embedding:{eq}"
            cached_vector = cache.get(session_id, cache_key) if session_id else None
            if cached_vector:
                query_vectors[i] = cached_vector
            else:
                uncached_queries.append(eq)
                uncached_indices.append(i)

        if uncached_queries:
            new_vectors = self.embedder.embed_texts(uncached_queries)
            for idx, vec in zip(uncached_indices, new_vectors):
                query_vectors[idx] = vec
                if session_id:
                    cache_key = f"embedding:{expanded_queries[idx]}"
                    cache.set(session_id, cache_key, vec)

        # 逐个检索（embedding 已经批量完成）
        for eq, query_vector in zip(expanded_queries, query_vectors):
            # 两层检索
            hits = self._two_level_retrieval(
                eq, query_vector, session_id, target_paper, graph_paper_ids or None
            )

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
                chunk_section = h.get("section", "").lower()
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

            # 构造 metadata（从独立字段）
            metadata = {
                "section": hit.get("section", ""),
                "page": hit.get("page", 0),
                "heading": hit.get("heading", ""),
            }

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
                "graph_context": {"paper_ids": graph_paper_ids} if graph_paper_ids else None,
            }

        return {
            "retrieved_chunks": retrieved,
            "error": None,
            "graph_context": {"paper_ids": graph_paper_ids} if graph_paper_ids else None,
        }
